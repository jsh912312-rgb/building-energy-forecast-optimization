# -*- coding: utf-8 -*-
"""
Building 802 전기 부하 최적화 v2 — 계절별 TOU 요금 + 수요요금 + 계약전력 초과페널티
====================================================================================
v1(기존 코드)과의 차이:
  v1: minimize Σ(price_t * load_t)                              -- 에너지요금만
  v2: minimize Σ(price_t * load_t) + demand_rate*P + penalty*excess
      -- 에너지요금 + 수요요금(최대부하 kW 기준) + 계약전력 초과 페널티

핵심 아이디어:
  - P(peak demand)를 LP 변수로 새로 도입하고, load_t <= P 를 모든 시간에 강제하면
    P는 자동으로 '그날의 최댓값'이 된다 (LP로 max를 표현하는 표준 기법).
  - P에 수요요금 단가를 곱해 목적함수에 더하면, LP가 "피크를 낮추는 것 자체"에
    경제적 동기를 갖게 되어, v1에서 썼던 인위적 peak_cap 제약이 필요 없어진다.
  - 계약전력을 넘긴 만큼(excess)에는 훨씬 비싼 할증 단가를 매겨, 계약전력 초과가
    최적해에서 자연스럽게 회피되도록 만든다.

TOU 단가·수요요금·계약전력 값은 전부 근사치(placeholder)이며,
실제 사업장의 계약종별 최신 고시 요금표로 반드시 교체해야 한다.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import pulp

ARTIFACT_DIR = "artifacts"
TARGET_BUILDING = 802
SEASON = "summer"  # 'summer' / 'winter' / 'spring_fall' 중 선택

# ====================================================================
# 1. 데이터 불러오기 + building 802 평일 대표 프로필 추출 (기존 코드 그대로)
# ====================================================================
with open(f"{ARTIFACT_DIR}/meta.json", encoding="utf-8") as f:
    meta = json.load(f)
df = pd.read_csv(f"{ARTIFACT_DIR}/df_features.csv", parse_dates=["timestamp"])
booster = lgb.Booster(model_file=f"{ARTIFACT_DIR}/model_electricity.txt")
feature_cols = meta["feature_cols"]
features_m = [c for c in feature_cols if c != "meter"]

b = df[(df["building_id"] == TARGET_BUILDING) & (df["meter"] == 0)].copy()
b = b.dropna(subset=features_m).sort_values("timestamp")
b["dow"] = b["timestamp"].dt.dayofweek
weekday = b[b["dow"] < 5]

pred_log = booster.predict(weekday[features_m], num_iteration=booster.best_iteration)
weekday = weekday.copy()
weekday["predicted_kwh"] = np.expm1(pred_log).clip(min=0)

predicted_t = weekday.groupby("hour")["predicted_kwh"].mean().values  # 평일 대표 프로필 24개 값


# ====================================================================
# 2. 계절별 · 시간대별 TOU 에너지요금 (원/kWh 상대값)
#    한전 산업용(을) 고압 요금 체계를 단순화: 여름/겨울철이 봄가을철보다 비쌈
# ====================================================================
def get_tou_price_table(season: str) -> np.ndarray:
    base = {"off": 85, "mid": 115, "peak": 165}  # 봄가을철 기준 단가 (원/kWh, 근사치)
    season_multiplier = {"summer": 1.35, "winter": 1.25, "spring_fall": 1.00}
    if season not in season_multiplier:
        raise ValueError("season은 'summer'/'winter'/'spring_fall' 중 하나")
    m = season_multiplier[season]

    def tier(hour):
        if 10 <= hour < 12 or 13 <= hour < 17:
            return base["peak"] * m
        elif 9 <= hour < 10 or 12 <= hour < 13 or 17 <= hour < 23:
            return base["mid"] * m
        else:
            return base["off"] * m

    return np.array([tier(h) for h in range(24)])


prices = get_tou_price_table(SEASON)


# ====================================================================
# 3. 수요요금 · 계약전력 파라미터 (근사치 — 실제 계약종별 요금표로 교체 필요)
# ====================================================================
demand_rate = 8300                          # 원/kW, 최대부하 1kW당 수요요금 단가
contract_power = predicted_t.mean() * 1.15  # 계약전력(kW), 평균부하의 1.15배로 근사 설정
overage_penalty_rate = demand_rate * 3      # 계약전력 초과분은 3배 할증


# ====================================================================
# 4. 물리/운영 제약 (기존 코드와 동일한 논리)
# ====================================================================
baseline_min = predicted_t.min() * 0.95
SHIFT_RATIO = 0.15  # 무한 재배치 방지, 시간당 이동 가능 폭을 예측치의 ±15%로 제한


# ====================================================================
# 5. LP 정식화 (PuLP) — 변수: load_t(24개) + P(최대부하) + excess(초과분)
# ====================================================================
hours = range(24)
prob = pulp.LpProblem("electricity_v2_demand_charge_802", pulp.LpMinimize)

load = pulp.LpVariable.dicts("load", hours, lowBound=0)
P = pulp.LpVariable("peak_demand", lowBound=0)
excess = pulp.LpVariable("contract_excess", lowBound=0)

# 목적함수: 에너지요금 + 수요요금 + 초과페널티
prob += (
    pulp.lpSum(prices[t] * load[t] for t in hours)
    + demand_rate * P
    + overage_penalty_rate * excess
)

# 총 사용량 보존
prob += pulp.lpSum(load[t] for t in hours) == sum(predicted_t)

for t in hours:
    prob += load[t] >= baseline_min
    prob += load[t] <= predicted_t[t] * (1 + SHIFT_RATIO)
    prob += load[t] >= predicted_t[t] * (1 - SHIFT_RATIO)
    prob += load[t] <= P                       # P가 최댓값이 되도록 강제

prob += excess >= P - contract_power            # 계약전력 초과분 정의

prob.solve(pulp.PULP_CBC_CMD(msg=0))

optimized_t = np.array([load[t].value() for t in hours])
P_opt = P.value()
excess_opt = excess.value()


# ====================================================================
# 6. Before/After 비용 분해 (에너지요금 / 수요요금 / 초과페널티 / 총비용)
# ====================================================================
def cost_breakdown(load_arr, prices, demand_rate, contract_power, overage_penalty_rate):
    energy_cost = float((load_arr * prices).sum())
    peak = float(load_arr.max())
    excess_v = max(0.0, peak - contract_power)
    demand_cost = demand_rate * peak
    overage_cost = overage_penalty_rate * excess_v
    total = energy_cost + demand_cost + overage_cost
    return energy_cost, demand_cost, overage_cost, total, peak, excess_v


e_before, d_before, o_before, total_before, peak_before, excess_before = cost_breakdown(
    predicted_t, prices, demand_rate, contract_power, overage_penalty_rate)
e_after, d_after, o_after, total_after, peak_after, _ = cost_breakdown(
    optimized_t, prices, demand_rate, contract_power, overage_penalty_rate)


# ====================================================================
# 7. 결과 출력 (요청하신 포맷 그대로)
# ====================================================================
print("=" * 60)
print(f"전기 트랙 v2 — 수요요금 포함 LP 최적화 결과 (건물: {TARGET_BUILDING}, 계절: {SEASON})")
print("=" * 60)
print(f"계약전력: {contract_power:.1f} kW / 수요요금단가: {demand_rate:,}원/kW")
print("-" * 60)
print(f"{'항목':<12}{'Before':>15}{'After':>15}")
print(f"{'에너지요금':<12}{e_before:>15,.0f}{e_after:>15,.0f}")
print(f"{'수요요금':<12}{d_before:>15,.0f}{d_after:>15,.0f}")
print(f"{'초과페널티':<12}{o_before:>15,.0f}{o_after:>15,.0f}")
print("-" * 60)
print(f"{'총비용':<12}{total_before:>15,.0f}{total_after:>15,.0f}")
print(f"\n총 절감액: {total_before - total_after:,.0f}원 "
      f"({(1 - total_after/total_before)*100:.2f}%)")
print(f"최대부하(peak): {peak_before:.1f} -> {peak_after:.1f} kWh "
      f"({(1 - peak_after/peak_before)*100:.2f}% 감소)")
print(f"계약전력 초과분(excess): {excess_before:.1f} -> {excess_opt:.1f} kWh")
print("=" * 60)

result = pd.DataFrame({
    "hour": list(hours),
    "price_tier": prices,
    "predicted_kwh": predicted_t.round(1),
    "optimized_kwh": optimized_t.round(1),
})
print("\n" + result.to_string(index=False))
