# -*- coding: utf-8 -*-
"""
Building 802 전기 부하 최적화 v3-scipy — 고정/유연부하 분리를 scipy.linprog로 구현
====================================================================
electricity_building802_v3_fixed_flexible.py(PuLP 버전)와 수학적으로 완전히 동일한
문제를 scipy.optimize.linprog로 풀도록 변환한 버전.

변수 벡터 x = [flexible_0, flexible_1, ..., flexible_23, P, excess]  (총 26개)
  - flexible_t : t시각에 배치하는 유연부하량 (0 ~ max_hourly_flexible)
  - P          : 최대부하(peak demand)
  - excess     : 계약전력 초과분

목적함수 (최소화):
  Σ price_t * flexible_t + demand_rate * P + overage_penalty_rate * excess
  (fixed_t * price_t 는 상수라 최적화에 영향 없으므로 목적함수에서는 생략하고,
   최종 비용 리포트에서만 다시 더해준다)
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linprog

ARTIFACT_DIR = "artifacts"
TARGET_BUILDING = 802
SEASON = "summer"

# ====================================================================
# 1. 데이터 불러오기 + 평일 대표 프로필 (기존과 동일)
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
weekday = b[b["dow"] < 5].copy()
pred_log = booster.predict(weekday[features_m], num_iteration=booster.best_iteration)
weekday["predicted_kwh"] = np.expm1(pred_log).clip(min=0)
predicted_t = weekday.groupby("hour")["predicted_kwh"].mean().values
n = 24


# ====================================================================
# 2. 계절별 TOU 요금 + 수요요금 + 계약전력 (동일)
# ====================================================================
def get_tou_price_table(season):
    base = {"off": 85, "mid": 115, "peak": 165}
    m = {"summer": 1.35, "winter": 1.25, "spring_fall": 1.00}[season]

    def tier(h):
        if 10 <= h < 12 or 13 <= h < 17:
            return base["peak"] * m
        elif 9 <= h < 10 or 12 <= h < 13 or 17 <= h < 23:
            return base["mid"] * m
        return base["off"] * m
    return np.array([tier(h) for h in range(24)])


prices = get_tou_price_table(SEASON)
demand_rate = 8300
contract_power = predicted_t.mean() * 1.15
overage_penalty_rate = demand_rate * 3


# ====================================================================
# 3. 고정부하 / 유연부하 분리 (동일)
# ====================================================================
fixed_ratio = 0.60
cap_multiplier = 3.0

fixed_t = predicted_t * fixed_ratio
flexible_pool_t = predicted_t * (1 - fixed_ratio)
flexible_pool_total = flexible_pool_t.sum()
max_hourly_flexible = (flexible_pool_total / 24) * cap_multiplier


# ====================================================================
# 4. scipy.linprog 정식화
#    변수 순서: [flexible_0..23, P, excess]  (n+2 = 26개)
# ====================================================================
n_vars = n + 2
IDX_P = n
IDX_EXCESS = n + 1

# 목적함수 계수
c = np.zeros(n_vars)
c[:n] = prices               # flexible_t 계수 = 시간대별 단가
c[IDX_P] = demand_rate
c[IDX_EXCESS] = overage_penalty_rate

# 등식 제약: 유연부하 풀 총량 보존
A_eq = np.zeros((1, n_vars))
A_eq[0, :n] = 1
b_eq = [flexible_pool_total]

# 부등식 제약 A_ub @ x <= b_ub
#  (a) fixed_t + flexible_t - P <= 0   ->  flexible_t - P <= -fixed_t   (24개 행)
#  (b) P - excess <= contract_power                                     (1개 행)
A_ub = np.zeros((n + 1, n_vars))
b_ub = np.zeros(n + 1)

for t in range(n):
    A_ub[t, t] = 1
    A_ub[t, IDX_P] = -1
    b_ub[t] = -fixed_t[t]

A_ub[n, IDX_P] = 1
A_ub[n, IDX_EXCESS] = -1
b_ub[n] = contract_power

# 변수별 상하한
bounds = [(0, max_hourly_flexible)] * n + [(0, None), (0, None)]

res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
if not res.success:
    raise RuntimeError(f"LP 최적화 실패: {res.message}")

flexible_opt = res.x[:n]
P_opt = res.x[IDX_P]
excess_opt = res.x[IDX_EXCESS]
optimized_t = fixed_t + flexible_opt


# ====================================================================
# 5. Before/After 비용 분해 (PuLP 버전과 동일한 방식)
# ====================================================================
def cost_breakdown(load_arr):
    energy_cost = float((load_arr * prices).sum())
    peak = float(load_arr.max())
    excess_v = max(0.0, peak - contract_power)
    demand_cost = demand_rate * peak
    overage_cost = overage_penalty_rate * excess_v
    return energy_cost, demand_cost, overage_cost, energy_cost + demand_cost + overage_cost, peak, excess_v


e_before, d_before, o_before, total_before, peak_before, excess_before = cost_breakdown(predicted_t)
e_after, d_after, o_after, total_after, peak_after, _ = cost_breakdown(optimized_t)

print("=" * 66)
print(f"전기 v3-scipy — 고정/유연부하 분리 LP 결과 (건물: {TARGET_BUILDING}, 계절: {SEASON})")
print("=" * 66)
print(f"고정부하 비율: {fixed_ratio:.0%} | 유연부하 풀 총량: {flexible_pool_total:.1f} kWh/일")
print(f"계약전력: {contract_power:.1f} kW / 수요요금단가: {demand_rate:,}원/kW")
print("-" * 66)
print(f"{'항목':<12}{'Before':>15}{'After':>15}")
print(f"{'에너지요금':<12}{e_before:>15,.0f}{e_after:>15,.0f}")
print(f"{'수요요금':<12}{d_before:>15,.0f}{d_after:>15,.0f}")
print(f"{'초과페널티':<12}{o_before:>15,.0f}{o_after:>15,.0f}")
print("-" * 66)
print(f"{'총비용':<12}{total_before:>15,.0f}{total_after:>15,.0f}")
print(f"\n총 절감액: {total_before - total_after:,.0f}원 "
      f"({(1 - total_after/total_before)*100:.2f}%)")
print(f"최대부하(peak): {peak_before:.1f} -> {peak_after:.1f} kWh "
      f"({(1 - peak_after/peak_before)*100:.2f}% 감소)")
print(f"계약전력 초과분(excess): {excess_before:.1f} -> {excess_opt:.1f} kWh")
print("=" * 66)
