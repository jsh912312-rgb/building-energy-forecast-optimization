# -*- coding: utf-8 -*-
"""
Building 802 전기 최적화 — SHIFT_RATIO 민감도 분석
====================================================================
목적: v2 결과에서 계약전력 초과분(excess)이 158.0 -> 70.3으로 줄었지만
      완전히 0이 되지 않은 이유가 'SHIFT_RATIO=0.15가 valley 시간대의
      부하 흡수 여력을 너무 좁게 제한했기 때문'인지 확인한다.

      SHIFT_RATIO를 0.15부터 점진적으로 늘려가며 동일한 LP를 반복 실행해,
      - 어느 지점에서 excess가 0이 되는지
      - 그 지점의 shift_ratio가 물리적으로 현실적인 값인지(예: 50% 이상이면
        비현실적 -> "시간 이동만으로는 해결 불가"라는 결론이 타당해짐)
      를 판단하기 위한 근거를 만든다.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import pulp

ARTIFACT_DIR = "artifacts"
TARGET_BUILDING = 802
SEASON = "summer"

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
baseline_min = predicted_t.min() * 0.95


def solve_for_shift_ratio(shift_ratio):
    hours = range(24)
    prob = pulp.LpProblem("sensitivity", pulp.LpMinimize)
    load = pulp.LpVariable.dicts("load", hours, lowBound=0)
    P = pulp.LpVariable("P", lowBound=0)
    excess = pulp.LpVariable("excess", lowBound=0)

    prob += (pulp.lpSum(prices[t] * load[t] for t in hours)
             + demand_rate * P + overage_penalty_rate * excess)
    prob += pulp.lpSum(load[t] for t in hours) == sum(predicted_t)
    for t in hours:
        prob += load[t] >= baseline_min
        prob += load[t] <= predicted_t[t] * (1 + shift_ratio)
        prob += load[t] >= predicted_t[t] * (1 - shift_ratio)
        prob += load[t] <= P
    prob += excess >= P - contract_power

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    optimized = np.array([load[t].value() for t in hours])
    total_cost = (pulp.value(prob.objective))
    return optimized.max(), excess.value(), total_cost


print(f"계약전력: {contract_power:.1f} kW (참고: 원래 예측 최대부하 = {predicted_t.max():.1f} kW)")
print("=" * 70)
print(f"{'shift_ratio':>12}{'최적화 후 peak':>18}{'excess(초과분)':>18}{'총비용(원)':>16}")
print("-" * 70)

results = []
for ratio in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00]:
    peak, excess_v, cost = solve_for_shift_ratio(ratio)
    results.append((ratio, peak, excess_v, cost))
    flag = "  <- 계약전력 이하 달성" if excess_v < 0.5 else ""
    print(f"{ratio:>12.2f}{peak:>18.1f}{excess_v:>18.1f}{cost:>16,.0f}{flag}")

print("=" * 70)
