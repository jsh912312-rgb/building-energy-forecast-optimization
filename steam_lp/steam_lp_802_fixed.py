import pulp
import pandas as pd

predicted_t = [
    1711.3, 1721.3, 1721.1, 1756.1, 1847.8, 1972.7, 2439.0, 3274.0,
    3333.4, 3011.8, 2766.2, 2637.3, 2569.7, 2541.6, 2602.7, 2561.5,
    2482.9, 2337.4, 2257.3, 1845.8, 1734.7, 1774.1, 1748.3, 1692.4,
]

hours = range(24)
baseline_min = min(predicted_t) * 0.95
ramp_up_limit = 950
ramp_down_limit = 200

# 핵심 추가 제약: 각 시간대 실제 필요 열량 대비 이 이상은 못 벗어남
# (건물 예열 여유 정도를 반영 — 여기선 실제 필요량의 15%까지만 유연하다고 가정)
DEVIATION_RATIO = 0.15

prob = pulp.LpProblem("steam_peak_shaving_802_fixed", pulp.LpMinimize)
load = pulp.LpVariable.dicts("load", hours, lowBound=0)
peak = pulp.LpVariable("peak_load", lowBound=0)

prob += peak

prob += pulp.lpSum(load[t] for t in hours) == sum(predicted_t)

for t in hours:
    prob += load[t] >= baseline_min
    prob += peak >= load[t]
    prev = (t - 1) % 24
    prob += load[t] - load[prev] <= ramp_up_limit
    prob += load[prev] - load[t] <= ramp_down_limit
    # deviation 제약: 실제 필요 열량에서 너무 못 벗어남
    prob += load[t] <= predicted_t[t] * (1 + DEVIATION_RATIO)
    prob += load[t] >= predicted_t[t] * (1 - DEVIATION_RATIO)

prob.solve(pulp.PULP_CBC_CMD(msg=0))

result = pd.DataFrame({
    "hour": list(hours),
    "predicted_kwh": [round(p, 1) for p in predicted_t],
    "optimized_kwh": [round(load[t].value(), 1) for t in hours],
})
result["diff"] = (result["optimized_kwh"] - result["predicted_kwh"]).round(1)

print(result.to_string(index=False))
print(f"\n원본 peak: {max(predicted_t):.1f} kWh")
print(f"최적화 후 peak: {peak.value():.1f} kWh")
print(f"peak 완화율: {(1 - peak.value()/max(predicted_t))*100:.1f}%")
