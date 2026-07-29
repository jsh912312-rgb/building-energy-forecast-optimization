import pulp

predicted_t = [
    1711.3, 1721.3, 1721.1, 1756.1, 1847.8, 1972.7, 2439.0, 3274.0,
    3333.4, 3011.8, 2766.2, 2637.3, 2569.7, 2541.6, 2602.7, 2561.5,
    2482.9, 2337.4, 2257.3, 1845.8, 1734.7, 1774.1, 1748.3, 1692.4,
]
hours = range(24)
baseline_min = min(predicted_t) * 0.95
ramp_up_limit = 950
ramp_down_limit = 200

for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    prob = pulp.LpProblem("s", pulp.LpMinimize)
    load = pulp.LpVariable.dicts("load", hours, lowBound=0)
    peak = pulp.LpVariable("peak", lowBound=0)
    prob += peak
    prob += pulp.lpSum(load[t] for t in hours) == sum(predicted_t)
    for t in hours:
        prob += load[t] >= baseline_min
        prob += peak >= load[t]
        prev = (t - 1) % 24
        prob += load[t] - load[prev] <= ramp_up_limit
        prob += load[prev] - load[t] <= ramp_down_limit
        prob += load[t] <= predicted_t[t] * (1 + ratio)
        prob += load[t] >= predicted_t[t] * (1 - ratio)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    reduction = (1 - peak.value() / max(predicted_t)) * 100
    print(f"deviation_ratio={ratio:.2f} -> peak={peak.value():.1f}, 완화율={reduction:.1f}%")
