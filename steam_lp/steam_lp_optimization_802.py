"""
Building 802 steam 부하 최적화 — ramp-rate 제약 LP
----------------------------------------------------
설계 근거 (실측 데이터 기반):
  - baseline ~1750 kWh, peak ~3334 kWh (h8), peak/baseline = 1.95
  - 상승 구간(h6->h8): 약 900 kWh/hour 상승
  - 하강 구간(h8->h20): 약 133 kWh/hour 하강 (훨씬 완만)
  - 설비 단위 sub-metering이 없어 MILP(on/off) 대신,
    "시간당 변화량 자체를 제약"하는 LP로 재정의

목적함수: Minimize peak_load (일일 최대 부하 자체를 낮춤)
변수: load_t  (t=0..23, 조정된 시간별 부하)
제약:
  - Sum(load_t) = Sum(predicted_t)          하루 총 사용량 보존
  - load_t >= baseline_min                   최소 운영량 이하로 못 내림
  - load_t - load_(t-1) <= ramp_up_limit     시간당 최대 상승폭 제한
  - load_(t-1) - load_t <= ramp_down_limit   시간당 최대 하강폭 제한
  - peak_load >= load_t (모든 t)             peak_load는 최대값
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import pulp

ARTIFACT_DIR = "artifacts"
TARGET_BUILDING = 802

# ---- 1. 예측값 로드 (겨울철 대표일 기준) ----
with open(f"{ARTIFACT_DIR}/meta.json", encoding="utf-8") as f:
    meta = json.load(f)

df = pd.read_csv(f"{ARTIFACT_DIR}/df_features.csv", parse_dates=["timestamp"])
booster = lgb.Booster(model_file=f"{ARTIFACT_DIR}/model_steam.txt")
feature_cols = meta["feature_cols"]
features_m = [c for c in feature_cols if c != "meter"]

b = df[(df["building_id"] == TARGET_BUILDING) & (df["meter"] == 2)].copy()
b = b.dropna(subset=features_m).sort_values("timestamp")
pred_log = booster.predict(b[features_m], num_iteration=booster.best_iteration)
b["predicted_kwh"] = np.expm1(pred_log).clip(min=0)

winter = b[b["air_temperature"] < 0]
predicted_t = winter.groupby("hour")["predicted_kwh"].mean().values  # 24개 값 (대표일 프로필)

# ---- 2. 물리적 제약 상수 (실측 데이터로부터 도출) ----
baseline_min = predicted_t.min() * 0.95   # 최소 운영량은 관측 최저치의 95%까지만 허용
ramp_up_limit = 950     # h6->h8 실측 상승폭(약 900) + 여유
ramp_down_limit = 200   # h8->h20 실측 하강폭(약 133) + 여유

# ---- 3. LP 정식화 ----
hours = range(24)
prob = pulp.LpProblem("steam_peak_shaving_802", pulp.LpMinimize)

load = pulp.LpVariable.dicts("load", hours, lowBound=0)
peak = pulp.LpVariable("peak_load", lowBound=0)

prob += peak  # 목적함수: peak 최소화

# 총 사용량 보존
prob += pulp.lpSum(load[t] for t in hours) == sum(predicted_t)

for t in hours:
    prob += load[t] >= baseline_min
    prob += peak >= load[t]
    prev = (t - 1) % 24
    prob += load[t] - load[prev] <= ramp_up_limit
    prob += load[prev] - load[t] <= ramp_down_limit

prob.solve(pulp.PULP_CBC_CMD(msg=0))

result = pd.DataFrame({
    "hour": list(hours),
    "predicted_kwh": predicted_t.round(1),
    "optimized_kwh": [round(load[t].value(), 1) for t in hours],
})

print(result.to_string(index=False))
print(f"\n원본 peak: {predicted_t.max():.1f} kWh")
print(f"최적화 후 peak: {peak.value():.1f} kWh")
print(f"peak 완화율: {(1 - peak.value()/predicted_t.max())*100:.1f}%")
