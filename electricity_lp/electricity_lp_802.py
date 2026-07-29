"""
Building 802 전기 부하 최적화 — TOU 요금 기반 LP
---------------------------------------------------
목적함수: Minimize Σ (price_t x load_t)   -- 전기요금 최소화
변수: load_t (t=0~23, 조정된 시간별 부하)
제약:
  Σload_t = Σpredicted_t              하루 총 사용량 보존 (옮기기만 함)
  load_t >= baseline_min                최소 운영량 이하로 못 내림
  |load_t - predicted_t| <= shift_limit  급격한 이동 방지 (연속변수라 ramp 대신 이 형태로 충분:
                                          전기는 저장 없이 즉시 소비라 물리적 관성이 없음)

TOU 단가는 실제 계약 요금표가 없어 가정치 사용 (한국 산업용 고압 통상 3단계 구조 참고)
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import pulp

ARTIFACT_DIR = "artifacts"
TARGET_BUILDING = 802

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

# ---- TOU 단가 (가정치, 상대값) ----
def tou_price(hour):
    if 10 <= hour < 12 or 13 <= hour < 17:
        return 2.0   # 최대부하
    elif 9 <= hour < 10 or 12 <= hour < 13 or 17 <= hour < 23:
        return 1.5   # 중간부하
    else:
        return 1.0   # 경부하

prices = [tou_price(h) for h in range(24)]

baseline_min = predicted_t.min() * 0.95
SHIFT_RATIO = 0.15  # steam과 동일한 논리: 무한 재배치 방지, 유연성 가정치

hours = range(24)
prob = pulp.LpProblem("electricity_cost_min_802", pulp.LpMinimize)
load = pulp.LpVariable.dicts("load", hours, lowBound=0)

prob += pulp.lpSum(prices[t] * load[t] for t in hours)
prob += pulp.lpSum(load[t] for t in hours) == sum(predicted_t)

for t in hours:
    prob += load[t] >= baseline_min
    prob += load[t] <= predicted_t[t] * (1 + SHIFT_RATIO)
    prob += load[t] >= predicted_t[t] * (1 - SHIFT_RATIO)

prob.solve(pulp.PULP_CBC_CMD(msg=0))

result = pd.DataFrame({
    "hour": list(hours),
    "price_tier": prices,
    "predicted_kwh": predicted_t.round(1),
    "optimized_kwh": [round(load[t].value(), 1) for t in hours],
})

before_cost = sum(prices[t] * predicted_t[t] for t in hours)
after_cost = sum(prices[t] * load[t].value() for t in hours)

print(result.to_string(index=False))
print(f"\n원본 요금(상대값): {before_cost:.1f}")
print(f"최적화 후 요금(상대값): {after_cost:.1f}")
print(f"요금 절감율: {(1 - after_cost/before_cost)*100:.1f}%")
