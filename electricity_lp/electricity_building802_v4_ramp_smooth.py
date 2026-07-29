# -*- coding: utf-8 -*-
"""
Building 802 전기 부하 최적화 v4 — ramp 제약 + smoothness penalty 추가
====================================================================
v3(고정/유연부하 분리 scipy.linprog)에서 나온 결과가 시간대 경계에서
급격히 튀는(지그재그) 비현실적인 형태였던 문제를 해결하기 위해,
아래 두 가지를 LP에 추가한 버전.

  (1) Ramp 제약 (하드 제약)
      : 인접 시간대 간 사용량 변화폭을 |load_t - load_(t-1)| <= ramp_limit 로 강제.
        실제 설비/공조는 물리적으로 한 시간 만에 조 단위로 널뛰기를 할 수 없으므로,
        "최적해가 존재할 수 있는 영역" 자체를 현실적인 범위로 좁히는 역할.

  (2) Smoothness penalty (소프트 페널티, 목적함수에 추가)
      : ramp 제약만으로는 "허용 범위 안에서 계단식으로 급변"하는 해도 여전히
        나올 수 있음. 인접 시간대 변화량의 절대값 합(총변동, total variation)에
        비용을 매겨 목적함수에 더함으로써, 비용 최적해들 중 "더 매끄러운" 해를
        선호하도록 만듦.

  scipy.linprog는 절대값을 직접 다루지 못하므로, 표준적인 LP 선형화 기법으로
  보조변수 diff_t >= |optimized_t - optimized_(t-1)| 를 도입한다
  (diff_t >= x, diff_t >= -x 두 부등식 + 목적함수에서 diff_t를 최소화 방향으로 사용
   -> 최적해에서는 diff_t가 자동으로 |x|와 같아짐).

변수 벡터 x = [flexible_0..23, P, excess, diff_0..23]  (총 24+2+24 = 50개)
  - flexible_t : t시각에 배치하는 유연부하량
  - P          : 최대부하(peak demand)
  - excess     : 계약전력 초과분
  - diff_t     : t와 t-1(wrap-around로 23->0 포함) 사이 총부하 변화량의 절대값 상한
"""

import json
import platform
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager


def set_korean_font():
    system = platform.system()
    if system == "Darwin":
        candidates = ["AppleGothic", "Apple SD Gothic Neo"]
    elif system == "Windows":
        candidates = ["Malgun Gothic"]
    else:
        candidates = ["Noto Sans CJK KR", "NanumGothic", "Noto Sans CJK JP"]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return name
    for f in font_manager.fontManager.ttflist:
        if any(k in f.name for k in ("Gothic", "Nanum", "CJK", "Malgun", "Apple")):
            matplotlib.rcParams["font.family"] = f.name
            return f.name
    print("[경고] 한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
    return None


set_korean_font()
matplotlib.rcParams["axes.unicode_minus"] = False

ARTIFACT_DIR = "artifacts"
TARGET_BUILDING = 802
SEASON = "summer"

# ====================================================================
# 1. 데이터 불러오기 + 평일 대표 프로필 (v3와 동일)
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
# 2. 계절별 TOU 요금 + 수요요금 + 계약전력 (v3와 동일)
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
demand_rate_monthly = 8300
billing_days = 30
demand_rate = demand_rate_monthly / billing_days
contract_power = predicted_t.mean() * 1.15
overage_penalty_rate = demand_rate * 3


# ====================================================================
# 3. 고정부하 / 유연부하 분리 (v3와 동일)
# ====================================================================
fixed_ratio = 0.60
cap_multiplier = 3.0

fixed_t = predicted_t * fixed_ratio
flexible_pool_t = predicted_t * (1 - fixed_ratio)
flexible_pool_total = flexible_pool_t.sum()
max_hourly_flexible = (flexible_pool_total / 24) * cap_multiplier


# ====================================================================
# 4. [NEW] Ramp 제약 & Smoothness penalty 하이퍼파라미터
# ====================================================================
# ramp_limit: 한 시간 사이에 허용하는 "총부하(fixed+flexible)" 최대 변화폭(kWh).
#   -> 계약전력의 일정 비율로 정의. 너무 빡빡하면 유연부하를 원하는 시간대로
#      옮길 여지 자체가 사라지므로(=최적화가 무력화), 실무적으로는
#      15~25% 선에서 시작해 조정하는 걸 권장.
ramp_limit_pct = 0.20
ramp_limit = contract_power * ramp_limit_pct

# smoothness_weight: 인접 시간대 변화량 1kWh당 부과하는 가상의 "불안정 비용"(원/kWh).
#   -> 에너지요금 단가(85~223원/kWh)와 같은 축 위에 있는 값이므로,
#      가격 단가보다 너무 크게 잡으면 TOU 차익거래 효과 자체가 사라져
#      "그냥 평평한 그래프"가 나온다. 절감 효과와 매끄러움 사이의 트레이드오프이므로
#      0(끔) ~ 가격 단가 수준까지 스윕하며 감도분석 하는 걸 권장.
smoothness_weight = 8.0

# wrap-around 여부: 대표 하루가 "매일 반복"된다고 보고 23시->0시 연결도
# ramp/smoothness 제약에 포함할지 여부. 대표 프로필이라는 전제상 True 권장.
USE_WRAP_AROUND = True


# ====================================================================
# 5. scipy.linprog 정식화
#    변수 순서: [flexible_0..23, P, excess, diff_0..23]  (n+2+n_diff)
# ====================================================================
transitions = [(t, t - 1) for t in range(1, n)]
if USE_WRAP_AROUND:
    transitions.append((0, n - 1))  # 23시 -> 0시(다음날)
n_diff = len(transitions)

IDX_P = n
IDX_EXCESS = n + 1
IDX_DIFF0 = n + 2
n_vars = n + 2 + n_diff

# ---- 목적함수 ----
c = np.zeros(n_vars)
c[:n] = prices
c[IDX_P] = demand_rate
c[IDX_EXCESS] = overage_penalty_rate
c[IDX_DIFF0:IDX_DIFF0 + n_diff] = smoothness_weight

# ---- 등식 제약: 유연부하 풀 총량 보존 ----
A_eq = np.zeros((1, n_vars))
A_eq[0, :n] = 1
b_eq = [flexible_pool_total]

# ---- 부등식 제약 A_ub @ x <= b_ub ----
rows = []
rhs = []

# (a) fixed_t + flexible_t - P <= 0
for t in range(n):
    row = np.zeros(n_vars)
    row[t] = 1
    row[IDX_P] = -1
    rows.append(row)
    rhs.append(-fixed_t[t])

# (b) P - excess <= contract_power
row = np.zeros(n_vars)
row[IDX_P] = 1
row[IDX_EXCESS] = -1
rows.append(row)
rhs.append(contract_power)

# (c) [NEW] ramp 하드 제약: -ramp_limit <= optimized_t - optimized_(t-1) <= ramp_limit
#     optimized_t - optimized_(t-1) = (flexible_t - flexible_(t-1)) + (fixed_t - fixed_(t-1))
for (t, tp) in transitions:
    fixed_diff = fixed_t[t] - fixed_t[tp]
    # flexible_t - flexible_tp <= ramp_limit - fixed_diff
    row = np.zeros(n_vars)
    row[t] += 1
    row[tp] -= 1
    rows.append(row)
    rhs.append(ramp_limit - fixed_diff)
    # -(flexible_t - flexible_tp) <= ramp_limit + fixed_diff
    row = np.zeros(n_vars)
    row[t] -= 1
    row[tp] += 1
    rows.append(row)
    rhs.append(ramp_limit + fixed_diff)

# (d) [NEW] smoothness 선형화: diff_k >= |optimized_t - optimized_tp|
for k, (t, tp) in enumerate(transitions):
    fixed_diff = fixed_t[t] - fixed_t[tp]
    diff_idx = IDX_DIFF0 + k
    # optimized_t - optimized_tp - diff_k <= 0
    row = np.zeros(n_vars)
    row[t] += 1
    row[tp] -= 1
    row[diff_idx] -= 1
    rows.append(row)
    rhs.append(-fixed_diff)
    # -(optimized_t - optimized_tp) - diff_k <= 0
    row = np.zeros(n_vars)
    row[t] -= 1
    row[tp] += 1
    row[diff_idx] -= 1
    rows.append(row)
    rhs.append(fixed_diff)

A_ub = np.vstack(rows)
b_ub = np.array(rhs)

# ---- 변수별 상하한 ----
bounds = [(0, max_hourly_flexible)] * n + [(0, None), (0, None)] + [(0, None)] * n_diff

res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
if not res.success:
    raise RuntimeError(
        f"LP 최적화 실패: {res.message}\n"
        f"-> ramp_limit({ramp_limit:.1f}kWh)이 너무 빡빡하지 않은지 확인하세요. "
        f"고정부하 자체의 시간당 변화폭보다 ramp_limit이 작으면 실행 불가능한 문제가 됩니다."
    )

flexible_opt = res.x[:n]
P_opt = res.x[IDX_P]
excess_opt = res.x[IDX_EXCESS]
optimized_t = fixed_t + flexible_opt


# ====================================================================
# 6. Before/After 비용 분해 + 매끄러움 지표
# ====================================================================
def cost_breakdown(load_arr):
    energy_cost = float((load_arr * prices).sum())
    peak = float(load_arr.max())
    excess_v = max(0.0, peak - contract_power)
    demand_cost = demand_rate * peak
    overage_cost = overage_penalty_rate * excess_v
    return energy_cost, demand_cost, overage_cost, energy_cost + demand_cost + overage_cost, peak, excess_v


def ramp_stats(load_arr):
    diffs = np.abs(np.diff(np.append(load_arr, load_arr[0]) if USE_WRAP_AROUND else load_arr))
    return float(diffs.max()), float(diffs.mean()), float(diffs.sum())


e_before, d_before, o_before, total_before, peak_before, excess_before = cost_breakdown(predicted_t)
e_after, d_after, o_after, total_after, peak_after, _ = cost_breakdown(optimized_t)
ramp_max_before, ramp_mean_before, tv_before = ramp_stats(predicted_t)
ramp_max_after, ramp_mean_after, tv_after = ramp_stats(optimized_t)

print("=" * 70)
print(f"전기 v4 — ramp 제약 + smoothness penalty LP 결과 (건물: {TARGET_BUILDING}, 계절: {SEASON})")
print("=" * 70)
print(f"고정부하 비율: {fixed_ratio:.0%} | 유연부하 풀 총량: {flexible_pool_total:.1f} kWh/일")
print(f"계약전력: {contract_power:.1f} kW | ramp_limit: {ramp_limit:.1f} kWh/h "
      f"({ramp_limit_pct:.0%} of 계약전력) | smoothness_weight: {smoothness_weight}")
print("-" * 70)
print(f"{'항목':<12}{'Before':>15}{'After':>15}")
print(f"{'에너지요금':<12}{e_before:>15,.0f}{e_after:>15,.0f}")
print(f"{'수요요금':<12}{d_before:>15,.0f}{d_after:>15,.0f}")
print(f"{'초과페널티':<12}{o_before:>15,.0f}{o_after:>15,.0f}")
print("-" * 70)
print(f"{'총비용':<12}{total_before:>15,.0f}{total_after:>15,.0f}")
print(f"\n총 절감액: {total_before - total_after:,.0f}원 "
      f"({(1 - total_after/total_before)*100:.2f}%)")
print(f"최대부하(peak): {peak_before:.1f} -> {peak_after:.1f} kWh")
print(f"계약전력 초과분(excess): {excess_before:.1f} -> {excess_opt:.1f} kWh")
print("-" * 70)
print("[매끄러움 지표 — 시간당 변화폭]")
print(f"  최대 ramp : {ramp_max_before:.1f} -> {ramp_max_after:.1f} kWh/h  (제약 상한 {ramp_limit:.1f})")
print(f"  평균 ramp : {ramp_mean_before:.1f} -> {ramp_mean_after:.1f} kWh/h")
print(f"  총변동(TV): {tv_before:.1f} -> {tv_after:.1f} kWh")
print("=" * 70)


# ====================================================================
# 7. Before / After 시각화 (ramp 상한 밴드 추가)
# ====================================================================
hours_arr = np.arange(24)


def get_tou_bands():
    off_peak_hours = list(range(23, 24)) + list(range(0, 9))
    peak_price_hours = list(range(10, 12)) + list(range(13, 17))
    mid_hours = [h for h in range(24) if h not in off_peak_hours and h not in peak_price_hours]
    return off_peak_hours, peak_price_hours, mid_hours


off_peak_hours, peak_price_hours, mid_hours = get_tou_bands()

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(12, 10.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1]}
)

band_colors = {"off": "#E1F5EE", "mid": "#FAEEDA", "peak": "#FAECE7"}
for h in range(24):
    c_ = band_colors["off"] if h in off_peak_hours else (
        band_colors["peak"] if h in peak_price_hours else band_colors["mid"])
    ax1.axvspan(h - 0.5, h + 0.5, color=c_, zorder=0)

ax1.plot(hours_arr, predicted_t, "o-", color="#993C1D", linewidth=2,
          label="Before (LightGBM 예측)")
ax1.plot(hours_arr, optimized_t, "o-", color="#0F6E56", linewidth=2,
          label="After (v4 LP: ramp+smoothness)")
ax1.fill_between(hours_arr, fixed_t, color="gray", alpha=0.18,
                   label=f"고정부하(fixed, {fixed_ratio:.0%})")
ax1.axhline(contract_power, color="#1F4E8C", linestyle="--", linewidth=1.4,
            label=f"계약전력={contract_power:.0f}kW")
ax1.axhline(P_opt, color="#0F6E56", linestyle=":", linewidth=1.4,
            label=f"최적화 후 최대부하(P)={P_opt:.0f}kW")

ax1.set_ylabel("전력사용량 (kWh)")
ax1.set_title(f"Building {TARGET_BUILDING} · 전기 부하 재배치 — v4 (ramp 제약 + smoothness penalty)")
ax1.legend(loc="upper left", fontsize=8.5)
ax1.grid(alpha=0.3)

# 시간당 변화폭(ramp) 비교
diff_before = np.diff(np.append(predicted_t, predicted_t[0]))
diff_after = np.diff(np.append(optimized_t, optimized_t[0]))
ax2.axhline(ramp_limit, color="crimson", linestyle="--", linewidth=1.2, label=f"ramp_limit=±{ramp_limit:.0f}")
ax2.axhline(-ramp_limit, color="crimson", linestyle="--", linewidth=1.2)
ax2.bar(hours_arr - 0.15, diff_before, width=0.3, color="#C97B5A", label="Before Δ")
ax2.bar(hours_arr + 0.15, diff_after, width=0.3, color="#3E9C82", label="After Δ")
ax2.set_ylabel("시간당 변화량\n(kWh/h)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

ax3.step(hours_arr, prices, where="mid", color="#5F5E5A")
ax3.set_ylabel("에너지요금\n(원/kWh)")
ax3.set_xlabel("시간(hour)")
ax3.set_xticks(hours_arr)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("electricity_building802_v4_ramp_smooth.png", dpi=150)
print("\n차트 저장 완료: electricity_building802_v4_ramp_smooth.png")
