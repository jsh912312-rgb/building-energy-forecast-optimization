"""
전기(meter=0) 최적화용 building 선정 스크리닝 — 전체 building 대상
--------------------------------------------------------------------
문서에 나온 후보 리스트는 사용하지 않고, meter=0 전체 building을 대상으로
증기 트랙과 동일한 방식(전수 스크리닝 후 필터링)으로 새로 선정.

전기 트랙 선정 기준 (증기와 다름):
  - 온도 상관관계 대신 "평일/주말 패턴 차이"가 핵심
    (전기 부하는 온도보다 업무시간/운영시간에 더 좌우되기 때문)
  - peak_to_mean으로 이동 가능한 뾰족한 피크가 있는지 확인
  - completeness, site 노이즈(특히 site 13, site 0 단위이슈) 체크
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ARTIFACT_DIR = "artifacts"
ELECTRICITY_METER_ID = 0
BAD_SITES = [13]  # steam에서 확인된 노이즈 사이트. site 0은 별도 단위이슈 체크 필요(아래 참고)


def load_artifacts():
    with open(f"{ARTIFACT_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(f"{ARTIFACT_DIR}/df_features.csv", parse_dates=["timestamp"])
    booster = lgb.Booster(model_file=f"{ARTIFACT_DIR}/model_electricity.txt")
    return df, booster, meta


def screen_electricity_buildings(df: pd.DataFrame, booster: lgb.Booster, meta: dict, top_n: int = 20) -> pd.DataFrame:
    feature_cols = meta["feature_cols"]
    features_m = [c for c in feature_cols if c != "meter"]

    elec = df[df["meter"] == ELECTRICITY_METER_ID].copy()
    elec = elec.dropna(subset=features_m)

    if elec.empty:
        raise ValueError("meter==0 데이터가 없습니다. df_features.csv 확인 필요")

    pred_log = booster.predict(elec[features_m], num_iteration=booster.best_iteration)
    elec["predicted_kwh"] = np.expm1(pred_log).clip(min=0)
    elec["dow"] = elec["timestamp"].dt.dayofweek
    elec["is_weekend"] = elec["dow"] >= 5

    full_span = df.groupby("building_id")["timestamp"].agg(lambda x: x.max() - x.min())

    rows = []
    for bid, g in elec.groupby("building_id"):
        n_hours = len(g)
        span = full_span.get(bid, pd.Timedelta(0))
        expected_hours = span.total_seconds() / 3600 + 1 if span else np.nan
        completeness = n_hours / expected_hours if expected_hours else np.nan
        if n_hours < 24 * 60:  # 최소 두 달치는 있어야 신뢰 가능
            continue

        weekday_mean = g.loc[~g["is_weekend"], "predicted_kwh"].mean()
        weekend_mean = g.loc[g["is_weekend"], "predicted_kwh"].mean()
        weekday_weekend_gap = (weekday_mean - weekend_mean) / weekend_mean if weekend_mean > 0 else np.nan

        mean_load = g["predicted_kwh"].mean()
        p95_load = g["predicted_kwh"].quantile(0.95)
        peak_to_mean = p95_load / mean_load if mean_load > 0 else np.nan

        site = g["site_id"].iloc[0]

        rows.append({
            "building_id": bid,
            "site_id": site,
            "n_hours": n_hours,
            "completeness": round(completeness, 3) if completeness == completeness else np.nan,
            "weekday_weekend_gap": round(weekday_weekend_gap, 2) if weekday_weekend_gap == weekday_weekend_gap else np.nan,
            "peak_to_mean": round(peak_to_mean, 2),
            "mean_predicted_kwh": round(mean_load, 1),
        })

    result = pd.DataFrame(rows)

    # 절대 사용량 왜곡 방지 (steam 스크리닝 때와 동일한 이유)
    min_load = result["mean_predicted_kwh"].median()

    filtered = result[
        (result["completeness"] > 0.9)
        & (~result["site_id"].isin(BAD_SITES))
        & (result["weekday_weekend_gap"] > 0.15)   # 평일이 주말보다 뚜렷이 높음
        & (result["peak_to_mean"] > 1.3)
        & (result["mean_predicted_kwh"] >= min_load)
    ].copy()

    filtered["score"] = (
        filtered["weekday_weekend_gap"] * 0.4
        + filtered["peak_to_mean"] * 0.35
        + filtered["completeness"] * 0.25
    )
    filtered = filtered.sort_values("score", ascending=False)

    print(f"electricity 전체 building 수: {elec['building_id'].nunique()}")
    print(f"필터 통과 building 수: {len(filtered)}")
    print(filtered.head(top_n).to_string(index=False))

    return filtered.head(top_n)


if __name__ == "__main__":
    df, booster, meta = load_artifacts()
    candidates = screen_electricity_buildings(df, booster, meta)
