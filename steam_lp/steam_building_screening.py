"""
  - completeness : 해당 building의 관측 시간 수 / 해당 building 전체 관측 가능 시간 수
  - temp_corr     : predicted_kwh 와 air_temperature 의 상관관계 (열관성 방향성 확인용)
  - peak_to_mean  : predicted_kwh 의 p95 / mean (on/off 급등 패턴 강도)
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ARTIFACT_DIR = "artifacts"
STEAM_METER_ID = 2
BAD_SITES = [14]  # 이전 분석에서 오차 2~5배로 확인된 사이트


def load_artifacts():
    with open(f"{ARTIFACT_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(f"{ARTIFACT_DIR}/df_features.csv", parse_dates=["timestamp"])
    booster = lgb.Booster(model_file=f"{ARTIFACT_DIR}/model_steam.txt")

    return df, booster, meta


def screen_steam_buildings(df: pd.DataFrame, booster: lgb.Booster, meta: dict, top_n: int = 15) -> pd.DataFrame:
    feature_cols = meta["feature_cols"]
    # steam 모델은 meter 컬럼 없이 학습됨 (노트북 cell 17 참고)
    features_m = [c for c in feature_cols if c != "meter"]

    steam = df[df["meter"] == STEAM_METER_ID].copy()
    steam = steam.dropna(subset=features_m)  # lag/rolling 초기 NaN 구간 제외

    if steam.empty:
        raise ValueError("meter==2 데이터가 없습니다. df_features.csv 확인 필요")

    pred_log = booster.predict(steam[features_m], num_iteration=booster.best_iteration)
    steam["predicted_kwh"] = np.expm1(pred_log).clip(min=0)

    # building별 전체 관측 가능 시간 수 (completeness 계산 기준)
    full_span = df.groupby("building_id")["timestamp"].agg(lambda x: x.max() - x.min())

    rows = []
    for bid, g in steam.groupby("building_id"):
        n_hours = len(g)

        span = full_span.get(bid, pd.Timedelta(0))
        expected_hours = span.total_seconds() / 3600 + 1 if span else np.nan
        completeness = n_hours / expected_hours if expected_hours else np.nan

        temp_corr = g["predicted_kwh"].corr(g["air_temperature"])

        mean_load = g["predicted_kwh"].mean()
        p95_load = g["predicted_kwh"].quantile(0.95)
        peak_to_mean = p95_load / mean_load if mean_load > 0 else np.nan

        site = g["site_id"].iloc[0]

        rows.append({
            "building_id": bid,
            "site_id": site,
            "n_hours": n_hours,
            "completeness": round(completeness, 3) if completeness == completeness else np.nan,
            "temp_corr": round(temp_corr, 3),
            "peak_to_mean": round(peak_to_mean, 2),
            "mean_predicted_kwh": round(mean_load, 1),
        })

    result = pd.DataFrame(rows)

    # 절대 사용량 하한선: mean_predicted_kwh가 너무 작으면 peak_to_mean 비율이
    # 분모(mean)의 미세한 흔들림에도 과도하게 튀어서 "가짜 급등 패턴"처럼 보일 수 있음.
    # -> 전체 steam building의 mean_predicted_kwh 중앙값 이상인 건물만 남긴다
    min_load = result["mean_predicted_kwh"].median()

    filtered = result[
        (result["completeness"] > 0.9)
        & (~result["site_id"].isin(BAD_SITES))
        & (result["temp_corr"] < -0.2)
        & (result["peak_to_mean"] > 1.3)
        & (result["n_hours"] >= 24 * 30)
        & (result["mean_predicted_kwh"] >= min_load)
    ].copy()

    filtered["score"] = (
        filtered["completeness"] * 0.3
        + filtered["peak_to_mean"] * 0.4
        + (-filtered["temp_corr"]) * 0.3
    )
    filtered = filtered.sort_values("score", ascending=False)

    print(f"steam 전체 building 수: {steam['building_id'].nunique()}")
    print(f"필터 통과 building 수: {len(filtered)}")
    print(filtered.head(top_n).to_string(index=False))

    return filtered.head(top_n)


if __name__ == "__main__":
    df, booster, meta = load_artifacts()
    candidates = screen_steam_buildings(df, booster, meta)
