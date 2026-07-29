"""
전기 트랙 상위 후보들의 건물 용도(primary_use) 확인
------------------------------------------------------
building_metadata.csv (또는 원본 raw 메타데이터 파일)에 있는 primary_use를
이전 스크리닝 결과에 조인해서, 사무용(Office) 말고 다른 용도의 건물을 찾는다.
"""

import pandas as pd

# 이전 스크리닝(site 0 제외)에서 나온 상위 후보들
candidates = [802, 799, 299, 481, 650, 798, 560, 676, 735, 957, 723, 652, 580, 586]

# 원본 building_metadata.csv 경로 - 실제 파일 위치에 맞게 수정 필요
META_PATH = "building_metadata.csv"

meta = pd.read_csv(META_PATH)
result = meta[meta["building_id"].isin(candidates)][
    ["building_id", "site_id", "primary_use", "square_feet"]
].sort_values("building_id")

print(result.to_string(index=False))
print("\n용도별 개수:")
print(result["primary_use"].value_counts())
