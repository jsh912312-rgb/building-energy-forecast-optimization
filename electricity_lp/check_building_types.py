
import pandas as pd

candidates = [802, 799, 299, 481, 650, 798, 560, 676, 735, 957, 723, 652, 580, 586]

META_PATH = "building_metadata.csv"

meta = pd.read_csv(META_PATH)
result = meta[meta["building_id"].isin(candidates)][
    ["building_id", "site_id", "primary_use", "square_feet"]
].sort_values("building_id")

print(result.to_string(index=False))
print("\n용도별 개수:")
print(result["primary_use"].value_counts())
