import json, collections
rows = json.load(open("dataset_pillar2_mid_v42_tpsl_all_volk05r06_8-40_cut20250627.json"))
print(collections.Counter(r["output"]["trade_simulation"].get("outcome") for r in rows))