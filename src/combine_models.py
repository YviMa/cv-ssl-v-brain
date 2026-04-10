import argparse
import os
import pandas as pd 
import yaml
import json
import numpy as np
from scipy.stats import ttest_rel, false_discovery_control
import pyarrow as pa
import pyarrow.parquet as pq
from utils import hash_config, agg_ttest, pool_lhrh

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-a', action='store_true')
group.add_argument('--config')
args = parser.parse_args()

run_all = args.a
dir_list = args.config

if run_all:
    dir_list = [os.path.join("results", dir) for dir in os.listdir("results") if os.path.isdir(os.path.join("results", dir))]
    file_name = "model_comp_all.parquet"
else:
    with open(dir_list, 'r') as f:
        dir_list = yaml.safe_load(f)
    config_hash = hash_config(dir_list)
    file_name = "model_comp_" + config_hash + ".parquet"

comp_dfs_rsa = []
comp_dfs_reg = []
   
for idx_1, dir_1 in enumerate(dir_list):
    print("outer index", idx_1)
    if dir_1 == os.path.join("results", "model_comparison"):
            continue
    rsa_table_1 = pq.read_table(os.path.join(dir_1, "eval_df_rsa.parquet"), partitioning=None)
    rsa_df_1 = rsa_table_1.to_pandas()
    rsa_df_1.reset_index(inplace=True, drop=True)

    meta_rsa_1 = rsa_table_1.schema.metadata["custom_meta".encode()]
    meta_rsa_1 = json.loads(meta_rsa_1)

    rsa_df_1 = pool_lhrh(rsa_df_1)
    rsa_df_1=rsa_df_1.loc[rsa_df_1.groupby(["ROI"])["R"].idxmax(), ["ROI", "Model", "Layer", "R", "R_array", "SEM", "Significance", "%R", "LNC", "UNC"]]

    rsa_df_1.reset_index(inplace=True, drop=True)
     
    rsa_df_1["model_name"] = meta_rsa_1["name"]
    rsa_df_1["time_window"] = int(meta_rsa_1["time_window"])
    rsa_df_1["crop_size"] = int(meta_rsa_1["crop_size"])
    rsa_df_1["center_crop"] = meta_rsa_1["center_crop"]

    comp_dfs_rsa.append(rsa_df_1)

    reg_table_1 = pq.read_table(os.path.join(dir_1, "eval_df_reg.parquet"), partitioning=None)
    reg_df_1 = reg_table_1.to_pandas()
    reg_df_1.reset_index(inplace=True, drop=True)

    meta_reg_1 = reg_table_1.schema.metadata["custom_meta".encode()]
    meta_reg_1 = json.loads(meta_reg_1)

    reg_df_1 = pool_lhrh(reg_df_1)
    reg_df_1=reg_df_1.loc[reg_df_1.groupby(["ROI"])["R"].idxmax(), ["ROI", "Model", "Layer", "R", "R_array", "SEM", "Significance", "%R", "LNC", "UNC"]]
    reg_df_1.reset_index(inplace=True, drop=True)
     
    reg_df_1["model_name"] = meta_reg_1["name"]
    reg_df_1["time_window"] = int(meta_reg_1["time_window"])
    reg_df_1["crop_size"] = int(meta_reg_1["crop_size"])
    reg_df_1["center_crop"] = meta_reg_1["center_crop"]
    comp_dfs_reg.append(reg_df_1)


comp_rsa = pd.concat(comp_dfs_rsa, ignore_index=True)
comp_reg = pd.concat(comp_dfs_reg, ignore_index=True)

comp_rsa_table = pa.Table.from_pandas(comp_rsa)
comp_reg_table = pa.Table.from_pandas(comp_reg)

pq.write_table(comp_rsa_table, os.path.join("results", "model_comparison", "rsa_"+file_name))
pq.write_table(comp_reg_table, os.path.join("results", "model_comparison", "reg_"+file_name))

print("done")