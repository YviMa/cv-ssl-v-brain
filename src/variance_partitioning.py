import argparse
import os
import pyarrow as pa
import numpy as np
import pandas as pd
import json
import pyarrow.parquet as pq
from scipy.stats import sem
import yaml
from utils import hash_config
from net2brain.evaluations.variance_partitioning_analysis import VPA

all_models = []
results_dir = 'results/Dwivedi2024EEG/'
results_list = os.listdir(results_dir)
for dir in results_list:
    if dir == "model_comparison":
        continue
    data_table = pq.read_table(results_dir + dir + "/eval_df_rsa.parquet", partitioning=None)
    meta = data_table.schema.metadata["custom_meta".encode()]
    meta = json.loads(meta)
    df = data_table.to_pandas()

    df["model_name"] = meta["name"]
    df["time_window"] = int(meta["time_window"]) if meta["time_window"] !='None' else pd.NA
    df["crop_size"] = int(meta["crop_size"]) if meta["crop_size"] !='None' else pd.NA
    df["center_crop"] = meta["center_crop"] if isinstance(meta["center_crop"], bool) else pd.NA
    all_models.append(df)

all_models = pd.concat(all_models)
all_models = all_models.reset_index(drop=True)
all_models["subj_mean"] = all_models["R_array"].apply(list).apply(lambda x: np.mean(x,axis=0))
all_models["subj_sem"] = all_models["R_array"].apply(list).apply(lambda x: sem(x,axis=0))
all_models["peak"] = all_models["subj_mean"].apply(np.max)

max_layers = all_models.groupby("Model")["peak"].idxmax()
which_layers = all_models.loc[max_layers.to_numpy(),["Model", "Layer", "subj_mean", "subj_sem", "time_window", "crop_size", "center_crop", "model_name"]]

#var_1_df = which_layers.loc[(which_layers["time_window"]==0) & (which_layers["crop_size"]==540),:]
#var_2_df = which_layers.loc[(which_layers["time_window"]==0) & (which_layers["crop_size"]!=540),:]
#var_3_df = which_layers.loc[(which_layers["time_window"]!=0) & (which_layers["crop_size"]==540),:]
#var_4_df = which_layers.loc[(which_layers["time_window"]!=0) & (which_layers["crop_size"]!=540),:]

var_1_df = which_layers.loc[(which_layers["time_window"]==0) & (which_layers["crop_size"]==540),:]
var_2_df = which_layers.loc[(which_layers["time_window"]==0) & (which_layers["crop_size"]==224) & (which_layers["center_crop"]==False),:]
var_3_df = which_layers.loc[(which_layers["time_window"]==15) & (which_layers["crop_size"]==224) & (which_layers["center_crop"]==False),:]
var_4_df = which_layers.loc[(which_layers["time_window"]==15) & (which_layers["crop_size"]==540),:]

rdm_dir = "tmp/rdms/Dwivedi2024EEG/"
var_1_rdms = [rdm_dir + model_name + "_rdm" + "/RDM_" + best_layer.replace(".", "_")+ ".npz" for model_name, best_layer in list(var_1_df[["Model", "Layer"]].itertuples(index=False, name=None))]
var_2_rdms = [rdm_dir + model_name + "_rdm" + "/RDM_" + best_layer.replace(".", "_")+ ".npz" for model_name, best_layer in list(var_2_df[["Model", "Layer"]].itertuples(index=False, name=None))]
var_3_rdms = [rdm_dir + model_name + "_rdm" + "/RDM_" + best_layer.replace(".", "_")+ ".npz" for model_name, best_layer in list(var_3_df[["Model", "Layer"]].itertuples(index=False, name=None))]
var_4_rdms = [rdm_dir + model_name + "_rdm" + "/RDM_" + best_layer.replace(".", "_")+ ".npz" for model_name, best_layer in list(var_4_df[["Model", "Layer"]].itertuples(index=False, name=None))]

independent_variables = [var_1_rdms, var_2_rdms, var_3_rdms, var_4_rdms]

variable_names = ["no cv, no ts", "cv, no ts", "no cv, ts", "cv, ts"]

dependent_variable_dir = "../../data/Dwivedi2024EEG/eeg_rdms_decoding.npy"
VPA_eval = VPA(dependent_variable_dir, independent_variables, variable_names)
dataframe = VPA_eval.evaluate(average_models=True)
dataframe["Values"] = dataframe["Values"].apply(lambda x: x.tolist())
dataframe["Significance"] = dataframe["Significance"].apply(lambda x: x.tolist())
#print(dataframe.iloc[0,2])
#dataframe.to_csv("results/Dwivedi2024EEG/model_comparison/variance_partitioning/vpa_results.csv", sep="\t")
data_table = pa.Table.from_pandas(dataframe)
pq.write_table(data_table, "results/Dwivedi2024EEG/model_comparison/variance_partitioning/vpa_results_one_of_each.parquet")
# Filter the dataframe to include only the unique variances and the shared variance by all variables
#dataframe = dataframe.query("Variable in ['y1234', 'y1', 'y2', 'y3', 'y4']").reset_index(drop=True)