import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from utils import hash_config
from scipy.stats import spearmanr

print("imports done")

config_path = "configs/872_shared_total/"
rdm_path = "tmp/rdms/"

time_windows = [0, 5, 10, 15, 20, 25]
models = ["resnet50", "vit_base"]

results_list = []

for model in models:
    for time_window_1 in time_windows:
        print("model", model, "time_window 1: ", time_window_1)
        idx = time_windows.index(time_window_1)
        time_windows_2 = time_windows.copy()
        time_windows_2.pop(time_windows.index(time_window_1))
        for time_window_2 in time_windows_2:
            name_1 = model + "_t="+str(time_window_1)+"_gs=224"
            name_2 = model + "_t="+str(time_window_2)+"_gs=224"
            hash_1 = hash_config(os.path.join(config_path, name_1+".yaml"))
            hash_2 = hash_config(os.path.join(config_path, name_2+".yaml"))
            rdms_path_1 = os.path.join(rdm_path, name_1+"_"+hash_1+"_rdm")
            rdms_path_2 = os.path.join(rdm_path, name_2+"_"+hash_2+"_rdm")

            for rdm_name_1 in os.listdir(rdms_path_1):
                for rdm_name_2 in os.listdir(rdms_path_2):
                    data_1 = np.load(os.path.join(rdms_path_1, rdm_name_1))
                    data_2 = np.load(os.path.join(rdms_path_2, rdm_name_2))
                    rdm_1 = data_1["rdm"]
                    rdm_2 = data_2["rdm"]
                    corr = spearmanr(rdm_1, rdm_2)[0]
                    results_dict = {"model": model, "time_window_1": time_window_1, "time_window_2": time_window_2, "Layer_1": str(data_1["layer_name"]), "Layer_2": str(data_2["layer_name"]), "R": corr}
                    results_list.append(results_dict)

results_df = pd.DataFrame(results_list)
results_table = pa.Table.from_pandas(results_df)
pq.write_table(results_table, 'results/872_shared_total/model_to_model.parquet')