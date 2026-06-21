import numpy as np
import os
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting import Plotter



results_dir = 'results/BOLDmoments/RSA/model_comparison/'

roi_cat=pd.read_csv('roi_categorization.csv', index_col=0)
roi_cat=roi_cat.rename(columns={"stream": "location/stream"})
roi_cat.set_index("ROI", inplace=True)


rsa_df_path = os.path.join(results_dir, "rsa_model_comp_all.parquet")
#path_dict = {"reg": reg_df_path, "rsa": rsa_df_path}

rsa_table = pq.read_table(rsa_df_path, partitioning=None)
rsa_df = rsa_table.to_pandas()

plotter = Plotter()
selectivity = "face-selective"
cs = 224

'''df = rsa_df.copy()
g=plotter.plot("line_plot", 
            df, 
            figsize=(4,1.5),
            init_kwargs={"x_var": "time_window", 
                        "other_value": cs, 
                        "sel": selectivity}, 
            format_kwargs={"roi_cat": roi_cat})
save_path = results_dir + "test_rsa_R_dep_on_time_window_gs="+str(cs)+"_"+str(selectivity)+".pdf"
plotter.save(g, save_path)
plt.close()'''
selectivities = list(roi_cat["selectivity"].unique())
selectivities.remove("word-selective")
method_dict = {"rsa": rsa_df}
for method in method_dict.keys():
    for selectivity in selectivities: 
        print(selectivity)
        crop_sizes = [112, 224, 540]
        #crop_sizes = [224]
        time_windows = [0, 15]
        df = method_dict[method]
        for cs in crop_sizes:
            print("crop_size=", cs)
            g=plotter.plot("line_plot", 
                        df, 
                        figsize=(4,1.5),
                        init_kwargs={"x_var": "time_window", 
                                    "other_value": cs, 
                                    "sel": selectivity}, 
                        format_kwargs={"roi_cat": roi_cat})
            save_path = results_dir + method + "_R_dep_on_time_window_gs="+str(cs)+"_"+str(selectivity)+".pdf"
            plotter.save(g, save_path)
            plt.close()
        
        for tw in time_windows:
            print("time_windows=", tw)

            g=plotter.plot("line_plot", 
                        df, 
                        figsize=(4,1.5),
                        init_kwargs={"x_var": "crop_size", 
                                    "other_value": tw, 
                                    "sel": selectivity}, 
                        format_kwargs={"roi_cat": roi_cat})
            save_path = results_dir + method + "_R_dep_on_crop_size_t="+str(tw)+"_"+str(selectivity)+".pdf"
            plotter.save(g, save_path)