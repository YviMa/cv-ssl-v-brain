import numpy as np
import os
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import yaml
from plotting import Plotter



results_dir = 'results/872_shared_total/model_comparison/'

roi_cat=pd.read_csv('roi_categorization.csv', index_col=0)
roi_cat=roi_cat.rename(columns={"stream": "location/stream"})
roi_cat.set_index("ROI", inplace=True)


reg_df_path = os.path.join(results_dir, "reg_model_comp_all.parquet")
rsa_df_path = os.path.join(results_dir, "rsa_model_comp_all.parquet")
#path_dict = {"reg": reg_df_path, "rsa": rsa_df_path}
reg_table = pq.read_table(reg_df_path, partitioning=None)
reg_df = reg_table.to_pandas()

rsa_table = pq.read_table(rsa_df_path, partitioning=None)
rsa_df = rsa_table.to_pandas()

plotter = Plotter()

method_dict = {"rsa": rsa_df, "reg": reg_df}
for method in method_dict.keys():
    for selectivity in roi_cat["selectivity"].unique(): 
        crop_sizes = [112, 224, 540]
        time_windows = [0, 15]
        df = method_dict[method]
        for cs in crop_sizes:
            print("crop_size=", cs)
            g=plotter.plot("line_plot_diff", 
                        df, 
                        figsize=(4,1.5),
                        init_kwargs={"x_var": "time_window", 
                                    "other_value": cs, 
                                    "sel": selectivity}, 
                        format_kwargs={"roi_cat": roi_cat})
            save_path = results_dir+"/improved_lineplots/" + method + "_diff_R_dep_on_time_window_gs="+str(cs)+"_"+str(selectivity)+".pdf"
            plotter.save(g, save_path)
            plt.close()
        
        for tw in time_windows:
            print("time_windows=", tw)

            g=plotter.plot("line_plot_diff", 
                        df, 
                        figsize=(4,1.5),
                        init_kwargs={"x_var": "crop_size", 
                                    "other_value": tw, 
                                    "sel": selectivity}, 
                        format_kwargs={"roi_cat": roi_cat})
            save_path = results_dir+"/improved_lineplots/" + method + "_diff_R_dep_on_crop_size_t="+str(tw)+"_"+str(selectivity)+".pdf"
            plotter.save(g, save_path)