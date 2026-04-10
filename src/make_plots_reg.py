import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from utils import strip_hash, agg_ttest
import json
import os
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp, sem, ttest_rel, false_discovery_control
from ipywidgets import widgets
import pyarrow.parquet as pq
#from net2brain.evaluations.plotting import Plotting
#from net2brain.evaluations.rsa import RSA
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap



available_dirs = os.listdir('results')
'''
for dir in available_dirs:
    print(dir)
    if dir=="model_comparison":
        continue

    reg_df_path = os.path.join("results",dir, "eval_df_reg.parquet")
    roi_cat=pd.read_csv('roi_categorization.csv', index_col=0)
    roi_cat=roi_cat.rename(columns={"stream": "location/stream"})

    reg_table = pq.read_table(reg_df_path, partitioning=None)
    reg_df = reg_table.to_pandas()
    reg_df.reset_index(inplace=True, drop=True)
    meta_reg = reg_table.schema.metadata["custom_meta".encode()]
    meta_reg = json.loads(meta_reg)

    reg_df_fdr = reg_df.copy()
    reg_df_fdr["Significance"] = false_discovery_control(reg_df_fdr["Significance"].to_numpy())
    roi_list_lh = reg_df_fdr["ROI"].apply(lambda x: x.split("_lh")[0]).unique()
    roi_list_rh = reg_df_fdr["ROI"].apply(lambda x: x.split("_rh")[0]).unique()
    intersect_roi = np.intersect1d(roi_list_lh, roi_list_rh)

    pool_or_not = reg_df_fdr.copy()
    pool_or_not["ROI_non_handed"] = pool_or_not["ROI"].apply(lambda x: x[:-3])
    pool_or_not = pool_or_not.loc[pool_or_not["ROI_non_handed"].isin(intersect_roi),:]
    pool_or_not = pool_or_not.groupby(["ROI_non_handed", "Layer"])["R_array"].agg(agg_ttest)
    pool_or_not = pool_or_not.reset_index()
    pool_or_not=pool_or_not.rename(columns={"R_array": "p-value"})

    pool_or_not["p-value_FDR"]=false_discovery_control(pool_or_not["p-value"])
    non_poolable=pool_or_not.loc[pool_or_not["p-value_FDR"]<0.05,["ROI_non_handed"]]

    reg_pooled_fdr=reg_df_fdr.copy()
    reg_pooled_fdr["ROI_non_handed"] = reg_pooled_fdr["ROI"].apply(lambda x: x[:-3])
    mask = reg_pooled_fdr[["ROI_non_handed"]].apply(tuple, axis=1).isin(non_poolable.apply(tuple, axis=1))
    reg_pooled_fdr = reg_pooled_fdr.loc[~mask,:]
    reg_pooled_fdr = reg_pooled_fdr.groupby(["ROI_non_handed", "Layer"])[["R", "%R", "R_array","LNC", "UNC"]].agg('mean')
    reg_pooled_fdr=reg_pooled_fdr.reset_index()
    #print(reg_df_fdr.loc[0,"Model"])
    reg_pooled_fdr.insert(2, "Model", reg_df_fdr.loc[0,"Model"])
    reg_pooled_fdr.rename(columns={"ROI_non_handed":"ROI"}, inplace=True)
    reg_non_pooled_fdr=reg_df_fdr.copy()
    reg_non_pooled_fdr["ROI_non_handed"] = reg_non_pooled_fdr["ROI"].apply(lambda x: x[:-3])
    mask = reg_non_pooled_fdr[["ROI_non_handed"]].apply(tuple, axis=1).isin(non_poolable.apply(tuple, axis=1))
    reg_non_pooled_fdr = reg_non_pooled_fdr.loc[mask,:]
    reg_non_pooled_fdr.drop(["ROI_non_handed", "Significance", "SEM"], axis=1, inplace=True)
    reg_pooled_fdr=pd.concat([reg_pooled_fdr, reg_non_pooled_fdr])

    significance = false_discovery_control(reg_pooled_fdr["R_array"].apply(lambda x: ttest_1samp(x,0)[1]))
    sem_arr = reg_pooled_fdr["R_array"].apply(sem)
    reg_pooled_fdr.insert(6, "Significance", significance)
    reg_pooled_fdr.insert(8, "SEM", sem_arr)

    reg_pooled_fdr.reset_index(inplace=True, drop=True)

    plotting_df_reg = reg_df_fdr.copy()
    plotting_df_reg = plotting_df_reg.loc[plotting_df_reg.groupby('ROI')["R"].idxmax()]
    plotting_df_reg.sort_values("R", inplace=True)
    plotting_df_reg = plotting_df_reg.explode("R_array")
    plotting_df_reg = plotting_df_reg.rename(columns = {"R": "R_mean", "%R": "%R_mean", "R_array": "R"})
    plotting_df_reg.reset_index(inplace=True, drop=True)

    fig, ax = plt.subplots(figsize = (16,6))

    sns.boxplot(x="ROI", y="R", hue="Model", palette='pastel', data=plotting_df_reg, showmeans=True, meanprops={"marker":"_", 
                        "markeredgecolor":"red",
                        "markersize":"20"}, ax=ax)

    rois=plotting_df_reg["ROI"].unique()

    #rois, idx = np.unique(plotting_df_reg["ROI"].to_numpy(), return_index=True)
    x = np.arange(len(rois))
    y = plotting_df_reg.groupby("ROI", sort=False)["R"].agg("max")+0.01
    y.reset_index(inplace=True, drop=True)
    mask = plotting_df_reg.groupby("ROI", sort=False)["Significance"].agg("first") < 0.05
    mask.reset_index(inplace=True, drop=True)
    ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

    #ax.hlines(plotting_df["LNC"].iloc[idx], x-0.3, x+0.3, color='k', ls='--')
    #ax.hlines(plotting_df["UNC"].iloc[idx], x-0.3, x+0.3, color='k', ls='--')

    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    current_handles_labels = ax.get_legend_handles_labels()
    obj_list = [current_handles_labels[0][0], Line2D([0], [0], color="red", lw=1)]
    label_list = [current_handles_labels[1][0], "mean"]
    plt.legend(obj_list, label_list, fontsize=14, frameon=True)
    ax.set_title("reg: R distributions across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_R_distribution_all_ROIs.pdf"), format='PDF', bbox_inches='tight')
    #plt.show()

    fig, ax = plt.subplots(figsize=(16,6))
    sns.barplot(x="ROI", y="R", hue="Model", palette='pastel', data=plotting_df_reg, errorbar='se', capsize=0.2, errwidth=1)
    rois=plotting_df_reg["ROI"].unique()
    x = np.arange(len(rois))
    y = plotting_df_reg.groupby("ROI", sort=False)["R"].agg("mean")+0.06
    y.reset_index(inplace=True, drop=True)
    mask = plotting_df_reg.groupby("ROI", sort=False)["Significance"].agg("first") < 0.05
    mask.reset_index(inplace=True, drop=True)
    ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

    #ax.hlines(plotting_df_reg.groupby("ROI").agg("first")["LNC"].reindex(rois), x-0.35, x+0.35, color='k', ls='dotted', label="noise ceiling")
    #ax.hlines(plotting_df_reg.groupby("ROI").agg("first")["UNC"].reindex(rois), x-0.35, x+0.35, color='k', ls='dotted')
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    plt.legend(frameon=True, fontsize=14)
    ax.set_title("reg: Mean R across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_Mean_R_all_ROIs.pdf"), format='PDF', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(16,6))
    sns.barplot(x="ROI", y="R", hue="Model", palette='pastel', data=plotting_df_reg, errorbar='se', capsize=0.2, errwidth=1)
    rois=plotting_df_reg["ROI"].unique()
    x = np.arange(len(rois))
    y = plotting_df_reg.groupby("ROI", sort=False)["R"].agg("mean")+0.06
    y.reset_index(inplace=True, drop=True)
    mask = plotting_df_reg.groupby("ROI", sort=False)["Significance"].agg("first") < 0.05
    mask.reset_index(inplace=True, drop=True)
    ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

    #ax.hlines(plotting_df_reg.groupby("ROI").agg("first")["LNC"].reindex(rois), x-0.35, x+0.35, color='k', ls='dotted', label="noise ceiling")
    #ax.hlines(plotting_df_reg.groupby("ROI").agg("first")["UNC"].reindex(rois), x-0.35, x+0.35, color='k', ls='dotted')
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    plt.legend(frameon=True, fontsize=14)
    ax.set_title("reg: Mean R across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_Mean_R_all_ROIs.pdf"), format='PDF', bbox_inches='tight')

    plotting_pooled = reg_pooled_fdr.copy()
    plotting_pooled = plotting_pooled.loc[plotting_pooled.groupby('ROI')["R"].idxmax()]
    plotting_pooled.sort_values("R", inplace=True)
    plotting_pooled = plotting_pooled.explode("R_array")
    plotting_pooled = plotting_pooled.rename(columns = {"R": "R_mean", "%R": "%R_mean", "R_array": "R"})
    plotting_pooled.reset_index(inplace=True, drop=True)

    fig, ax = plt.subplots(figsize = (16,6))

    sns.boxplot(x="ROI", y="R", hue="Model", palette='pastel', data=plotting_pooled, showmeans=True, meanprops={"marker":"_", 
                        "markeredgecolor":"red",
                        "markersize":"36"}, ax=ax)


    rois=plotting_pooled["ROI"].unique()
    x = np.arange(len(rois))
    y = plotting_pooled.groupby("ROI", sort=False)["R"].agg("max")+0.01
    y.reset_index(inplace=True, drop=True)
    mask = plotting_pooled.groupby("ROI", sort=False)["Significance"].agg("first") < 0.05
    mask.reset_index(inplace=True, drop=True)
    ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

    #ax.hlines(plotting_df["LNC"].iloc[idx], x-0.3, x+0.3, color='k', ls='--')
    #ax.hlines(plotting_df["UNC"].iloc[idx], x-0.3, x+0.3, color='k', ls='--')

    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    current_handles_labels = ax.get_legend_handles_labels()
    obj_list = [current_handles_labels[0][0], Line2D([0], [0], color="red", lw=1)]
    label_list = [current_handles_labels[1][0], "mean"]
    plt.legend(obj_list, label_list, fontsize=14, frameon=True)
    ax.set_title("reg: R distributions across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_R_distribution_pooled.pdf"), format='PDF', bbox_inches='tight')
    #plt.show()

    fig, ax = plt.subplots(figsize=(16,6))
    sns.barplot(x="ROI", y="R", hue="Model", palette='pastel', data=plotting_pooled, errorbar='se', capsize=0.2, errwidth=1)
    rois=plotting_pooled["ROI"].unique()
    x = np.arange(len(rois))
    y = plotting_pooled.groupby("ROI", sort=False)["R"].agg("mean")+0.05
    y.reset_index(inplace=True, drop=True)
    mask = plotting_pooled.groupby("ROI", sort=False)["Significance"].agg("first") < 0.05
    mask.reset_index(inplace=True, drop=True)
    ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

    #ax.hlines(plotting_pooled.groupby("ROI")["LNC"].agg("first").reindex(rois), x-0.35, x+0.35, color='k', ls='dotted', label="noise ceiling")
    #ax.hlines(plotting_pooled.groupby("ROI")["UNC"].agg("first").reindex(rois), x-0.35, x+0.35, color='k', ls='dotted')
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    plt.legend(frameon=True, fontsize=14)
    ax.set_title("reg: Mean R across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_Mean_R_pooled.pdf"), format='PDF', bbox_inches='tight')

    table = pd.pivot_table(reg_pooled_fdr, values="R", columns="Layer", index="ROI")
    sorted=table.max(axis=1).sort_values(ascending=False)
    table_sorted = table.loc[sorted.index,:]

    p_table = pd.pivot_table(reg_pooled_fdr, values="Significance", columns="Layer", index="ROI") < 0.05
    p_table_sorted = p_table.loc[sorted.index,:]
    p_table_sorted[p_table_sorted==1]="*"

    #annot = pd.pivot_table(reg_pooled_fdr, values="SEM", columns="Layer", index="ROI")
    #annot_sorted = annot.loc[sorted.index,:]

    fig, ax = plt.subplots(figsize=(4,8))
    sns.heatmap(table_sorted, cmap=sns.color_palette("viridis", as_cmap=True), annot=p_table_sorted, fmt="")
    cbar = ax.collections[0].colorbar
    cbar.set_label("R", rotation=0, labelpad=15)
    ax.set_xticklabels([str(i) for i in range(1,len(table.columns)+1)], rotation=0)
    ax.tick_params(axis='both',labelsize=14)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=12)
    cax.set_label("R")
    ax.set_title(f"{meta_reg['name']}, $\Delta T$={meta_reg['time_window']}, crop={meta_reg['crop_size']}", pad=20)
    plt.savefig(os.path.join("results",dir, "reg_R_by_Layer_pooled.pdf"), format='PDF', bbox_inches='tight')
    #sns.heatmap(table_sorted, cmap=sns.cubehelix_palette(start=1.6, rot=-.9, as_cmap=True, dark=0.3), vmin=0, vmax=0.2)

    roi_cat.set_index("ROI", inplace=True)
    plotting_pooled["processing_stage"]=[roi_cat.loc[roi.split("_")[0], "processing_stage"] for roi in plotting_pooled["ROI"].to_list()]
    plotting_pooled["location/stream"]=[roi_cat.loc[roi.split("_")[0], "location/stream"] for roi in plotting_pooled["ROI"].to_list()]

    plotting_pooled = reg_pooled_fdr.copy()
    plotting_pooled = plotting_pooled.loc[plotting_pooled.groupby('ROI')["R"].idxmax()]
    plotting_pooled.reset_index(inplace=True, drop=True)

    #palette = sns.color_palette("pastel")
    palette = sns.color_palette("Blues", n_colors=5)

    fig, ax = plt.subplots(figsize=(16,6))

    count_v = 0
    count_d = 0
    loc_v = 1
    loc_d = 12
    xtick_loc = []
    xtick_labels = []
    rois = plotting_pooled.sort_values('R')["ROI"].to_numpy()

    for roi in rois:
        if roi_cat.loc[roi.split("_")[0],"location/stream"]=="ventral":
            loc = loc_v + count_v
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[2], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_v += 0.6
        else:
            loc = loc_d + count_d
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[4], width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_d += 0.6

    #ax.plot([1 ,1,count_v+loc_v-0.6, count_v+loc_v-0.6], [-.15, -.25, -.25, -.15], linewidth=1, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.plot([loc_d ,loc_d,count_d+loc_d-0.6, count_d+loc_d-0.6], [-.15, -.25, -.25, -.15], linewidth=1, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.text(loc_v+(count_v)/2-0.6, -.3, "ventral", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)
    #ax.text(loc_d+(count_d)/2-0.6, -.3, "doregl", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)

    y = plotting_pooled.sort_values('R')["R"]+0.05
    mask = plotting_pooled.sort_values('R')["Significance"] < 0.05
    ax.scatter(np.array(xtick_loc)[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')


    ax.set_xticks(xtick_loc, xtick_labels)
    ax.grid(visible=True, which='major', axis='y')
    ax.grid(visible=False, axis='x')
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    patches = [mpl.patches.Patch(color=palette[2], label="ventral"), mpl.patches.Patch(color=palette[4], label="dorsal")]
    plt.legend(handles=patches, frameon=True, fontsize=14)
    ax.set_title("reg: Mean R across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_Mean_R_ventral_dorsal.pdf"), format='PDF', bbox_inches='tight')

    palette = sns.color_palette("viridis", n_colors=4)

    fig, ax = plt.subplots(figsize=(16,6))

    count_e = 0
    count_m = 0
    count_l = 0
    loc_e = 1
    loc_m = 6
    loc_l = 9
    xtick_loc = []
    xtick_labels = []
    rois = plotting_pooled.sort_values('R')["ROI"].to_numpy()

    for roi in rois:
        if roi_cat.loc[roi.split("_")[0],"processing_stage"]=="early":
            loc = loc_e + count_e
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[1], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_e += 0.6
        elif roi_cat.loc[roi.split("_")[0],"processing_stage"]=="mid":
            loc = loc_m + count_m
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[2], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_m += 0.6
        else:
            loc = loc_l + count_l
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[3], width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_l += 0.6

    #ax.plot([1 ,1,count_v+loc_v-0.6, count_v+loc_v-0.6], [-.15, -.25, -.25, -.15], linewidth=1, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.plot([loc_d ,loc_d,count_d+loc_d-0.6, count_d+loc_d-0.6], [-.15, -.25, -.25, -.15], linewidth=1, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.text(loc_v+(count_v)/2-0.6, -.3, "ventral", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)
    #ax.text(loc_d+(count_d)/2-0.6, -.3, "doregl", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)

    y = plotting_pooled.sort_values('R')["R"]+0.05
    mask = plotting_pooled.sort_values('R')["Significance"] < 0.05
    ax.scatter(np.array(xtick_loc)[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')


    ax.set_xticks(xtick_loc, xtick_labels)
    ax.grid(visible=True, which='major', axis='y')
    ax.grid(visible=False, axis='x')
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    patches = [mpl.patches.Patch(color=palette[1], label="early"), 
            mpl.patches.Patch(color=palette[2], label="mid"),
            mpl.patches.Patch(color=palette[3], label="late")]
    plt.legend(handles=patches, frameon=True, fontsize=14)
    ax.set_title("reg: Mean R across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_Mean_R_early_mid_late.pdf"), format='PDF', bbox_inches='tight')

    palette = sns.color_palette("plasma", n_colors=6)

    fig, ax = plt.subplots(figsize=(16,6))

    count_0 = 0
    count_b = 0
    count_f = 0
    count_p = 0
    count_w = 0
    loc_0 = 1
    loc_b = 7
    loc_f = 9
    loc_p = 12
    loc_w = 15
    xtick_loc = []
    xtick_labels = []
    rois = plotting_pooled.sort_values('R')["ROI"].to_numpy()

    for roi in rois:
        if roi_cat.loc[roi.split("_")[0],"selectivity"]=="basic features":
            loc = loc_0 + count_0
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[1], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_0 += 0.6
        elif roi_cat.loc[roi.split("_")[0],"selectivity"]=="body-selective":
            loc = loc_b + count_b
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[2], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_b += 0.6
        elif roi_cat.loc[roi.split("_")[0],"selectivity"]=="face-selective":
            loc = loc_f + count_f
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[3], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_f += 0.6
        elif roi_cat.loc[roi,"selectivity"]=="place-selective":
            loc = loc_p + count_p
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[4], 
                width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_p += 0.6
        else:
            loc = loc_w + count_w
            xtick_loc.append(loc)
            xtick_labels.append(roi)
            ax.bar(loc, plotting_pooled["R"].loc[plotting_pooled["ROI"]==roi],
                color=palette[5], width=0.5, 
                yerr=plotting_pooled["SEM"].loc[plotting_pooled["ROI"]==roi], 
                capsize=2)
            count_w += 0.6

    #ax.plot([1 ,1,count_v+loc_v-0.6, count_v+loc_v-0.6], [-.15, -.25, -.25, -.15], linewidth=1, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.plot([loc_d ,loc_d,count_d+loc_d-0.6, count_d+loc_d-0.6], [-.15, -.25, -.25, -.15], linewidth=1, color='k', transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.text(loc_v+(count_v)/2-0.6, -.3, "ventral", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)
    #ax.text(loc_d+(count_d)/2-0.6, -.3, "doregl", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)

    y = plotting_pooled.sort_values('R')["R"]+0.05
    mask = plotting_pooled.sort_values('R')["Significance"] < 0.05
    ax.scatter(np.array(xtick_loc)[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')


    ax.set_xticks(xtick_loc, xtick_labels)
    ax.grid(visible=True, which='major', axis='y')
    ax.grid(visible=False, axis='x')
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    patches = [mpl.patches.Patch(color=palette[1], label="basic features"), 
            mpl.patches.Patch(color=palette[2], label="body-selective"),
            mpl.patches.Patch(color=palette[3], label="face-selective"),
            mpl.patches.Patch(color=palette[4], label="place-selective"),
            mpl.patches.Patch(color=palette[5], label="word-selective")]
    plt.legend(handles=patches, frameon=True, fontsize=14)
    ax.set_title("reg: Mean R across subjects (n=8) from best layer for each ROI", fontsize=18)
    plt.savefig(os.path.join("results",dir, "reg_Mean_R_by_selectivity.pdf"), format='PDF', bbox_inches='tight')

    plot_all_layers = reg_pooled_fdr.copy()
    plot_all_layers["processing_stage"]=[roi_cat.loc[roi.split("_")[0], "processing_stage"] for roi in plot_all_layers["ROI"].to_list()]
    plot_all_layers["location/stream"]=[roi_cat.loc[roi.split("_")[0], "location/stream"] for roi in plot_all_layers["ROI"].to_list()]
    plot_all_layers["selectivity"]=[roi_cat.loc[roi.split("_")[0], "selectivity"] for roi in plot_all_layers["ROI"].to_list()]

    plot_all_layers=plot_all_layers.explode("R_array")

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12,15))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    sns.barplot(data=plot_all_layers.loc[plot_all_layers["selectivity"]=="basic features",:], x="Layer", y="R_array", hue="ROI", ax=ax1, capsize=0.5)
    ax1.legend(frameon=True)
    ax1.set_xticklabels(np.arange(1,12))
    ax1.set_title("ROIs selective for basic features")
    ax1.set_ylabel("R")
    ax1.set_ylim(0,0.5)

    ax2 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=plot_all_layers.loc[plot_all_layers["selectivity"]=="body-selective",:], x="Layer", y="R_array", hue="ROI", ax=ax2, capsize=0.5)
    ax2.legend(frameon=True)
    ax2.set_xticklabels(np.arange(1,12))
    ax2.set_title("Body-selective ROIs")
    ax2.set_ylabel("R")
    ax2.set_ylim(0,0.5)

    ax3 = fig.add_subplot(gs[1, 1])
    sns.barplot(data=plot_all_layers.loc[plot_all_layers["selectivity"]=="face-selective",:], x="Layer", y="R_array", hue="ROI", ax=ax3, capsize=0.5)
    ax3.legend(frameon=True)
    ax3.set_xticklabels(np.arange(1,12))
    ax3.set_title("Face-selective ROIs")
    ax3.set_ylabel("R")
    ax3.set_ylim(0,0.5)

    ax4 = fig.add_subplot(gs[2, 0])
    sns.barplot(data=plot_all_layers.loc[plot_all_layers["selectivity"]=="place-selective",:], x="Layer", y="R_array", hue="ROI", ax=ax4, capsize=0.5)
    ax4.legend(frameon=True)
    ax4.set_xticklabels(np.arange(1,12))
    ax4.set_title("Place-selective ROIs")
    ax4.set_ylabel("R")
    ax4.set_ylim(0,0.5)

    ax5 = fig.add_subplot(gs[2, 1])
    sns.barplot(data=plot_all_layers.loc[plot_all_layers["selectivity"]=="word-selective",:], x="Layer", y="R_array", hue="ROI", ax=ax5, capsize=0.5)
    ax5.legend(frameon=True)
    ax5.set_xticklabels(np.arange(1,12))
    ax5.set_title("Place-selective ROIs")
    ax5.set_ylabel("R")
    ax5.set_ylim(0,0.5)

    plt.suptitle(f"R across Layers for all ROIs", fontsize=18, y=0.92)
    plt.savefig(os.path.join("results",dir, "reg_all_layers.pdf"), format='PDF', bbox_inches='tight')
'''




comp_dir = "results/model_comparison"
reg_comp_table = pq.read_table(os.path.join(comp_dir, "reg_model_comp_all.parquet"), partitioning=None)
reg_comp = reg_comp_table.to_pandas()

for roi in reg_comp["ROI"].unique():
    roi_df = reg_comp.loc[reg_comp["ROI"]==roi,:].copy()
    roi_df.reset_index(inplace=True, drop=True)
    # fdr correct within ROI
    roi_df["Significance"]=false_discovery_control(roi_df["Significance"])

    M = np.zeros((26,26))
    sign_corr = np.zeros((26,26))
    for i, row_1 in roi_df.iterrows():
        for j, row_2 in roi_df.iterrows():
            M[i,j]=ttest_rel(row_1["R_array"], row_2["R_array"])[1]
            sign_corr[i,j]=np.sign(row_1["R"] - row_2["R"])

    tril_idx = np.tril_indices(26, k=-1)
    p_value_FDR = false_discovery_control(M[tril_idx], axis=None)
    M=np.zeros((26,26))
    M[tril_idx[0], tril_idx[1]]=p_value_FDR
    p_value_df=pd.DataFrame(M, index=roi_df["Model"], columns=roi_df["Model"])
    p_value_df.mask(p_value_df>=0.05, 0.051, inplace=True)
    p_value_df=p_value_df*sign_corr

    colors1 = plt.cm.Blues(np.linspace(0, 0.8, 6))
    colors1[0,:]=[0.6, 0.6, 0.6, 1]
    colors2 = plt.cm.Reds_r(np.linspace(0.2, 1, 6))
    colors2[-1,:]=[0.6, 0.6, 0.6, 1]
    colors = np.vstack((colors1, colors2))
    mymap = ListedColormap(colors)

    mask = np.triu(np.ones(p_value_df.shape, dtype=bool))
    palette = sns.color_palette("coolwarm", n_colors=8)
    fig, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(p_value_df, annot=False, fmt=".2f", cmap=mymap, cbar=True, ax=ax, linewidths=0.5, vmin=-0.06, vmax=0.06, mask=mask)
    model_labels = pd.Series(p_value_df.index).apply(strip_hash).to_list()
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(model_labels, fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.set_label('p-value', rotation=0, labelpad=15)
    cbar_ticklabels = list(np.round(np.arange(-0.050, 0.051, 0.01),2))
    cbar_ticks = list(np.arange(-0.05, 0.051, 0.01))
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels, fontsize=12)
    cbar.set_label("p-value", fontsize=16)
    ax.set_title(f"P-values of pairwise model comparison, ROI: {roi}", fontsize=18)
    plt.savefig(f"results/model_comparison/reg_{roi}_all_models_p_values.pdf", format='PDF', bbox_inches='tight')

    roi_df_expl = roi_df.sort_values("R")
    roi_df_expl = roi_df_expl.explode("R_array")
    roi_df_expl.reset_index(inplace=True, drop=True)

    palette = sns.color_palette('pastel')
    fig, ax = plt.subplots(figsize=(16,6))
    sns.barplot(x="Model", y="R_array", color=palette[0], data=roi_df_expl, errorbar='se', capsize=0.2, errwidth=1)

    models=roi_df_expl.index.unique()
    x = np.arange(len(model_labels))
    y = roi_df_expl.groupby("Model", sort=False)["R"].agg("first")+0.04
    y.reset_index(inplace=True, drop=True)
    mask = roi_df_expl.groupby("Model", sort=False)["Significance"].agg("first") < 0.05
    mask.reset_index(inplace=True, drop=True)
    ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

    #ax.hlines(roi_df_expl.groupby("Model").agg("first")["LNC"], x-0.35, x+0.35, color='k', ls='dotted', label="noise ceiling")
    #ax.hlines(roi_df_expl.groupby("Model").agg("first")["UNC"], x-0.35, x+0.35, color='k', ls='dotted')
    ax.set_xticklabels(model_labels)
    ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
    plt.setp(ax.get_xticklabels(),ha='right')
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("R (pearson)", fontsize=16)
    ax.set_xlabel("ROI", fontsize=16)
    ax.set_title(f"reg: Mean R across subjects (n=8) for all models, ROI: {roi}", fontsize=18)
    plt.savefig(f"results/model_comparison/reg_Mean_R_all_models_{roi}.pdf", format='PDF', bbox_inches='tight')