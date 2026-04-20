import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyarrow as pa
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from utils import strip_hash, agg_ttest
import json
import os
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp, sem, ttest_rel, false_discovery_control
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec
from abc import ABC, abstractmethod
# boxplot x="ROI", y="R", hue="Model"
# barplot x="ROI", y="R", hue="Model"
# barplot x="Model", y="R"
# grouped barplot by ventral/dorsal, by selectivity and by early/mid/late
# all layers plot for all ROIs by selectivity
class Plotter():
    def __init__(self):
        self.PLOT_REGISTRY = {
            "box": BoxPlot,
            "roi_bar": BarPlot,
            "grouped_bar": GroupedBar,
            "all_layers_grouped": AllLayers,
            "p_value_heatmap": PValuePlot,
            "layer_heatmap": LayerHeatmap,
            "R_heatmap": ModelPlot,
            "model_bar": ModelPlot
        }      

    def plot(self, plot_type, dataframe, x=None, y=None, hue=None, figsize=None, save=False, save_path=None, plot_kwargs=dict(), format_kwargs=dict()):
        PlotCLS = self.PLOT_REGISTRY[plot_type]
        plot_cls = PlotCLS()
        df = plot_cls.format_df(dataframe, **format_kwargs)
        ax = plot_cls.plot(df, figsize, **plot_kwargs)
        plot_cls.style(ax)
        plot_cls.annotate(df, ax)

        if save == True:
            plt.savefig(save_path, format='PDF', bbox_inches='tight')
        return ax


class Plot(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def format_df():
        pass

class ROIPlot(Plot):
    def __init__(self):
        super().__init__()
    def format_df(self, dataframe):
        df = dataframe.loc[dataframe.groupby('ROI')["R"].idxmax()]
        df.sort_values("R", inplace=True)
        df = df.explode("R_array")
        df = df.rename(columns = {"R": "R_mean", "%R": "%R_mean", "R_array": "R"})
        df.reset_index(inplace=True, drop=True)
        return df
    def style(self, ax):
        ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel("R (pearson)", fontsize=16)
        ax.set_xlabel("ROI", fontsize=16)

    def annotate(self, dataframe, ref_point, x, ax):
        y = dataframe.groupby("ROI", sort=False)["R"].agg(ref_point)+0.03
        y.reset_index(inplace=True, drop=True)
        mask = dataframe.groupby("ROI", sort=False)["Significance"].agg("first") < 0.05
        mask.reset_index(inplace=True, drop=True)
        ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

class BarPlot(ROIPlot):
    def __init__(self):
        super().__init__()
    def plot(self, dataframe, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="ROI", y="R", hue="Model", palette='pastel', data=dataframe, errorbar='se', capsize=0.2, errwidth=1, ax=ax)
        return ax
    def style(self, ax):
        super().style(ax)
        current_handles_labels = ax.get_legend_handles_labels()
        plt.legend([current_handles_labels[0][0]], 
                   [strip_hash(current_handles_labels[1][0])], frameon=True, fontsize=14)

    def annotate(self,dataframe, ax):
        x = np.arange(len(dataframe["ROI"].unique()))
        super().annotate(dataframe, "mean", x, ax)
    
class BoxPlot(ROIPlot):
    def __init__(self):
        super().__init__()

    def plot(self,dataframe, figsize, markersize="36"):
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x="ROI", y="R", hue="Model", data=dataframe, palette="pastel", showmeans=True, meanprops={"marker":"_", "markeredgecolor":"red","markersize":markersize}, ax=ax)
        return ax

    def style(self, ax):
        super().style(ax)
        current_handles_labels = ax.get_legend_handles_labels()
        obj_list = [current_handles_labels[0][0], Line2D([0], [0], color="red", lw=1)]
        label_list = [strip_hash(current_handles_labels[1][0]), "mean"]
        plt.legend(obj_list, label_list, fontsize=14, frameon=True)
        ax.set_title("reg: R distributions across subjects (n=8) from best layer for each ROI", fontsize=18)
    
    def annotate(self,dataframe, ax):
        pass
        #x = np.arange(len(dataframe["ROI"].uniqe()))
        #super.annotate(dataframe, "max", x, ax)

class GroupedBar(ROIPlot):
    def __init__(self):
        super().__init__()
        self.palette_dict = {
            "location/stream": sns.color_palette("Blues", n_colors=5),
            "processing_stage": sns.color_palette("viridis", n_colors=4),
            "selectivity": sns.color_palette("plasma", n_colors=6),
            }

        self.color_dicts = {
            "location/stream": {
                "ventral": self.palette_dict["location/stream"][2], 
                "dorsal": self.palette_dict["location/stream"][4]
                },
            "processing_stage": {
                "early": self.palette_dict["processing_stage"][1], 
                "mid": self.palette_dict["processing_stage"][2], 
                "late": self.palette_dict["processing_stage"][3]
                },
            "selectivity": {
                "basic features": self.palette_dict["selectivity"][1], 
                "body-selective": self.palette_dict["selectivity"][2], 
                "face-selective": self.palette_dict["selectivity"][3],
                "place-selective": self.palette_dict["selectivity"][4],
                "word-selective": self.palette_dict["selectivity"][5]
                }
        }

    def format_df(self, dataframe, roi_cat, cat=None):
        df = super().format_df(dataframe)
        df.set_index("ROI", inplace=True)
        df[cat]=[roi_cat.loc[roi.split("_")[0], cat] for roi in df.index.to_list()]
        df = df.groupby(df.index)[["R_mean", "SEM", "Significance", cat]].agg("first").sort_values(cat)
        self.cat = cat
        return df

    def plot(self, dataframe, figsize, **plot_kwargs):
        color_dict = self.color_dicts[self.cat]
        n_bars = dataframe.groupby(self.cat).size().apply((lambda x: x+2))
        pos=n_bars
        n_bars = n_bars.cumsum()-pos
        x = n_bars.to_dict()

        width = 0.8
        xticks = []
        xticklabels = []
        fig, ax = plt.subplots(figsize=figsize)

        for j, group in enumerate(dataframe[self.cat].unique()):
            group_df = dataframe.loc[dataframe[self.cat]==group, ["R_mean", "SEM"]].sort_values("R_mean")
            for i, roi in enumerate(group_df.index):
                ax.bar(
                    x[group] + i * (width+0.2),
                    group_df.loc[roi, "R_mean"],
                    width=width,
                    color=color_dict[group],
                    yerr=group_df.loc[roi, "SEM"],
                    capsize=2,
                    label=group
                )
                xticks.append(x[group] + i * (width+0.2))
                xticklabels.append(roi)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        return ax

    def annotate(self, dataframe, ax):
        sig_df = dataframe[["R_mean", self.cat]]
        n_bars = dataframe.groupby(self.cat).size().apply((lambda x: x+2))
        pos=n_bars
        n_bars = n_bars.cumsum()-pos
        x = n_bars.to_dict()
        width=0.8
        for j, group in enumerate(dataframe[self.cat].unique()):
            group_df = dataframe.loc[dataframe[self.cat]==group, ["Significance", "R_mean"]].sort_values("R_mean")
            for i, roi in enumerate(group_df.index):
                if group_df.loc[roi, "Significance"] < 0.05:
                    ax.scatter(
                        x[group] + i * (width+0.2),
                        group_df.loc[roi, "R_mean"]+0.04,
                        color="b",
                        marker="*"
                    )
    
    def style(self, ax):
        super().style(ax)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=14)
        pass

class LayerHeatmap(Plot):
    def __init__(self):
        super().__init__()
    def format_df(self, dataframe):
        table = pd.pivot_table(dataframe, values="R", columns="Layer", index="ROI")
        if dataframe.loc[0,"Model"].startswith("vit"):
            table.columns = pd.Series([table.columns[i].split(".")[2] for i in range(len(table.columns))]).astype("int32")
        table.sort_index(axis=1, inplace=True)
        sorted_rois=table.max(axis=1).sort_values(ascending=False)
        table_sorted = table.loc[sorted_rois.index,:]
        p_table = pd.pivot_table(dataframe, values="Significance", columns="Layer", index="ROI") < 0.05
        p_table_sorted = p_table.loc[sorted_rois.index,:]
        p_table_sorted[p_table_sorted==1]="*"
        self.p_table=p_table_sorted
        return table_sorted
    
    def annotate(self, dataframe, ax):
        pass

    def style(self, ax):
        cbar = ax.collections[0].colorbar
        cbar.set_label("R", labelpad=15)
        ax.set_xticklabels([str(i) for i in range(1,len(ax.get_xticklabels())+1)])
        ax.tick_params(axis='both',labelsize=14, rotation=0)
        cax = ax.figure.axes[-1]
        cax.tick_params(labelsize=12, rotation=0)
        cax.set_label("R")
    
    def plot(self, dataframe):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(dataframe, cmap=sns.color_palette("viridis", as_cmap=True), annot=self.p_table, fmt="", ax=ax)
        return ax

class AllLayers(Plot):
    def __init__():
        super.__init__()
    def format_df(self, dataframe, roi_cat=None):
        df=dataframe.copy()
        df["processing_stage"]=[roi_cat.loc[roi.split("_")[0], "processing_stage"] for roi in df["ROI"].to_list()]
        df["location/stream"]=[roi_cat.loc[roi.split("_")[0], "location/stream"] for roi in df["ROI"].to_list()]
        df["selectivity"]=[roi_cat.loc[roi.split("_")[0], "selectivity"] for roi in df["ROI"].to_list()]
        if df.loc[0,"Model"].iloc[0].startswith("vit"):
            df["Layer"] = df["Layer"].apply(lambda x: x.split(".")[2]).astype("int32")
        df=df.explode("R_array")
        return df
    def plot(self, dataframe):
        fig = plt.figure(figsize=(12,15))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3)
        return fig

    def style(self, fig):
        pass
    def annotate(self, dataframe, fig):
        pass
class ModelPlot(Plot):
    def __init__():
        super.__init__()
    def format_df(dataframe):
        df = dataframe.copy()
        df["condition_tuples"] = list(zip(df["model_name"], df["time_window"], df["crop_size"], df["center_crop"]))
        df = df.explode("R_array")
        df.reset_index(inplace=True, drop=True)
        df.sort_values(["model_name", "time_window", "crop_size", "center_crop"], inplace=True)
        return df

class PValuePlot(ModelPlot):
    def __init__():
        super.__inti__()
        
    def format_df(dataframe):
        df = dataframe.copy()
        df.index = pd.MultiIndex.from_tuples(df.index)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.sort_index(axis=0, level=[0, 1, 2, 3])   # rows
        df = df.sort_index(axis=1, level=[0, 1, 2, 3])
        return df

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import matplotlib.pyplot as plt
    import json
    import os
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_1samp, sem, ttest_rel, false_discovery_control
    import pyarrow.parquet as pq

    data_path = 'results/resnet50_t=5_gs=224_99e6c1e4'

    reg_df_path = os.path.join(data_path, "eval_df_reg.parquet")
    roi_cat=pd.read_csv('roi_categorization.csv', index_col=0)
    roi_cat=roi_cat.rename(columns={"stream": "location/stream"})
    roi_cat.set_index("ROI", inplace=True)


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
    def grouping_lr(x):
        if x.str.endswith('_lh'):
            return 0
        elif x.str.endswith('_rh'):
            return 1

    def agg_ttest(x):
        
        y=x.reset_index(drop=True)
        return ttest_rel(y.iloc[0],y.iloc[1])[1]

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

    #plotter = Plotter()
    #plotter.plot("box", reg_pooled_fdr, figsize=(16,6), save=True, save_path='test_plots/box.pdf')

    #plotter = Plotter()
    #plotter.plot("roi_bar", reg_pooled_fdr, figsize=(16,6), save=True, save_path='test_plots/bar.pdf')

    #plotter = Plotter()
    #plotter.plot("layer_heatmap", reg_pooled_fdr, figsize=(4,6), save=True, save_path='test_plots/layer_heatmap.pdf')

    plotter = Plotter()
    plotter.plot("grouped_bar", reg_pooled_fdr, figsize=(16,6), save=True, save_path='test_plots/grouped_barplot.pdf', format_kwargs={"roi_cat": roi_cat, "cat": "selectivity"})