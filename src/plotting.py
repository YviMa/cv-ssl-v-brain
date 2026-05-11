import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from utils import strip_hash, agg_ttest
import os
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp, sem, ttest_rel, false_discovery_control
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec
from abc import ABC, abstractmethod
from matplotlib.colors import ListedColormap

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
            "model_bar": ModelBarPlot,
            "line_plot": LinePlot,
            "line_plot_diff": LinePlotDiff,
            "activity_plot": ActivityPlot
        }      

    def plot(self, plot_type, dataframe, x=None, y=None, hue=None, figsize=None, init_kwargs=dict(), plot_kwargs=dict(), format_kwargs=dict()):
        PlotCLS = self.PLOT_REGISTRY[plot_type]
        plot_cls = PlotCLS(**init_kwargs)
        df = plot_cls.format_df(dataframe, **format_kwargs)
        ax = plot_cls.plot(df, figsize, **plot_kwargs)
        plot_cls.style(df, ax)
        plot_cls.annotate(df, ax)
        return ax
    
    def save(self, ax, save_path):
        if isinstance(ax, Axes):
            fig = ax.get_figure()
        elif isinstance(ax, sns.axisgrid.FacetGrid):
            fig = ax.fig
        else:
            fig = ax
        fig.savefig(save_path, format='PDF', bbox_inches='tight')


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
    def style(self, dataframe, ax):
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
    def style(self, dataframe, ax):
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

    def style(self, dataframe, ax):
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
    
    def style(self, dataframe, ax):
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

    def style(self, dataframe, ax):
        cbar = ax.collections[0].colorbar
        cbar.set_label("R", labelpad=15)
        ax.set_xticklabels([str(i) for i in range(1,len(ax.get_xticklabels())+1)])
        ax.tick_params(axis='both',labelsize=14, rotation=0)
        cax = ax.figure.axes[-1]
        cax.tick_params(labelsize=12, rotation=0)
        cax.set_label("R")
    
    def plot(self, dataframe, figsize, **plot_kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(dataframe, cmap=sns.color_palette("viridis", as_cmap=True), annot=self.p_table, fmt="", ax=ax)
        return ax

class AllLayers(Plot):
    def __init__(self):
        super().__init__()
        self.x = "Layer"
        self.y = "R_array"
        self.hue = "ROI"
    def format_df(self, dataframe, roi_cat=None):
        df=dataframe.copy()
        df["processing_stage"]=[roi_cat.loc[roi.split("_")[0], "processing_stage"] for roi in df["ROI"].to_list()]
        df["location/stream"]=[roi_cat.loc[roi.split("_")[0], "location/stream"] for roi in df["ROI"].to_list()]
        df["selectivity"]=[roi_cat.loc[roi.split("_")[0], "selectivity"] for roi in df["ROI"].to_list()]
        df=df.explode("R_array")
        if df.loc[0,"Model"].iloc[0].startswith("vit"):
            df["Layer"] = df["Layer"].apply(lambda x: x.split(".")[2]).astype("int32")
        return df
    def plot(self, dataframe, figsize, **plot_kwargs):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3)
        nrows, ncols = gs.get_geometry()
        cats = sorted(dataframe["selectivity"].unique())
        ii, ij = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
        ixpairs = list(np.reshape(np.stack([ii, ij], axis=2), (-1,2)))
        ixpairs.pop(0)
        for idx, ixpair in enumerate(ixpairs):
            if ixpair[0]==0:
                ax = fig.add_subplot(gs[ixpair[0],:])
            else:
                ax = fig.add_subplot(gs[ixpair[0], ixpair[1]])
            cat=cats[idx]
            sns.barplot(data=dataframe.loc[dataframe["selectivity"]==cat,:], x=self.x, y=self.y, hue=self.hue, ax=ax, capsize=0.5)
            ax.set_title(f"{cat} ROIs")

        return fig

    def style(self, dataframe, fig):
        for ax in fig.axes:
            ax.legend(frameon=True)
            ax.set_xticklabels(np.arange(1,12))
            ax.set_ylabel("R")
            ax.set_ylim(0,0.6)

        plt.suptitle(f"R across Layers for all ROIs", fontsize=18, y=0.92)

    def annotate(self, dataframe, fig):
        cats = sorted(dataframe["selectivity"].unique())
        for idx, ax in enumerate(fig.axes):
            cat=cats[idx]
            df = dataframe.copy()
            df = df.loc[dataframe["Significance"]<0.05, :]
            df["annotation"] = df["R"]+0.07
            sns.stripplot(data=df.loc[df["selectivity"]==cat,:], x=self.x, y="annotation", hue=self.hue, dodge=True, jitter=False, ax=ax, marker='*', s=10, legend=False)

class ActivityPlot(AllLayers):
    def __init__(self):
        super().__init__()
        self.x = "ROI"
        self.y = "Activity_array"
        self.hue = "category"
    def style(self, dataframe, fig):
        super().style(dataframe, fig)
        for ax in fig.axes:
            ax.set_ylabel("Mean activity")
        plt.suptitle(f"Mean activity per ROI", fontsize=18, y=0.92)

    def annotate(self, dataframe, fig):
        return None

class ActivityPlot(AllLayers):
    def __init__(self):
        super().__init__()
        self.x = "ROI"
        self.y = "Activity_array"
        self.hue = "category"
    def style(self, dataframe, fig):
        super().style(dataframe, fig)
        for ax in fig.axes:
            ax.set_ylabel("Mean activity")
        plt.suptitle(f"Mean activity per ROI", fontsize=18, y=0.92)

    def annotate(self, dataframe, fig):
        return None

class ModelPlot(Plot):
    def __init__(self):
        super().__init__()
    def format_df(self,dataframe):
        df = dataframe.copy()
        df["condition_tuples"] = list(zip(df["model_name"], df["time_window"], df["crop_size"], df["center_crop"]))
        df = df.explode("R_array")
        df.reset_index(inplace=True, drop=True)
        df.sort_values(["model_name", "time_window", "crop_size", "center_crop"], inplace=True)
        return df
    def create_model_labels(self,cond_tuple):
        if cond_tuple[3]==False:
            return f"{cond_tuple[0]}_t={cond_tuple[1]}_gs={cond_tuple[2]}"
        else:
            return f"{cond_tuple[0]}_t={cond_tuple[1]}_gs={cond_tuple[2]}"+"_center_crop"

class PValuePlot(ModelPlot):
    def __init__(self):
        super().__init__()
        colors1 = plt.cm.Blues(np.linspace(0, 0.8, 6))
        colors1[0,:]=[0.6, 0.6, 0.6, 1]
        colors2 = plt.cm.Reds_r(np.linspace(0.2, 1, 6))
        colors2[-1,:]=[0.6, 0.6, 0.6, 1]
        colors = np.vstack((colors1, colors2))
        self.cmap = ListedColormap(colors)
        palette = sns.color_palette("coolwarm", n_colors=8)
        
    def format_df(self, dataframe):
        return dataframe

    def plot(self, dataframe, figsize, roi=""):
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones(dataframe.shape, dtype=bool))
        sns.heatmap(dataframe, annot=False, fmt=".2f", cmap=self.cmap, cbar=True, ax=ax, linewidths=0.5, vmin=-0.06, vmax=0.06, mask=mask)
        #roi = dataframe["ROI"].unique()
        ax.set_title(f"P-values of pairwise model comparison, ROI: {roi}", fontsize=18)
        return ax

    def style(self, dataframe, ax):
        model_labels_x = pd.Series(dataframe.index.to_series().apply(self.create_model_labels))
        model_labels_y = pd.Series(dataframe.columns.to_series().apply(self.create_model_labels))
        ax.set_xticklabels(model_labels_x, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(model_labels_y, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = ax.collections[0].colorbar
        cbar.set_label('p-value', rotation=0, labelpad=15)
        cbar_ticklabels = list(np.round(np.arange(-0.050, 0.051, 0.01),2))
        cbar_ticks = list(np.arange(-0.05, 0.051, 0.01))
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels, fontsize=12)
        cbar.set_label("p-value", fontsize=16)
        
    def annotate(self, dataframe, ax):
        pass

class ModelBarPlot(ModelPlot):
    def __init__(self):
        super().__init__()
        self.palette = sns.color_palette('pastel')

    def format_df(self, dataframe):
        df=super().format_df(dataframe)
        return df

    def plot(self, dataframe, figsize, roi=""):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="Model", y="R_array", color=self.palette[0], data=dataframe, errorbar='se', capsize=0.2, errwidth=1)
        ax.set_title(f"reg: Mean R across subjects (n=8) for all models, ROI: {roi}", fontsize=18)
        return ax

    def style(self, dataframe, ax):
        ax.set_xticklabels(self.create_model_labels(dataframe))
        ax.tick_params(axis='x', labelrotation=45, size=12, labelsize=12)
        plt.setp(ax.get_xticklabels(),ha='right')
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel("R (pearson)", fontsize=16)
        ax.set_xlabel("ROI", fontsize=16)

    def create_model_labels(self, dataframe):
        #models=dataframe.index.unique()
        model_labels = pd.Series(dataframe["Model"].unique()).apply(strip_hash).to_list()
        return model_labels

    def annotate(self,dataframe, ax):
        x = np.arange(len(dataframe["Model"].unique()))
        y = dataframe.groupby("Model", sort=False)["R"].agg("first")+0.04
        y.reset_index(inplace=True, drop=True)
        mask = dataframe.groupby("Model", sort=False)["Significance"].agg("first") < 0.05
        mask.reset_index(inplace=True, drop=True)
        ax.scatter(x[mask],y[mask], marker=(8,2,0), lw=0.5, color='b')

class AllModelsHeatmap(ModelPlot):
    def __init__(self):
        super().__init__()

    def format(self, dataframe):
        df = reg_comp.sort_values(by="R")
        df["condition_tuple"]=list(zip(df["model_name"], df["time_window"], df["crop_size"], df["center_crop"]))
        df = pd.pivot_table(data=df, values="R", index="ROI", columns="condition_tuple", sort=False)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.sort_index(axis=1, level=[0, 1, 2, 3])   
        return df

    def plot(self, dataframe, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(dataframe, ax=ax)
        return ax

    def style(self, dataframe, ax):
        model_labels_x = pd.Series(dataframe.columns.to_series().apply(self.create_model_labels))
        ax.set_xticklabels(model_labels_x, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.set_title("R values for all Models and all ROIs, ROIs sorted by max. sum of R over all models")
        ax.set_ylabel("")
        cbar = ax.collections[0].colorbar
        cbar.set_label('R', rotation=0, labelpad=15)
        cbar.set_ticklabels(np.round(cbar.get_ticks(),2), fontsize=12)
    
class LinePlot(Plot):
    def __init__(self, x_var, other_value, sel, row=None):
        super().__init__()
        self.x_var = x_var
        if self.x_var == "time_window":
            self.other = "crop_size"
        else:
            self.other = "time_window"
        self.other_value = other_value
        self.sel = sel
        self.row = row
    def format_df(self, dataframe, roi_cat=None):
        df = dataframe.copy()
        df["ROI_non_handed"] = df["ROI"].apply(lambda x: x.split("_")[0])
        df["selectivity"]=[roi_cat.loc[roi, "selectivity"] for roi in df["ROI_non_handed"].to_list()]
        df=df.loc[(df["selectivity"]==self.sel) & 
                    (df["center_crop"]==False) & 
                    (df[self.other]==self.other_value),:]
        df=df.explode("R_array")
        return df
    
    def plot(self, dataframe, figsize):
        height, aspect = figsize
        g = sns.FacetGrid(dataframe, row=self.row, col="model_name", hue="ROI", height=height, aspect=aspect)
        g.map(sns.lineplot, self.x_var, "R_array", lw=4, errorbar='se', marker="o", markersize=4)
        #g.map(sns.scatterplot, self.x_var, "R", markersize=4)
        return g
        
    def style(self, dataframe, g):
        max_y = dataframe["R"].max()
        g.set(ylim=(0,max_y+0.1))
        #g.set_titles("{col_name}")
        g.set_xlabels(self.x_var)
        g.set_ylabels("R")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.add_legend()

        plt.suptitle(f"R depending on {self.x_var} for {self.sel} ROIs, {self.other} = {self.other_value}", fontsize=18, y=1.08)
    def annotate(self, dataframe, g):
        '''def plot_sig(data, **kws):
            sig = data.loc[data["Significance"] < 0.05]

            if sig.empty:
                return

            ax = plt.gca()  # current facet axis

            ax.scatter(sig[self.x_var], sig["dR_mean"] + 0.03, marker="*", s=80, color="black", zorder=10)

        g.map_dataframe(plot_sig)'''
        pass

class LinePlotDiff(LinePlot):
    def __init__(self, x_var, other_value, sel, row=None):
        super().__init__(x_var, other_value, sel, row)
    def format_df(self, dataframe, roi_cat=None):
        df = dataframe.copy()
        #df["ROI_non_handed"] = df["ROI"].apply(lambda x: x.split("_")[0])
        roi_list = df["ROI"].apply(lambda x: x.split("_")[0]).to_list()
        df["selectivity"]=[roi_cat.loc[roi, "selectivity"] for roi in roi_list]
        df=df.loc[(df["selectivity"]==self.sel) & 
                    (df["center_crop"]==False) & 
                    (df[self.other]==self.other_value),:]
        df = df.sort_values(["model_name", "ROI", self.x_var])
        #x_var_vals = df[self.x_var].unique()
        #x_var_vals_shifted = np.roll(x_var_vals, 1)
        #pairs = [[x1, x2] for x1, x2 in zip(x_var_vals[1:], x_var_vals_shifted[1:])]
        #for pair in pairs:
            #row_1 = df.loc[df[self.x_var]==pair[0],:]
            #row_2 = df.loc[df[self.x_var]==pair[1],:]
            #print(row_1.groupby(["model_name", "ROI"]))
            #groups_1 = row_1.groupby(["model_name", "ROI"]).agg('fist')
            #groups_2 = row_2.groupby(["model_name", "ROI"]).agg('fist')

        df["R_array"] = df["R_array"].apply(np.array)
        if "category" in df.columns:
            df["R_prev"] = df.groupby(["model_name", "ROI", "category"])["R_array"].shift() 
        else:
            df["R_prev"] = df.groupby(["model_name", "ROI"])["R_array"].shift() 

        df[self.x_var] = pd.to_numeric(df[self.x_var])
        df = df.reset_index(drop=True)
        df = df.dropna(subset=["R_prev"]).copy()
        df["dR_array"] = df.apply(
            lambda row: row["R_array"] - row["R_prev"],
            axis=1
        )    
        df["dR_mean"] = df["dR_array"].apply(np.mean)
        df["Significance"] = df["dR_array"].apply(lambda x: ttest_1samp(x,0)[1])
        df["Significance"] = false_discovery_control(df["Significance"].to_numpy())

        df = df.explode(["dR_array"])

        df = df.rename(columns={"dR_array": "dR"})
        df["dR"] = df["dR"].astype(float)  
        #print(df.head(10))
        return df
    
    def plot(self, dataframe, figsize):
        height, aspect = figsize
        g = sns.FacetGrid(dataframe, row=self.row, col="model_name", hue="ROI", height=height, aspect=aspect)
        g.map(sns.lineplot, self.x_var, "dR", lw=5, errorbar='se', marker="o", markersize=7)
        #g.map(sns.scatterplot, self.x_var, "dR_mean")
        return g
    def annotate(self, dataframe, g):
        #df = dataframe.groupby(["model_name", "ROI"])["dR"].agg(lambda x: ttest_1samp(x,0)[1])
        rois = dataframe["ROI"].unique()
        offs = np.arange(len(rois))*0.03+0.02
        roi_offsets = {roi: off for roi, off in zip(rois, offs)}
        def plot_significance(data, **kws):
            sig_data = data.loc[data["Significance"] < 0.05]
            
            if not sig_data.empty:
                current_roi = sig_data["ROI"].iloc[0]
                offset = roi_offsets.get(current_roi, 0.03) 
                
                sns.scatterplot(
                    data=sig_data,
                    x=self.x_var,
                    y=sig_data["dR_mean"]+offset,
                    marker="*",
                    s=200,
                    color=kws.get("color"), 
                    legend=False,
                    ax=plt.gca()
                )

        g.map_dataframe(plot_significance)
        
        
    def style(self, dataframe, g):
        super().style(dataframe,g)
        max_y = dataframe["dR_mean"].max()
        min_y = dataframe["dR_mean"].min()
        g.set(ylim=(min_y-0.1,max_y+0.15))
    

'''if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import matplotlib.pyplot as plt
    import json
    import os
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_1samp, sem, ttest_rel, false_discovery_control
    import pyarrow.parquet as pq
    
    roi_cat=pd.read_csv('roi_categorization.csv', index_col=0)
    roi_cat=roi_cat.rename(columns={"stream": "location/stream"})
    roi_cat.set_index("ROI", inplace=True)

    #data_path = 'results/872_shared_total/resnet50_t=5_gs=224_99e6c1e4'

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

    #plotter = Plotter()
    #plotter.plot("grouped_bar", reg_pooled_fdr, figsize=(16,6), save=True, save_path='test_plots/grouped_barplot.pdf', format_kwargs={"roi_cat": roi_cat, "cat": "selectivity"})

    #plotter = Plotter()
    #plotter.plot("all_layers_grouped", reg_pooled_fdr, figsize=(12,10), save=True, save_path='test_plots/all_layers.pdf', format_kwargs={"roi_cat": roi_cat})


    comp_dir = "results/872_shared_total/model_comparison"
    reg_comp_table = pq.read_table(os.path.join(comp_dir, "reg_model_comp_all.parquet"), partitioning=None)
    reg_comp = reg_comp_table.to_pandas()
    roi_df = reg_comp.loc[reg_comp["ROI"]=="EBA",:].copy()
    roi_df["Significance"]=false_discovery_control(roi_df["Significance"])
    roi_df["condition_tuples"] = list(zip(roi_df["model_name"], roi_df["time_window"], roi_df["crop_size"], roi_df["center_crop"]))
    #roi_df.reset_index(inplace=True, drop=True)
    roi_df = roi_df.sort_values("condition_tuples").reset_index(drop=True)
    
    M = np.zeros((26,26))
    sign_corr = np.zeros((26,26))
    for i, row_1 in roi_df.iterrows():
        for j, row_2 in roi_df.iterrows():
            M[i,j]=ttest_rel(row_1["R_array"], row_2["R_array"])[1]
            sign_corr[i,j]=np.sign(row_1["R"] - row_2["R"])

    tril_idx = np.tril_indices(26, k=-1)
    p_value_FDR = false_discovery_control(M[tril_idx].flatten())
    M=np.zeros((26,26))
    M[tril_idx]=p_value_FDR
    p_value_df=pd.DataFrame(M, index=roi_df["condition_tuples"], columns=roi_df["condition_tuples"])
    p_value_df.mask(p_value_df>=0.05, 0.051, inplace=True)
    p_value_df=p_value_df*sign_corr

    #plotter = Plotter()
    #plotter.plot("p_value_heatmap", p_value_df, figsize=(12,10), save=True, save_path='test_plots/pval_fdr.pdf', plot_kwargs={"roi": "EBA"})

    #plotter = Plotter()
    #plotter.plot("model_bar", roi_df, figsize=(12,6), save=True, save_path='test_plots/model_bar.pdf', plot_kwargs={"roi": "EBA"})

    #plotter = Plotter()
    #plotter.plot("line_plot", reg_comp, figsize=(4,1.3), save=True, save_path='test_plots/line_plot.pdf', init_kwargs={"x_var" :"time_window", "other_value": 224, "sel": "body-selective"}, format_kwargs={"roi_cat": roi_cat})

    plotter = Plotter()
    plotter.plot("line_plot_diff", reg_comp, figsize=(4,1.3), save=True, save_path='test_plots/line_plot_diff.pdf', init_kwargs={"x_var" :"time_window", "other_value": 224, "sel": "body-selective"}, format_kwargs={"roi_cat": roi_cat})'''