import time

print("starting imports")

t = time.time()
import numpy as np
print("numpy done", time.time() - t)

t = time.time()
import os
import matplotlib.pyplot as plt
print("matplotlib", time.time() - t)

t = time.time()
import pandas as pd
print("pandas done", time.time() - t)
from matplotlib.gridspec import GridSpec
from abc import ABC, abstractmethod
import seaborn as sns
print("imports done")

def get_shared_rois_nsd():
    '''
    Retreives the names of the ROIs that are available for all 8 subjects
    '''
    sub01_path = '../../data/Algonauts23_shared_Net2Brain/subj01/rdms'
    file_list = os.listdir(sub01_path)
    roi_names = []
    for filename in file_list:
        roi_names.append(filename.split("_subj01")[0])

    for subj in range(2, 9):
        subj_path = '../../data/Algonauts23_shared_Net2Brain/subj0' + str(subj)+'/rdms'
        file_list = os.listdir(subj_path)
        subj_roi_names = []
        for filename in file_list:
            subj_roi_names.append(filename.split("_subj0")[0])
        intersection = list(set(subj_roi_names) & set(roi_names))
        roi_names[:] = intersection
    
    return roi_names

class Plotter():
    def __init__(self):
        self.PLOT_REGISTRY = {
            "all_layers_grouped": AllLayers,
            "activity_plot": ActivityPlot
        }      

    def plot(self, plot_type, dataframe, x=None, y=None, hue=None, figsize=None, save=False, save_path=None, init_kwargs=dict(), plot_kwargs=dict(), format_kwargs=dict()):
        PlotCLS = self.PLOT_REGISTRY[plot_type]
        plot_cls = PlotCLS(**init_kwargs)
        df = plot_cls.format_df(dataframe, **format_kwargs)
        ax = plot_cls.plot(df, figsize, **plot_kwargs)
        plot_cls.style(df, ax)
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
        self.y = "activity"
        self.hue = "category"
    def format_df(self, dataframe, roi_cat=None):
        df=dataframe.copy()
        df["processing_stage"]=[roi_cat.loc[roi.split("_")[0], "processing_stage"] for roi in df["ROI"].to_list()]
        df["location/stream"]=[roi_cat.loc[roi.split("_")[0], "location/stream"] for roi in df["ROI"].to_list()]
        df["selectivity"]=[roi_cat.loc[roi.split("_")[0], "selectivity"] for roi in df["ROI"].to_list()]
        df=df.explode("activity")
        return df
    def style(self, dataframe, fig):
        for ax in fig.axes:
            ax.legend(frameon=True)
            #ax.set_xticklabels(np.arange(1,12))
            ax.set_ylabel("Mean activity")
            #ax.set_ylim(0,0.6)

        plt.suptitle(f"Mean activity per ROI", fontsize=18, y=0.92)

    def annotate(self, dataframe, fig):
        return None
    

cats = ["objects", "scenes", "animals", "people"]
data_path = "../../data/Algonauts23_shared_Net2Brain/"
shared_rois = get_shared_rois_nsd()
activity_list = []
for cat in cats:
    for subj in range(1,9):
        subj_dir = os.path.join(data_path, "subj0"+str(subj), "rois")
        for roi in shared_rois:
            f = os.path.join(data_path, cat, "subj0"+str(subj), "rois", roi+"_"+"subj0"+str(subj)+".npy")
            data = np.load(f)
            mean = np.mean(data)
            roi_dict = {"category": cat, "subject": subj, "roi": roi, "activity": mean}
            activity_list.append(roi_dict)

activity_df = pd.DataFrame(activity_list)
activity_df = activity_df.groupby(["roi", "category"]).agg(list)
activity_df["activity"] = activity_df["activity"].apply(lambda x: np.array(x))
activity_df.reset_index(inplace=True)
activity_df.drop("subject", axis=1, inplace=True)

#activity_df.to_csv('activity_by_category.csv', index=False)
#activity_df = pd.read_csv('activity_by_category.csv')
roi_list_lh = activity_df["roi"].apply(lambda x: x.split("_lh")[0]).unique()
roi_list_rh = activity_df["roi"].apply(lambda x: x.split("_rh")[0]).unique()
intersect_roi = list(np.intersect1d(roi_list_lh, roi_list_rh))
activity_pooled = activity_df.copy()

activity_pooled["ROI_non_handed"] = activity_pooled["roi"].apply(lambda x: x[:-3])
activity_pooled = activity_pooled.loc[activity_pooled["ROI_non_handed"].isin(intersect_roi),:]
activity_pooled.drop("roi", axis=1, inplace=True)
print(activity_pooled.head())
activity_pooled["activity"]= activity_pooled.groupby(["ROI_non_handed", "category"])["activity"].transform('mean')
activity_pooled.reset_index(inplace=True)
activity_pooled.rename(columns={"ROI_non_handed":"ROI"}, inplace=True)

roi_cat=pd.read_csv('roi_categorization.csv', index_col=0)
roi_cat=roi_cat.rename(columns={"stream": "location/stream"})
roi_cat.set_index("ROI", inplace=True)

plotter = Plotter()
plotter.plot("activity_plot", activity_pooled, figsize=(12,10), save=True, save_path='test_plots/activity_plot.pdf', format_kwargs={"roi_cat": roi_cat})

