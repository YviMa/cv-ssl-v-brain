import hashlib
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, sem, ttest_rel, false_discovery_control
import os
from os import listdir
from net2brain.rdm.feature_iterator import nsorted

def compose_model_dir(model_name, time_window, crop_size, center_crop=False):
    '''
    Creates model checkpoint file name from config parameters.
    '''
    if center_crop == False:
        return model_name + "_t=" + str(time_window) + "_gs=" + str(crop_size) + ".ckpt"
    else:
        return model_name + "_t=" + str(time_window) + "_gs=" + str(crop_size) + "_center_crop" + ".ckpt"

def hash_config(config_path):
    '''
    Computes hash from config file.
    '''
    with open(config_path, 'rb') as f:
        hash_func = hashlib.new("shake128")
        hash_func.update(f.read())
        file_hash = hash_func.hexdigest(length=4)
    return file_hash

def compose_dir_name(config_path, model_name, time_window, crop_size, center_crop=False):
    '''
    Creates the name of results directory from config parameters and config hash.
    '''
    if center_crop == False:
        config_hash = hash_config(config_path)
        model_specs = model_name + "_t=" + str(time_window) + "_gs=" + str(crop_size) 
        results_dir = model_specs+"_"+config_hash
        return results_dir
    else:
        config_hash = hash_config(config_path)
        model_specs = model_name + "_t=" + str(time_window) + "_gs=" + str(crop_size) 
        results_dir = model_specs+"_"+config_hash + "_center_crop"
        return results_dir

def get_shared_rois_nsd():
    '''
    Retreives the names of the ROIs that are available for all 8 subjects
    '''
    sub01_path = '../../data/Algonauts23_shared_Net2Brain/subj01/rdms'
    file_list = listdir(sub01_path)
    roi_names = []
    for filename in file_list:
        roi_names.append(filename.split("_subj01")[0])

    for subj in range(2, 9):
        subj_path = '../../data/Algonauts23_shared_Net2Brain/subj0' + str(subj)+'/rdms'
        file_list = listdir(subj_path)
        subj_roi_names = []
        for filename in file_list:
            subj_roi_names.append(filename.split("_subj0")[0])
        intersection = list(set(subj_roi_names) & set(roi_names))
        roi_names[:] = intersection
    
    return roi_names

def get_shared_rois_nsd_all(subj_numbers):
    '''
    Retreives names of the ROIs that are available for the 4 subjects that viewed the full
    set of stimuli in the complete NSD data.
    subj_numbers must be a list of ints
    '''
    first_path = '../../data/DatasetAlgonauts_NSD/Algonauts23_Net2Brain/subj0'+str(subj_numbers[0])+'/rois'
    file_list = listdir(first_path)
    roi_names = []
    for filename in file_list:
        roi_names.append(filename.split("_subj01")[0])

    for subj in subj_numbers[1:]:
        subj_path = '../../data/DatasetAlgonauts_NSD/Algonauts23_Net2Brain/subj0' + str(subj)+'/rois'
        file_list = listdir(subj_path)
        subj_roi_names = []
        for filename in file_list:
            subj_roi_names.append(filename.split("_subj0")[0])
        intersection = list(set(subj_roi_names) & set(roi_names))
        roi_names[:] = intersection
    
    return roi_names

def r_list(a):
    # helper function for brining encoding dataframe to same format as RSA dataframe
    return list(a)
def get_p_value(a):
    # helper function computing p-value across subjects from linear encoding
    return ttest_1samp(a,0)[1]

def summarize_subjects(shared_rois, eval_df):
    '''
    Creates a common dataframe for all subjects that shares the same
    format as the dataframe returned by RSA().
    '''
    roi_dfs = []
    for roi in shared_rois:
        mask=eval_df["ROI"].str.startswith(roi)
        sub_df = eval_df.loc[mask,:]
        roi_df =sub_df.groupby('Layer')["R"].aggregate(r_list)
        roi_df=roi_df.reset_index()
        roi_df=roi_df.rename(columns = {"R": "R_array"})
        roi_df.insert(0,"ROI", roi)
        roi_df.insert(2,"Model", sub_df["Model"].iloc[0])
        roi_df.insert(3, "R", roi_df["R_array"].map(np.mean))
        roi_df.insert(4, "%R", np.nan)
        roi_df["Significance"]=roi_df["R_array"].map(get_p_value)
        roi_df["SEM"]=roi_df["R_array"].map(sem)
        roi_df["LNC"]=np.nan
        roi_df["UNC"]=np.nan
        roi_dfs.append(roi_df)

    eval_df = pd.concat(roi_dfs)
    eval_df.reset_index(inplace=True, drop=True)
    return eval_df

def fix_naming(dataframe):
    '''
    Adjusts the entries of ROI and Layer of the linear encoding results
    dataframe to match the naming conventions of the dataframe returned by RSA().
    '''
    df = dataframe.copy()
    new_ROIs = df["ROI"].map(lambda x: x.split(") ")[1])
    new_ROIs = new_ROIs.map(lambda x: x.split("_RDMs")[0])
    df["ROI"] = new_ROIs

    new_layers = df["Layer"].map(lambda x: x.split("_",maxsplit=1)[1])
    new_layers = new_layers.map(lambda x: x.split(".npz")[0])
    new_layers = new_layers.map(lambda x: x.replace("_", "."))
    df["Layer"] = new_layers

    return df

def agg_ttest(x):
    y=x.reset_index(drop=True)
    return ttest_rel(y.iloc[0],y.iloc[1])[1]

def pool_lhrh(dataframe1):
    """
    Pools left- and right- hand side of ROIs and selects best layer each.

    Returns: Dataframe with one row per ROI containing best layers for both models and correlations.
    """
    roi_list_lh = dataframe1["ROI"].apply(lambda x: x.split("_lh")[0]).unique()
    roi_list_rh = dataframe1["ROI"].apply(lambda x: x.split("_rh")[0]).unique()
    intersect_roi = list(np.intersect1d(roi_list_lh, roi_list_rh))
    # saw that for vit_base_t=0_gs_112 we can't pool V3d lh/rh due to significant difference
    # therefore can't pool for any model for comparison
    intersect_roi.remove("V3d")

    pool = dataframe1.copy()

    pool["ROI_non_handed"] = pool["ROI"].apply(lambda x: x[:-3])
    pool = pool.loc[pool["ROI_non_handed"].isin(intersect_roi),:]
    pool= pool.groupby(["ROI_non_handed", "Layer"])[["R", "%R", "R_array","LNC", "UNC"]].agg('mean')
    pool.reset_index(inplace=True)
    pool.insert(2, "Model", dataframe1.loc[0,"Model"])
    pool.rename(columns={"ROI_non_handed":"ROI"}, inplace=True)

    no_pool = dataframe1.copy()
    no_pool["ROI_non_handed"] = no_pool["ROI"].apply(lambda x: x[:-3])
    no_pool=no_pool.loc[~no_pool["ROI_non_handed"].isin(intersect_roi),:]
    no_pool.drop(["ROI_non_handed", "Significance", "SEM"], axis=1, inplace=True)
    
    pooled=pd.concat([pool, no_pool])
    pooled["SEM"]=pooled["R_array"].apply(sem)
    pooled["Significance"]=pooled["R_array"].apply(lambda x: ttest_1samp(x,0)[1])
    pooled.reset_index(inplace=True, drop=True)

    return pooled

def strip_hash(x):
    splits = x.split("_")
    if len(splits)==4 or len(splits)==6:
        splits.pop(3)
    elif len(splits)==5 or len(splits)==7:
        splits.pop(4)
    splits = [s + "_" for s in splits[:-1]] + [splits[-1]]
    return "".join(splits)

def create_subset_mask_872(folder, all_img_dir):
    '''
    Creates mask for the full 872 image brain RDMs to select only the rows and columns that correspond 
    to the stimuli in the provided folder.

    Returns: np.array of bool
    '''
    file_list = listdir(folder)
    file_list = [x for x in file_list if os.path.isfile(os.path.join(folder, x))]

    total_img_list = listdir(all_img_dir)
    total_img_list = [x for x in total_img_list if os.path.isfile(os.path.join(all_img_dir, x))]

    file_list_sorted = nsorted(file_list)
    total_img_list_sorted = nsorted(total_img_list)

    mask = [1 if file in file_list_sorted else 0 for file in total_img_list_sorted]

    return np.array(mask).astype('bool')
