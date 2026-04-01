import hashlib
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, sem
from os import listdir

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
    sub01_path = '../data/Algonauts23_shared_Net2Brain/subj01/rdms'
    file_list = listdir(sub01_path)
    roi_names = []
    for filename in file_list:
        roi_names.append(filename.split("_subj01")[0])

    for subj in range(2, 9):
        subj_path = '../data/Algonauts23_shared_Net2Brain/subj0' + str(subj)+'/rdms'
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
