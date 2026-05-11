import numpy as np
import pandas as pd
import os 
from scipy.spatial.distance import squareform
from utils import get_shared_rois_nsd,create_subset_mask_872

# making a list of ROI names that are shared across all subjects

roi_names = get_shared_rois_nsd()

# creating a new brain_data directory with one npz file per ROI 
# each npz file has size (n_subjects, n_stimuli, n_stimuli) as in other net2brain datasets

cat_list = ["objects", "scenes", "animals", "people"]
data_path = '../../data/Algonauts23_shared_Net2Brain/'
img_list = []
for cat in cat_list:
    data_dir = f'../../data/Algonauts23_shared_Net2Brain/{cat}/images'
    file_list = os.listdir(data_dir)
    file_list = [x for x in file_list if os.path.isfile(os.path.join(data_dir, x))]
    for img in file_list:
        algonauts_id = img.split("-")[1].split("_")[0]
        nsd_id = img.split("-")[2].split(".")[0]
        category = cat
        img_dict={"algonauts_id": algonauts_id, 
         "nsd_id": nsd_id, 
         "file_name": img, 
         "category": cat}
        img_list.append(img_dict)

img_df = pd.DataFrame(img_list)
img_df.to_csv('nsd_shared_categories.csv', index=False)