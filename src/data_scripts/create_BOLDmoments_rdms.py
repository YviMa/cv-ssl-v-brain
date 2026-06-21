import os
import sys
import pandas as pd
import numpy as np
from net2brain.rdm_creation import RDMCreator
from net2brain.rdm.dist import correlation
from net2brain.rdm.dist_utils import dist
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import get_shared_rois_BOLDmoments


rdm_path = "../../data/BOLDmoments/rdms/"
all_subj_dir = "../../data/BOLDmoments/ds005165/derivatives/versionB/MNI152/prepared_allvoxel_pkl"
subj_dir_list = os.listdir(all_subj_dir)
roi_list = get_shared_rois_BOLDmoments()
print(roi_list)
for roi in roi_list:
    rdm_list = []
    for subj_dir in subj_dir_list:
        data = np.load(os.path.join(all_subj_dir, subj_dir, subj_dir+"_roi-"+roi + "_betas_normalized.pkl" ), allow_pickle=True)
        train_data = np.mean(data["train_data_allvoxel"], axis=1)
        test_data = np.mean(data["test_data_allvoxel"], axis=1)
        
        combined_data = np.concatenate([train_data, test_data], axis = 0)
        rdm = dist(combined_data, metric="correlation", chunk_size=None, verbose=True, device='cpu')
        rdm_list.append(rdm)

    all_subj_rdm = np.stack(rdm_list, axis = 0)
    np.savez(os.path.join(rdm_path, roi + "_RDMs.npz"), all_subj_rdm)
