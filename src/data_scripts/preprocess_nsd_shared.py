import numpy as np
import os 
from scipy.spatial.distance import squareform
from utils import get_shared_rois_nsd

# making a list of ROI names that are shared across all subjects

roi_names = get_shared_rois_nsd()

# creating a new brain_data directory with one npz file per ROI 
# each npz file has size (n_subjects, n_stimuli, n_stimuli) as in other net2brain datasets
data_path = '../data/Algonauts23_shared_Net2Brain/'
save_dir = '../data/Algonauts23_shared_Net2Brain/brain_data'
for roi in roi_names:
    print(roi)
    roi_rdm = np.zeros((8, 872, 872))
    for subj in range(1, 9):
        subj_path = os.path.join(data_path, "subj0"+str(subj), "rdms")
        roi_filename = roi+"_subj0"+str(subj)+".npy"
        subj_roi = squareform(np.load(os.path.join(subj_path, roi_filename)))
        roi_rdm[subj-1,:,:] = subj_roi
    roi
    save_name = roi+"_RDMs.npz"
    np.savez(os.path.join(save_dir, save_name), roi_rdm)
