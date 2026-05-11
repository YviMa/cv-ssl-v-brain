import numpy as np
import os 
from scipy.spatial.distance import squareform
from utils import get_shared_rois_nsd,create_subset_mask_872

# making a list of ROI names that are shared across all subjects

roi_names = get_shared_rois_nsd()

# creating a new brain_data directory with one npz file per ROI 
# each npz file has size (n_subjects, n_stimuli, n_stimuli) as in other net2brain datasets

cat_list = ["objects", "scenes", "animals", "people"]
data_path = '../../data/Algonauts23_shared_Net2Brain/'
for cat in cat_list:
    save_dir = f'../../data/Algonauts23_shared_Net2Brain/{cat}/images'
    cat_mask = create_subset_mask_872(save_dir, os.path.join(data_path, 'images'))
    #print(np.sum(cat_mask))
    os.makedirs(os.path.join(save_dir, 'brain_data'), exist_ok=True)
    for roi in roi_names:
        print(roi)
        save_name = roi+"_RDMs.npz"
        roi_rdm = np.zeros((8, 100, 100))
        for subj in range(1, 9):
            os.makedirs(os.path.join(save_dir, f'subj0{subj}', 'rois'), exist_ok=True)
            subj_path = os.path.join(data_path, "subj0"+str(subj))
            roi_filename = roi+"_subj0"+str(subj)+".npy"
            full_rdm = np.load(os.path.join(subj_path, "rdms", roi_filename))
            subj_roi = squareform(full_rdm)[np.ix_(cat_mask, cat_mask)]
            roi_rdm[subj-1,:,:] = subj_roi
            roi_raw = np.load(os.path.join(subj_path, "rois", roi_filename))
            roi_cat = roi_raw[cat_mask,:]
            np.save(os.path.join(save_dir, f'subj0{subj}', 'rois', f'{roi}_subj0{subj}.npy'), roi_cat)
        np.savez(os.path.join(save_dir, 'brain_data', save_name), roi_rdm)

