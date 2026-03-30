import yaml
import argparse
import torch
import json
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from os import makedirs
from os.path import join
from cv_ssl.solo.methods import METHODS
from omegaconf import OmegaConf
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from net2brain.evaluations.rsa import RSA
from net2brain.evaluations.encoding import Linear_Encoding
from utils import compose_dir_name, compose_model_dir, get_shared_rois_nsd, summarize_subjects, fix_naming

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

config_path = args.config

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# loading the model
model_config = config["model"]
model_dir = model_config["directory"]
model_name = model_config["name"]
time_window = model_config["time_window"]
crop_size = model_config["crop_size"]
center_crop = model_config["center_crop"]

checkpoint_path = join(model_dir, compose_model_dir(model_name, time_window, crop_size, center_crop))
ckpt = torch.load(checkpoint_path, weights_only=False)
cfg = OmegaConf.create(ckpt["args"])

model = METHODS[cfg["method"]](cfg)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
#device = torch.device("cuda:0")
#model.to(device)

# feature extraction
data_config = config["data"]
stimuli_path = data_config["stimuli_path"]

layers_to_extract = config["feature_extraction"]["layers"]

dir_name = compose_dir_name(config_path, model_name, time_window, crop_size, center_crop)
feature_path = "tmp/features/" + dir_name + "_feat" 
fx = FeatureExtractor(model=model, device='cpu') 
fx.extract(data_path=stimuli_path, save_path = feature_path, layers_to_extract=layers_to_extract, consolidate_per_layer=False)

# rdm creation
rdm_path = "tmp/rdms/" + dir_name + "_rdm"
rdm_creator = RDMCreator(verbose=True, device='cpu') 
rdm_creator.create_rdms(feature_path=feature_path, save_path=rdm_path, save_format='npz')

# evaluation
brain_rdms = data_config["brain_rdms"] 
analysis_config = config["analysis"]
results_dir = "results/" + dir_name
makedirs(results_dir, exist_ok=True)

with open(join(results_dir, "config.yaml"), 'w') as f:
    yaml.dump(config, f)

metadata = {**model_config, "layers": layers_to_extract}
metadata_key = 'custom_meta'

if analysis_config["rsa"]["execute"] == True:
    evaluation =  RSA(rdm_path, brain_rdms, model_name=dir_name, squared=False)
    distance_metric = analysis_config["rsa"]["distance_metric"]
    eval_df = evaluation.evaluate(distance_metric=distance_metric)
    eval_df = fix_naming(eval_df)
    eval_table = pa.Table.from_pandas(eval_df)

    metadata["evaluation"]="rsa"
    metadata_json = json.dumps(metadata)
    original_meta = eval_table.schema.metadata
    combined_meta = {metadata_key.encode(): metadata_json.encode(), **original_meta}
    eval_table = eval_table.replace_schema_metadata(combined_meta)
    pq.write_table(eval_table, join(results_dir,"eval_df_rsa.parquet"))

    # in order to avoid rerunning evaluations for model comparison later
    with open(join(results_dir,"RSA_instance.pkl"), 'wb') as f:
        pickle.dump(evaluation, f)

if analysis_config["reg"]["execute"]==True:
    reg_config = analysis_config["reg"]
    roi_paths = [join(data_config["data_dir"],"subj0"+str(j), "rois") for j in range(1,9)]

    eval_df = Linear_Encoding(feat_path=feature_path,  
                roi_path=roi_paths,
                model_name=dir_name,
                trn_tst_split=reg_config["trn_tst_split"],
                n_folds=reg_config["n_folds"],
                n_components=reg_config["n_components"],
                batch_size=reg_config["batch_size"],
                random_state=42,
                save_path=results_dir,
                file_name="le_results", 
                return_correlations=False,
                average_across_layers=False)
      
    shared_rois = get_shared_rois_nsd()
    eval_df_all_subj = summarize_subjects(shared_rois, eval_df)

    eval_table = pa.Table.from_pandas(eval_df_all_subj)
    metadata["evaluation"]="reg"
    metadata_json = json.dumps(metadata)
    original_meta = eval_table.schema.metadata
    combined_meta = {metadata_key.encode(): metadata_json.encode(), **original_meta}
    eval_table = eval_table.replace_schema_metadata(combined_meta)
    pq.write_table(eval_table, join(results_dir,'eval_df_reg.parquet'))

