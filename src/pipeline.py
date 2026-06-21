import yaml
import argparse
import torch
import json
import pickle
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import torchvision.models as torchvision_models
from os import makedirs, listdir
from os.path import join, split
from cv_ssl.solo.methods import METHODS
from omegaconf import OmegaConf
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from net2brain.evaluations.rsa import RSA
from net2brain.evaluations.encoding import Linear_Encoding
from Moco import vits
from utils import compose_dir_name, compose_model_dir, get_shared_rois_nsd, get_shared_rois_BOLDmoments, summarize_subjects, fix_naming

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

config_path = args.config

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# loading the model
model_config = config["model"]
model_dir = model_config.pop("directory")
model_name = model_config["name"]
netset = model_config["netset"]
time_window = model_config["time_window"]
crop_size = model_config["crop_size"]
center_crop = model_config["center_crop"]

data_config = config["data"]
stimuli_path = data_config["stimuli_path"]

layers_to_extract = config["feature_extraction"]["layers"]
extraction_kwargs = config["feature_extraction"]["extraction_kwargs"]

if netset == "cv_ssl":
    dir_name = compose_dir_name(config_path, model_name, time_window, crop_size, center_crop)
    feature_path = "tmp/features/" + dir_name + "_feat"
    checkpoint_path = join(model_dir, compose_model_dir(model_name, time_window, crop_size, center_crop))
    ckpt = torch.load(checkpoint_path, weights_only=False)
    cfg = OmegaConf.create(ckpt["args"])

    model = METHODS[cfg["method"]](cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    #device = torch.device("cuda:0")
    #model.to(device)

    # feature extraction
    fx = FeatureExtractor(model=model, device='cpu') 
    
elif netset == "moco":
    if model_name == "vit_base":
        model = vits.__dict__['vit_base']()
        checkpoint_path = join(model_dir, "vit-b-300ep.pth.tar")
        dir_name = "moco_vit_base_imagenet"
        feature_path = "tmp/features/"+dir_name+"_feat"
        linear_keyword = "head"
    else:
        model = torchvision_models.__dict__['resnet50']()
        checkpoint_path = join(model_dir, "r-50-1000ep.pth.tar")
        dir_name = "moco_resnet50_imagenet"
        feature_path = "tmp/features/"+dir_name+"_feat"
        linear_keyword="fc"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg=model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(msg)
    for n, p in model.named_parameters():
        if p.is_meta:
            print("META:", n)
    fx = FeatureExtractor(model=model, device='cpu') 
    #all_layers = fx.get_all_layers()
    #with open("res_layers", 'w') as f:
        #json.dump(all_layers, f)
else:
    fx = FeatureExtractor(model=model_name, netset=netset, device='cpu')
    dir_name = netset + "_" + model_name
    feature_path = "tmp/features/" + dir_name + "_feat"
#all_layers = fx.get_all_layers()
#with open("res_layers", 'w') as f:
    #json.dump(all_layers, f)
fx.extract(data_path=stimuli_path, 
           save_path = feature_path, 
           layers_to_extract=layers_to_extract, 
           consolidate_per_layer=False,
           **extraction_kwargs)

# evaluation
brain_rdms = data_config["brain_rdms"] 
analysis_config = config["analysis"]
results_dir = data_config["results_dir"] + dir_name
makedirs(results_dir, exist_ok=True)

with open(join(results_dir, "config.yaml"), 'w') as f:
    yaml.dump(config, f)

metadata = {**model_config, "layers": layers_to_extract, "category": data_config["category"]}
metadata_key = 'custom_meta'

if analysis_config["rsa"].pop("execute"):
    #dm creation
    rdm_path = "tmp/rdms/" + split(data_config["data_dir"])[1] +"/"+ dir_name + "_rdm"
    rdm_creator = RDMCreator(verbose=True, device='cpu') 
    rdm_creator.create_rdms(feature_path=feature_path, save_path=rdm_path, save_format='npz')

    kwargs = analysis_config["rsa"]["kwargs"]
    evaluation =  RSA(rdm_path, brain_rdms, model_name=dir_name, squared=False, **kwargs)
    distance_metric = analysis_config["rsa"]["distance_metric"]
    eval_df = evaluation.evaluate(distance_metric=distance_metric)
    eval_df = fix_naming(eval_df)
    eval_table = pa.Table.from_pandas(eval_df)

    metadata["evaluation"]="rsa"
    metadata = {**metadata, **analysis_config["rsa"]}
    metadata_json = json.dumps(metadata)
    original_meta = eval_table.schema.metadata
    combined_meta = {metadata_key.encode(): metadata_json.encode(), **original_meta}
    eval_table = eval_table.replace_schema_metadata(combined_meta)
    pq.write_table(eval_table, join(results_dir,"eval_df_rsa.parquet"))

    # in order to avoid rerunning evaluations for model comparison later
    with open(join(results_dir,"RSA_instance.pkl"), 'wb') as f:
        pickle.dump(evaluation, f)

if analysis_config["reg"].pop("execute"):
    reg_config = analysis_config["reg"]
    #roi_paths = [join(data_config["data_dir"],"subj0"+str(j), "rois") for j in range(1,9)]
    roi_paths = [join(data_config["data_dir"],"sub-0"+str(j)) for j in range(1,9)]
    roi_paths += [join(data_config["data_dir"],"sub-10")]
    #print(roi_paths)
    #roi_paths = data_config["data_dir"]

    eval_df = Linear_Encoding(feat_path=feature_path,  
                roi_path=roi_paths,
                model_name=dir_name,
                trn_tst_split=reg_config["trn_tst_split"],
                custom_trn_tst=reg_config["custom_trn_tst"],
                n_folds=reg_config["n_folds"],
                n_components=reg_config["n_components"],
                batch_size=reg_config["batch_size"],
                random_state=42,
                save_path=results_dir,
                file_name="le_results", 
                return_correlations=False,
                average_across_layers=False)
    #shared_rois = get_shared_rois_nsd()
    shared_rois = get_shared_rois_BOLDmoments()
    eval_df_all_subj = summarize_subjects(shared_rois, eval_df)

    eval_table = pa.Table.from_pandas(eval_df_all_subj)
    metadata["evaluation"]="reg"
    metadata = {**metadata, **analysis_config["reg"]}
    metadata_json = json.dumps(metadata)
    original_meta = eval_table.schema.metadata
    combined_meta = {metadata_key.encode(): metadata_json.encode(), **original_meta}
    eval_table = eval_table.replace_schema_metadata(combined_meta)
    pq.write_table(eval_table, join(results_dir,'eval_df_reg.parquet'))