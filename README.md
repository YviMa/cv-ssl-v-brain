Comparing semantic representations of biologically inspired self-supervised models to brain data using the Net2Brain toolbox.

# Structure
Overview over most important subfolders

```
cv-ssl-v-brain/
├─ configs/ # yaml config files for running pipeline.py
│
├─ src/ # my code
│ └─ pipeline.py # script for running RSA/encoding steps
| └─ preprocess.py # script for bringing NSD data into right format for RSA
| └─ utils.py # utilities for pipeline.py
│
├─ cv_ssl/ # Tim's code 
│ 
├─ results/ # one subfolder per model, same naming convention as Tim
|
└─inspect.ipynb # for inspecting results and plotting
```
# Results Format
Each folder results/ is for **one** model, both RSA and linear encoding. It includes:
- copy of the yaml config file called "config.yaml"
- pandas dataframe from RSA saved as parquet called `eval_df_rsa.parquet`
    - contains metadata with model name, time_frames, layers, etc., key: `'custom_meta'`
    - ROI and layer names are automatically changed to match output of linear encoding using `fix_naming` from `utils.py`
- pickled instance of `RSA` class from Net2Brain after evaluation called "RSA_instance.pkl"
    - for comparing models later without having to redo evaluation
- pandas dataframe from Linear Encoding saved as parquet called `eval_df_reg.parquet`
    - contains metadata with model name, time_frames, layers, etc., key: `'custom_meta'`
    - results are concatenated into same format as RSA output using `summarize_subjects` from `utils.py`
- raw results dataframe from Linear Encoding called `le_results.csv`
    - saved by default when running `Linear_Encoding()` from Net2Brain
- `le_results.npy'
    - saved by default when running `Linear_Encoding()` from Net2Brain

The main results files to refer to for plotting/further inspection are `eval_df_reg.parquet` and `eval_df_rsa.parquet`. They can be loaded in the following way:

```python

import json
import pandas as pd
import pyarrow.parquet as pq

results_table = pq.read_table("/path/to/results.parquet", partitioning=None)
results_df = results_table.to_pandas()
metadata = json.loads(rsa_table.schema.metadata["custom_meta".encode()]) # dictionary

```
# Installation
```bash
# create environment
conda create --name env_name python==3.10
conda activate env_name

# install Net2Brain
pip install -U git+https://github.com/cvai-roig-lab/Net2Brain

# newer version of timm needed than Net2Brain requires
pip uninstall -y timm

# install additional requirements
pip install requirements.txt
```

