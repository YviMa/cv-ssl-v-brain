import os
import random
from pathlib import Path

import h5py
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

if __name__ == '__main__':
    download_link = "https://www.kaggle.com/datasets/jessicali9530/coil100"

    output_dir = Path("") # <-- change this to the path where you want to save the processed dataset
    output_dir.mkdir(exist_ok=True, parents=True)


    data_dir = Path("/coil-100/") # <-- change this to the path where you extracted the COIL100 dataset
    if not data_dir.exists():
        raise FileNotFoundError("Please download the COIL100 dataset from {} and extract it to {}".format(download_link, data_dir))
    imgs = [f for f in os.listdir(data_dir) if f.startswith("obj")]
    imgs = shuffle(imgs, random_state=0)
    labels = [int(fn.split("__")[0][3:])-1 for fn in imgs]

    train_idx = [5*random.randint(0,72-1) for _ in range(100)]


    train_size = len(train_idx)
    val_size = len(imgs) - train_size

    train_h5 = h5py.File(output_dir / "train.h5", "w")
    val_h5 = h5py.File(output_dir / "val.h5", "w")
    train_images = train_h5.create_dataset(f"images",shape=(train_size,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    train_targets = train_h5.create_dataset(f"targets",shape=(train_size,),dtype=np.int32)

    val_images = val_h5.create_dataset(f"images",shape=(val_size,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    val_targets = val_h5.create_dataset(f"targets",shape=(val_size,),dtype=np.int32)

    bar = tqdm(range(len(imgs)), total=len(imgs), desc=f"process")
    print(train_size, val_size, np.unique(np.array(labels)))
    print(train_idx)
    cpt_train = 0
    cpt_val = 0
    for i in bar:
        file, target = imgs[i], labels[i]
        id = int(imgs[i].split("__")[1][:-4])
        if id == train_idx[target]:
            train_images[cpt_train] = np.fromfile(data_dir / file, dtype=np.uint8)
            train_targets[cpt_train] = target
            cpt_train += 1
        else:
            val_images[cpt_val] = np.fromfile(data_dir / file, dtype=np.uint8)
            val_targets[cpt_val] = target
            cpt_val += 1
    train_h5.close()
    val_h5.close()




