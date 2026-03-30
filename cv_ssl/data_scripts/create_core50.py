from pathlib import Path
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    download_link = "http://bias.csr.unibo.it/maltoni/download/core50/core50_350x350.zip"

    dir = Path("path_to_core50_350x350") # <-- change this to the path where you extracted the Core50 dataset
    if not dir.exists():
        raise FileNotFoundError("Please download the Core50 dataset from {} and extract it to {}".format(download_link, dir))

    corr = [Path('/s3/o43/C_03_43_209.png')]

    total = len(list(filter(lambda x: not x.stem.startswith("."), dir.rglob("*.png"))))

    with h5py.File(dir / "core50_arr.h5", "w") as h5_file, tqdm(total=total) as bar:
        for bg_p in filter(lambda x: x.is_dir() and not x.stem.startswith("."), dir.iterdir()):
            bg_group = h5_file.create_group(bg_p.stem)

            images = list(filter(lambda x: not x.stem.startswith('.'), bg_p.rglob("*.png")))

            total_images = len(images)
            img_dataset = bg_group.create_dataset("images",
                                                  shape=(total_images, 350, 350, 3),
                                                  dtype=np.uint8
                                                  )
            target_dataset = bg_group.create_dataset("targets",
                                                     shape=(total_images,),
                                                     dtype=np.int32)
            for i, images_p in enumerate(images):
                try:
                    img_dataset[i] = np.array(Image.open(images_p))
                except Exception as e:
                    print("Corrupted", images_p)
                    continue

                target_dataset[i] = int(images_p.parent.stem[1:]) - 1
                bar.update(n=1)