import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import exifread

from PIL import Image
import rawpy
# import imageio 

import numpy as np

import h5py
import cv2

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import pickle

EPS = 1e-7
INPUT_SIZE = 256

JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe"}
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
SUPPORTED_EXTENSIONS = JPEG_EXTENSIONS | RAW_EXTENSIONS | {".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_image(path):
    """Load any supported image as an RGB uint8 array.

    Grayscale, palette and CMYK images are converted to RGB so that downstream
    code can rely on a 3-channel array.
    """
    if os.path.splitext(path)[1].lower() in RAW_EXTENSIONS:
        with rawpy.imread(path) as raw:
            return raw.postprocess()

    with Image.open(path) as handle:
        return np.array(handle.convert("RGB"))


def center_crop_square(img):
    """Crop the long edge symmetrically so the result is square.

    The original implementation used ``img[(h-w)//2:-(h-w)//2]``, which yields an
    empty array whenever the two sides differ by zero or one pixel (``-0`` is not
    "the end"). This form is identical for the even case the model was trained on
    and correct for the rest.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        top = (h - w) // 2
        return img[top:top + w, :]
    left = (w - h) // 2
    return img[:, left:left + h]


def preprocess_image(img):
    """Match the preprocessing the pretrained checkpoint was trained with.

    Portrait orientation, centre crop to square, cubic resize to 256x256, then
    min-max normalisation into [0, 1]. Changing any of this invalidates the
    published weights, so training and inference share this one function.
    """
    h, w = img.shape[:2]
    if h < w:
        img = np.transpose(img, (1, 0, 2))

    img = center_crop_square(img)
    img = cv2.resize(img, dsize=(INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)

    v_min, v_max = img.min(), img.max()
    return (img - v_min) / (v_max - v_min + EPS)


def find_images(root, recursive=False):
    """List supported image files under ``root``, sorted, skipping everything else.

    Accepts a single file path as well as a directory. Directories, sidecar files
    (.xmp, .txt) and hidden files are skipped rather than handed to the RAW
    decoder, which used to fail with an opaque LibRawIOError.
    """
    if os.path.isfile(root):
        return [root]

    if not os.path.isdir(root):
        raise FileNotFoundError(f"No such file or directory: {root}")

    paths = []
    if recursive:
        for directory, _, names in os.walk(root):
            paths.extend(os.path.join(directory, name) for name in names)
    else:
        paths = [os.path.join(root, name) for name in os.listdir(root)]

    images = [
        path for path in paths
        if os.path.isfile(path)
        and not os.path.basename(path).startswith(".")
        and os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images)


class ImageFolder(Dataset):
    """Images on disk, preprocessed for inference. No labels required."""

    def __init__(self, root_dir, recursive=False, return_raw=True):
        self.root_dir = root_dir
        self.eps = EPS
        self.return_raw = return_raw

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.files = find_images(root_dir, recursive=recursive)
        if not self.files:
            raise FileNotFoundError(
                f"No supported images found in '{root_dir}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        img = load_image(image_path)

        sample = {"img": self.transform(preprocess_image(img)), "path": image_path}
        if self.return_raw:
            sample["raw_img"] = img
        return sample


class FocalLengthDataset(Dataset):
    def __init__(self, root_dir, transform=None, hdf5_path="data/imgdataset.h5", focal_length_path="data/split_file.pickle", force_recompute=False, mode="train", split_mode="rand",
        append_new_data=False, recompute_split=False, in_memory=False, force_focal_length=None):
        self.root_dir = root_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self.eps = 1e-7 
        self.in_memory = in_memory
        self.force_focal_length = force_focal_length

        # check existence
        if not append_new_data:
            with h5py.File(hdf5_path, 'r') as hf:
                if "imgs" not in hf.keys() or "focal_length" not in hf.keys() or force_recompute:
                    doprep = True
                else:
                    if hf["imgs"].shape[0] != hf["focal_length"].shape[0]:
                        doprep = True
                    else:
                        doprep = False
            if doprep:
                self.doprep()
                recompute_split = True

        else:
            self.append_data()
            recompute_split = True

        # prepare variables
        with h5py.File(hdf5_path, 'r') as hf:
            self.focal_length = hf["focal_length"][:]
            if in_memory:
                self.imgs = hf["imgs"][:]

        # organize splitting of samples
        if force_recompute or recompute_split:
            # samples with focal length 0
            invalid_mask = np.ones_like(self.focal_length)
            invalid_mask *= self.focal_length!=0

            # generate split file
            idx = np.arange(len(self.focal_length))

            # mask invalid data
            idx = idx[invalid_mask==1]
            valid_focal_length = self.focal_length[invalid_mask==1]

            if split_mode=="rand":
                X_train_idx, X_test_idx, _, y_train = train_test_split(idx, valid_focal_length, test_size=0.2, random_state=1)
                X_train_idx, X_val_idx, _, _ = train_test_split(X_train_idx, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
            elif split_mode=="time":
                # Chronological split. The 20-30% band is deliberately left out of
                # every split as a buffer, so that near-duplicate shots from the
                # same session cannot straddle the val/train boundary.
                n = len(idx)
                X_test_idx, X_val_idx, X_train_idx = idx[:int(n*0.1)], idx[int(n*0.1):int(n*0.2)], idx[int(n*0.3):]
            else:
                raise ValueError(f"Unknown split_mode '{split_mode}', expected 'rand' or 'time'")

            split_dict = {"train": X_train_idx, "test": X_test_idx, "val": X_val_idx}

            os.makedirs(os.path.dirname(focal_length_path) or ".", exist_ok=True)
            with open(focal_length_path, 'wb') as handle:
                pickle.dump(split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(focal_length_path, 'rb') as handle:
                split_dict = pickle.load(handle)

        if mode not in split_dict:
            raise ValueError(f"Unknown mode '{mode}', expected one of {sorted(split_dict)}")
        self.X_idx = np.sort(np.asarray(split_dict[mode]))
        self.y = self.focal_length[self.X_idx]

        # print("init finished")


    def append_data(self):


        Focal_lengths = []
        self.images = []
        valid = False

        count = 0
        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                image_path = os.path.join(path, name)
                if image_path.__contains__("."):
                    if image_path.split(".")[1] not in ["xmp", "MOV", "tif", "TIF", "tiff", "TIFF"]:
                        count += 1

        with h5py.File(self.hdf5_path, 'a') as hf:
            # resize to new shape
            old_shape = hf["imgs"].shape[0]
            hf["imgs"].resize((old_shape + count), axis = 0) 
            Focal_lengths = list(hf["focal_length"])

            i = 0
            dset = hf["imgs"]
            for path, subdirs, files in os.walk(self.root_dir):
                for name in files:
                    image_path = os.path.join(path, name)
                    # print(image_path)
                    if image_path.__contains__("."):
                        if image_path.split(".")[1] not in ["xmp", "MOV", "tif", "TIF", "tiff", "TIFF"]:

                            with open(image_path, 'rb') as f: 
                                
                                tags = exifread.process_file(f)
                                if "EXIF FocalLengthIn35mmFilm" in tags:
                                    focal_length = int(str(tags["EXIF FocalLengthIn35mmFilm"]))
                                    valid = True
                                elif self.force_focal_length is not None:
                                    focal_length = self.force_focal_length
                                    valid = True
                                else: 
                                    print("No Tag")
                                    valid = False

                            if valid:
                                if image_path.split(".")[1] in ["jpg", "JPG"]:
                                    with Image.open(image_path) as f: 
                                        img = np.array(f)
                                else:
                                    with rawpy.imread(image_path) as raw:  
                                        try:
                                            img = raw.postprocess()
                                        except:
                                            continue

                                h, w, _ = img.shape
                                if h<w:
                                    img = np.transpose(img, (1,0,2))

                                img = center_crop_square(img)
                                res = cv2.resize(img, dsize=(INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

                                dset[old_shape+i] = np.transpose(res,(2,0,1))
                                Focal_lengths.append(focal_length)
                                i+=1
                                print(old_shape+i)
                                # if i ==10:
                                #     break
            dset.resize((i+old_shape,3,256,256))
            del hf["focal_length"]
            flengths = hf.create_dataset('focal_length', data=Focal_lengths)

            # hf["focal_length"].resize((i + hf["focal_length"].shape[0]), axis = 0) 
            # hf["focal_length"][-len(Focal_lengths):] = Focal_lengths


    def doprep(self):
        Focal_lengths = []
        self.images = []
        valid = False

        count = 0
        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                image_path = os.path.join(path, name)
                if image_path.split(".")[1] not in  ["xmp", "MOV", "tif", "TIF", "tiff", "TIFF"]:
                    count += 1
                # print(image_path)

        # hf = h5py.File('data.h5', 'w')
        # f = h5py.File(hdf5_path, 'w')
        with h5py.File(self.hdf5_path, 'w') as hf:
            
            # hf.create_dataset
            # uint8, not int8: pixel values run 0-255, and h5py silently clamps
            # anything above 127 when the dataset is signed. The published
            # checkpoint was trained through that clamp, so rebuilding the cache
            # with this fix changes the input distribution -- retrain rather than
            # mixing a new cache with the old weights.
            dset = hf.create_dataset('imgs',  shape=(count,3,INPUT_SIZE,INPUT_SIZE),
                maxshape=(None,3,INPUT_SIZE,INPUT_SIZE), chunks=(8,3,INPUT_SIZE,INPUT_SIZE),
                dtype=np.uint8, compression="gzip")

        i = 0
        with h5py.File(self.hdf5_path, 'a') as hf:

            dset = hf["imgs"]
            for path, subdirs, files in os.walk(self.root_dir):
                for name in files:
                    image_path = os.path.join(path, name)
                    # print(image_path)
                    if image_path.split(".")[1] not in ["xmp", "MOV", "tif", "TIF", "tiff", "TIFF"]:

                        with open(image_path, 'rb') as f: 
                            
                            tags = exifread.process_file(f)
                            if "EXIF FocalLengthIn35mmFilm" in tags: 
                                focal_length = int(str(tags["EXIF FocalLengthIn35mmFilm"]))
                                valid = True
                            else: 
                                print("No Tag")
                                valid = False

                        if valid:
                            if image_path.split(".")[1] in ["jpg", "JPG"]:
                                with Image.open(image_path) as f: 
                                    img = np.array(f)
                            else:
                                with rawpy.imread(image_path) as raw:  
                                    try:
                                        img = raw.postprocess()
                                    except:
                                        continue

                            h, w, _ = img.shape
                            if h<w:
                                img = np.transpose(img, (1,0,2))

                            img = center_crop_square(img)
                            res = cv2.resize(img, dsize=(INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

                            dset[i] = np.transpose(res,(2,0,1))
                            Focal_lengths.append(focal_length)
                            i+=1
                            print(i)
                            # if i ==30:
                            #     break
                    # else:
                    #     print("No Tag found:", image_path)
            dset.resize((i,3,256,256))
            flengths = hf.create_dataset('focal_length', data=Focal_lengths)

        print("finished with", str(i), "samples") 


                        

    def __len__(self):
        return len(self.X_idx)

    def __getitem__(self, idx):

        # pick from the train, test, val ordering
        dataidx = self.X_idx[idx]   

        # access data
        if self.in_memory:
            img = self.imgs[dataidx]
        else:
            with h5py.File(self.hdf5_path, 'r') as hf:
                img = hf["imgs"][dataidx]

        focal_length = self.focal_length[dataidx]

        # reshape 
        img = np.transpose(img.astype(np.float32),(1,2,0))

        # noramlize
        v_min, v_max = img.min(), img.max()
        new_min, new_max = 0.0, 1.0
        img = (img - v_min)/(v_max - v_min + self.eps)*(new_max - new_min) + new_min

        # apply transforms and augmentations
        if self.transform:
            img = self.transform(img)

        return {'img': img, 'y': focal_length}

def build_cache(argv=None):
    """Build the HDF5 training cache from a directory of photographs.

        python dataset.py --data-dir ~/Pictures/2022 --hdf5-path data/imgdataset4.h5

    Only images whose EXIF carries FocalLengthIn35mmFilm are kept, unless
    --force-focal-length supplies a value for images that lack the tag.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=build_cache.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', required=True, help='Directory of photographs to ingest')
    parser.add_argument('--hdf5-path', default='data/imgdataset4.h5', help='Output HDF5 cache')
    parser.add_argument('--split-file', default='data/split_file4.pickle', help='Output split file')
    parser.add_argument('--split-mode', default='time', choices=['time', 'rand'])
    parser.add_argument('--append', action='store_true', help='Add to an existing cache instead of rebuilding')
    parser.add_argument('--force-recompute', action='store_true', help='Rebuild even if the cache looks complete')
    parser.add_argument('--force-focal-length', type=float, default=None,
                        help='Focal length to assume for images without the EXIF tag')
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.hdf5_path) or '.', exist_ok=True)
    if not os.path.exists(args.hdf5_path):
        with h5py.File(args.hdf5_path, 'w'):
            pass

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = FocalLengthDataset(
        root_dir=args.data_dir,
        transform=data_transform,
        hdf5_path=args.hdf5_path,
        focal_length_path=args.split_file,
        force_recompute=args.force_recompute,
        split_mode=args.split_mode,
        force_focal_length=args.force_focal_length,
        append_new_data=args.append,
        recompute_split=True,
    )

    print(f"Cache ready: {len(dataset)} training samples in {args.hdf5_path}")
    return dataset


if __name__ == '__main__':
    build_cache()

