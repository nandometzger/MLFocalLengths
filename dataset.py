import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import exifread

from PIL import Image
import rawpy
import imageio 

import numpy as np

import h5py
import cv2

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import pickle

# import tkinter as tk
# import libraw

class FocalLengthDataset(Dataset):
    def __init__(self, root_dir, transform=None, hdf5_path=None, focal_length_path=None, force_recompute=False, mode="train", split_mode="rand",
        append_new_data=False, recompute_split=False, in_memory=False):
        self.root_dir = root_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self.eps = 1e-7 
        self.in_memory = in_memory

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
            self.focal_length = self.focal_length[invalid_mask==1]

            if split_mode=="rand":
                X_train_idx, X_test_idx, y_train, y_test = train_test_split(idx, self.focal_length, test_size=0.2, random_state=1)
                X_train_idx, X_val_idx, y_train, y_val = train_test_split(X_train_idx, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
            elif split_mode=="time":
                n = len(idx)
                X_train_idx, X_test_idx, X_val_idx = idx[:int(n*0.7)], idx[int(n*0.7):int(n*0.8)], idx[int(n*0.8):]
                y_train, y_test, y_val = idx[:int(n*0.7)], idx[int(n*0.7):int(n*0.8)], idx[int(n*0.8):] 

                split_dict = { "train": X_train_idx,  "test": X_test_idx, "val": X_val_idx  }

            with open('data/split_file.pickle', 'wb') as handle:
                pickle.dump(split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open('data/split_file.pickle', 'rb') as handle:
                split_dict = pickle.load(handle)
            
        # decise on the split
        if mode=="train":
            self.X_idx = split_dict["train"]
        elif mode=="test":
            self.X_idx = split_dict["test"]
        elif mode=="val":
            self.X_idx = split_dict["val"]
        self.X_idx.sort()
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
                                h, w, _ = img.shape 

                                img = img[(h-w)//2:-(h-w)//2, :]
                                res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                                # im_resized = img.resize((256, 256))

                                dset[old_shape+i] = np.transpose(res,(2,0,1))
                                Focal_lengths.append(focal_length)
                                i+=1
                                print(old_shape+i)
                                # if i ==10:
                                #     break
                else:
                    # Continue if the inner loop wasn't broken.
                    continue    
                break             
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
            dset = hf.create_dataset('imgs',  shape=(count,3,256,256), 
                maxshape=(None,3,256,256), chunks=(8,3,256,256), dtype=np.int8, compression="gzip")

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
                            h, w, _ = img.shape 

                            img = img[(h-w)//2:-(h-w)//2, :]
                            res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                            # im_resized = img.resize((256, 256))

                            dset[i] = np.transpose(res,(2,0,1))
                            Focal_lengths.append(focal_length)
                            i+=1
                            print(i)
                            # if i ==30:
                            #     break
                else:
                    # Continue if the inner loop wasn't broken.
                    continue    
                break            
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

if __name__=='__main__':

        
    # Create a transform to preprocess the data
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    # dataset = FocalLengthDataset(root_dir=r'C:\Users\nando\Pictures\Lightroom_backuped\Lightroom Catalog-v12 Smart Previews.lrdata\E', transform=data_transform)
    # dataset = FocalLengthDataset(
    #     root_dir=r'D:\Photo_collection_ssd',
    #     transform=data_transform, hdf5_path="data/imgdataset3.h5", focal_length_path='data/split_file3.pickle',
    #     force_recompute=False,  split_mode="time", append_new_data=True)

    dataset = FocalLengthDataset(
        root_dir=r'D:\Photo_collection_ssd',
        transform=data_transform, hdf5_path="data/imgdataset3.h5", focal_length_path='data/split_file3.pickle',
        force_recompute=False,  split_mode="time")

    # Create a dataloader to feed the dataset into a model
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    a = dataset[0]
    # Iterate through the dataloader
    for batch in tqdm(dataloader):
        images = batch['img']
        focal_lengths = batch['y']

