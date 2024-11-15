# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import json
import torch
import dnnlib
import matplotlib.pyplot as plt
import random
import pdb

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        dataset_main_name,
        dataset_main_name_cond,
        raw_shape,              # Shape of the raw image data (NCHW).
        dataset_main_name_back=None,
        cond_norm=1,
        gt_norm=1,
        use_offsets=False,
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?

    ):

        self._name = name
        self.dataset_main_name = dataset_main_name
        self.dataset_main_name_cond = dataset_main_name_cond
        self.dataset_main_name_back = dataset_main_name_back
        self.cond_norm = cond_norm
        self.gt_norm = gt_norm
        #self.num_offsets = num_offsets
        self.use_offsets = use_offsets
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        #if (max_size is not None) and (self._raw_idx.size > max_size):
            # np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
        #    self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self.xflip = xflip

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):

        idx_str = str(idx).zfill(4)  # This will ensure the index is always 4 digits long

        cond = np.load(f'{self.dataset_main_name_cond}_{idx_str}.npy') / self.cond_norm
        #print(cond.shape)
        if not self.use_offsets:
            print("use only zero offset:")
            if len(cond.shape) > 2:
                rtm_chan = int(round(cond.shape[0]/2))
                print(rtm_chan)
                cond = cond[rtm_chan,:,:]
            cond = cond[np.newaxis,...]

        #load in background model 
        if not (self.dataset_main_name_back == None):
            print("using background ")
            background = np.load(f'{self.dataset_main_name_back}_{idx_str}.npy')  / self.gt_norm
            if len(background.shape) < 3:
                background = background[np.newaxis,...] 
            cond = np.concatenate([cond, background], axis=0)

        #insert mask
        if not (self.dataset_main_name_back == None):
            print("using mask ")
            mask = np.ones((1, 256, 512))
            # if len(background.shape) < 3:
            #     background = background[np.newaxis,...] 
            cond = np.concatenate([cond, mask], axis=0)

        target_image = np.load(f'{self.dataset_main_name}_{idx_str}.npy')
        target_image = target_image[np.newaxis,...]  / self.gt_norm

        if self.xflip:
            rand_num = np.random.uniform()
            if rand_num > 0.5:
                if target_image.shape[-1] > 0:
                    target_image = np.flip(target_image, 2)
                if cond.shape[-1] > 0:
                    cond = np.flip(cond, 2)

        #need the copy if you have the flip
        return torch.from_numpy(target_image.copy()), torch.from_numpy(cond.copy())


    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = False, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
            # print("Inside Directory")
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
            # print("Inside Zip File")
        else:
            # print("Neither zip or directory")
            raise IOError('Path must point to a directory or zip')

        self._image_fnames = sorted(fname for fname in self._all_fnames if os.path.splitext(fname)[1] == '.npy')
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape,**super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            elif self._file_ext(fname) == '.pt':
                image = torch.load(f)
            elif self._file_ext(fname) == '.npy':
                image = np.load(f)
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
