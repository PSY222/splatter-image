#refer to srn file and build the mipo360 datasetloader.
#refer to gaussian splatting tactics
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

SHAPENET_DATASET_ROOT = None # Change this to your data directory
assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

class MIP360Dataset(Dataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name

        self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
                                                                              cfg.data.category,
                                                                              cfg.data.category,
                                                                              dataset_name))
      