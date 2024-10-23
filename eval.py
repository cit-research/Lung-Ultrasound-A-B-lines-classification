import argparse
from sympy import O
import wandb
import pprint
from pathlib import Path
import os
import shutil
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

from datetime import datetime
import torch
import os
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
import monai
import monai.transforms as transforms
from monai.data import DataLoader, CacheDataset
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)




from utils.CvatMonaiDatasetClassification import CvatMonaiDatasetClassification
from utils.Augumentations import RandomNoise, GrayscaleNormalize, RandomSpeckleNoise
from utils.DatasetLoader import DataLoaderABLine

from sklearn import metrics
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

import time 
import configparser

from utils.Evaluation import Evaluation

# Load config values
config = configparser.ConfigParser()
config.read('./configs/global.cfg')

PATH_TMP_FOLDER_OUTPUT = config.get('Global', 'path_temp_folder_output')
PROJECT_NAME = config.get('Global', 'project_name')
PATH_TMP_FOLDER_MODEL = config.get('Global', 'path_temp_folder_models') 

def log(payload):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp, "LOG from ",__name__, ":" ,payload)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log(f'Detected devce is {device}')
    log(f'Detected devce name is {torch.cuda.get_device_name()}')
    log("Eval test run")
    num_classes = 2
    RUN_NAME = "valiant-sweep-3"
    model = None
    if model is None:
        # Define the ResNet-18 model

            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, num_classes)  # Change the output layer to match the number of classes in your dataset
            model.to(device)
            # Donwload train weights
            result = model.load_state_dict(torch.load("{model_path}/model_weights_{model_name}.pth".format(model_path= PATH_TMP_FOLDER_MODEL, model_name=RUN_NAME)))
            log(result)

    evaluation = Evaluation(model, test_loader, df_test, RUN_NAME)