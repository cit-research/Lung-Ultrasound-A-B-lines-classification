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
f

import warnings
warnings.filterwarnings("ignore")

from utils.CvatMonaiDatasetClassification import CvatMonaiDatasetClassification
from utils.Augumentations import RandomNoise, GrayscaleNormalize, RandomSpeckleNoise
from utils.DatasetLoader import DataLoaderABLine, DataLoaderABLineFrame
from utils.Evaluation import Evaluation

from sklearn import metrics
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

import time 
import configparser


# ------------------------------------- 
# GLOBAL VARIABLES 
# Load config values
config = configparser.ConfigParser()
config.read('configs/global.cfg')
PROJECT_NAME = config.get('Global', 'project_name')
JOB_TYPE = config.get('Global', 'job_type')
ARTIFACT_NAME = config.get('Global', 'artifacts_name') #name of verzion dataset you need
PATH_TMP_FOLDER = config.get('Global', 'path_temp_folder') #name of verzion dataset you need
TRAIN = config.getboolean('Global', 'train') 
TEST = config.getboolean('Global', 'test') 
PATH_TMP_FOLDER_DATA = config.get('Global', 'path_temp_folder_data') 
PATH_TMP_FOLDER_MODEL = config.get('Global', 'path_temp_folder_models') 

# ------------------------------------- 
# CREATE FOLDERS
Path(PATH_TMP_FOLDER).mkdir(parents=True, exist_ok=True)
Path(PATH_TMP_FOLDER_DATA).mkdir(parents=True, exist_ok=True)



# Path(PATH_FOLDER_ARTIFACTS).mkdir(parents=True, exist_ok=True)
# Path(PATH_FOLDER_DATA).mkdir(parents=True, exist_ok=True)

# config class
class SimpleNamespace:
    def __init__(self,fold_id,experiment_class,dataset_revision,probe_types,wandb_project,dataset_filtering,random_seed,batch_size,resolution,learning_rate,num_epochs,validation_split,dropout,patience,model,class_weight,l2_loss,l1_activate,l1_lambda,normalise_intensity,augumentation_type,spatial_transforms) -> None:
        self.fold_id=fold_id
        self.experiment_class=experiment_class
        self.dataset_revision=dataset_revision
        self.probe_types=probe_types
        self.wandb_project=wandb_project
        self.dataset_filtering=dataset_filtering
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.resolution = resolution
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.dropout = dropout
        self.patience = patience
        self.model = model
        self.class_weight = class_weight
        self.l2_loss = l2_loss
        self.l1_activate = l1_activate
        self.l1_lambda = l1_lambda
        self.normalise_intensity = normalise_intensity
        self.augumentation_type = augumentation_type
        self.spatial_transforms = spatial_transforms

config_defaults = SimpleNamespace(
    fold_id=0,
    experiment_class = "a",
    dataset_revision=8,
    probe_types="probe_all",
    dataset_filtering="Yes",
    random_seed=1234,
    batch_size = 16,
    resolution="500,600",
    learning_rate = 0.0001,
    num_epochs = 50,
    validation_split = 0.2,
    dropout = 0,
    patience = 5,
    model = "inception-v3",
    class_weight = 7.0,
    l2_loss = 0.0001,
    l1_activate = True,
    l1_lambda = 0.0001,
    wandb_project=PROJECT_NAME,
    normalise_intensity="(0,1)",
    augumentation_type="a_t_only_basic",
    spatial_transforms="resize"
) # type: ignore

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_id', type=int, default=config_defaults.fold_id)
    parser.add_argument('--experiment_class', type=str, default=config_defaults.experiment_class)
    parser.add_argument('--dataset_revision', type=int, default=config_defaults.dataset_revision)
    parser.add_argument('--probe_types', type=str, default=config_defaults.probe_types)
    parser.add_argument('--dataset_filtering', type=str, default=config_defaults.dataset_filtering)
    parser.add_argument('--random_seed', type=int, default=config_defaults.random_seed)


    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--resolution', type=str, default=config_defaults.resolution)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--num_epochs', type=int, default=config_defaults.num_epochs)
    parser.add_argument('--validation_split', type=float, default=config_defaults.validation_split)
    parser.add_argument('--dropout', type=int, default=config_defaults.dropout)
    parser.add_argument('--patience', type=int, default=config_defaults.patience)
    parser.add_argument('--model', type=str, default=config_defaults.model)

    parser.add_argument('--class_weight', type=float, default=config_defaults.class_weight)
    parser.add_argument('--l2_loss', type=float, default=config_defaults.l2_loss)
    parser.add_argument('--l1_activate', type=bool, default=config_defaults.l1_activate)
    parser.add_argument('--l1_lambda', type=float, default=config_defaults.l1_lambda)
    parser.add_argument('--normalise_intensity', type=str, default=config_defaults.normalise_intensity)
    parser.add_argument('--augumentation_type', type=str, default=config_defaults.augumentation_type)
    parser.add_argument('--spatial_transforms', type=str, default=config_defaults.spatial_transforms)

    parser.add_argument('--wandb_project', type=str, default=config_defaults.wandb_project)
    return parser.parse_args()

def log(payload):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp, "LOG from ",__name__, ":" ,payload)

if __name__ == "__main__":
    RUN_ID = "dry-sweep-288"
    log("RUN")
    args = parse_args()

    if TRAIN == False and TEST == False:
        log("if not TRAIN and not TEST:")
        log("Nothing to do here. Im quit!")
        exit()

    elif TRAIN and TEST == False: 
        # TRAIN/VALID
        log("else if TRAIN and not TEST:")
        wandb.init(project=PROJECT_NAME, config=args, job_type=JOB_TYPE)
    
    elif TRAIN == False and TEST:
        # EVAL
        log("else if not TRAIN and TEST:")
        if RUN_ID == "":
            log("Run ID is empty. Im quit!")
            exit()
        wandb.init(project=PROJECT_NAME, config=args, job_type=JOB_TYPE, resume=True, id=RUN_ID)


    elif TRAIN and TEST:
        # TRAIN/VALID
        # TEST
        log("else if TRAIN and TEST:")
        wandb.init(project=PROJECT_NAME, config=args, job_type=JOB_TYPE)


    else:
        log("Nothing to do here. Im quit!")
        exit()


    
    # log("WanDB Init")
    # with wandb.init(project=PROJECT_NAME, config=args, job_type=JOB_TYPE):
    #     log(config_defaults.__dict__)
    # ----------------------------------------
    # PREPAIR DATA - GLOBAL
    # - download artifacts
    # - if dataset not exist downlaod from sourse
    # - apply  merging datasets and create train/valid/subset
    # - data is avaliable under for example: dataset_loader.df_train
    # ----------------------------------------
    probe_name = wandb.config["probe_types"]
    print(probe_name)
    art_name = f'kkui-lung-{wandb.config["experiment_class"]}line-{probe_name}'
    log("Running dataset -> " + art_name)
    # if wandb.config["experiment_class"] == "a":
    #     art_name = "kkui-lung-aline-lumify"
    # elif wandb.config["experiment_class"] == "b":
    #     art_name = "kkui-lung-bline-lumify"
    # else:
    #     log("artifact name do not exit")
    # exit()
    dataset_loader = DataLoaderABLineFrame(art_name, "latest")
    # dataset_loader = DataLoaderABLine(art_name, wandb.config["probe_types"])
    dataset_loader.download_artifacts(wandb=wandb)
    dataset_loader.download_dataset(copy_from_nas=True)
    dataset_loader.load_dataset(wandb=wandb)
    # # #### PER FRAME need to implement later keep here for store code
    wandb.config["subset_train_frames_count_0"] = dataset_loader.df_train.loc[dataset_loader.df_train["label"]==0].shape[0]
    wandb.config["subset_train_frames_count_1"] = dataset_loader.df_train.loc[dataset_loader.df_train["label"]==1].shape[0]
    wandb.config["subset_train_frames_count_2"] = dataset_loader.df_train.loc[dataset_loader.df_train["label"]==2].shape[0]
    
    wandb.config["subset_valid_frames_count_0"] = dataset_loader.df_valid.loc[dataset_loader.df_valid["label"]==0].shape[0]
    wandb.config["subset_valid_frames_count_1"] = dataset_loader.df_valid.loc[dataset_loader.df_valid["label"]==1].shape[0]
    wandb.config["subset_valid_frames_count_2"] = dataset_loader.df_valid.loc[dataset_loader.df_valid["label"]==2].shape[0]

    wandb.config["subset_test_frames_count_0"] = dataset_loader.df_test.loc[dataset_loader.df_test["label"]==0].shape[0]
    wandb.config["subset_test_frames_count_1"] = dataset_loader.df_test.loc[dataset_loader.df_test["label"]==1].shape[0]
    wandb.config["subset_test_frames_count_2"] = dataset_loader.df_test.loc[dataset_loader.df_test["label"]==2].shape[0]
    # exit()
    
    # ----------------------------------------
    # GET DEVICE
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        wandb.config["device_name"] = torch.cuda.get_device_name()
    else: 
        wandb.config["device_name"] = "cpu"
    log(f'Detected devce is {device}')
    log(f'Detected devce name is {torch.cuda.get_device_name()}')



    # -------------------------------------------------------
    # PREPAIR DATASET and DATALOADER for TRAIN/VALID/TEST
    # -------------------------------------------------------
    if wandb.config["model"] == "resnet-50":
        batch_size = wandb.config["batch_size"] 
    else:
        batch_size = wandb.config["batch_size"] * 2
    train_dataset = CvatMonaiDatasetClassification(dataframe=dataset_loader.df_train, 
                                                server_path=dataset_loader.path_cvat_folder, 
                                                resolution = tuple(map(int,wandb.config["resolution"].split(','))), 
                                                device=device, 
                                                transform_spatial_type=wandb.config["spatial_transforms"],
                                                transforms_augumentation_type=wandb.config["augumentation_type"], 
                                                transform_output_type=wandb.config["normalise_intensity"])
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True, collate_fn=monai.data.list_data_collate)
    
    val_dataset = CvatMonaiDatasetClassification(dataframe=dataset_loader.df_valid, 
                                                server_path=dataset_loader.path_cvat_folder, 
                                                resolution = tuple(map(int,wandb.config["resolution"].split(','))), 
                                                device=device, 
                                                transform_spatial_type=wandb.config["spatial_transforms"],
                                                transforms_augumentation_type="empty_transforms", 
                                                transform_output_type=wandb.config["normalise_intensity"])
    val_loader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False, collate_fn=monai.data.list_data_collate)

  


    if TRAIN:
        # -------------------------------------------------------
        # PREPAIR MODEL
        # -------------------------------------------------------
        ### TEMP  GLOBAL VARAIBLES
        best_model_state_dict = None


        ### PREPAIR MODEL
        print("LOG -> Prepir model")
        # Define the ResNet-18 model

        if wandb.config["model"] == "resnet-18":
            log("LOG -> ResNet-18 model")
            model = torchvision.models.resnet18(pretrained=True)
            print(model)
            model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
            print(model)

            if wandb.config["dropout"] > 0:
                print("LOG -> Droppout is activated")
                # Add dropout to the fully connected layer (fc)
                dropout_prob = wandb.config["dropout"]  # Adjust this value as per your preference
                model.fc = nn.Sequential(
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Dropout(dropout_prob, inplace=True),
                    nn.Linear(512, 2)  # Change the output layer to match the number of classes
                )
            print(model)
            model.to(device)

        elif wandb.config["model"] == "resnet-34":
            log("LOG -> ResNet-34 model")
            model = torchvision.models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
            print(model)

            if wandb.config["dropout"] > 0:
                print("LOG -> Droppout is activated")
                # Add dropout to the fully connected layer (fc)
                dropout_prob = wandb.config["dropout"]  # Adjust this value as per your preference
                model.fc = nn.Sequential(
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Dropout(dropout_prob, inplace=True),
                    nn.Linear(512, 2)  # Change the output layer to match the number of classes
                )
            print(model)
            model.to(device)

        elif wandb.config["model"] == "resnet-50":
            log("LOG -> ResNet-50 model")
            model = torchvision.models.resnet50(pretrained=True)
            print(model)
            # model.fc = nn.Linear(2048, 2)
            model.fc = nn.Sequential(
                    nn.Linear(2048, 512),

    
                    nn.Linear(512, 2)  # Change the output layer to match the number of classes
                )
            # model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
            print(model)

            if wandb.config["dropout"] > 0:
                print("LOG -> Droppout is activated")
                # Add dropout to the fully connected layer (fc)
                dropout_prob = wandb.config["dropout"]  # Adjust this value as per your preference
                model.fc = nn.Sequential(
                     nn.Linear(2048, 512),
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Dropout(dropout_prob, inplace=True),
                    nn.Linear(512, 2)  # Change th)  # Change the output layer to match the number of classes
                )
            print(model)
            model.to(device)

        elif wandb.config["model"] == "inception-v3":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            print(model)
            model.fc = nn.Sequential(
                     nn.Linear(2048, 512),
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Linear(512, 2)  # Change th)  # Change the output layer to match the number of classes
                )
            model.to(device)

        else:
            print("LOG ERROR -> Model was not defined")
            exit()
        

        print("LOG -> Reguralisation & loss")
        w = [1.0, wandb.config["class_weight"]]
        l2_loss = wandb.config["l2_loss"]
        l1 = wandb.config["l1_activate"]
        l1_lambda = wandb.config["l1_lambda"]  # Adjust this value based on the desired strength of L1 regularization
        fine_tune_enable = True
        patience = wandb.config["patience"]
        
        def l1_regularization(model):
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.norm(param, 1)  # Add L1 norm of each parameter to the loss
            return l1_loss


        class_weights = torch.tensor(w)  # Replace the values with appropriate weights for your classes
        class_weights = class_weights.to(device)


        if wandb.config["class_weight"] == 0:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"], weight_decay=l2_loss)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        best_model_state_dict = None
        ### TRAIN
        print("LOG -> Train")
        fine_tune_activated = False
        total_step = len(train_loader)
        # with torch.profiler.profile(
        # schedule=torch.profiler.schedule(
        #     wait=1,
        #     warmup=1,
        #     active=3,
        #     repeat=1
        # ),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/mh731nk/_projects/usg-ab-lines-classification/tnsrdir/myresnet")) as prof:
        for epoch in range(wandb.config["num_epochs"]):
            # Record the start time
            start_time = time.time()
            # Training
            model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                # images = images.to(device)
                labels = labels.to(device)
            
                # Forward pass
                if wandb.config["model"] == "inception-v3":
                    outputs, _ = model(images)
                else:
                    outputs = model(images)

                if l1 != 0:
                    # log("LOG -> L1 regularization is activated")
                    loss = criterion(outputs, labels) + l1_lambda * l1_regularization(model)  # Add L1 regularization to the loss
                else:
                    # log("LOG -> L1 regularization is defualt")
                    loss = criterion(outputs, labels)
                train_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
        
                #INCEPTION NET 
                # Forward pass
                # outputs, _ = model(images)  # Retrieve the InceptionOutputs tuple
                # logits = outputs.logits  # Get the logits from the InceptionOutputs object
                # loss = criterion(logits, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 10 == 0:
                    print(f'Training: Epoch [{epoch + 1}/{wandb.config["num_epochs"]}], Step [{i + 1}/{total_step}], Loss: {loss.item()/100:.4f}')
                    
            train_accuracy = (train_correct / train_total) * 100
            avg_train_loss = train_loss / len(train_loader)

            
            # Validation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    if wandb.config["model"] == "inception-v3":
                        outputs= model(images)
                    else:
                        outputs = model(images)

                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                val_accuracy = (correct / total) * 100
                avg_val_loss = val_loss / len(val_loader)
                
                print(f'Validation: Epoch [{epoch + 1}/{wandb.config["learning_rate"]}], Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
                
                # Early Stopping Check and Save the Best Model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    print(patience)
                    best_model_state_dict = model.state_dict()  # Save the current model state
                else:
                    epochs_without_improvement += 1
                    print("Early stopping triggered count:" + str(epochs_without_improvement))

                    # -------------
                    # Need to check this problems 
                    if fine_tune_enable:
                        if fine_tune_activated == False and epochs_without_improvement >= patience: 
                            print("Fine_tune_actiavte!")
                            fine_tune_activated = True
                            epochs_without_improvement = 0
                            patience = patience - 4
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = wandb.config["learning_rate"] / 10
                                print(param_group['lr'])
                            # l1_lambda = l1_lambda / 10
                            # l2_loss = l2_loss / 10
                            #load model
                            if best_model_state_dict:
                                model.load_state_dict(best_model_state_dict)
                        else:
                            if epochs_without_improvement >= patience:
                                print(f'FT Early stopping at epoch {epoch + 1}. Validation loss has not improved for {patience} consecutive epochs.')
                                wandb.run.summary["stopping"] = epoch
                                break
                    
                    else:
                        if epochs_without_improvement >= patience:
                            print(f'Early stopping at epoch {epoch + 1}. Validation loss has not improved for {patience} consecutive epochs.')
                            wandb.run.summary["stopping"] = epoch
                            break

            # EPOCHS LOG
            # Record the end time
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print("LOG -> Time per epoch")
            print(elapsed_time)
            wandb.log({
                "train_loss": avg_train_loss, 
                "train_acc": train_accuracy,
                "valid_loss": avg_val_loss, 
                "valid_acc": val_accuracy,
                "epoch": epoch,
                "epoch_time": elapsed_time
            })
        # Save the profiler results to a file
        # prof.export_chrome_trace(PATH_TO_CHROME_TRACE_FILE)

        print("LOG -> Store model")
        print(wandb.run.name)
        # Store model
        if best_model_state_dict:
            model.load_state_dict(best_model_state_dict)
            torch.save(model.state_dict(), PATH_TMP_FOLDER_MODEL+'/model_weights_{model_name}.pth'.format(model_name=wandb.run.name))
            print("LOG -> Store model")
        if TEST == False:
            wandb.finish()


    # --------------------------------------------------------------------
    # EVALUATION / TEST
    # --------------------------------------------------------------------
    if TEST:
        log("INIT model evaluation")
        # # RUN IF IS ONLY TESTING
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # log(f'Detected devce is {device}')
        # log(f'Detected devce name is {torch.cuda.get_device_name()}')
        log("Eval test run")
        num_classes = 2

        # LOAD MODEL FROM STORAGE - this is not good model could be different .. need to creat function
        if wandb.config["model"] == "resnet-18":
            log("LOG -> ResNet-18 model")
            model = torchvision.models.resnet18(pretrained=True)
            print(model)
            model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
            print(model)

            if wandb.config["dropout"] > 0:
                print("LOG -> Droppout is activated")
                # Add dropout to the fully connected layer (fc)
                dropout_prob = wandb.config["dropout"]  # Adjust this value as per your preference
                model.fc = nn.Sequential(
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Dropout(dropout_prob, inplace=True),
                    nn.Linear(512, 2)  # Change the output layer to match the number of classes
                )
            print(model)
            model.to(device)
        elif wandb.config["model"] == "resnet-34":
            log("LOG -> ResNet-34 model")
            model = torchvision.models.resnet34(pretrained=True)
            print(model)
            model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
            print(model)

            if wandb.config["dropout"] > 0:
                print("LOG -> Droppout is activated")
                # Add dropout to the fully connected layer (fc)
                dropout_prob = wandb.config["dropout"]  # Adjust this value as per your preference
                model.fc = nn.Sequential(
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Dropout(dropout_prob, inplace=True),
                    nn.Linear(512, 2)  # Change the output layer to match the number of classes
                )
            print(model)
            model.to(device)
        elif wandb.config["model"] == "resnet-50":
            log("LOG -> ResNet-50 model")
            model = torchvision.models.resnet50(pretrained=True)
            print(model)
            # model.fc = nn.Linear(2048, 2)
            model.fc = nn.Sequential(
                    nn.Linear(2048, 512),
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
    
                    nn.Linear(512, 2)  # Change the output layer to match the number of classes
                )
            # model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
            print(model)

            if wandb.config["dropout"] > 0:
                print("LOG -> Droppout is activated")
                # Add dropout to the fully connected layer (fc)
                dropout_prob = wandb.config["dropout"]  # Adjust this value as per your preference
                model.fc = nn.Sequential(
                     nn.Linear(2048, 512),
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Linear(512, 2)  # Change th)  # Change the output layer to match the number of classes
                )
            print(model)
            model.to(device)
        elif wandb.config["model"] == "inception-v3":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            print(model)
            model.fc = nn.Sequential(
                     nn.Linear(2048, 512),
                    # nn.Dropout(dropout_prob, inplace=True),
                    # nn.Linear(512, 128),
                    nn.Linear(512, 2)  # Change th)  # Change the output layer to match the number of classes
                )
            model.to(device)

        else:
            print("LOG ERROR -> Model was not defined")
            exit()
          # Change the output layer to match the number of classes in your dataset
        model.to(device)
        # Donwload trained weights
        result = model.load_state_dict(torch.load("{model_path}/model_weights_{model_name}.pth".format(model_path=PATH_TMP_FOLDER_MODEL, model_name=wandb.run.name)))
        log(result)

        test_dataset = CvatMonaiDatasetClassification(dataframe=dataset_loader.df_test, 
                                                server_path=dataset_loader.path_cvat_folder, 
                                                resolution = tuple(map(int,wandb.config["resolution"].split(','))), 
                                                device=device, 
                                                transform_spatial_type=wandb.config["spatial_transforms"],
                                                transforms_augumentation_type="empty_transforms", 
                                                transform_output_type=wandb.config["normalise_intensity"])
        test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False, collate_fn=monai.data.list_data_collate)


        evaluator = Evaluation(model=model,
                    dataset_loader=dataset_loader,
                    loader=test_loader, 
                    device=device,
                    run_name=wandb.run.name,
                    model_type=wandb.config["model"]
                        )


        log("!WARNING: threshold is hardcoded!")
        prediction_threshold = 0.5
        video_frame_votes_threshold = 0.5
        evaluator.model_eval(threshold=prediction_threshold)

        result_metrics = evaluator.calculate_eval_metric_per_video(mode = "binary")
        log(result_metrics.__dict__)
        log(result_metrics.balanced_accuracy)
        wandb.run.summary["accuracy"] = result_metrics.accuracy
        wandb.run.summary["balanced_accuracy"] = result_metrics.balanced_accuracy
        wandb.run.summary["precision"] = result_metrics.precision
        wandb.run.summary["recall"] = result_metrics.recall
        wandb.run.summary["f1_score"] = result_metrics.f1_score
        wandb.run.summary["support"] = result_metrics.support
        wandb.run.summary["confusion_matrix"] = result_metrics.confusion_matrix

        result_metrics = evaluator.calculate_eval_metric_per_frame(wandb=wandb, mode="binary",video_frame_votes_threshold=video_frame_votes_threshold)
        

        wandb.finish()