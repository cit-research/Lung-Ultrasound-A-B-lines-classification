
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


from utils.CustomDataset import CustomDataset
from utils.Augumentations import RandomNoise, GrayscaleNormalize, RandomSpeckleNoise

from sklearn import metrics
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

import time 
# from torch.profiler import profile, record_function, ProfilerActivity
# import torch.profiler

# GLOBAL VARIABLES 

PROJECT_NAME = "KKUI-Lung-ablines-classification"
JOB_TYPE = "develop"
ARTIFACT_NAME = "kkui-lung-ablines:"  #name of verzion dataset you need
TMP_DATA_FOLDER = "/data/data_mh731nk/experiments_tmp"
PATH_FOLDER_ARTIFACTS = TMP_DATA_FOLDER+"/"+PROJECT_NAME+"/aritifacts"
PATH_FOLDER_DATA = TMP_DATA_FOLDER+"/"+PROJECT_NAME+"/data"
SYNOLOGY_FOLDER = "/data_nas_synology/"
PATH_MODELS_DATA = TMP_DATA_FOLDER+"/"+PROJECT_NAME+"/models/"


Path(TMP_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
Path(TMP_DATA_FOLDER+"/"+PROJECT_NAME).mkdir(parents=True, exist_ok=True)
Path(PATH_FOLDER_ARTIFACTS).mkdir(parents=True, exist_ok=True)
Path(PATH_FOLDER_DATA).mkdir(parents=True, exist_ok=True)

# config class
class SimpleNamespace:
    def __init__(self,fold_id,experiment_class,dataset_revision,probe_types,wandb_project,dataset_filtering,random_seed,batch_size,resolution,learning_rate,num_epochs,validation_split,dropout,patience,model,class_weight,l2_loss,l1_activate,l1_lambda) -> None:
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



config_defaults = SimpleNamespace(
    fold_id=0,
    experiment_class = "a",
    dataset_revision=8,
    probe_types="probe_lumify",
    dataset_filtering="No",
    random_seed=1234,
    batch_size = 16,
    resolution="512,1024",
    learning_rate = 0.001,
    num_epochs = 100,
    validation_split = 0.2,
    dropout = 0,
    patience = 5,
    model = "resnet-34",
    class_weight = 5.0,
    l2_loss = 0.000001,
    l1_activate = True,
    l1_lambda = 0.0001,
    wandb_project=PROJECT_NAME,
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


    parser.add_argument('--wandb_project', type=str, default=config_defaults.wandb_project)
    return parser.parse_args()


def train(conifg=config_defaults):
    pass

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('fork')# good solution !!!!
    # PIPELINE
    # LOAD DATASET
    
    args = parse_args()
    print(SimpleNamespace(**args.__dict__))
    print(args)
    print(PROJECT_NAME)

    
    with wandb.init(project=PROJECT_NAME, config=args, job_type=JOB_TYPE):

        ### TEMP  GLOBAL VARAIBLES
        best_model_state_dict = None
        ### CONTINUE
        print("LOG -> Project was initialized")
        pprint.pprint(config_defaults.__dict__)

        wandb.config["new_param"] = "hello"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("LOG -> Device GPU load")
        print(device)
        wandb.config["device"] = device
        if device == "cuda":
            wandb.config["device_name"] = torch.cuda.get_device_name()
        else: 
            wandb.config["device_name"] = "cpu"

        print("LOG -> Download artifacts")
        ARTIFACT_NAME = ARTIFACT_NAME+wandb.config["probe_types"]
        print("LOG -> Artifactt name: " + ARTIFACT_NAME)


        artifact = wandb.use_artifact(ARTIFACT_NAME)
        print("LOG -> Artifact aliases: ")
        pprint.pprint(artifact.aliases)
        print("LOG -> Artifact metadata")
        pprint.pprint(artifact.metadata)
        DATASET_REVISION = "revision_" + str(artifact.metadata["dataset_revision"])

        # DATASET
        ## Download artifacts 
        PATH_ARTIFACTS = artifact.download(root="{path_artifacts_folder}/{artifact_name}".format(path_artifacts_folder=PATH_FOLDER_ARTIFACTS,artifact_name=ARTIFACT_NAME))
        print("LOG -> Artifafct path: " + PATH_ARTIFACTS)
        
        os.path.join
        ## CHECK/CREATE REVISION PROJECT
        if os.path.exists(PATH_FOLDER_DATA) and os.path.isdir(PATH_FOLDER_DATA):
            print(f"LOG -> The folder at {PATH_FOLDER_DATA} exists.")

            if os.path.exists(PATH_FOLDER_DATA+"/"+DATASET_REVISION) and os.path.isdir(PATH_FOLDER_DATA+"/"+DATASET_REVISION):
                print(f"LOG -> The folder at {DATASET_REVISION} exists.")
            else:
                print(f"LOG -> The folder at {DATASET_REVISION} does not exist.")
                Path(PATH_FOLDER_DATA+"/"+DATASET_REVISION).mkdir(parents=True, exist_ok=True)
                print("LOG -> Folder created")

        else:
            print(f"LOG -> The folder at {PATH_FOLDER_DATA} does not exist.")

        ## CECK CREATE AND DONWLAOD neccesary files

        for folder_name in artifact.metadata["neccessary_folders"]:
            print(folder_name)

            if os.path.exists(os.path.join(PATH_FOLDER_DATA,DATASET_REVISION,folder_name)) and os.path.isdir(os.path.join(PATH_FOLDER_DATA,DATASET_REVISION,folder_name)):
                print(f"LOG -> The folder at {PATH_FOLDER_DATA} exists.")
            else:
                print(f"LOG -> The folder at {PATH_FOLDER_DATA} do not exist need to copy.")
                source_folder = os.path.join(SYNOLOGY_FOLDER, artifact.metadata["synology_path"],DATASET_REVISION,"cropped")
                destination_folder = os.path.join(PATH_FOLDER_DATA,DATASET_REVISION,"cropped")
                shutil.copytree(source_folder, destination_folder)
                print("LOG -> File transfer done")
        
        ### DATASET PATH 
        TMP_DATA_FOLDER_CVAT = os.path.join(PATH_FOLDER_DATA, DATASET_REVISION)

        ## LOAD supported files
        ### LOAD DF df_full
        df_full_merged = pd.read_csv(os.path.join(PATH_FOLDER_ARTIFACTS,ARTIFACT_NAME,"df_frames_full.csv"))
        print(df_full_merged.shape)

        ### LOAD JSON for fold
        # Open and read the JSON file
        with open(os.path.join(PATH_FOLDER_ARTIFACTS,ARTIFACT_NAME,"video","fold_" + str(wandb.config["fold_id"])+".json"), "r") as json_file:
            fold_json_data = json.load(json_file)
        pprint.pprint(fold_json_data)
        df_fold_videos = pd.DataFrame(data=fold_json_data)
        print(df_fold_videos.shape)
        fold_video_name = list(df_fold_videos["name"])
        pprint.pprint(df_fold_videos)

        ### WARNING: FILTERING FILES
        ### NOW NO FILTER
        if  wandb.config["dataset_filtering"] == "No":
            print("LOG -> Dataset filtering - NO")
            print(df_full_merged.shape)
            # df_full_merged_fold = df_full_merged.loc[df_full_merged["name_cvat"].isin(df_fold_videos["name"])]
            df_result = pd.merge(df_full_merged, df_fold_videos, left_on="name_cvat", right_on="name",how="right")
            # print(df_full_merged_fold.shape)
            print(df_result.shape)
        else:
            print("LOG -> taky filtering este nemam zvaliduj kod nieco zle aprsuje z configu")
            df_result = pd.merge(df_full_merged, df_fold_videos, left_on="name_cvat", right_on="name",how="right")

        pprint.pprint(df_result.columns)
        ### LETER ADD VIDEO BALANCE
        df_result["label_old"] = df_result["label"] # keep old label
        # ### WARNING: CLASS SWAPPING
        # 0: no_lines
        # 1: a_lines
        # 2: b_lines
        # for A_lines experiment -> b_lines change to 0
        # for B_lines experiment -> a_lins change to 0
        if wandb.config["experiment_class"] == "a":
            print("LOG -> Expeirment type Alines. Label was changed")
            df_result.loc[df_result["label_old"] == 2, "label"] = 0
        elif wandb.config["experiment_class"] == "b":
            print("LOG -> Expeirment type Vlines. Label was changed")
            df_result.loc[df_result["label_old"] == 2, "label"] = 1
            df_result.loc[df_result["label_old"] == 1, "label"] = 0
            # df_result.loc[df_result["label_old"] == 0, "label"] = 0
        else:
            print("LOG -> frong experiment type")

        ### SPLIT TRAIN / TEST
        df_train = df_result.loc[df_result["subset"] == "train"]
        print("LOG -> DATASET -> Per frame counts")
        print("TRAINT", df_train.loc[df_train["label_old"]==0].shape[0], df_train.loc[df_train["label_old"]==1].shape[0], df_train.loc[df_train["label_old"]==2].shape[0])
   
        df_train, df_valid = train_test_split(df_train, test_size = 0.2, stratify=df_train["label_old"], random_state=wandb.config["random_seed"])
        print("TRAINT (after split)", df_train.loc[df_train["label_old"]==0].shape[0], df_train.loc[df_train["label_old"]==1].shape[0], df_train.loc[df_train["label_old"]==2].shape[0])
        print("VALID (after split)", df_valid.loc[df_valid["label_old"]==0].shape[0], df_valid.loc[df_valid["label_old"]==1].shape[0], df_valid.loc[df_valid["label_old"]==2].shape[0])
        df_test = df_result.loc[df_result["subset"] == "test"]
        print("TEST", df_test.loc[df_test["label_old"]==0].shape[0], df_test.loc[df_test["label_old"]==1].shape[0], df_test.loc[df_test["label_old"]==2].shape[0])
        

        # ### Generate statistic of dataset
        # #### PER FRAME
        wandb.config["subset_train_frames_count_0"] = df_train.loc[df_train["label"]==0].shape[0]
        wandb.config["subset_train_frames_count_1"] = df_train.loc[df_train["label"]==1].shape[0]
        wandb.config["subset_train_frames_count_2"] = df_train.loc[df_train["label"]==2].shape[0]
        
        wandb.config["subset_valid_frames_count_0"] = df_valid.loc[df_valid["label"]==0].shape[0]
        wandb.config["subset_valid_frames_count_1"] = df_valid.loc[df_valid["label"]==1].shape[0]
        wandb.config["subset_valid_frames_count_2"] = df_valid.loc[df_valid["label"]==2].shape[0]

        wandb.config["subset_test_frames_count_0"] = df_test.loc[df_test["label"]==0].shape[0]
        wandb.config["subset_test_frames_count_1"] = df_test.loc[df_test["label"]==1].shape[0]
        wandb.config["subset_test_frames_count_2"] = df_test.loc[df_test["label"]==2].shape[0]



        ### Define transformati
        print("LOG -> Define transformations")

        # mus by nakopirovat piramo do dataloadera 
        train_transform = transforms.Compose([
            # # MUST BE 
            # transforms.EnsureChannelFirst(), 
            # transforms.Resize(resolution),
            transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3),
            transforms.RandAdjustContrast(prob=0.5, gamma=(0.5, 4.5)),
            transforms.RandHistogramShift(num_control_points=10, prob=0.5),
            transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5),

            transforms.RandFlip(prob=0.5, spatial_axis=1),
            # transforms.HistogramNormalize(num_bins=256, min=0, max=255)
            transforms.RandGibbsNoise(prob=0.5, alpha=(0.4, 1.0)),
            
            transforms.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.4),
            transforms.RandRotate(range_x=10.0, prob=0.5, keep_size=True),
            transforms.RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5),prob=0.8),
                                        
            transforms.RandAffine(prob=0.5),
            transforms.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.03, 0.03)),
            transforms.Rand2DElastic(spacing=(20,20), magnitude_range=(1,2), prob=0.4),
            transforms.RandCoarseDropout(holes=5, spatial_size=(30,30), fill_value=0,prob=0.6),
            transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3)
            # transforms.HistogramNormalize(num_bins=256, min=0, max=255)
            
            # # MUST BE 
            # transforms.ToTensor(),
            # transforms.NormalizeIntensity()
        ])

        val_transform = transforms.Compose([
            # # MUST BE 
            # transforms.EnsureChannelFirst(), 
            # transforms.Resize(resolution),
            # transforms.HistogramNormalize(num_bins=256, min=0, max=255)
            # # MUST BE 
            # transforms.ToTensor(),
            # transforms.NormalizeIntensity()
        ])
        

        ### SEPARETE DATASETS
        print("LOG -> Prepair datasets and dataloader object")
        # Create custom datasets and data loaders for training, validation, and test
        print("___________________RESOLUTION")
        print(tuple(map(int,wandb.config["resolution"].split(','))))
        train_dataset = CustomDataset(df_train, TMP_DATA_FOLDER_CVAT, resolution = tuple(map(int,wandb.config["resolution"].split(','))), device=device, transform=train_transform, class_for_augumentation=None)
        train_loader = DataLoader(train_dataset, num_workers=0, batch_size=wandb.config["batch_size"], shuffle=True, collate_fn=monai.data.list_data_collate)

        val_dataset = CustomDataset(df_valid, TMP_DATA_FOLDER_CVAT,resolution = tuple(map(int,wandb.config["resolution"].split(','))), device=device ,transform=val_transform, class_for_augumentation=1)
        val_loader = DataLoader(val_dataset, num_workers=0, batch_size=wandb.config["batch_size"], shuffle=False, collate_fn=monai.data.list_data_collate)

        test_dataset = CustomDataset(df_test, TMP_DATA_FOLDER_CVAT, resolution = tuple(map(int,wandb.config["resolution"].split(','))), device=device ,transform=val_transform, class_for_augumentation=None)
        test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False, collate_fn=monai.data.list_data_collate)

        
        ### PREPAIR MODEL
        print("LOG -> Prepir model")
        # Define the ResNet-18 model

        if wandb.config["model"] == "resnet-18":
            model = torchvision.models.resnet18(pretrained=True)
            print(model)

            model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
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
            model.to(device)
        elif wandb.config["model"] == "resnet-34":
            model = torchvision.models.resnet34(pretrained=True)
            print(model)

            model.fc = nn.Linear(512, 2)  # Change the output layer to match the number of classes in your dataset
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
                outputs = model(images)

                if l1_lambda != 0:
                    loss = criterion(outputs, labels) + l1_lambda * l1_regularization(model)  # Add L1 regularization to the loss
                else:
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
                    print(f'Training: Epoch [{epoch + 1}/{wandb.config["num_epochs"]}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
                    
            train_accuracy = (train_correct / train_total) * 100
            avg_train_loss = train_loss / len(test_loader)

            
            # Validation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
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

                        
                    if fine_tune_enable:
                        if fine_tune_activated == False and epochs_without_improvement >= patience: 
                            print("Fine_tune_actiavte!")
                            fine_tune_activated = True
                            epochs_without_improvement = 0
                            patience = patience + 2
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = wandb.config["learning_rate"] / 10
                                print(param_group['lr'])
                            l1_lambda = l1_lambda / 10
                            l2_loss = l2_loss / 10
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
            print(end_time)
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
            torch.save(model.state_dict(), PATH_MODELS_DATA+'model_weights_{model_name}.pth'.format(model_name=wandb.run.name))

        print("LOG -> Store model")
        ### Test the model
        threshold = 0.5  # You can adjust this threshold based on your needs
        model.eval()
        test_acc = 0
        test_output = []
        test_binnary = []


        print("LOG -> Start test")
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # Apply thresholding
                predicted_probs = torch.softmax(outputs, dim=1)
                predicted_class = (predicted_probs[:, 1] > threshold).long()  # Change the index (1) based on your classes
                test_binnary.append(int(predicted_class))
                test_output.extend(predicted_probs[:, 1].cpu().numpy())

                total += labels.size(0)
                correct += (predicted_class == labels).sum().item()
                # break

            accuracy = (correct / total) * 100
            test_acc = accuracy
            print(f'Test Accuracy: {accuracy:.2f}%')

        print("LOG -> Store accuracy")
        if wandb.run is not None:
            print("LOG -> Store accuracy")
            wandb.run.summary["test_accuracy"] = test_acc
        
        print("LOG -> DF operations")
        df_test["predicted"] = test_binnary
        df_test["predicted_raw"] = test_output
        # Assuming you have y_true and y_pred arrays

        y_true = df_test["label"]
        y_pred = df_test["predicted"]

        print("LOG -> Calculatin another metrics")
        # 1. Accuracy
        accuracy = metrics.accuracy_score(y_true, y_pred)

        # 2. Precision, Recall, F1-score, Support (for each class)
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_true, y_pred)

        print("LOG -> Calculatin another metrics")
        if wandb.run is not None:
            print("LOG -> Calculatin another metrics")
            wandb.run.summary["precision_0"] = precision[0]
            wandb.run.summary["precision_1"] = precision[1]
            wandb.run.summary["recall_0"] = recall[0]
            wandb.run.summary["recall_1"] = recall[1]
            wandb.run.summary["f1_score_0"] = f1_score[0]
            wandb.run.summary["f1_score_1"] = f1_score[1]
            wandb.run.summary["f1_score_avg"] = (f1_score[0] + f1_score[1])/2
            wandb.run.summary["support_0"] = support[0]
            wandb.run.summary["support_1"] = support[1]

        if wandb.run is not None:
        # 3. Confusion Matrix
            confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
            wandb.run.summary["cf_0:0"] = confusion_matrix[0,0]
            wandb.run.summary["cf_0:1"] = confusion_matrix[0,1]
            wandb.run.summary["cf_1:0"] = confusion_matrix[1,0]
            wandb.run.summary["cf_1:1"] = confusion_matrix[1,1]
            wandb.run.summary["confusion_matrix"] = confusion_matrix

        df_test.to_csv("local.csv")
        # 4. Classification Report
        # classification_report = metrics.classification_report(y_true, y_pred)

        # # 5. ROC AUC Score (if binary classification)
        # roc_auc = metrics.roc_auc_score(y_true, y_pred)

        # # 6. Area Under the Precision-Recall Curve (if binary classification)
        # pr_auc = metrics.average_precision_score(y_true, y_pred)