import pprint
import os
import configparser
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd
import json
from sklearn.model_selection import train_test_split
#
# Load config values
config = configparser.ConfigParser()
config.read('./configs/global.cfg')

PROJECT_NAME = config.get('Global', 'project_name')
PATH_TMP_FOLDER_ARTIFACTS = config.get('Global', 'path_temp_folder_artifacts') #name of verzion dataset you need
PATH_TMP_FOLDER_DATA = config.get('Global', 'path_temp_folder_data') #name of verzion dataset you need
# ARTIFACT_NAME = config.get('Global', 'artifacts_name')
PATH_SYNOLOGY_FOLDER = config.get('Global', 'path_synology_folder')

class DatasetLoader():
    # ----------
    # Output -> from dateset must be 3 DataFrames for test train and valid.
    # Input -> Metadata for dataset is downlaoder from wandb project.
    # Dataloader is disagned for using CVAT proprocesing repository. 
    # ----------
    def __init__(self, artifact_name, artifact_tag):
        self.log("Created")
        #self.path_for_temp_folder = path_for_temp_folder # this folder is using for string artifact (dataset metadadta), dataset, models and outputs
        #self.artifat_name = artifact_name # whole name or dataset artifat (only metadata) for example "kkui-lung-ablines"
        self.artifact_tag = artifact_tag # like "latest", "v6" or "probes_all"
        self.artifact_full_name = artifact_name + ":" + self.artifact_tag
        self.log("Artifacts full name: {}".format(self.artifact_full_name))
        self.artifact = {}
        self.dataset_revision = "revision_" # describe version of cvat dataset processing
        self.path_artifact = ""
        self.path_cvat_folder = ""
        Path(PATH_TMP_FOLDER_ARTIFACTS+"/"+PROJECT_NAME).mkdir(parents=True, exist_ok=True)
        Path(PATH_TMP_FOLDER_DATA+"/"+PROJECT_NAME).mkdir(parents=True, exist_ok=True)

    def log(self, payload):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp, "LOG from ",self.__class__.__name__, ":" ,payload)
        
    def download_artifacts(self, wandb):
        # Dowload artifacts (metadata of dataset)
        self.artifact = wandb.use_artifact(self.artifact_full_name)
        self.dataset_revision = "revision_" + str(self.artifact.metadata["dataset_revision"])
        self.path_cvat_folder = os.path.join(PATH_TMP_FOLDER_DATA,PROJECT_NAME,self.dataset_revision) # need to be created here becouse a need know revision number
        
        # Download metadata files form WanDB server -> get path to data, 
        self.path_artifact = self.artifact.download(root=os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name))
        self.log("Artifacts (dataset metadata) is located od path: {}".format(self.path_artifact))
        self.log(self.artifact.metadata)

    def download_dataset(self, copy_from_nas=False):
        # Check temp folders if exist. If not folders will be created. 
        if os.path.exists(PATH_TMP_FOLDER_DATA) and os.path.isdir(PATH_TMP_FOLDER_DATA):
            self.log(f"The folder at {PATH_TMP_FOLDER_DATA} exists.")

            if os.path.exists(os.path.join(PATH_TMP_FOLDER_DATA,PROJECT_NAME,self.dataset_revision)) and os.path.isdir(os.path.join(PATH_TMP_FOLDER_DATA,PROJECT_NAME,self.dataset_revision)):
                self.log(f"The folder at {self.dataset_revision} exists.")
            else:
                self.log(f"The folder at {self.dataset_revision} does not exist.")
                Path(os.path.join(PATH_TMP_FOLDER_DATA,PROJECT_NAME,self.dataset_revision)).mkdir(parents=True, exist_ok=True)
                self.log("Folder created")

        else:
            self.log(f"The folder at {PATH_TMP_FOLDER_DATA} does not exist.")

        # Checking if file files is neccesery files is exist in destionation temoporary folders
        for folder_name in self.artifact.metadata["neccessary_folders"]:
            self.log("Folder {} is exacuted".format(folder_name))

            if os.path.exists(os.path.join(self.path_cvat_folder,folder_name)) and os.path.isdir(os.path.join(self.path_cvat_folder,folder_name)):
                self.log(f"The folder at {os.path.join(self.path_cvat_folder,folder_name)} exists.")
            else:
                if copy_from_nas:
                    self.log(f"The folder at {os.path.join(self.path_cvat_folder,folder_name)} do not exist need to copy.")
                    source_folder = os.path.join(PATH_SYNOLOGY_FOLDER, self.artifact.metadata["synology_path"],self.dataset_revision,"cropped")
                    destination_folder = os.path.join(self.path_cvat_folder,folder_name)
                    shutil.copytree(source_folder, destination_folder)
                    self.log("File transfer done")
                else:
                    self.log("Data from next steps must coppiend manualy. Code will be interupted")
                    exit()


##################################################################################################################
############################################ WITH GENERATED DATA
#################################################################################################################

class DataLoaderABLineFrameWithGenerated(DatasetLoader):
    def __init__(self, artifact_name, artifact_tag, generated_file_metadata_json, generated_use_for_subset="test"):
        super().__init__(artifact_name, artifact_tag)
        self.df_full_merged = pd.DataFrame()
        self.df_fold_videos = pd.DataFrame()
        self.generated_file_metadata_json = generated_file_metadata_json
        self.generated_use_for_subset = generated_use_for_subset
   

    
    def load_dataset(self,wandb):
        # --------------------------
        # is necessary use for different dataset types different loaders
        # Future decision on metadata from artifacts
        # df_full_merged must contain only frame with labels and subset variable
        # --------------------------
        if True:
            return self.dataset_transformer_per_video(wandb)



     
    def dataset_transformer_per_video(self,wandb):
        self.df_full_merged = pd.read_csv(os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name,"df_frames_full.csv"))
        self.df_full_merged = self.df_full_merged.drop(['label'], axis=1, errors='ignore')
        self.log(f"DataFrame full merge has shape: {str(self.df_full_merged.shape)}")
        self.log(f"Loading dataframe fold number {str(wandb.config['fold_id'])}")

        
        frames_all = []
        # LOADING read data from artefacts
        with open(os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name,"video","fold_" + str(wandb.config["fold_id"])+".json"), "r") as json_file:
            fold_json_data = json.load(json_file)
        print("i am using real data")

        for video in fold_json_data:
                for frame in video["frames_only_label"]:
                    frame["label"] = video["label"]
                    frame["subset"] = video["subset"]
                    frames_all.append(frame)

        # LOADING generated data
        if self.generated_file_metadata_json is not None:
            for path in self.generated_file_metadata_json:
                print("i am using also generated aritficail data")
                with open(path, "r") as json_file:
                    fold_json_data = json.load(json_file)
                
                for video in fold_json_data:
                    for frame in video["frames_only_label"]:
                        frame["label"] = video["label"]
                        frame["subset"] = self.generated_use_for_subset
                        frames_all.append(frame)



        self.df_full_merged = pd.DataFrame(frames_all)
        self.df_full_merged["label_old"] = self.df_full_merged["label"] # keep old label
#           
        if wandb.config["experiment_class"] == "a":
            self.log("Expeirment type Alines. Labels was changed")
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 2, "label"] = 0
        elif wandb.config["experiment_class"] == "b":
            self.log("Expeirment type Blines. Labels was changed")
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 2, "label"] = 1
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 1, "label"] = 0
            # self.df_full_merge.loc[self.df_full_merge["label_old"] == 0, "label"] = 0
        else:
            self.log("Wrong experiment type. Exit()")
            exit()
        
        ### SPLIT TRAIN / TEST
        ### ! WARNING THIS MUST BE IMPROVED AN USED DURING PREPARING DATASET FOR SPLITING ON EACH SUBSET!
        df_train = self.df_full_merged.loc[self.df_full_merged["subset"] == "train"]
        self.log("DATASET Per frame counts")
        self.log(f'TRAINT: {df_train.loc[df_train["label_old"]==0].shape[0]}, {df_train.loc[df_train["label_old"]==1].shape[0]}, {df_train.loc[df_train["label_old"]==2].shape[0]}')
   
        df_train, df_valid = train_test_split(df_train, test_size = 0.2, stratify=df_train["label_old"], random_state=wandb.config["random_seed"])
        self.log(f'TRAINT (after split), {df_train.loc[df_train["label_old"]==0].shape[0]}, {df_train.loc[df_train["label_old"]==1].shape[0]}, {df_train.loc[df_train["label_old"]==2].shape[0]}')
        self.log(f'VALID (after split), {df_valid.loc[df_valid["label_old"]==0].shape[0]}, {df_valid.loc[df_valid["label_old"]==1].shape[0]}, {df_valid.loc[df_valid["label_old"]==2].shape[0]}')
        df_test = self.df_full_merged.loc[self.df_full_merged["subset"] == "test"]
        self.log(f'TEST, {df_test.loc[df_test["label_old"]==0].shape[0]}, {df_test.loc[df_test["label_old"]==1].shape[0]}, {df_test.loc[df_test["label_old"]==2].shape[0]}')
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test


        print(self.df_full_merged.shape)
        print(self.df_full_merged.shape)
        print(self.df_full_merged.columns)






##########################################################################################################################################################








class DataLoaderABLineFrame(DatasetLoader):
    def __init__(self, artifact_name, artifact_tag):
        super().__init__(artifact_name, artifact_tag)
        self.df_full_merged = pd.DataFrame()
        self.df_fold_videos = pd.DataFrame()
   

    
    def load_dataset(self,wandb):
        # --------------------------
        # is necessary use for different dataset types different loaders
        # Future decision on metadata from artifacts
        # df_full_merged must contain only frame with labels and subset variable
        # --------------------------
        if True:
            return self.dataset_transformer_per_video(wandb)



     
    def dataset_transformer_per_video(self,wandb):
        self.df_full_merged = pd.read_csv(os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name,"df_frames_full.csv"))
        self.df_full_merged = self.df_full_merged.drop(['label'], axis=1, errors='ignore')
        self.log(f"DataFrame full merge has shape: {str(self.df_full_merged.shape)}")
        self.log(f"Loading dataframe fold number {str(wandb.config['fold_id'])}")

        # frames_train = []
        # frames_test = []



        with open(os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name,"video","fold_" + str(wandb.config["fold_id"])+".json"), "r") as json_file:
            fold_json_data = json.load(json_file)

        frames_all = []
        for video in fold_json_data:
                for frame in video["frames_only_label"]:
                    frame["label"] = video["label"]
                    frame["subset"] = video["subset"]
                    frames_all.append(frame)


        
        self.df_full_merged = pd.DataFrame(frames_all)
        self.df_full_merged["label_old"] = self.df_full_merged["label"] # keep old label
#           
        if wandb.config["experiment_class"] == "a":
            self.log("Expeirment type Alines. Labels was changed")
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 2, "label"] = 0
        elif wandb.config["experiment_class"] == "b":
            self.log("Expeirment type Blines. Labels was changed")
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 2, "label"] = 1
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 1, "label"] = 0
            # self.df_full_merge.loc[self.df_full_merge["label_old"] == 0, "label"] = 0
        else:
            self.log("Wrong experiment type. Exit()")
            exit()
        
        ### SPLIT TRAIN / TEST
        ### ! WARNING THIS MUST BE IMPROVED AN USED DURING PREPARING DATASET FOR SPLITING ON EACH SUBSET!
        df_train = self.df_full_merged.loc[self.df_full_merged["subset"] == "train"]
        self.log("DATASET Per frame counts")
        self.log(f'TRAINT: {df_train.loc[df_train["label_old"]==0].shape[0]}, {df_train.loc[df_train["label_old"]==1].shape[0]}, {df_train.loc[df_train["label_old"]==2].shape[0]}')
   
        df_train, df_valid = train_test_split(df_train, test_size = 0.2, stratify=df_train["label_old"], random_state=wandb.config["random_seed"])
        self.log(f'TRAINT (after split), {df_train.loc[df_train["label_old"]==0].shape[0]}, {df_train.loc[df_train["label_old"]==1].shape[0]}, {df_train.loc[df_train["label_old"]==2].shape[0]}')
        self.log(f'VALID (after split), {df_valid.loc[df_valid["label_old"]==0].shape[0]}, {df_valid.loc[df_valid["label_old"]==1].shape[0]}, {df_valid.loc[df_valid["label_old"]==2].shape[0]}')
        df_test = self.df_full_merged.loc[self.df_full_merged["subset"] == "test"]
        self.log(f'TEST, {df_test.loc[df_test["label_old"]==0].shape[0]}, {df_test.loc[df_test["label_old"]==1].shape[0]}, {df_test.loc[df_test["label_old"]==2].shape[0]}')
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test


        print(self.df_full_merged.shape)
        print(self.df_full_merged.shape)
        print(self.df_full_merged.columns)

class DataLoaderABLine(DatasetLoader):
    def __init__(self, artifact_tag):
        super().__init__(artifact_tag)
        self.df_full_merged = pd.DataFrame()
        self.df_fold_videos = pd.DataFrame()
   

    
    def load_dataset(self,wandb):
        # --------------------------
        # is necessary use for different dataset types different loaders
        # Future decision on metadata from artifacts
        # df_full_merged must contain only frame with labels and subset variable
        # --------------------------
        if True:
            return self.dataset_transformer_per_video(wandb)


       

    def dataset_transformer_per_video(self,wandb):
        self.df_full_merged = pd.read_csv(os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name,"df_frames_full.csv"))
        self.df_full_merged = self.df_full_merged.drop(['label'], axis=1, errors='ignore')
        self.log(f"DataFrame full merge has shape: {str(self.df_full_merged.shape)}")

        self.log(f"Loading dataframe fold number {str(wandb.config['fold_id'])}")
        with open(os.path.join(PATH_TMP_FOLDER_ARTIFACTS,self.artifact_full_name,"video","fold_" + str(wandb.config["fold_id"])+".json"), "r") as json_file:
            fold_json_data = json.load(json_file)
        self.df_fold_videos = pd.DataFrame(data=fold_json_data)
        self.log(f"DataDrame fold video  has shape: {str(self.df_fold_videos.shape)}")
        
        self.df_full_merged = pd.merge(self.df_full_merged, self.df_fold_videos, left_on="name_cvat", right_on="name",how="right")
        
        self.log(self.df_full_merged.columns)
        self.log(self.df_full_merged.shape)
        self.df_full_merged["label_old"] = self.df_full_merged["label"] # keep old label
        # ### WARNING: CLASS SWAPPING
        # 0: no_lines
        # 1: a_lines
        # 2: b_lines
        # for A_lines experiment -> b_lines change to 0
        # for B_lines experiment -> a_lins change to 0
        if wandb.config["experiment_class"] == "a":
            self.log("Expeirment type Alines. Labels was changed")
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 2, "label"] = 0
        elif wandb.config["experiment_class"] == "b":
            self.log("Expeirment type Blines. Labels was changed")
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 2, "label"] = 1
            self.df_full_merged.loc[self.df_full_merged["label_old"] == 1, "label"] = 0
            # self.df_full_merge.loc[self.df_full_merge["label_old"] == 0, "label"] = 0
        else:
            self.log("Wrong experiment type. Exit()")
            exit()

        ### SPLIT TRAIN / TEST
        ### ! WARNING THIS MUST BE IMPROVED AN USED DURING PREPARING DATASET FOR SPLITING ON EACH SUBSET!
        df_train = self.df_full_merged.loc[self.df_full_merged["subset"] == "train"]
        self.log("DATASET Per frame counts")
        self.log(f'TRAINT: {df_train.loc[df_train["label_old"]==0].shape[0]}, {df_train.loc[df_train["label_old"]==1].shape[0]}, {df_train.loc[df_train["label_old"]==2].shape[0]}')
   
        df_train, df_valid = train_test_split(df_train, test_size = 0.2, stratify=df_train["label_old"], random_state=wandb.config["random_seed"])
        self.log(f'TRAINT (after split), {df_train.loc[df_train["label_old"]==0].shape[0]}, {df_train.loc[df_train["label_old"]==1].shape[0]}, {df_train.loc[df_train["label_old"]==2].shape[0]}')
        self.log(f'VALID (after split), {df_valid.loc[df_valid["label_old"]==0].shape[0]}, {df_valid.loc[df_valid["label_old"]==1].shape[0]}, {df_valid.loc[df_valid["label_old"]==2].shape[0]}')
        df_test = self.df_full_merged.loc[self.df_full_merged["subset"] == "test"]
        self.log(f'TEST, {df_test.loc[df_test["label_old"]==0].shape[0]}, {df_test.loc[df_test["label_old"]==1].shape[0]}, {df_test.loc[df_test["label_old"]==2].shape[0]}')
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        # # #### PER FRAME need to implement later keep here for store code
        # wandb.config["subset_train_frames_count_0"] = df_train.loc[df_train["label"]==0].shape[0]
        # wandb.config["subset_train_frames_count_1"] = df_train.loc[df_train["label"]==1].shape[0]
        # wandb.config["subset_train_frames_count_2"] = df_train.loc[df_train["label"]==2].shape[0]
        
        # wandb.config["subset_valid_frames_count_0"] = df_valid.loc[df_valid["label"]==0].shape[0]
        # wandb.config["subset_valid_frames_count_1"] = df_valid.loc[df_valid["label"]==1].shape[0]
        # wandb.config["subset_valid_frames_count_2"] = df_valid.loc[df_valid["label"]==2].shape[0]

        # wandb.config["subset_test_frames_count_0"] = df_test.loc[df_test["label"]==0].shape[0]
        # wandb.config["subset_test_frames_count_1"] = df_test.loc[df_test["label"]==1].shape[0]
        # wandb.config["subset_test_frames_count_2"] = df_test.loc[df_test["label"]==2].shape[0]

