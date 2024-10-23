import torch
import configparser
from pathlib import Path
import os
from monai.data import DataLoader, CacheDataset
from sklearn import metrics
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import monai
import time 
from datetime import datetime
import numpy as np

from utils.CvatMonaiDatasetClassification import CvatMonaiDatasetClassificationWithMask

class MetricsResult:
    def __init__(self, accuracy, balanced_accuracy, precision, recall, f1_score, support, confusion_matrix):
        self.accuracy = accuracy
        self.balanced_accuracy = balanced_accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        if support is None:
            self.support = -999
        else:
            self.support = support
        self.confusion_matrix = confusion_matrix


# Load config values
config = configparser.ConfigParser()
config.read('./configs/global.cfg')

PATH_TMP_FOLDER_OUTPUT = config.get('Global', 'path_temp_folder_output')
PROJECT_NAME = config.get('Global', 'project_name')
class Evaluation():
    def __init__(self, model, dataset_loader, loader, device, run_name, model_type):
        self.model = model
        self.dataset_loader = dataset_loader
        self.test_loader = loader
        self.device = device
        self.df_test =  self.dataset_loader.df_test
        self.run_name = run_name
        self.model_type = model_type
        self.df_test["predicted"] = 0
        self.df_test["predicted_raw"] = 0

        self.videos_result_metrics = []

        # Create directories
        Path(PATH_TMP_FOLDER_OUTPUT).mkdir(parents=True, exist_ok=True)
        Path(PATH_TMP_FOLDER_OUTPUT+"/"+PROJECT_NAME).mkdir(parents=True, exist_ok=True)
        self.path_project_output = os.path.join(PATH_TMP_FOLDER_OUTPUT, PROJECT_NAME, self.run_name)
        Path(self.path_project_output).mkdir(parents=True, exist_ok=True)
 
    
    def log(self, payload):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp, "LOG from ",self.__class__.__name__, ":" ,payload)
    
    def model_eval(self, threshold):
        self.log("Model evaluationg started. Return common fast statistics.")
        self.log("Test dataframe will be stored with predictions.")
        self.model.eval()
        test_acc = 0
        test_output = []
        test_binnary = []

        # Model evalutaion pipeline
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.model_type == "inception-v3":
                    outputs = self.model(images)
                else:
                    outputs = self.model(images)
                # Apply thresholding
                predicted_probs = torch.softmax(outputs, dim=1)
                predicted_class = (predicted_probs[:, 1] > threshold).long()  # Change the index (1) based on your classes
                test_binnary.append(int(predicted_class))
                test_output.extend(predicted_probs[:, 1].cpu().numpy())

                total += labels.size(0)
                correct += (predicted_class == labels).sum().item()

            accuracy = (correct / total) * 100
            test_acc = accuracy
            self.log(f'Test Accuracy: {accuracy:.2f}%')
            print(test_binnary)
            self.df_test["predicted"] = test_binnary
            self.df_test["predicted_raw"] = test_output

            self.log("Test dataframe stored.")
            self.df_test.to_csv(os.path.join(self.path_project_output, "df_test.csv"), index=False)
            return self.df_test

    def calculate_eval_metric_per_video(self, mode):
        # ----------------------------------------------------------------
        # Calculate results for all test frame 
        # TO DO: add curvers
        # ----------------------------------------------------------------
        accuracy = metrics.accuracy_score(self.df_test["label"], self.df_test["predicted"])
        ba = metrics.balanced_accuracy_score(self.df_test["label"], self.df_test["predicted"])
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(self.df_test["label"], self.df_test["predicted"], average="macro")
        confusion_matrix = metrics.confusion_matrix(self.df_test["label"], self.df_test["predicted"])
        self.log("Accuracy")
        self.log(accuracy)
        self.log("Ballanced accuracy")
        self.log(ba)
        self.log("Precision")
        self.log(precision)
        self.log("Recall")
        self.log(recall)
        self.log("F1_score")
        self.log(f1_score)
        self.log("Support")
        self.log(support)
        self.log("Confusion matrix")
        self.log(confusion_matrix)


        self.log("BINNARY ------------------")
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(self.df_test["label"], self.df_test["predicted"], average="binary")
        self.log("Accuracy")
        self.log(accuracy)
        self.log("Ballanced accuracy")
        self.log(ba)
        self.log("Precision")
        self.log(precision)
        self.log("Recall")
        self.log(recall)
        self.log("F1_score")
        self.log(f1_score)
        self.log("Support")
        self.log(support)
        self.log("Confusion matrix")
        self.log(confusion_matrix)
        self.log("MICRO ------------------")
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(self.df_test["label"], self.df_test["predicted"], average="micro")
        self.log("Accuracy")
        self.log(accuracy)
        self.log("Ballanced accuracy")
        self.log(ba)
        self.log("Precision")
        self.log(precision)
        self.log("Recall")
        self.log(recall)
        self.log("F1_score")
        self.log(f1_score)
        self.log("Support")
        self.log(support)
        self.log("Confusion matrix")
        self.log(confusion_matrix)
        self.log("MACRO ------------------")
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(self.df_test["label"], self.df_test["predicted"], average="macro")
        self.log("Accuracy")
        self.log(accuracy)
        self.log("Ballanced accuracy")
        self.log(ba)
        self.log("Precision")
        self.log(precision)
        self.log("Recall")
        self.log(recall)
        self.log("F1_score")
        self.log(f1_score)
        self.log("Support")
        self.log(support)
        self.log("Confusion matrix")
        self.log(confusion_matrix)
        self.log("WEIGHTED ------------------")
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(self.df_test["label"], self.df_test["predicted"], average="weighted")
        self.log("Accuracy")
        self.log(accuracy)
        self.log("Ballanced accuracy")
        self.log(ba)
        self.log("Precision")
        self.log(precision)
        self.log("Recall")
        self.log(recall)
        self.log("F1_score")
        self.log(f1_score)
        self.log("Support")
        self.log(support)
        self.log("Confusion matrix")
        self.log(confusion_matrix)

        self.log("Stored metrics is calculated via {mode} mode")
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(self.df_test["label"], self.df_test["predicted"], average=mode)
        self.per_frame_result = MetricsResult(accuracy, ba, precision, recall, f1_score, support, confusion_matrix)
        self.per_frame_class_report = metrics.classification_report(self.df_test["label"], self.df_test["predicted"])
        return self.per_frame_result
    
    def calculate_eval_metric_per_frame(self, wandb, mode, video_frame_votes_threshold, store_output=True):
        # ----------------------------------------------------------------
        # per vide statistics
        # per video lookup frames

        # get video coints
        video_names_list = set(self.df_test["name_cvat"])
        self.log(f"Uniq videos for test {len(video_names_list)}")

        video_accuracy_list = []
        video_ba_accuracy_list = []
        video_precision_list = []
        video_recall_list = []
        # Iterate over all videos names
        for video_name in video_names_list:
            # Get lookup frames only for specific video
            df_tmp = self.df_test.loc[self.df_test["name_cvat"] == video_name]
            self.log(f"Video {video_name}")
            self.log(f"Video frame count{df_tmp.shape[0]}")

            # Calc metrics for video
            accuracy = metrics.accuracy_score(df_tmp["label"], df_tmp["predicted"])
            ba = metrics.balanced_accuracy_score(df_tmp["label"], df_tmp["predicted"])
            precision, recall, f1_score, support = metrics.precision_recall_fscore_support(df_tmp["label"], df_tmp["predicted"], average=mode)
            confusion_matrix = metrics.confusion_matrix(df_tmp["label"], df_tmp["predicted"])
            metrics_object = MetricsResult(accuracy, ba, precision, recall, f1_score, support, confusion_matrix)
            print(metrics_object.__dict__)
            self.videos_result_metrics.append(metrics_object)

            video_accuracy_list.append(accuracy)
            video_ba_accuracy_list.append(ba)
            video_precision_list.append(precision)
            video_recall_list.append(recall)
            
            
            if store_output:
                # Load Image and mask , create subbplot and plot
                # Future add saliancy maps
                local_datafset = CvatMonaiDatasetClassificationWithMask(dataframe=df_tmp, 
                                                    server_path=self.dataset_loader.path_cvat_folder, 
                                                    resolution = tuple(map(int,wandb.config["resolution"].split(','))), 
                                                    device=self.device, 
                                                    transform_spatial_type=wandb.config["spatial_transforms"],
                                                    transforms_augumentation_type="empty_transforms", 
                                                    transform_output_type=wandb.config["normalise_intensity"],
                                                    load_mask=True)
                local_loader = DataLoader(local_datafset, num_workers=0, batch_size=1, shuffle=False, collate_fn=monai.data.list_data_collate)

                # os.mkdir(os.path.join(self.path_project_output, video_name), exist=True)
                Path(os.path.join(self.path_project_output, video_name)).mkdir(parents=True, exist_ok=True)
                for return_object in local_loader:
                    image = return_object["image"].to(self.device)
                    labels = return_object["image"].to(self.device)
                    
                    # Get saliency maps
                    image.requires_grad_()
                    output = self.model(image)
                    # Catch the output
                    output_idx = output.argmax()
                    output_max = output[0, output_idx]
                    # Do backpropagation to get the derivative of the output based on the image
                    output_max.backward()
                    # Retireve the saliency map and also pick the maximum value from channels on each pixel.
                    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
                    saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
                    # saliency = saliency.reshape(resolution)
                    sa_map = saliency.cpu().detach().numpy().T

                    # Create a 1x2 subplot layout
                    plt.subplots(figsize=(30,15))
                    plt.title(str(return_object["label"]))
                    plt.suptitle("Label old {} - Label {} - Predicted {} / Predicted raw {}".format(str(return_object["label_old"][0]), str(return_object["label"][0]), str(return_object["prediction"][0]), str(return_object["prediction_raw"][0])), fontsize=20)
                    plt.subplot(1, 3, 1)
                    img_for_plot = image.cpu().detach().numpy()[0]
                    # img_for_plot = img_for_plot.astype(np.uint8)
                    plt.imshow(img_for_plot[0].T,cmap="gray")  # Use 'cmap' argument for grayscale images
                    plt.title('Image')
                    
                    plt.subplot(1, 3, 2)
                    # plt.imshow(mask.T, cmap="gray")  # Use 'cmap' argument for grayscale images
                    plt.imshow(img_for_plot[0].T,cmap="gray")
                    plt.title('Mask')

                    plt.subplot(1, 3, 3)
                    # print(sal_map(model, sm_transform, resolution, img_path).shape)
                    # sm = np.resize(sal_map(model, sm_transform, resolution, img_path),image.cpu().detach().numpy().T.shape)
                    plt.imshow(sa_map, cmap='hot')  # Use 'cmap' argument for grayscale images
                    plt.title('Saliency Map')

                    
                    plt.savefig(os.path.join(self.path_project_output, video_name, f"{str(return_object['name_cvat'][0])}_{str(return_object['prediction_raw'][0])}.png"))

        video_accuracy_list = np.array([video_accuracy_list])
        video_ba_accuracy_list = np.array([video_ba_accuracy_list])
        video_precision_list = np.array([video_precision_list])
        video_recall_list = np.array([video_recall_list])

        return np.mean(video_accuracy_list), np.mean(video_ba_accuracy_list), np.mean(video_precision_list), np.mean(video_recall_list)
