from monai.transforms import LoadImage
import monai.transforms as transforms
from monai.data import DataLoader, CacheDataset
import numpy as np
from datetime import datetime
from monai.transforms import LoadImage
import torch
from utils.UsgTransforms import UsgTransforms
# monai dataset + specific for APVV lung data from pandas dataframe
# --------------------------
# CVAT MONAI DATASET
# Specific for CVAT processing
# As input is used df_full_merged or frames_label_full.csv
# For classification is neccesary ["label"] -> GT
# Path from
# --------------------------


class CvatMonaiDatasetClassification(CacheDataset):
    def __init__(self, dataframe, server_path, resolution, device, transform_spatial_type, transforms_augumentation_type, transform_output_type, load_mask=False):
        self.data = dataframe

        # Convert data from indexic to list -> it is faster
        self.path_list = list(self.data["frame_cropped_path"])
        self.label_list = list(self.data["label"])
        self.resolution = resolution
        self.device = device
        self.server_path = server_path + "/"
        self.transform_spatial_type = transform_spatial_type
        self.transforms_augumentation_type = transforms_augumentation_type
        self.transform_output_type = transform_output_type
        self.load_mask = load_mask
        self.load_image = LoadImage(image_only=True)
        self.usg_augumentation_transform = UsgTransforms()
        
        # Helpers for trigers print only one times -> delete inf future
        

        # TRANSFORMS
        self.transforms_init = self.compose_init_transforms()
        self.transforms_spatial = self.composoe_spatial_transforms(self.transform_spatial_type)
        self.transforms_augumentation = self.compose_augumentation_transforms(self.transforms_augumentation_type)
        self.transforms_output = self.compose_output_transforms(self.transform_output_type)


        # print(len(self.transforms_augumentation))
        # exit()

    def __len__(self):
        return len(self.data)
    
    def log(self, payload):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp, "LOG from ",self.__class__.__name__, ":" ,payload)


    def compose_init_transforms(self):
        list_of_transforms_init = []

        # Neccesarry transform for  mono channel image
        list_of_transforms_init.append(transforms.EnsureChannelFirst())

        # Device chcek. If is cuda avaliable send data to cuda and make faster transformation on GPU
        if self.device == torch.device('cuda'):
            list_of_transforms_init.append(transforms.ToDevice(device=self.device))

        return list_of_transforms_init

    def composoe_spatial_transforms(self, transform_spatial_type):
        list_of_transofrms_spatial = []
        if transform_spatial_type == "padding_spatial":
            self.log(f"composoe_spatial_transforms -> padding_spatial")
            list_of_transofrms_spatial.append(transforms.SpatialPad(spatial_size=self.resolution, mode="replicate"))
        elif transform_spatial_type == "resize":
            self.log(f"composoe_spatial_transforms -> resize")
            list_of_transofrms_spatial.append(transforms.Resize(spatial_size=(self.resolution[0],self.resolution[1])))
        else:
            pass 
        return list_of_transofrms_spatial
        
    
    def compose_augumentation_transforms(self, transform_type):
        if transform_type == "a_t_default":
            augu = self.usg_augumentation_transform.a_t_default()
            self.log(f"Leng of augumetation pipeline is {len(augu)}")
            return augu
        elif transform_type == "a_t_without_intensity":
            augu = self.usg_augumentation_transform.a_t_without_intensity()
            self.log(f"Leng of augumetation pipeline is {len(augu)}")
            return augu
        elif transform_type == "a_t_without_noises":
            augu = self.usg_augumentation_transform.a_t_without_noises()
            self.log(f"Leng of augumetation pipeline is {len(augu)}")
            return augu
        elif transform_type == "a_t_only_basic":
            augu = self.usg_augumentation_transform.a_t_only_basic()
            self.log(f"Leng of augumetation pipeline is {len(augu)}")
            return augu
        elif transform_type == "empty_transforms":
            return self.usg_augumentation_transform.empty_transforms()
        elif transform_type == "crop_30":
            self.log("crop_30 augumentation but gordcored in gettintem after load image")
            return self.usg_augumentation_transform.a_t_default()
        else:
            
            self.log(f"DONT HAVE THIS TYPE OF AUGUMTENATION TRANSFORMATION ->  {transform_type}   <-. KILL PROGRAM")
            # exit()

    def compose_output_transforms(self, transform_output_type):  
        list_of_transformation_output = []
        list_of_transformation_output.append(transforms.ToTensor())
        
        if transform_output_type == "(0,1)":
            self.log("Output normalization (0,1)")
            list_of_transformation_output.append(transforms.ScaleIntensityRange(a_min=0,a_max=255,b_min=0,b_max=1))
        elif transform_output_type == "(-1,1)":
            self.log("Output normalization (-1,1)")
            list_of_transformation_output.append(transforms.ScaleIntensityRange(a_min=0,a_max=255,b_min=-1,b_max=1))
        elif transform_output_type == "zero_centering":
            self.log("DONT HAVE THIS TYPE OF OUPUT TRANSFORMATION. KILL PROGRAM")
            exit()
        else:
            self.log("DONT HAVE THIS TYPE OF OUPUT TRANSFORMATION. KILL PROGRAM")
            exit()
        return list_of_transformation_output

        

    def __getitem__(self, index):
        # Find exact item per index and load image
        img_path = self.server_path + self.path_list[index]
        label = self.label_list[index]
        
        # Check resolution of loadet image
        # UWAGA HARDCODET FOR FAST TRAIN
        # if self.transforms_augumentation_type == "crop_30":
        image = self.load_image(img_path)
        # self.log(image.shape)
        image = image[20:-10,30:-30]
        # self.log(image.shape)
        width, height = image.shape[0], image.shape[1]

        # ------------------------------------------------------
        # TRANSFORMS 
        # Continuos appending of functions
        list_of_transforms_per_frame = []

         # If is loader image higher as resolution of input to neural net / padding requirements
        if width > self.resolution[0]:
            if self.transform_spatial_type != "resize":
                image = image[20:-10,30:-30]
                list_of_transforms_per_frame.append(transforms.Resize(spatial_size=(self.resolution[0]-20,self.resolution[1]-20)))
            

        list_of_trans = self.transforms_init + list_of_transforms_per_frame + self.transforms_spatial + self.transforms_augumentation + self.transforms_output

        # print(list_of_trans)
        # Compose all transfromr together
        result_transform = transforms.Compose(list_of_trans)

        image = result_transform(image)
        # from 1channel iba stack same info to 3 channels (RGB for pretranied resnets)
        image = image.expand(3, -1, -1)
        # self.log(image.get_device())
        return image, label, 


class CvatMonaiDatasetClassificationWithMask(CvatMonaiDatasetClassification):
    def __init__(self, dataframe, server_path, resolution, device, transform_spatial_type, transforms_augumentation_type, transform_output_type, load_mask=False):
        super().__init__(dataframe, server_path, resolution, device, transform_spatial_type, transforms_augumentation_type, transform_output_type, load_mask)
        # self.mask_list = list(self.data["mask_cropped_path"])
        self.mask_list = list(self.data["frame_cropped_path"]) # temp fix
        self.name_cvat_list = list(self.data["name_cvat"])
        self.label_old_list = list(self.data["label_old"])
        self.prediction_list = list(self.data["predicted"])
        self.prediction_raw_list = list(self.data["predicted_raw"])

        # TRANSFORMS
        self.transforms_init = self.compose_init_transforms()
        self.transforms_spatial = self.composoe_spatial_transforms(self.transform_spatial_type)
        self.transforms_augumentation = self.compose_augumentation_transforms(self.transforms_augumentation_type)
        self.transforms_output = self.compose_output_transforms(self.transform_output_type)

    def __getitem__(self, index):
        super().__getitem__(index)
        # Find exact item per index and load image
        img_path = self.server_path + self.path_list[index]
        mask_path = self.server_path + self.path_list[index]
        label = self.label_list[index]

        image = self.load_image(img_path)
        image = image[20:-10,30:-30]
        mask = self.load_image(mask_path)
        

        # Check resolution of loadet image
        width, height = image.shape[0], image.shape[1]

       # ------------------------------------------------------
        # TRANSFORMS 
        # Continuos appending of functions
        list_of_transforms_per_frame = []

         # If is loader image higher as resolution of input to neural net / padding requirements
        if width > self.resolution[0]:
            if self.transform_spatial_type != "resize":
                image = image[20:-10,30:-30]
                list_of_transforms_per_frame.append(transforms.Resize(spatial_size=(self.resolution[0]-20,self.resolution[1]-20)))
            

        list_of_trans = self.transforms_init + list_of_transforms_per_frame + self.transforms_spatial + self.transforms_augumentation + self.transforms_output

        # print(list_of_trans)
        # Compose all transfromr together
        result_transform = transforms.Compose(list_of_trans)

        image = result_transform(image)
        # from 1channel iba stack same info to 3 channels (RGB for pretranied resnets)
        image = image.expand(3, -1, -1)
        # self.log(image.get_device())


        return {
            'image': image,
            'label': label,
            'mask': mask,
            'name_cvat': self.name_cvat_list[index],
            'label_old': self.label_old_list[index],
            'prediction': self.prediction_list[index],
            'prediction_raw': self.prediction_raw_list[index]
        }