# Benchmarking Augmentation to Enhance A-Line and B-Line Classification with Sparse Data

This repository contains a dataset and processing pipeline for benchmarking augmentation methods to enhance **A-line** and **B-line** classification under sparse data conditions. The data is annotated using [CVAT](https://www.cvat.ai/) and organized into three main folders:

- **A-lines** – Videos containing clear A-line patterns  
- **B-lines** – Videos with visible B-line artifacts  
- **No lines** – Videos where no relevant line patterns are present  

Each video is accompanied by corresponding CVAT-exported annotations.

##  Pipeline

The full data processing and analysis pipeline is included in this repository. It handles:

-  Loading and preprocessing annotated data  
-  Parsing CVAT exports  
-  Preparing the dataset for training or evaluation  
-  Evaluation of the classification results
