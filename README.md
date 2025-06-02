# Lung-Ultrasound-A-B-lines-classification

This repository contains a dataset and processing pipeline related to the benchmarking augmentation methods to enhance A-Line and B-Line classification with sparse data. The data is organized into three main folders:

A-lines – Videos containing clear A-line patterns
B-lines – Videos with visible B-line artifacts
No lines – Videos where no relevant line patterns are present

Each video in the dataset is accompanied by annotations exported from CVAT.

Pipeline
The full data processing and analysis pipeline is included in this repository. It handles:

Loading and preprocessing annotated data
Parsing CVAT exports
Preparing the dataset for training or evaluation
Evaluation of the results
