# apr26_bds_int_anomaly

# MVTec AD Anomaly Detection and Classification

This project contains models for anomaly detection of the MVTec AD dataset. The main pipeline will detect the object category, will further decide if a given image shows an object with a anomaly compared to good samples of the same object and in the last step classifies the specific defect type for that object.

## Features

- SVM object router for object category clasification
- PatchCore model anomaly detection for each object category
- convolutional auto encoder (CAE) for anomaly detection
- object specific SVM defect classifier for the classification of diffrent defect types

## Installation

``` bash
git clone https://github.com/DataScientest-Studio/apr26_bds_int_anomaly
cd apr26_bds_int_anomaly
pip install -r requirements.txt
```

## How to use the pipeline

For now, the way to go is to use and config the mvtec_ad_pipeline.py file.

Open the file and change the global variable DATASET_ROOT_PATH to the path where your MVTec AD dataset is located. The dataset should be stored in a folder structure like ".../MVTec_AD/archive" on your machine, containing all the subfolders for the diffrent  object categories.

Change the IMAGE_PATH variable to a specific image of the dataset.

Running the script will then download all the models if needed, checks for the object type via the SVM object router classifier, loads the object specific PatchCore model and detects if a anomaly is present on that image. If a anomaly was detected it will load the defect classifier and will predict the defect class of that image.

## Project Structure

apr26_bds_int_anomaly/
|- Data Audit/ # contains the descriptiv analysis of the dataset
|- metrics/ # contains the metrics for the defect classifiers for each object type
|- modesl/ # contains the weights for the object router and defect classifier
|- scripts.py # all the scripts used to train and test the models
|- README.md
|- LICENSE
|- requirements.txt 

## Environment

- Python 3.12.3

## License

MIT License

## Contribution

Ayoub Hamdoun
Christopher Jager
Mohamed B.
Romeal Nansi