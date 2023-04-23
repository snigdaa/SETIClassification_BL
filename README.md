# Classification of radio signals of the galactic center as potential radiosignatures

### Authors: Snigdaa S Sethuram, Bin Xia

#### This project was created as part of the final project for the graduate-level Computational Physics course taught by Dr. John Wise, Spring 2023

All data used for training and testing of the final convolutional neural network (CNN) was obtained from the Berkeley SETI Research Center's Kaggle data release [1](https://www.kaggle.com/competitions/seti-breakthrough-listen/data). The data is not included in this repository.

### If you would like to train and test your own model based on this architecture, please download the data from source and follow these steps:

* Run `preprocess.py` on the source data to obtain processed data files.

* update `fin` path in `ModelSetup/main.py` and `ModelSetup/test.py` to reflect where you have stored your processed files

* run `python3 ModelSetup/main.py` to create a trained model

* run `python3 ModelSetup/test.py` to test your created model

### If you are using the pre-trained model available in `model_seti.pt`, simply use the model as is in `ModelSetup/test.py` after updating the desired test data if necessary
