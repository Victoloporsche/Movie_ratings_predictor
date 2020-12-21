# Movie_ratings_predictor

This repository provides a Pythonic implementation of Movie Ratings prediction by utilizing Python concept of OOP. This repository is is refactored to fit any classification
based problem and consists of the Machine Learning Pipelines (Data Exploratory, Feature Engineering, Feature Selection, Cross Validation, Hyperparameter Optimization, 
Model training and prediction)

# Installation

This implementation is written with Python version 3.6 with the listed packages in the requirements.txt file

Clone this repository with git clone https://github.com/Victoloporsche/Movie_ratings_predictor.git
With Virtual Environent, use : a) pip install virtualenv b) cd path-to-the-cloned-repository c) virtualenv myenv d) source myenv/bin/activate e) pip install -r requirements.txt
With Conda Environment, use: a) cd path-to-the-cloned-repository b) conda create --name myenv c) source activate myenv d) pip install -r requirements.txt

# Implementation Folder:

 1) The Input files consists of the ratings, movie and cast datasts which require cleaning. The dataset can be found here: https://www.kaggle.com/rounakbanik/the-movies-dataset
 2) Model folder consists of the trained model 
 3) Output folder consists of the cleaned input data as well as the encoded categorical features 
 4) src folder consists of the python and jupyter files
 
 The order of running this repository is:
data_cleaning.ipynb: This file cleans the input files and outputs a cleaned data which is used for the model
data_exploration.py : This classs provides a detailed information about the dataset
feature_engineering.py: This class performs feature engineering techniques on the data
feature_selection.py: This class selects the n-best features for the model
preprocessed_data.py: This class combines step 3 and 4
models.py: This class performs cross validation, hyperparameter optimization and model training as well as prediction
main.py: This class combines all the previous classes
example_movie_ratings_predictor.ipynb: This provides the jupyter documentation of the model.
Next Step: Deploying this model as a web based application for automated movie ratings prediction

More modifications and commits would be made to this repository from time to time. Kindly reach out if you have any questions or improvements.
