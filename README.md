# Deep-FeedForward-Neural-Network

## Description
This project features, data & feature engineering, data preprocessing, and a sequential neural network. 
Credit to the book Neural Network Projects with Python by James Loy. The file main.py contains all the functions
in the order executed. The data engineering and model structure can be viewed in main.py. The model predicts
the price of taxi fares in New York City based on 17 data points, with an error of ~$3.50. 

## Installation
* Pip install tensorflow (built with 2.5.0)
* Pip install numpy (built with 1.19.5)
### Dataset
1. Go to https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data
2. Click on the 'Download All' button
3. After the download is complete, unzip the zip file and move the file 'train.csv' into your project folder.
4. Rename the file 'train.csv' as 'NYC_taxi.csv'

## Usage
Running the main.py will load the data, engineer the features in such data, preprocess the data and then 
train the model on the data. The model is then tested and measured with RMSE.  

To get better usage out of this project the functions should be read and understood in the order executed. 
Alongside the book Neural Network Projects with Python by James Loy. 

## Neural Network Details
Input layer, units=17  
Hidden layer 1, units=128, activation=relu  
Hidden layer 2, units=64, activation=relu  
Hidden layer 3, units=32, activation=relu  
Hidden layer 4, units=8, activation=relu  
Output layer, units=1, activation=relu  
Optimizer Adam, Loss MSE

Testing: ~$3.50 error range

## Credits
* Author: James Loy
* Modified & Studied by: Lee Taylor

## Note
I found this project interesting to learn as not only did it feature a deep neural network, but also 
gave me insight on how to perform data engineering effectively for future projects. 
