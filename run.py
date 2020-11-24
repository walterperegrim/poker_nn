import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from poker_NN_prototype import PokerNN
from sklearn.model_selection import train_test_split
import sys
import time


def run():
    data = pd.read_csv("game_data.csv")
    data_mat = data.to_numpy()
    limited_data = data_mat[0:60000]
    labels = limited_data[:,5]
    dataFrame = pd.read_csv("encoded_data.csv", header=None)
    encoded_data = dataFrame.values
    encoder = LabelEncoder()
    encoder.fit(labels)     #fit label encoder
    print(encoder.classes_)
    encodedLabels = encoder.transform(labels)       #transform labels into numerical representation
    binaryLabels = np_utils.to_categorical(encodedLabels)   #transform to one-hot encoded binary matrix
    inputFeatures = encoded_data.astype(float)
    print(inputFeatures.shape, binaryLabels.shape)
    nn = PokerNN((110, 16, 3), inputFeatures, binaryLabels, ('relu', 'softmax'))
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, binaryLabels, test_size=0.33)
    start_time = time.time()
    t_score = nn.eval(20,256,X_train, X_test, y_train, y_test)
    print(t_score, time.time() - start_time)
    #nn.kFoldCrossValidation(200, 5)

run()