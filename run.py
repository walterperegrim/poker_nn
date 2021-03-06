import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from poker_NN_prototype import PokerNN
from sklearn.model_selection import train_test_split
import sys
import time
from nn_tuner import tune
from preprocessor import PreProcessor


NUM_FEATURES = 110
DATA_POINTS = 60000
FILENAME = "encoded_data.csv"
data = pd.read_csv("game_data.csv")
data_mat = data.to_numpy()
limited_data = data_mat[0:DATA_POINTS]
labels = limited_data[:,5]
dataFrame = pd.read_csv(FILENAME, header=None)
encoded_data = dataFrame.values
encoder = LabelEncoder()
encoder.fit(labels)     #fit label encoder
encodedLabels = encoder.transform(labels)       #transform labels into numerical representation
binaryLabels = np_utils.to_categorical(encodedLabels)   #transform to one-hot encoded binary matrix
inputFeatures = encoded_data.astype(float)

def run():
    nn = PokerNN((NUM_FEATURES, 16, 3), inputFeatures, binaryLabels, ('relu', 'softmax'))
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, binaryLabels, test_size=0.33)
    start_time = time.time()
    t_score = nn.eval(20,256,X_train, X_test, y_train, y_test, False)
    elapsed = time.time() - start_time
    return t_score, elapsed
    #nn.kFoldCrossValidation(200, 5)

def run_vary_epochs():
    nn = PokerNN((NUM_FEATURES, 16, 3), inputFeatures, binaryLabels, ('relu', 'softmax'))
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, binaryLabels, test_size=0.33)
    start_time = time.time()
    for i in range(1, 100):
        print("Epochs: ", i)
        t_score = nn.eval(i,256,X_train, X_test, y_train, y_test, False)
        print(t_score, time.time() - start_time)

def run_and_tune(tuner_type):
    x_train, x_test, y_train, y_test = train_test_split(inputFeatures, binaryLabels, test_size=0.33)
    return tune(2, x_train, y_train, x_test, y_test, tuner_type, NUM_FEATURES)

def compare():
    res1 = run()
    print("Before tuning:  ", res1)
    res2 = run_and_tune("random")
    print("After tuning:   ", res2)

def train_on_simulation():
    df = pd.read_csv("simulated_game_data.csv")
    df = pd.get_dummies(data=df, columns=['stage', 'action','play_style'])
    features = df.drop(['hand_quality'], axis=1)
    labels = df['hand_quality']
    nn = PokerNN((13, 16, 3), features, labels, ('relu', 'softmax'))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)
    t_score = nn.opponent_modeling(20,256,X_train, X_test, y_train, y_test)
    print(t_score)

def run_ordinal():
    nn = PokerNN((NUM_FEATURES, 16, 3), inputFeatures, binaryLabels, ('relu', 'softmax'))
    pre = PreProcessor()
    print("Encoding Ordinally......")
    inputs = pre.Ordinal()
    x_train, x_test, y_train, y_test = train_test_split(inputs, binaryLabels, test_size=0.33)
    tune(5, x_train, y_train, x_test, y_test, "random", NUM_FEATURES)

#train_on_simulation()
#score, t = run()
#print(score,t)
run()
#run_vary_epochs()
#run_and_tune("random")
#run_and_tune("hyperband")
#run_and_tune("bayes")
#compare()
#run_ordinal()
