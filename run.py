import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from keras.utils import np_utils
from poker_NN_prototype import PokerNN

DATA_POINTS = 60000

data = pd.read_csv("game_data.csv")
dataset = data.values
data_mat = data.to_numpy()

class PreProcessor:

    def label_data(self):
        done = False
        i = 0
        while not done:
            try:
                if 'folds' in data_mat[i, 5]:
                    data_mat[i, 5] = 'Fold'
                if 'calls' in data_mat[i, 5]:
                    data_mat[i, 5] = 'Call'
                if 'bets' in data_mat[i, 5]:
                    data_mat[i, 5] = 'Bet'
                if 'checks' in data_mat[i, 5]:
                    data_mat[i, 5] = 'Check'
                if 'raises' in data_mat[i, 5]:
                    data_mat[i, 5] = 'Raise'
                if 'allin' in data_mat[i, 5]:
                    data_mat[i, 5] = 'Allin'
                i += 1
                if i == data_mat.shape[0]:
                    done = True
            except:
                data_mat = np.delete(data_mat, i, 0)

card_types = ['Ac', 'Ad', 'Ah', 'As',
              '2c', '2d', '2h', '2s', 
              '3c', '3d', '3h', '3s',
              '4c', '4d', '4h', '4s',
              '5c', '5d', '5h', '5s',
              '6c', '6d', '6h', '6s',
              '7c', '7d', '7h', '7s',
              '8c', '8d', '8h', '8s',
              '9c', '9d', '9h', '9s',
              'Tc', 'Td', 'Th', 'Ts',
              'Jc', 'Jd', 'Jh', 'Js',
              'Qc', 'Qd', 'Qh', 'Qs',
              'Kc', 'Kd', 'Kh', 'Ks']
stages = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']



def OneHotEncoder(categories, data):
    res = np.zeros(len(categories))
    for i in range(len(data)):
        res[categories.index(data[i])] = 1
    return res


limited_data = data_mat[0:DATA_POINTS]
labels = limited_data[:,5]
'''
temp = []

for i in range(DATA_POINTS):
    s = [data_mat[i, 0]]
    temp = np.concatenate((temp, OneHotEncoder(stages, s)))
    hole_cards = literal_eval(data_mat[i, 1])
    temp = np.concatenate((temp, OneHotEncoder(card_types, hole_cards)))

    if data_mat[i, 2] != '-1':
        comm_cards = literal_eval(data_mat[i, 2])
        temp = np.concatenate((temp, OneHotEncoder(card_types, comm_cards)))
    else:
        temp = np.concatenate((temp, np.zeros(52)))
    
    arr = [data_mat[i, 3], data_mat[i, 4]]
    temp = np.concatenate((temp, arr))
    print(i)

encoded_data = temp.reshape(DATA_POINTS, 110)

np.savetxt('encoded_data.csv', encoded_data, delimiter=",")
'''
dataFrame = pd.read_csv("encoded_data.csv", header=None)
encoded_data = dataFrame.values

encoder = LabelEncoder()
#fit label encoder
encoder.fit(labels)
print(encoder.classes_)
#transform labels into numerical representation
encodedLabels = encoder.transform(labels)
#transform to one-hot encoded binary matrix
binaryLabels = np_utils.to_categorical(encodedLabels)
inputFeatures = encoded_data.astype(float)
print(inputFeatures.shape, binaryLabels.shape)
nn = PokerNN((110, 80, 6), inputFeatures, binaryLabels, ('relu', 'softmax'))
nn.kFoldCrossValidation(200, 5)
