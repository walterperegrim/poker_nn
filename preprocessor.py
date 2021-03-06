import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from poker_NN_prototype import PokerNN
from sklearn.model_selection import train_test_split


#number of data points to pull from data set
DATA_POINTS = 63235

class PreProcessor():

    def __init__(self):
        self.data = pd.read_csv("game_data.csv")
        self.dataset = self.data.values
        self.data_mat = self.data.to_numpy()
        self.limited_data = self.data_mat[0:DATA_POINTS]

    def labelEncoder(self):
        done = False
        i = 0
        while not done:
            try:
                if 'folds' in self.data_mat[i, 5]:
                    self.data_mat[i, 5] = 'Fold'
                if 'calls' in self.data_mat[i, 5]:
                    self.data_mat[i, 5] = 'Bet'
                if 'bets' in self.data_mat[i, 5]:
                    self.data_mat[i, 5] = 'Bet'
                if 'checks' in self.data_mat[i, 5]:
                    self.data_mat[i, 5] = 'Check'
                if 'raises' in self.data_mat[i, 5]:
                    self.data_mat[i, 5] = 'Bet'
                if 'allin' in self.data_mat[i, 5]:
                    self.data_mat[i, 5] = 'Bet'
                i += 1
                if i == self.data_mat.shape[0]:
                    done = True
            except:
                self.data_mat = np.delete(self.data_mat, i, 0)
        limited_data = self.data_mat[0:DATA_POINTS]
        return limited_data[:,5]
    
    def encode_datapoint(self, stage, hole_cards, comm_cards, hand_strength, hand_potential):
        card_types = [['A','c'], ['A','d'], ['A','h'],['A', 's'],
                      ['2','c'], ['2','d'], ['2','h'],['2', 's'],
                      ['3','c'], ['3','d'], ['3','h'],['3', 's'],
                      ['4','c'], ['4','d'], ['4','h'],['4', 's'],
                      ['5','c'], ['5','d'], ['5','h'],['5', 's'],
                      ['6','c'], ['6','d'], ['6','h'],['6', 's'], 
                      ['7','c'], ['7','d'], ['7','h'],['7', 's'],
                      ['8','c'], ['8','d'], ['8','h'],['8', 's'],
                      ['9','c'], ['9','d'], ['9','h'],['9', 's'],
                      ['T','c'], ['T','d'], ['T','h'],['T', 's'],
                      ['J','c'], ['J','d'], ['J','h'],['J', 's'],
                      ['Q','c'], ['Q','d'], ['Q','h'],['Q', 's'],
                      ['K','c'], ['K','d'], ['K','h'],['K', 's']] 
        stages = ['hole', 'flop', 'turn', 'river']
        temp = []
        stage_enc = np.zeros(4)
        stage_enc[stages.index(stage)] = 1
        temp = np.concatenate((temp, stage_enc))
        temp = np.concatenate((temp, self.OneHotEncoder(card_types, hole_cards)))
    
        if not comm_cards:
            temp = np.concatenate((temp, np.zeros(52)))
        else:
            temp = np.concatenate((temp, self.OneHotEncoder(card_types, comm_cards)))

        hand_stats = [hand_potential, hand_strength]
        temp = np.concatenate((temp, hand_stats))
        return temp

    def OneHotEncoder(self, categories, data):
        res = np.zeros(len(categories))
        for i in range(len(data)):
            res[categories.index(data[i])] = 1
        return res

    def featureEncoder(self):
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


        temp = []
        for i in range(DATA_POINTS):
            s = [self.data_mat[i, 0]]
            whole_hand = np.zeros(52)
            temp = np.concatenate((temp, self.OneHotEncoder(stages, s)))
            hole_cards = literal_eval(self.data_mat[i, 1])
            encoded_hole_cards = self.OneHotEncoder(card_types, hole_cards)
            temp = np.concatenate((temp, encoded_hole_cards))
            whole_hand += encoded_hole_cards
            if self.data_mat[i, 2] != '-1':
                comm_cards = literal_eval(self.data_mat[i, 2])
                encoded_comm_cards = self.OneHotEncoder(card_types, comm_cards)
                temp = np.concatenate((temp, encoded_comm_cards))
                whole_hand += encoded_comm_cards
            else:
                temp = np.concatenate((temp, np.zeros(52)))
            temp = np.concatenate((temp, whole_hand))
            arr = [self.data_mat[i, 3], self.data_mat[i, 4]]
            temp = np.concatenate((temp, arr))
            print(i)
        encoded_data = temp.reshape(DATA_POINTS, 162)
        np.savetxt('F:\\poker-nn\\encoded_data.csv', encoded_data, delimiter=",")
    def Ordinal(self):
        X = self.dataset[0:60000, :-1]
        #y = self.dataset[:,-1]
        #y = y.reshape((len(y), 1))
        oe = OrdinalEncoder()
        oe.fit(X)
        x_enc = oe.transform(X)
        return x_enc

#proc = PreProcessor()
#proc.featureEncoder()