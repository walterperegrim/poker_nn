import pandas
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband, RandomSearch, BayesianOptimization


class NN_Tuner(HyperModel):

    def __init__(self, input_dim, output_dim):
        self.inputs = input_dim
        self.outputs = output_dim

    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(units=hp.Int('units', min_value=8, max_value=512, step=8, default=16), 
                  input_dim=self.inputs,
                  activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu')))
        layers=hp.Int('layers', min_value=1, max_value=5, step=1, default=2)
        for _ in range(layers):
            model.add(
                Dense(units=hp.Int('units', min_value=8, max_value=512, step=8, default=16),
                    activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu')))
        model.add(Dense(self.outputs, activation='softmax'))
        model.compile(
            optimizer=optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-5, 
                    max_value=0.1,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss=hp.Choice('loss', 
                      values=['categorical_crossentropy', 
                              'kullback_leibler_divergence'],
                      default='categorical_crossentropy'),
            metrics=['accuracy']
        )
        return model


def tune(epochs, x_train, y_train, x_test, y_test, tuner_type, input_dim):
    hypermodel = NN_Tuner(input_dim, 3)
    tuner = None
    if tuner_type == "hyperband":
        tuner = Hyperband(
            hypermodel, 
            max_epochs = 30,
            objective ='val_accuracy',
            seed = 1, 
            executions_per_trial=2,
            directory='F:\\poker-nn\\hyperband',
            project_name='poker-nn'
        )
    if tuner_type == "random":
            tuner = RandomSearch(
            hypermodel, 
            objective ='val_accuracy',
            seed = 1, 
            max_trials = 10000,
            executions_per_trial=2,
            directory='F:\\poker-nn\\random_search',
            project_name='poker-nn'
        )
    if tuner_type == "bayes":
            tuner = BayesianOptimization(
            hypermodel, 
            objective ='val_accuracy',
            seed = 1, 
            max_trials = 5000,
            executions_per_trial=2,
            directory='F:\\poker-nn\\bayesian_opt',
            project_name='poker-nn'
        )
    

    
    #tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=epochs, validation_split=0.1)
    #tuner.results_summary()
    best= tuner.get_best_models(num_models=1)[0]

    loss, accuracy = best.evaluate(x_test, y_test, batch_size=256)
    #print("Testing Loss: ", loss)
    #print("Testing Accuracy: ", accuracy)
    return loss, accuracy


