import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


#simple 4 layer configurable neural network
class PokerNN:

    def __init__(self, structure, features, labels, activations):
        self.inputNodes = structure[0]
        self.hiddenNodes = structure[1]
        self.outputNodes = structure[2]
        self.hiddenActivation = activations[0]
        self.outputActivation = activations[1]
        self.features = features
        self.labels = labels


    def kFoldCrossValidation(self, epochs, batch_size):
        def model():
            model = Sequential()
            model.add(Dense(self.hiddenNodes, input_dim=self.inputNodes, activation=self.hiddenActivation))
            model.add(Dense(self.hiddenNodes, activation=self.hiddenActivation))
            model.add(Dense(self.hiddenNodes, activation=self.hiddenActivation))
            model.add(Dense(self.outputNodes, activation=self.outputActivation))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        estimator = KerasClassifier(build_fn=model,epochs=epochs, batch_size=batch_size, verbose=0)
        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, self.features, self.labels, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    def eval(self, epochs, batch_size, X_train, X_test, y_train, y_test):
        labels = ['Bet', 'Check', 'Fold']
        model = Sequential()
        model.add(Dense(self.hiddenNodes, input_dim=self.inputNodes, activation='tanh'))
        model.add(Dense(self.hiddenNodes, activation=self.hiddenActivation))
        model.add(Dense(self.hiddenNodes, activation=self.hiddenActivation))
        model.add(Dense(self.outputNodes, activation=self.outputActivation))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose=0)
        test_score = model.evaluate(X_test, y_test,batch_size=batch_size)
        #Converts call,raise to bet
        #yhats = [list(x) for x in model.predict(X_test,batch_size=batch_size)]
        #new_hats = [labels[np.argmax(yhat)] for yhat in yhats]
        #new_test = [labels[np.argmax(y)] for y in y_test]
        #new_hats = ['Bet' if h == 'Call' or h == 'Raise' else h for h in new_hats]
        #new_test = ['Bet' if h == 'Call' or h == 'Raise' else h for h in new_test]
        #x = np.sum([1 for i in range(len(new_test)) if new_test[i] == new_hats[i]]) / len(new_test)
        #print(test_score,x)
        return test_score


'''
#######testing with iris flower data set, another three-class problem. will be removed later #######
dataFrame = pandas.read_csv("iris.data", header=None)
dataset = dataFrame.values
inputFeatures = dataset[:,0:4].astype(float)
labels = dataset[:,4]
encoder = LabelEncoder()
#fit label encoder
encoder.fit(labels)
#transform labels into numerical representation
encodedLabels = encoder.transform(labels)
#transform to one-hot encoded binary matrix
print(labels, labels.shape)
binaryLabels = np_utils.to_categorical(encodedLabels)

print(binaryLabels.shape, inputFeatures.shape)
nn = PokerNN((4, 8, 3), inputFeatures, binaryLabels, ('relu', 'softmax'))
nn.kFoldCrossValidation(200, 5)
'''
