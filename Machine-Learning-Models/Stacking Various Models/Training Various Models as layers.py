from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Stack Net for classification models


class SNKFC:

    def __init__(self, models, k, s):
        self.models = models
        self.k = k
        self.s = s

    def fit(self, data, target):
        kf = KFold(n_splits=self.k)
    #    n1 = len(self.models)
        L = 0
        for model in self.models:
            n = len(model)
            i = 0
            layer_accuracy = []
            pred = pd.DataFrame()
            for train_index, val_index in kf.split(data):
                X_train, X_val = data.loc[train_index], data.loc[val_index]
                y_train, y_val = target[train_index], target[val_index]
                c = 0
                while c < n:
                    model[c].fit(X_train, y_train)
                    pred[i] = model[c].predict(X_val)
                    c = c+1
                    i = i+1
            for t in np.arange(kf.n_splits):
                if t == 0:
                    X = pred[np.arange(0, n)]
                else:
                    Y = pred[np.arange(t*n, (1+t)*n)].set_index(np.arange(
                        t * int(data.shape[0]/kf.n_splits),
                        (t+1)*int(data.shape[0]/kf.n_splits)
                    ))
                    for g in np.arange(n):
                        Y[g] = Y[(n*t)+g]
                        Y = Y.drop(n*t+g, axis=1)
                    X = pd.concat([X, Y])
                    Y = Y.drop(np.arange(0, n), axis=1)
            for v in np.arange(n):
                layer_accuracy.append(accuracy_score(target, X[v]))
                print(
                    f'''\n Accuracy Score of Layer-{L}
                            Model-{v}= {layer_accuracy[v]}'''
                )
                print('--------------------------------------------------\n')
            if self.s[L] == 1:
                data = pd.concat([data, X], axis=1)
            elif self.s[L] == 0:
                data = X.copy()
            L += 1
            print(f'''\nLayer{L} Starts Here:
                    --{L}--{L}--{L}--{L}--{L}--{L}--{L}--{L}--{L}--- \n''')

    def predict(self, data):
        pred = pd.DataFrame()
        for model in self.models:
            for n in np.arange(len(model)):
                pred[n] = model[n].predict(data)
            data = pred.copy()
            pred = pd.DataFrame()
        return data


# stack net for regression models


class SNKFR:

    def __init__(self, models, k, s):
        self.models = models
        self.k = k
        self.s = s

    def fit(self, data, target):
        kf = KFold(n_splits=self.k)
        # n1 = len(self.models)
        L = 0
        for model in self.models:
            n = len(model)
            i = 0
            rmse = []
            pred = pd.DataFrame()
            for train_index, val_index in kf.split(data):
                X_train, X_val = data.loc[train_index], data.loc[val_index]
                y_train, y_val = target[train_index], target[val_index]
                c = 0
                while c < n:
                    model[c].fit(X_train, y_train)
                    pred[i] = model[c].predict(X_val)
                    c = c+1
                    i = i+1
            for t in np.arange(kf.n_splits):
                if t == 0:
                    X = pred[np.arange(0, n)]
                else:
                    Y = pred[np.arange(t*n, (1+t)*n)].set_index(np.arange(
                        t * int(data.shape[0]/kf.n_splits),
                        (t+1)*int(data.shape[0]/kf.n_splits)
                    ))
                    for g in np.arange(n):
                        Y[g] = Y[(n*t)+g]
                        Y = Y.drop(n*t+g, axis=1)
                    X = pd.concat([X, Y])
                    Y = Y.drop(np.arange(0, n), axis=1)
            for v in np.arange(n):
                rmse.append(np.sqrt(mean_squared_error(target, X[v])))
                print(f'\n RMS Error of Layer-{L} Model-{v} = {rmse[v]}')
                print('--------------------------------------------------\n')
            if L < len(self.s):
                if self.s[L] == 1:
                    data = pd.concat([data, X], axis=1)
                elif self.s[L] == 0:
                    data = X.copy()
                print(f'''\n Layer {L} Ends Here:
                    --{L}--{L}--{L}--{L}--{L}--{L}--{L}--{L}--{L}--- \n''')
            L += 1

    def predict(self, data):
        pred = pd.DataFrame()
        L = 0
        for model in self.models:
            for n in np.arange(len(model)):
                pred[n] = model[n].predict(data)
            if L == len(self.s):
                return pred
            else:
                if self.s[L] == 1:
                    data = pd.concat([data, pred], axis=1)
                elif self.s[L] == 0:
                    data = pred.copy()
                pred = pd.DataFrame()
            L += 1
