import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import sys
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
        

class LogisticModel:
    def __init__(self):
        self.__parameters=None
    
    def fit(self, X_train, y_train):
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self, logreg, X_test, y_test):
        y_pred = logreg.predict(X_test)
        return y_pred
    
    def accuracy(self,traindata,X_train,y_train,testdata, y_test, y_pred):
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        modelCV = LogisticRegression()
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelCV, traindata[X_train], traindata[y_train], cv=kfold, scoring=scoring)
        print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
        
        print(classification_report(testdata[y_test], y_pred))
        
if __name__=="__main__":
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    finalfile = sys.argv[3]
    column_name = sys.argv[4]
    traindata = pd.read_csv(trainfile)
    testdata =  pd.read_csv(testfile)
    finaldf = pd.read_csv(finalfile)
    traindata = traindata.fillna(traindata.mean())
    testdata = testdata.fillna(testdata.mean())
    try:
        traindata = traindata.drop('None', axis=1)
    except:
        pass
    try:
        testdata = testdata.drop('None', axis=1)
    except:
        pass
    traindata = traindata.convert_objects(convert_numeric=True)
    testdata = testdata.convert_objects(convert_numeric=True)
    X_train = [traindata.columns[0], traindata.columns[1], traindata.columns[2]]
    y_train = traindata.columns[3]
    
    X_test = [testdata.columns[0], testdata.columns[1], testdata.columns[2]]
    y_test = testdata.columns[3]
    
    log = LogisticModel()
    logreg = log.fit(traindata[X_train], traindata[y_train])
    y_pred = log.predict(logreg, testdata[X_test], testdata[y_test])
    
    finaldf[column_name] = y_pred
    finaldf.to_csv(finalfile)
    
    # log.accuracy(traindata, X_train, y_train, testdata, y_test, y_pred)
    