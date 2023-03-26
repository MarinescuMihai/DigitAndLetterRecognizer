from GenerateDatasets import DigitRecognisionClass
import pandas as pd  
from sklearn import svm
from sklearn import metrics
import joblib
import os  
import numpy as np
from sklearn.utils import shuffle

if os.path.exists("csv/dataset_digits.csv"):
  os.remove("csv/dataset_digits.csv")

digitRecognision = DigitRecognisionClass()
digitRecognision.CollectDataForAllExistingDigits()

dataframe = pd.read_csv("csv/dataset_digits.csv")   
dataframe = dataframe.sample(frac=1).reset_index(drop=True) 
print(dataframe)

X = dataframe.drop(["label"],axis=1)   
Y = dataframe["label"] 

X_train, Y_train = X[0:55], Y[0:55] 
X_test, Y_test = X[55:], Y[55:]  

model = svm.SVC(kernel="linear")  
model.fit(X_train, Y_train)  
joblib.dump(model,"model/SVM_classifier")  

Y_pred = model.predict(X_test)  
print(Y_pred)
print("Acuratetea este:", metrics.accuracy_score(Y_test, Y_pred))
