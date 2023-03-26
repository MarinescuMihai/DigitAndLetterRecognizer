from zGenerateDatasets import DigitRecognisionClass
import pandas as pd  
from sklearn import svm
from sklearn import metrics
import joblib
import os  
import numpy as np
from sklearn.utils import shuffle

##Step 0: Colectarea datelor din imagini
if os.path.exists("csv/dataset_digits.csv"):
  os.remove("csv/dataset_digits.csv")

digitRecognision = DigitRecognisionClass()
digitRecognision.CollectDataForAllExistingDigits()

# ##Step 1: Colectarea datelor din csv
dataframe = pd.read_csv("csv/dataset_digits.csv")   # re retin datele din fisier-ul cvs
dataframe = dataframe.sample(frac=1).reset_index(drop=True) # se amesteca datele
print(dataframe)

# ##Step 2: Separarea instantelor si claselor
X = dataframe.drop(["label"],axis=1)   # copie a datelor din fisierul csv dar fara tabela label 0 = 'index' , 1='columns'
Y = dataframe["label"]  # se retin etichetele

X_train, Y_train = X[0:40], Y[0:40] # se retin primele 130 de instante si etigheta lor, pentru a fi folosite in a antrena clasificatorul
X_test, Y_test = X[40:], Y[40:]   # se retin ultimele instante ramase pentru testare

# ##Step 3: Construirea modelului 
model = svm.SVC(kernel="linear")  # declararea clasificatorului SVC(Support Vector Classification)
model.fit(X_train, Y_train)  # pasul de invatare
joblib.dump(model,"model/SVM_classifier")  # salvarea clasificatorului in cazul in care nu se mai doreste invatarea lui inca o data

# ##Step4 : Afisarea acuratetei
Y_pred = model.predict(X_test)  # predictiile date de clasificator
print(Y_pred)
print("Acuratetea este:", metrics.accuracy_score(Y_test, Y_pred))
