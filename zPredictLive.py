from zGenerateDatasets import DigitRecognisionClass
import joblib  
import cv2
import numpy as np
import csv
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if os.path.exists("csv/dataset_live_prediction.csv"):
  os.remove("csv/dataset_live_prediction.csv")

label = "7" 
drc = DigitRecognisionClass()
drc.ColectDataForLivePrediction(label)

dataframe = pd.read_csv("csv/dataset_live_prediction.csv")  
print(dataframe)

model = joblib.load("model\SVM_classifier")

Y_pred = model.predict(dataframe)  
print("PREDICTIA ESTE:", Y_pred)

img = mpimg.imread("temp/"+label+".jpg")
imgplot = plt.imshow(img)
plt.show()


