import cv2,io
import numpy as np
import csv
import glob

class DigitRecognisionClass:

	def CollectData(self,label,addTitle):
		dirList = glob.glob("orig_images/"+label+"/*.png")
		for img_path in dirList:  
			im = cv2.imread(img_path)  
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0) 
			roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA) 

			data=[]
			data.append(label) 
			rows,cols = roi.shape  
			
			for i in range(rows):
				for j in range(cols):
					k = roi[i,j]
					if k>100:  
						k=0   
					else:
						k=1	 

					data.append(k)

			with open("csv/dataset_digits.csv", "a", newline='') as f: 
				writer = csv.writer(f) 	
				if addTitle == True:
					addTitle = False
					header =["label"]
					for i in range(0,784):
						header.append("pixel"+str(i))
					writer.writerow(header)  
				
				if(data[1] > -1):
					writer.writerow(data)  

		return True

	def CollectDataForAllExistingDigits(self):
		ok = self.CollectData(label = "4",addTitle = True)
		if ok == True:
			self.CollectData(label = "5",addTitle = False)
		if ok == True:
			self.CollectData(label = "7",addTitle = False)
		if ok == True:
			self.CollectData(label = "A",addTitle = False)
	

	def ColectDataForLivePrediction(self,label):
		im = cv2.imread("temp/"+label+".png")
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
		roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

		data=[]
		rows,cols = roi.shape  
		
		for i in range(rows):
			for j in range(cols):
				k = roi[i,j]
				if k>100:
					k=0
				else:
					k=1	

				data.append(k)

		with open("csv/dataset_live_prediction.csv", "a", newline='') as f:
			writer = csv.writer(f) 	
			header =["pixel"]
			for i in range(0,783):
				header.append("pixel"+str(i))
			writer.writerow(header)  

			writer.writerow(data)  




		

