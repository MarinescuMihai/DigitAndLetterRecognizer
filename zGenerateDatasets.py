import cv2,io
import numpy as np
import csv
import glob

class DigitRecognisionClass:

	def CollectData(self,label,addTitle):
		dirList = glob.glob("orig_images/"+label+"/*.png") # directorul in care sunt imaginile cu fiecare cifra
		for img_path in dirList:  # se parcurge fiecare imagine din director
			im = cv2.imread(img_path)  # se deschide imaginea
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # se for schimba culorile din imaginini in alb-negru, deoarece imaginea trebuie sa fie cat mai simpla
			im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0) # se netezeste imaginea pentru a avea cat mai putin zbomot
			roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA) # se face rescale imaginii

			#cv2.imshow("window",roi)

			data=[] # se declara lista care v-a retine instantele
			data.append(label) # se adauga etigheta
			rows,cols = roi.shape  # dimensiunile imaginii
			
			# se adauga fiecare pixel in lista data
			for i in range(rows):
				for j in range(cols):
					k = roi[i,j]
					if k>100:  
						k=0   # se returneaza 0 in cazul in care pixelul este alb
					else:
						k=1	  # se returneaza 1 in cazul in care pixelul este negru

					data.append(k)

			with open("csv/dataset_digits.csv", "a", newline='') as f:  # se deschide/creaza fisierul csv care v-a retine instantele si etighetele
				writer = csv.writer(f) 	# clasa ajutatoare in popularea fisierului csv cu instante
				if addTitle == True:
					addTitle = False
					header =["label"]
					for i in range(0,784):
						header.append("pixel"+str(i))
					writer.writerow(header)  # se adauga numele atributelor
				
				if(data[1] > -1):
					writer.writerow(data)  # popularea fisieului csv cu instante

		return True

	def CollectDataForAllExistingDigits(self):
		ok = self.CollectData(label = "4",addTitle = True)
		if ok == True:
			self.CollectData(label = "5",addTitle = False)
		if ok == True:
			self.CollectData(label = "7",addTitle = False)
	

	def ColectDataForLivePrediction(self,label):
		im = cv2.imread("temp/"+label+".png")
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
		roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

		#cv2.imshow("window",roi)

		data=[]
		rows,cols = roi.shape  
		
		# #Add pixel one-by-one into data Array.
		for i in range(rows):
			for j in range(cols):
				k = roi[i,j]
				if k>100:
					k=0 # pixel alb
				else:
					k=1	# pixel negru

				data.append(k)

		with open("csv/dataset_live_prediction.csv", "a", newline='') as f:
			writer = csv.writer(f) 	
			header =["pixel"]
			for i in range(0,783):
				header.append("pixel"+str(i))
			writer.writerow(header)  

			writer.writerow(data)  




		

