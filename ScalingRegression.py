#! /Library/Frameworks/Python.framework/Versions/anaconda/bin/python

#=====================================================================#
# This is a basic polynomial regression prototype.  It is basic and   #
#   uses the stock market pricing of a random ticker.   The dataset   #
#   is oriented in time from most recent to the first price.          #
#   This is just to organize a new algorithm for some other project.  #
#   In general, it will make a random training set and CV set and     #
#   will find the best parameter fit over many different realizations #
#   This technique is similar to Monte Carlo/K-folds and bootstrapping#
#       Created by: nmg5038                                           #
#       Created on: 10/5/16                                           #
#=====================================================================#

def datasetCreation(whatMonth,data,data2,totalN, TrainingToCV=.8):
	proportions = round(TrainingToCV * totalN)

	TrainingSet = np.zeros((2,proportions))
	CrossValidationSet = np.zeros((2,totalN-proportions))
	
	count = 0
	count2 = 0
	for i in np.arange(1,13):
		locationsForMonth,=np.where(whatMonth == i)
		
		# Shuffle the index locations
		np.random.shuffle(locationsForMonth)
		
		allotmentForTrainingSet = round(locationsForMonth.size * TrainingToCV)
		allotmentForCV = locationsForMonth.size - allotmentForTrainingSet
		
		if (count2 + allotmentForCV) > (totalN-proportions):
			diff = (count2 + allotmentForCV) - (totalN-proportions)
			allotmentForCV = (totalN-proportions)-count2
			allotmentForTrainingSet += diff
		
		if (count + allotmentForTrainingSet) > (proportions):
			diff = (count + allotmentForTrainingSet) - (proportions)
			allotmentForTrainingSet = (proportions)-count
			allotmentForCV += diff
		
		locationForTrainingSet = locationsForMonth[:allotmentForTrainingSet]
		locationForCV = locationsForMonth[allotmentForTrainingSet:]
		
		TrainingSet[0,count:count+allotmentForTrainingSet] = data[locationForTrainingSet]
		TrainingSet[1,count:count+allotmentForTrainingSet] = data2[locationForTrainingSet]
		
		CrossValidationSet[0,count2:count2+allotmentForCV] = data[locationForCV]
		CrossValidationSet[1,count2:count2+allotmentForCV] = data2[locationForCV]
		
		count += allotmentForTrainingSet
		count2 += allotmentForCV
	return TrainingSet, CrossValidationSet
	
def parameterDetermination(TrainingSet,polynomalRegress = 2):
	shapex = TrainingSet.shape[0]
	shapey = TrainingSet.shape[1]

	
	Y = TrainingSet[1,:]
	A = np.ones((1,shapey)) # Accounting for Bias
	
	for i in np.arange(1,polynomalRegress+1):
		powerArr = np.zeros(shapey)+i
		polynomialX=np.power(TrainingSet[0,:],powerArr)
		A = np.vstack((A,polynomialX))
	
	A = np.matrix(A)
	Y = np.matrix(Y)

	parameters = np.dot(np.dot(A,A.T).I,np.dot(A,Y.T))

	return parameters

def crossValidationTest(CVSet,parameters,polynomalRegress = 2,test='MSSE'):
	shapex = CVSet.shape[0]
	shapey = CVSet.shape[1]
	n = shapey
	p = parameters.size
	
	Y = CVSet[1,:]
	A = np.ones((1,shapey)) # Accounting for Bias
	
	for i in np.arange(1,polynomalRegress+1):
		powerArr = np.zeros(shapey)+i
		polynomialX=np.power(CVSet[0,:],powerArr)
		A = np.vstack((A,polynomialX))
	
	A = np.matrix(A)
	Y = np.matrix(Y)	
	
	YHat = np.dot(parameters.T,A)
	
	error = Y - YHat
	SSE = np.dot(error,error.T)
	
	MSE = SSE / n
	RMSE = np.sqrt(MSE)
	
	AIC = n * np.log(SSE) - n * np.log(n) + 2. * p
	BIC = n * np.log(SSE) - n * np.log(n) + 2. * np.log(p)
	APC = (n+p)/(n*(n-p)) * SSE
	
	return MSE, RMSE, AIC, BIC, APC	

if __name__ == "__main__":
	
	import matplotlib.pyplot as plt
	import numpy as np
	import sys, os,time
	import csv,datetime
	
	with open('./montecarloex.csv', 'rb') as csvfile:											# Read in the historical data from NextEra Energy
		readinfile= csv.reader(csvfile, dialect='excel',quoting=csv.QUOTE_MINIMAL)			
		dates,open_p,high_p,low_p,close_p,volume_p=[],[],[],[],[],[]
		
		for i,row in enumerate(readinfile):
			if i > 0:
				dates.append(row[0])
				open_p.append(float(row[1]))
				high_p.append(float(row[2]))
				low_p.append(float(row[3]))
				close_p.append(float(row[4]))
				volume_p.append(float(row[5]))


	totalNumberOfPoints = len(open_p)
	
	ndays_since = np.zeros(totalNumberOfPoints)
	whatMonth = np.zeros(totalNumberOfPoints)
	
	# Determine what day this was since the start of the index and what month it is in.
	for i,date in enumerate(dates):
		whatMonth[i]=datetime.datetime(int(date[:4]),int(date[5:7]),int(date[8:])).month
		ndays_since[i]=(datetime.datetime(int(date[:4]),int(date[5:7]),int(date[8:]))-datetime.datetime(int(dates[-1][:4]),int(dates[-1][5:7]),int(dates[-1][8:]))).days
	
	# Reverse Arrays For increasing in time
	ndays_since = ndays_since[::-1]
	whatMonth = whatMonth[::-1]

	close_p = np.array(close_p)[::-1]
	open_p = np.array(open_p)[::-1]
	high_p = np.array(high_p)[::-1]
	low_p = np.array(low_p)[::-1]
	volume_m = np.array(volume_p)[::-1]
	
	# Determine the number of tests (arbitrary for now)
	numberOfTests = round(totalNumberOfPoints/10)
	
	# Allocate and initialize some arrays
	estParams = np.matrix(np.zeros((3,numberOfTests)))
	CVmetric = np.zeros(numberOfTests)
	CVmetric2 = np.zeros(numberOfTests)
	
	for testNumber in np.arange(numberOfTests):
	
		# Determine the training set and the cross validation set
		TrainSet, CV = datasetCreation(whatMonth, open_p,close_p,totalNumberOfPoints)
	
		# Determine the polynomial regression parameters with Training Set
		params = parameterDetermination(TrainSet,polynomalRegress = 2)
		
		# Store parameters for later
		estParams[:,testNumber] = params
	
		# Evaluate parameters against a cross-validation set
		MSETest, RMSETest, AICTest, BICTest, APCTest = crossValidationTest(CV,params,polynomalRegress = 2,test='MSSE')
		
		# Store some of the metrics in a holding array
		CVmetric[testNumber] = AICTest
		CVmetric2[testNumber] = MSETest
	
	# Find the minimum error metric (best scenario)
	bestEstimate, = np.where(CVmetric == np.nanmin(CVmetric))
	bestParameters = estParams[:,bestEstimate] 
	
	# Output
	print "The best parameters are (a_o + a_1 x + a_2 x^2):"
	print "		a_o =", bestParameters[0,:]
	print "		a_1 =", bestParameters[1,:]
	print "		a_2 =", bestParameters[2,:]
	
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(211)
	ax.plot(CVmetric)
	ax.plot(bestEstimate,CVmetric[bestEstimate],'o')
	ax.set_xlim([0,CVmetric.size])
	ax.set_xlabel("Random Fold Number")
	ax.set_ylabel("AIC")
	ax.set_title("Akaike's Information Criterion")

	ax = fig.add_subplot(212)
	ax.plot(CVmetric2)
	ax.set_xlim([0,CVmetric2.size])
	ax.plot(bestEstimate,CVmetric2[bestEstimate],'o')
	ax.set_xlabel("Random Fold Number")
	ax.set_ylabel("MSE")
	ax.set_title("Mean Squared Error")
	
	
	plt.show()
	sys.exit()

		