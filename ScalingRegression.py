#! /Library/Frameworks/Python.framework/Versions/anaconda/bin/python

class unit():
	def __init__(self,value,grad):
		self.value = value
		self.grad = grad
class multiply_gate():
	def __init__(self):
		print 'multiply_gate initialized'
		
	def forward(self,u0,u1):
		self.u0 = u0
		self.u1 = u1
		self.utop = unit(u0.value*u1.value,0.)
		return self.utop
	
	def backward(self):
		self.u0.grad += self.u1.value * self.utop.grad
		self.u1.grad += self.u0.value * self.utop.grad

class add_gate():
	def __init__(self):
		print 'add_gate initialized'
		
	def forward(self,u0,u1):
		self.u0 = u0
		self.u1 = u1
		self.utop = unit(u0.value+u1.value,0.)
		return self.utop
	
	def backward(self):
		self.u0.grad += 1 * self.utop.grad
		self.u1.grad += 1 * self.utop.grad

class sigmoid_gate():
	def __init__(self):
		print 'sigmoid_gate initialized'
		
	def sigmoid(self,x):
		return 1./(1.+np.exp(-x))
	
	def sigmoid_gradient(self,x):
		return x*(1.-x)
		
	def forward(self,u0):
		self.u0 = u0
		self.utop = unit(self.sigmoid(self.u0.value),0.)
		
		return self.utop
	
	def backward(self):
		s = self.sigmoid(self.u0.value)
		
		self.u0.grad += self.sigmoid_gradient(s) * self.utop.grad
		
	
def sigmoid(x):
	return 1./(1.+np.exp(-x))
def sigmoid_gradient(x):
	return x*(1.-x)
def bi_sigmoid(x):
	return 2./(1.+np.exp(-x)) - 1.
def bi_sigmoid_gradient(x):
	return .5*(1.+x)*(1.-x)
def gaussian_std(x):
	return (x-np.mean(x))/np.std(x)

def alt_std(x):
	return (x/np.max(x))*.8 + .1
def forward_gate(x,y):
	return x*y
def forward_gate_add(x,y):
	return x+y
def forward_circut(x,y,z):
	return forward_gate(forward_gate_add(x,y),z)

def alt_std_two(x):
	return np.min(x),np.max(x)-np.min(x),(x - np.min(x))/(np.max(x)-np.min(x)) 

def H_o(THETA,X):
	return np.dot(THETA,X)
	
def cost_funct(THETA,x,y):
	hypo = H_o(THETA,x)

	delta = hypo - y
	#delta_sq = (delta*delta.T)/(2.*delta.size)
	#print ((hypo * hypo.T)-(y * hypo.T)-(hypo * y.T)+(y * y.T))/(2.*delta.size)
	
	return delta#,delta_sq

def batch_gradient_descent(learning_rate,THETA,x,y):
	
	delta=cost_funct(THETA,x,y)
	current_cost = np.dot(delta,delta.T) / (2.*delta.size)
	THETA_new = THETA.copy()
	hold_update = np.zeros(THETA.size)
	
	for i in np.arange(THETA.size):
		hold_update[i] = learning_rate * np.dot(delta,x[i,:].T) / (delta.size)
		
	for i in np.arange(THETA.size):
		THETA_new[0,i] -= hold_update[i]
		
	return current_cost, THETA_new
	
	

	
	
if __name__ == "__main__":
	
	import matplotlib.pyplot as plt
	import numpy as np
	import netCDF4 as net
	import sys, os,time
	import csv,datetime
	import mpl_toolkits.basemap 
	from mpl_toolkits.basemap import Basemap

	
	
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

	ndays_since = np.zeros(np.array(open_p).size)
	for i,date in enumerate(dates):
		ndays_since[i]=(datetime.datetime(int(date[:4]),int(date[5:7]),int(date[8:]))-datetime.datetime(int(dates[-1][:4]),int(dates[-1][5:7]),int(dates[-1][8:]))).days
	
	ndays_since = ndays_since[::-1]
	close_p = np.array(close_p)[::-1]
	open_p = np.array(open_p)[::-1]
	high_p = np.array(high_p)[::-1]
	low_p = np.array(low_p)[::-1]
	volume_m = np.array(volume_p)[::-1]
	
	open_p_t = open_p[1:-1]
	high_p_t = high_p[1:-1]
	low_p_t = low_p[1:-1]
	volume_m_t = volume_m[1:-1]
	ndays_since_t = ndays_since[1:-1]
	close_p_t = close_p[:-2]
	close_p_truth = close_p[1:-1]
	#X_o = np.zeros(ndays_since_t.size)+1.
	#close_p_truth = np.matrix(close_p_truth)
	
	
	min_op,del_op,open_p_t = alt_std_two(open_p_t)
	min_cl,del_cl,close_p_t = alt_std_two(close_p_t)
	min_hi,del_hi,high_p_t = alt_std_two(high_p_t)
	min_lo,del_lo,low_p_t = alt_std_two(low_p_t)
	min_vol,del_vol,volume_m_t = alt_std_two(volume_m_t)
	min_ndy,del_ndys,ndays_since_t = alt_std_two(ndays_since_t)
	min_tru,del_tru,close_p_truth = alt_std_two(close_p[1:-1])
	close_p_truth = np.matrix(close_p_truth)
	X_o = np.zeros(ndays_since_t.size)+1.
	training_data = np.matrix(np.array([X_o,ndays_since_t,open_p_t,close_p_t,high_p_t,low_p_t,volume_m_t]))
	
	PARAMS = np.matrix(np.zeros((7)))
	
	test=np.matrix(np.array([X_o[3],ndays_since_t[3],open_p_t[3],close_p_t[3],high_p_t[3],low_p_t[3],volume_m_t[3]]))

	previous_cost = 9999.
	current_cost = 5
	dp = 20.
	i = 0
	while current_cost>1e-5:
				
		current_cost,PARAMS = batch_gradient_descent(1.15,PARAMS,training_data,close_p_truth)
		
		dp = current_cost - previous_cost
		
		previous_cost = current_cost
		if i % 10 == 0:
			print current_cost
		
		i+=1

	
	dp=np.dot(PARAMS,training_data)-close_p_truth
	
	# NORMAL EQUATIONS (1 step, no iterations!)	
	PARAMS2 = np.dot(np.dot((np.dot(training_data,training_data.T)).I,training_data),close_p_truth.T).T
	

	gradient_descent_est = np.dot(PARAMS,training_data).T
	normal_eqs_est = np.dot(PARAMS2,training_data).T
	
	
	
	plt.plot(normal_eqs_est-close_p_truth.T)
	#plt.plot(close_p_truth.T)
	plt.show()
	
	sys.exit()

		