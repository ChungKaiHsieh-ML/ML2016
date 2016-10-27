#!/usr/bin/env python
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import math
import csv
import time
import random

def read_data():
	'''
	read data
	'''
	#Training Data
	#read data
	columns_name = ['data_id']
	for i in range(57):
		columns_name.append(str(i))
	columns_name.append('label')
	training_data = pd.read_csv('data/spam_train.csv', sep=',' , encoding='latin1' , names=columns_name)

	#select feature
	columns_select = []
	for i in range(57):
		columns_select.append(str(i))
	training_feature = training_data.as_matrix(columns=columns_select).astype(dtype='float32')
	columns_select  = ['label']
	training_yhat     = training_data.as_matrix(columns=columns_select ).astype(dtype='float32')

	print(training_feature)
	print("training_feature.shape" + str(training_feature.shape))
	print(training_yhat)
	print("training_yhat.shape" + str(training_yhat.shape))

	training_feature_total = training_feature
	training_yhat_total = training_yhat

	validation_percent = 1/10.0
	training_percent = 1 - validation_percent 
	num_training = int(training_percent * training_feature.shape[0])
	indices = np.random.permutation(training_feature.shape[0])
	training_idx,validation_idx = indices[:num_training], indices[num_training:]

	training_feature ,validation_feature = training_feature[training_idx,:], training_feature[validation_idx,:]
	training_yhat ,validation_yhat = training_yhat[training_idx,:], training_yhat[validation_idx,:]

	print("training_feature.shape" + str(training_feature.shape))
	print("validation_feature.shape" + str(validation_feature.shape))
	print("training_yhat.shape" + str(training_yhat.shape))
	print("validation_yhat.shape" + str(validation_yhat.shape))


	# #Testing Data
	#read data
	columns_name = ['data_id']
	for i in range(57):
		columns_name.append(str(i))

	testing_data = pd.read_csv('data/spam_test.csv', sep=',', encoding='latin1', names=columns_name)

	columns_select = []
	for i in range(57):
		columns_select.append(str(i))

	testing_set = testing_data.as_matrix(columns=columns_select).astype(dtype='float32')
	print(testing_set)
	print("testing_set.shape" + str(testing_set.shape))

	print("training_feature_total.shape" + str(training_feature_total.shape) )
	print("training_feature.shape" + str( training_feature.shape) )
	print("validation_feature.shape" + str( validation_feature.shape) )

	return training_feature_total, training_yhat_total , training_feature , training_yhat, validation_feature, validation_yhat ,testing_set


'''
Algorithm
'''

def rss(array):
	return np.sqrt(array.T.dot(array))


#usage: 
#data_set=numpy.array([])


def feature_normalize(data_set):
	tmp = data_set
	for i in range(np.shape(tmp)[1]):
		tmp[: , i] = ( tmp[:,i] - np.mean(tmp[:,i])) / np.std(tmp[:,i])

	return tmp

def feature_normalize_2(data_set):
	tmp = data_set
	for i in range(np.shape(tmp)[1]):
		tmp[: , i] = ( tmp[:,i] - np.mean(tmp[:,i])) 

	return tmp


def sigmoid(X):

	den = 1.0 + np.exp (-1.0 * X)
	d = 1.0 / den

	return d  

def derivation_sigmoid(X):

	return sigmoid(X) * ( 1 - sigmoid(X))




def compute_accuracy_neural_network(X, y, w1 , w2):
	# forward propagation
	z1 = np.dot(w1, X.T )
	a1 = sigmoid(z1)
	z2 = np.dot(w2, a1)
	y_p = sigmoid(z2)

	y_p = y_p.T

	predictions = np.rint( y_p ).astype(int)

	tmp = predictions

	for i in range(predictions.shape[0] ):

		if( predictions[i] == y[i]):
			tmp[i] = 1
		else:
			tmp[i] = 0

	accurate_num = np.sum(tmp)
	accuracy = float(accurate_num)/len(predictions)
	print("[accurate:]" + str(accurate_num) + "/" + str(len(predictions)) + "  [accuracy:]" + str(float(accurate_num)/len(predictions))      )      

	return accuracy







#compute cost value
def compute_cost_neural_network(X ,y , w1 , w2):
	# m : data number
	m = X.shape[0]
	# theta = np.reshape(theta, (len(theta),1 ))
	cost = 0.0

	# forward propagation
	z1 = np.dot(w1, X.T )
	a1 = sigmoid(z1)
	z2 = np.dot(w2, a1)
	y_p = sigmoid(z2)

	y_p = y_p.T
	# J : cost 
	small_value = 10**-6
	J = (1.0/m) * ( - np.transpose(y).dot(  np.log( y_p + small_value )  )   - np.transpose( 1- y ).dot( np.log( 1 - y_p  + small_value  )   )     )


	return J[0][0]



#*************************
#-----Gradient (adagrad)-----
#*************************
 
def neural_network(X, y, X_validation, y_validation, w1 , w2, learning_rate, epochs):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking epochs gradient steps with learning_rate
	#-------------------------------------------------
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	X_validation_tmp = X_validation
	n_validation = X_validation.shape[0]
	bias_ones = np.ones( (n_validation,1) , dtype='float32')
	X_validation_tmp = np.append(X_validation, bias_ones, axis=1)

	# g_history = np.zeros( [len(theta), epochs] )
	cost_training_history = np.zeros(shape=(epochs, 1))
	cost_validation_history = np.zeros(shape=(epochs, 1))

	# print("X_validation.shape"  + str(X_validation.shape))
	# print("X_validation_tmp.shape"  + str(X_validation_tmp.shape))

	w1_adagrad_coeff = 0 * w1
	w2_adagrad_coeff = 0 * w2

	for i in range(epochs):
		# predictions = X_tmp.dot(theta)
		# theta_size = theta.shape[0]

		for j in range(n):

			# forward propagation
			z1 = np.dot(w1, X_tmp[j, : ].T )
			a1 = sigmoid(z1)
			z2 = np.dot(w2, a1)
			y_p = sigmoid(z2)

			#backpropagation

			dw2 = ( - ( y[j] - y_p)[0] *  a1 )

			dw1 = (  np.dot(   - ( y[j] - y_p)[0] *  (w2.T.reshape(w2.T.shape[0],1) * derivation_sigmoid(z1).reshape(z1.shape[0],1) )  , X_tmp[j,:].T.reshape(1,X_tmp[j,:].T.shape[0])   ) )
	


			w1_adagrad_coeff = np.sqrt( (w1_adagrad_coeff * w1_adagrad_coeff) + (dw1 * dw1) )
			w2_adagrad_coeff = np.sqrt( (w2_adagrad_coeff * w2_adagrad_coeff) + (dw2 * dw2) )

			w1 = w1 - learning_rate * dw1/w1_adagrad_coeff
			w2 = w2 - learning_rate * dw2/w2_adagrad_coeff

			if( (j %500) == 0 ):
				print("iteration numbers : " + str(i))
					# print("theta : " + str(theta))
				print(" [cost] - training    data   : " + str( compute_cost_neural_network(X_tmp, y, w1, w2)))
				training_accuracy =  compute_accuracy_neural_network(X_tmp, y, w1 , w2)
				print(" [cost] - validation data   : " + str( compute_cost_neural_network(X_validation_tmp, y_validation, w1, w2)))
				validation_accuracy =  compute_accuracy_neural_network(X_validation_tmp, y_validation, w1 , w2)
				# print(" [cost] - validation data : " + str( compute_cost(X_validation_tmp, y_validation, theta)))
				# compute_accuracy(X_validation_tmp, y_validation, theta)
				# cost_training_history[i, 0] = compute_cost(X_tmp, y, theta)
				# cost_validation_history[i,0] = compute_cost(X_validation_tmp,y_validation,theta)
				# validation_accuracy =  compute_accuracy(X_validation_tmp, y_validation, theta)

	return w1,w2, training_accuracy, validation_accuracy
	# return theta, cost_training_history, cost_validation_history , validation_accuracy




def testing_neural_network(X,w1,w2):
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	# forward propagation
	z1 = np.dot(w1, X_tmp.T )
	a1 = sigmoid(z1)
	z2 = np.dot(w2, a1)
	y_p = sigmoid(z2)

	y_p = y_p.T

	predictions = np.rint( y_p ).astype(int)

	return  predictions



def write_file(predictions,file_name):
	predictions.shape = (len(predictions) , 1)
	writer = open(file_name,"w")
	writer.write("id,label\n")
	for i in range(len(predictions)):
		str_tmp = str( i+1 )+"," + str(predictions[i][0]) + "\n"
		# print(str_tmp)
		writer.write(str_tmp)		
	writer.close()


if __name__ == '__main__':


	start_time = time.time()

	training_accuracy_th  = 0.95
	validation_accuracy_th = 0.96

	for k in range(100000):

		print("**************************")
		print("**************************")
		print("****  Data Changes  ****")
		print("**************************")
		print("**************************")
		# time.sleep(1)

		(training_feature_total, training_yhat_total, training_feature , training_yhat, validation_feature, validation_yhat ,testing_set) = read_data()
		# training_feature_total_normalized = feature_normalize(training_feature_total) 
		# training_feature_normalized = feature_normalize(training_feature) 
		# validation_feature_normalized = feature_normalize(validation_feature) 
		# testing_set_normalized = feature_normalize(testing_set)

		training_feature_total_normalized = feature_normalize(training_feature_total) 
		training_feature_normalized = feature_normalize(training_feature)  
		validation_feature_normalized = feature_normalize(validation_feature) 
		testing_set_normalized = feature_normalize(testing_set) 


		epochs = 5
		learning_rate = 0.8

		feature_num = 57
		mu, sigma = 0, 0.8
		hidden_layer_num = 3
		w1 = sigma * np.random.randn( hidden_layer_num , feature_num +1) + mu
		w2 = sigma * np.random.randn( 1 , hidden_layer_num ) + mu


		(w1,w2, training_accuracy, validation_accuracy) = neural_network(training_feature_normalized, training_yhat, validation_feature_normalized, validation_yhat, w1 , w2, learning_rate, epochs)
		

		if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):
			# (w1,w2, training_accuracy, validation_accuracy) = neural_network(training_feature_total_normalized, training_yhat_total, validation_feature_normalized, validation_yhat, w1 , w2, learning_rate, epochs)
			print(" ****** Hidden Layer " + str(hidden_layer_num) + "  *************")
			break
		else:
			print("accuracy is not enough ")


		epochs = 5
		learning_rate = 0.8

		feature_num = 57
		mu, sigma = 0, 0.8
		hidden_layer_num = 5
		w1 = sigma * np.random.randn( hidden_layer_num , feature_num +1) + mu
		w2 = sigma * np.random.randn( 1 , hidden_layer_num ) + mu


		(w1,w2, training_accuracy, validation_accuracy) = neural_network(training_feature_normalized, training_yhat, validation_feature_normalized, validation_yhat, w1 , w2, learning_rate, epochs)
		

		if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):
			# (w1,w2, training_accuracy, validation_accuracy) = neural_network(training_feature_total_normalized, training_yhat_total, validation_feature_normalized, validation_yhat, w1 , w2, learning_rate, epochs)
			print(" ****** Hidden Layer " + str(hidden_layer_num) + "  *************")
			break
		else:
			print("accuracy is not enough ")


		epochs = 5
		learning_rate = 0.8

		feature_num = 57
		mu, sigma = 0, 0.8
		hidden_layer_num = 5
		w1 = sigma * np.random.randn( hidden_layer_num , feature_num +1) + mu
		w2 = sigma * np.random.randn( 1 , hidden_layer_num ) + mu


		(w1,w2, training_accuracy, validation_accuracy) = neural_network(training_feature_normalized, training_yhat, validation_feature_normalized, validation_yhat, w1 , w2, learning_rate, epochs)
		

		if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):
			# (w1,w2, training_accuracy, validation_accuracy) = neural_network(training_feature_total_normalized, training_yhat_total, validation_feature_normalized, validation_yhat, w1 , w2, learning_rate, epochs)
			print(" ****** Hidden Layer " + str(hidden_layer_num) + "  *************")
			break
		else:
			print("accuracy is not enough ")


	# testing_set = (testing_set)
	output_filename = "prediction.csv"
	predictions = testing_neural_network(testing_set_normalized, w1, w2)
	write_file(predictions, output_filename)

	end_time = time.time()

	min_ = int(  (end_time - start_time)/60  )
	sec_ = (end_time - start_time)%60

	print("Consume : " + str(min_) + "mins " + str(sec_) + "secs" )




	# # print("starting point of  theta " + str(theta_tmp))

	# '''
	# Draw the cost function
	# '''
	# #gradient descent
	# plt.figure(1)
	# plot1 = plt.plot(range(len(cost_training_history[0:])) , cost_training_history[0:],'ro' ,label='$training$')
	# plot2 = plt.plot(range(len(cost_training_history[0:])) , cost_validation_history[0:],'g--',label ='$Validation$')

	# plt.title('Linear Regression')
	# plt.title('LR:' + str(learning_rate), loc='left')
	# plt.title('epochs :' + str(epochs), loc='right')
	# plt.ylabel('Cost')
	# plt.xlabel('epochs')
	# plt.xlim([0,len(cost_training_history[0:])])
	# plt.legend()
	# plt.show(block=True)

