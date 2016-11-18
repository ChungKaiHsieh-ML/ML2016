#!/usr/bin/env python
import numpy as np 
import pickle
import scipy
import math
import time
import theano
import keras
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,UpSampling2D,Input, Dense 
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.models import load_model


'''
Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python method2.py
'''
def read_data(path):
	tStart = time.time()

	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3

	print("Reading labeled data")
	#label data : X_train_label , y_train_label
	all_label_filename =  path +  "all_label.p"
	all_label = pickle.load(open(all_label_filename , 'rb'))
	all_label = np.array(all_label)

	#input image total class
	num_class = len(all_label)
	#input image total numver
	num_image = len(all_label[0])
	#total image nb
	nb_total_image = num_class * num_image

	#labeled image data structure declare
	X_train_label = np.zeros(shape=(nb_total_image,img_channels,img_rows,img_cols )).astype(dtype = 'float32')
	y_train_label = np.array([0] * num_class * num_image).astype(dtype='int').reshape(num_class * num_image,1)

	#feature
	for i in range(num_class):
		for j in range(num_image):
			tmp = np.array( all_label[i][j] ).astype(dtype='float32').reshape(1,img_channels,img_rows,img_cols)
			X_train_label[ i*num_image + j ] = tmp

		print("reading labeled data :" + str(500*i))



	#label
	for i in range(num_class):
		for j in range(num_image):
			y_train_label[ i*num_image + j] = i

	# np.save("X_train_label", X_train_label)
	# np.save("y_train_label",y_train_label)




	#unlabel data : X_train_unlabel
	all_unlabel_filename =  path + "all_unlabel.p"
	all_unlabel = pickle.load(open(all_unlabel_filename , 'rb'))
	all_unlabel = np.array(all_unlabel)


	#input image total number
	num_image = len(all_unlabel)
	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3


	#unlabel data structure declare :X_train_unlabel
	X_train_unlabel = np.zeros(shape = (num_image,img_channels,img_rows,img_cols) ).astype(dtype='float32')

	for i in range(num_image):
		tmp = np.array(all_unlabel[i]).astype(dtype='float32').reshape(img_channels,img_rows,img_cols)
		X_train_unlabel[i] = tmp

		if(i % 1000 == 0):
			print("reading ulabel data :" + str(i))

	# print("X_train_unlabel.shape" + str( X_train_unlabel.shape ))

	# np.save("X_train_unlabel", X_train_unlabel)



	#testing data : X_test , ID_test
	test_filename = path+"test.p"
	test = pickle.load(open(test_filename,'rb') )
	data_test = test['data'][:]
	ID_test = test['ID'][:]

	data_test = np.array(data_test)
	ID_test = np.array(ID_test)

	np_image = len(data_test)

	X_test = np.zeros(shape = (np_image, img_channels,img_rows,img_cols) ).astype(dtype='float32')

	for i in range(np_image):
		tmp = np.array(data_test[i]).astype(dtype='float32').reshape(img_channels,img_rows,img_cols)
		X_test[i] = tmp

		if(i % 1000 == 0):
			print("reading testing data :" + str(i) )


	tEnd = time.time()
	cost_time = tEnd - tStart
	print("Reading data cost " + str(cost_time) + "sec")

	return X_train_label,y_train_label,X_train_unlabel,X_test,ID_test


def add_unlabel_featrue2training(training_feature, training_yhat, unlabel_feature, model, confidence_th):
	unlabel_prediction = model.predict(unlabel_feature)
	# print("nb_unlabel_feature:" + str(len(unlabel_feature) ))
	# print("nb_unlabel_prediction:" + str(len(unlabel_prediction)))
	nb_unlabel_data = len(unlabel_prediction)

	delete_array = np.array([0]*nb_unlabel_data).astype(dtype='int')
	delete_index = np.array([]).astype(dtype='int')
	count = 0

	print("[ original data ] : training->" + str(len(training_feature)) + "  unlabel->"+str(len(unlabel_feature)) )



	for i in range(nb_unlabel_data):
		if(  np.amax( unlabel_prediction[i] ) >= confidence_th  ):

			delete_array[i] = 1
			max_index = np.argmax(unlabel_prediction[i])
			tmp_prediction = unlabel_prediction[i]
			tmp_prediction.fill(0)
			tmp_prediction[max_index] = 1.0

			tmp_feature =  unlabel_feature[i]
			# print(tmp_feature.shape)
			# dim = tmp_feature.shape[0] * tmp_feature.shape[1] * tmp_feature.shape[2]
			# tmp_feature = np.array( tmp_feature ).astype(dtype='float32').reshape(1,)
			training_feature = np.vstack((training_feature, tmp_feature ))


			tmp_prediction = np.array( tmp_prediction ).astype(dtype='float32').reshape(1,10)			
			training_yhat     = np.vstack((training_yhat, tmp_prediction ))

			count += 1
		if(i%1000 == 0 ):
			print("check unlabel confidence :" + str(i) +"  data")


	for i in range( nb_unlabel_data):
		if(delete_array[i] == 1):
			delete_index = np.append(delete_index, i )	
		if(i%1000 == 0 ):
			print("delete unlabel feature :" + str(i) +"  data")

	unlabel_feature = np.delete(unlabel_feature, delete_index, axis = 0 )	


	print("after confidence similar trimming:")
	print("total move :" + str(count) + "  data from unlabel to training ")
	print("[after data] : training->" + str(len(training_feature)) + "  unlabel->" + str(len(unlabel_feature)) )
	time.sleep(3)


	return training_feature, training_yhat, unlabel_feature


def split_data(training_feature, training_yhat, validation_percent):
	
	training_percent = 1 - validation_percent 
	num_training = int(training_percent * training_feature.shape[0])
	indices = np.random.permutation(training_feature.shape[0])
	training_idx,validation_idx = indices[:num_training], indices[num_training:]
	# print("training_feature.shape[0]",training_feature.shape[0])
	# print()

	training_feature ,validation_feature = training_feature[training_idx,:], training_feature[validation_idx,:]
	training_yhat ,validation_yhat = training_yhat[training_idx,:], training_yhat[validation_idx,:]

	return training_feature , training_yhat , validation_feature , validation_yhat


def write_file(ID,predictions,file_name):
	predictions.shape = (len(predictions) , 1)
	writer = open(file_name,"w")
	writer.write("ID,class\n")
	for i in range(len(predictions)):
		str_tmp = str( ID[i] )+"," + str(predictions[i][0]) + "\n"
		# print(str_tmp)
		writer.write(str_tmp)		
	writer.close()




if __name__ == "__main__":

	flag_read_data = True

	flag_train_autoencoder = True
	flag_save_encoder  = True
	flag_load_encoder = True
	flag_classifier_train_model = True
	flag_save_model = True

	flag_prediction = False
	flag_image_show = False


	path = sys.argv[1]
	output_model = sys.argv[2]

	'''
	read data
	'''

	if flag_read_data == True:
		(X_train_label,y_train_label,X_train_unlabel,X_test,ID_test) = read_data(path)
		# input image dimensions
		img_rows, img_cols = 32, 32
		# the CIFAR10 images are RGB
		img_channels = 3

		(training_feature , training_yhat , validation_feature , validation_yhat) = split_data(X_train_label , y_train_label , 0.1 )

		nb_classes = 10
		# convert class vectors to binary class matrices
		training_yhat = np_utils.to_categorical(training_yhat, nb_classes)
		validation_yhat = np_utils.to_categorical(validation_yhat, nb_classes)

		#normalized
		# training_feature -= 128
		# validation_feature -= 128
		training_feature /= 255.0 
		validation_feature /= 255.0

		#reshape from 3x32x32 to  1*3072
		# training_feature = training_feature.reshape(len(training_feature), np.prod( training_feature.shape[1:]) )
		# validation_feature = validation_feature.reshape(len(validation_feature), np.prod( validation_feature.shape[1:]) )


		# #unlabel data
		unlabel_feature = X_train_unlabel

		#normalized
		# unlabel_feature -= 128
		unlabel_feature /= 255.0 

		#reshape from 3x32x32 to  1*3072
		# unlabel_feature = unlabel_feature.reshape(len(unlabel_feature), np.prod( unlabel_feature.shape[1:]) )


		#testing data

		test_ID = ID_test

		test_feature = X_test
		test_feature /= 255.0





	'''
	Keras model
	'''

	if flag_train_autoencoder == True:
		print("flag_train_autoencoder == True")

		batch_size = 50
		nb_classes = 10
		nb_epoch = 50

		# input image dimensions
		img_rows, img_cols = 32, 32
		# the CIFAR10 images are RGB
		img_channels = 3

		input_dim = img_channels * img_rows * img_cols

		#this is the size of our encoded representations
		encoding_dim = 32*32

		'''
		Training model
		'''

		input_img = Input(shape=(img_channels, img_rows, img_cols))

		x = Convolution2D(16, 3, 3, activation = 'relu', border_mode = 'same')(input_img)
		x = MaxPooling2D(pool_size=(2,2), border_mode = 'same' )(x)
		x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(x)
		x = MaxPooling2D(pool_size=(2,2), border_mode = 'same')(x)
		x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(x)
		encoded = MaxPooling2D(pool_size=(2,2), border_mode = 'same' )(x)

		x = Convolution2D(8, 3, 3,activation = 'relu', border_mode = 'same')(encoded)
		x = UpSampling2D( (2,2) )(x)
		x = Convolution2D(8, 3, 3, activation = 'relu', border_mode = 'same')(x)
		x = UpSampling2D( (2,2) )(x)
		x = Convolution2D(16, 3, 3, activation='relu', border_mode= 'same')(x)
		x = UpSampling2D( (2,2) )(x)
		decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same')(x)

		autoencoder = Model(input_img, decoded)
		autoencoder.compile(loss='mse',
					optimizer='adam')


		#Let's create a separate encoder model

		#this model maps an input to its encoded representation
		encoder = Model(input=input_img, output=encoded)


		early_stop = EarlyStopping(monitor='val_loss' , patience=20, verbose=1)
		autoencoder.fit(training_feature,training_feature, nb_epoch = nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(validation_feature,validation_feature),callbacks=[early_stop] )



	'''
	Save encoder as np
	'''
	if flag_save_encoder == True:
		print("flag_save_encoder == True")
		X_train_label_autoencoder  = encoder.predict(training_feature)
		X_validation_label_autoencoder = encoder.predict(validation_feature)
		X_train_unlabel_autoencoder = encoder.predict(unlabel_feature)
		X_test_autoencoder = encoder.predict(test_feature)

		# np.save("X_train_label_autoencoder", X_train_label_autoencoder)
		# np.save("X_validation_label_autoencoder", X_validation_label_autoencoder)
		# np.save("X_train_unlabel_autoencoder", X_train_unlabel_autoencoder)
		np.save("X_test_autoencoder", X_test_autoencoder)

		tmp = np.array([0]* len(training_yhat))
		for i in range(len(training_yhat)):
			tmp[i] = np.argmax(training_yhat[i])
		training_yhat = tmp

		tmp = np.array([0]* len(validation_yhat))
		for i in range(len(validation_yhat)):
			tmp[i] = np.argmax(validation_yhat[i])
		validation_yhat = tmp


		# np.save("training_yhat_autoencoder", training_yhat)
		# np.save("validation_yhat_autoencoder", validation_yhat)


	'''
	Train classifier model
	'''
	if flag_classifier_train_model == True:

		# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
		X_train_label_autoencoder = X_train_label_autoencoder.reshape(len(X_train_label_autoencoder), np.prod(X_train_label_autoencoder.shape[1:])) 
		X_validation_label_autoencoder = X_validation_label_autoencoder.reshape(len(X_validation_label_autoencoder),np.prod(X_validation_label_autoencoder.shape[1:]))
		X_train_unlabel_autoencoder = X_train_unlabel_autoencoder.reshape(len(X_train_unlabel_autoencoder),np.prod(X_train_unlabel_autoencoder.shape[1:]))
		X_test_autoencoder = X_test_autoencoder.reshape(len(X_test_autoencoder),np.prod(X_test_autoencoder.shape[1:]))
		print(X_train_label_autoencoder.shape)
		print(X_validation_label_autoencoder.shape)
		print(X_test_autoencoder.shape)

		nb_classes = 10
		# convert class vectors to binary class matrices
		training_yhat = np_utils.to_categorical(training_yhat, nb_classes)
		validation_yhat = np_utils.to_categorical(validation_yhat, nb_classes)


		nb_classes = 10
		batch_size = 50
		nb_epoch = 500
		nb_epoch = 50


		for i in range(4):
			if i == 3:
				input_dim = X_train_label_autoencoder.shape[1]
				model = Sequential()
				model.add(Dense(256, input_dim=input_dim))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(512))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(1024))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(2048))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(nb_classes))
				model.add(Activation('softmax'))

				model.compile(loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])
				#early stop
				early_stop = EarlyStopping(monitor='val_loss' , patience=20, verbose=1)

				model.fit(X_train_label_autoencoder, training_yhat, batch_size = batch_size, nb_epoch = nb_epoch, shuffle=True, validation_data=(X_validation_label_autoencoder, validation_yhat), callbacks=[early_stop] )

			else:
				input_dim = X_train_label_autoencoder.shape[1]
				model = Sequential()
				model.add(Dense(256, input_dim=input_dim))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(512))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(1024))
				model.add(Activation('relu'))
				model.add(Dropout(0.25))

				model.add(Dense(nb_classes))
				model.add(Activation('softmax'))

				model.compile(loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])
				#early stop
				early_stop = EarlyStopping(monitor='val_loss' , patience=20, verbose=1)

				confidence_th = 0.995
				model.fit(X_train_label_autoencoder, training_yhat, batch_size = batch_size, nb_epoch = nb_epoch, shuffle=True, validation_data=(X_validation_label_autoencoder, validation_yhat), callbacks=[early_stop] )
				(X_train_label_autoencoder, training_yhat, X_train_unlabel_autoencoder) = add_unlabel_featrue2training(X_train_label_autoencoder, training_yhat, X_train_unlabel_autoencoder, model, confidence_th)
				del(model)


	if flag_save_model == True:
		model.save(output_model + '.h5')


	'''
	calculate predictionss
	'''
	if flag_prediction == True:
		print("flag_prediction == True")
		predictions_filename = "predictions.csv"
		predictions_tmp = model.predict(X_test_autoencoder)
		nb_predictions = len(predictions_tmp)

		predictions = np.array([0] * nb_predictions).astype(dtype='int')

		for i in range( nb_predictions ):
			predictions[i] = predictions_tmp[i].argmax()

		print("predictions_tmp.shape : ",predictions_tmp.shape)
		# predictionss = [round(x) for x in predictionss_tmp]

		write_file(test_ID,predictions, predictions_filename)







	'''
	image show
	'''
	if flag_image_show == True:
		print("flag_image_show == True")
		encoded_imgs = encoder.predict(test_feature)
		decoded_imgs = autoencoder.predict(test_feature)

		n = 10 #how many digits we will display
		plt.figure(1,figsize=(20, 4))
		for i in range(n):
			#display original
			ax = plt.subplot(3, n, i+1)
			# print("test_feature[i].reshape(img_channels,img_rows,img_cols)" + str(test_feature[i].reshape(img_channels,img_rows,img_cols).shape ))
			tmp =  test_feature[i].reshape(img_channels,img_rows,img_cols)
			tmp = np.dstack( (tmp[0] , tmp[1] , tmp[2])  )
			plt.imshow(tmp)
			# plt.gray()
			# plt.plot()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display reconstruction
			ax = plt.subplot(3, n, i + 1 + n)
			tmp = decoded_imgs[i].reshape(img_channels,img_rows,img_cols)
			tmp = np.dstack( (tmp[0] , tmp[1] , tmp[2]) )
			plt.imshow(tmp)
			# plt.plot()
			# plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display encoded
			ax = plt.subplot(3, n, i + 1 + n + n)
			print(encoded_imgs.shape)
			tmp = encoded_imgs[i]
			tmp[0] = tmp[0] + tmp[1] + tmp[2]
			tmp[3] = tmp[3] + tmp[4] + tmp[5]
			tmp[6] = tmp[6] + tmp[7]
			# tmp = encoded_imgs[i].reshape(img_channels,encoded_imgs[i].shape[1],encoded_imgs[i].shape[2])
			tmp = np.dstack( (tmp[0] , tmp[3] , tmp[6]) )
			plt.imshow(tmp)
			# plt.plot()
			# plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

		#plot trainning picture

		encoded_imgs = encoder.predict(training_feature)
		decoded_imgs = autoencoder.predict(training_feature)

		n = 10 #how many digits we will display
		plt.figure(2,figsize=(20, 4))
		for i in range(n):
			#display original
			ax = plt.subplot(3, n, i+1)
			# print("test_feature[i].reshape(img_channels,img_rows,img_cols)" + str(test_feature[i].reshape(img_channels,img_rows,img_cols).shape ))
			tmp =  training_feature[i].reshape(img_channels,img_rows,img_cols)
			tmp = np.dstack( (tmp[0] , tmp[1] , tmp[2])  )
			plt.imshow(tmp)
			# plt.gray()
			# plt.plot()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display reconstruction
			ax = plt.subplot(3, n, i + 1 + n)
			tmp = decoded_imgs[i].reshape(img_channels,img_rows,img_cols)
			tmp = np.dstack( (tmp[0] , tmp[1] , tmp[2]) )
			plt.imshow(tmp)
			# plt.plot()
			# plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display encoded
			ax = plt.subplot(3, n, i + 1 + n + n)
			print(encoded_imgs.shape)
			tmp = encoded_imgs[i]
			tmp[0] = tmp[0] + tmp[1] + tmp[2]
			tmp[3] = tmp[3] + tmp[4] + tmp[5]
			tmp[6] = tmp[6] + tmp[7]
			# tmp = encoded_imgs[i].reshape(img_channels,encoded_imgs[i].shape[1],encoded_imgs[i].shape[2])
			tmp = np.dstack( (tmp[0] , tmp[3] , tmp[6]) )
			plt.imshow(tmp)
			# plt.plot()
			# plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)





		plt.show()




	'''
	calculate predictionss
	'''
	# predictions_filename = "predictions.csv"
	# predictions_tmp = model.predict(test_feature)
	# nb_predictions = len(predictions_tmp)

	# predictions = np.array([0] * nb_predictions).astype(dtype='int')

	# for i in range( nb_predictions ):
	# 	predictions[i] = predictions_tmp[i].argmax()

	# print("predictions_tmp.shape : ",predictions_tmp.shape)
	# # predictionss = [round(x) for x in predictionss_tmp]

	# write_file(test_ID,predictions, predictions_filename)




