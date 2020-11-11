"""
INFECTED/UNINFETCED NEURAL NETWORK TRAINING CODE
WRITTEN BY MONIKA DELEKTA
s"""
import os,cv2, itertools, numpy as np
from sklearn.cross_validation import train_test_split
from keras import backend as K
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import adam
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#convert to grayscale and resize
def imageAlter(img):
	img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img=cv2.resize(img,(35,35))
	return img

#classes
def class_assign(directories, path):
	for folder in directories:
		images=os.listdir(path+'/'+ folder)
		for img in images:
			read_img=cv2.imread(path + '/'+ folder + '/'+ img )
			list_im.append(imageAlter(read_img))
	shape = np.array(list_im, dtype = np.float32)
	#normalize between 0 and 1
	shape = shape/255
	shape= np.expand_dims(shape, axis=1) 
	return shape

def assign_labels(imShape):
	mask = imShape.shape[0]
	class_labels = np.zeros(mask, dtype='int64')
	class_labels[0:470]=0 #uninfected
	class_labels[470: ]=1 #infected
	class_vector = np_utils.to_categorical(class_labels, classes)
	hor,vert = shuffle(imShape, class_vector, random_state=2)
	horTrain, horTest, vertTrain, vertTest = train_test_split(hor, vert, test_size=0.1, random_state=2)
	return horTrain, horTest, vertTrain, vertTest

def set_training_definition(imData):
	cnnArch = Sequential()
	cnnArch.add(Convolution2D(32, 3,3, border_mode='same',input_shape=(imData[0].shape)))
	cnnArch.add(Activation('relu'))
	cnnArch.add(Convolution2D(64, 3, 3))
	cnnArch.add(Activation('relu'))
	cnnArch.add(MaxPooling2D(pool_size=(2, 2)))
	cnnArch.add(Dropout(0.25))
	cnnArch.add(Flatten())
	cnnArch.add(Dense(128))
	cnnArch.add(Activation('relu'))
	cnnArch.add(Dropout(0.5))
	cnnArch.add(Dense(classes))
	cnnArch.add(Activation('softmax'))
	cnnArch.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
	return cnnArch

def training_process(horTrain, horTest, vertTrain, vertTest, archModel):

	#aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
	#height_shift_range=0.1, shear_range=0.2, zoom_range=0.1,
	#horizontal_flip=True, fill_mode="nearest")
	#train = archModel.fit_generator(aug.flow(horTrain, vertTrain, batch_size=16), validation_data=(horTest, vertTest), steps_per_epoch=len(horTrain) // 16,
	#epochs=epoch, verbose=1)

	train = archModel.fit(horTrain, vertTrain, batch_size=8, nb_epoch=epoch, validation_data=(horTest, vertTest))
	lossAcc = archModel.evaluate(horTest, vertTest, verbose=0)
	print('Test Loss:', lossAcc[0], 'Test accuracy:', lossAcc[1])
	archModel.save('CNN_with_background.hdf5')
	print("Model Saved")


K.set_image_dim_ordering('th')
path = os.path.dirname(__file__)+'/data'
directories = os.listdir(path)
epoch=30
classes = 2
list_im=[]

imShape = class_assign(directories, path)
horTrain, horTest, vertTrain, vertTest = assign_labels(imShape)
model = set_training_definition(imShape)
training_process(horTrain, horTest, vertTrain, vertTest, model)

