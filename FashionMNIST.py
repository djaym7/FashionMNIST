import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import fashion mnist dataset from keras################################################
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
data = tf.keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test) = data.load_data()


#Exploring data shows 60k 28x28 train images and 10,000 test images. Labels are from 0-9 1D
print(X_train.shape) 
#print(X_train[0])
print(X_test.shape) 
print(y_train.shape) 
print(y_train)
print(y_test.shape) 
print(set(y_train))

#Visualizing images
plt.imshow(X_train[0])
print(class_names[y_train[0]])


#############################################Scaling data and reshaping

#We know that the pixel value here ranges from 0-255, so divide it by 255 and all values will be in range 0-1
X_train = X_train / 255.0
X_test = X_test/255.0
w,h=28,28
X_train = X_train.reshape(X_train.shape[0], w, h, 1)
X_test = X_test.reshape(X_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



################################Saving data for resuing it later on 
import pickle

def SaveVar(var,name):
    pickle_out = open(name,"wb")
    pickle.dump(var,pickle_out)
    pickle_out.close()
#Restoring
def ResVar(name):
    pickle_in= open(name,"rb")
    return pickle.load(pickle_in)
'''    
SaveVar(X_train,'X_train')
SaveVar(X_test,'X_test')
SaveVar(y_train,'y_train')
SaveVar(y_test,'y_test')

X_train = ResVar('X_train')
X_test = ResVar('X_test')
y_train = ResVar('y_train')
y_test = ResVar('y_test')
'''
########################################## MODEL BUILDING


#Best Model 92% accuracy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Activation,Conv2D,MaxPooling2D,LeakyReLU,BatchNormalization
from tensorflow.keras.initializers import he_uniform


model = Sequential()

model.add(Conv2D(filters=64,kernel_size=2,input_shape=(28,28,1) )) 
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=2))  
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=2))  
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) 
model.add(Dense(256))  
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10)
