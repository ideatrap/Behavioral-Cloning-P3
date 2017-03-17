import csv
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


def consolidate_input (file_list):
    images = []
    measurements = []

    for file in file_list:
        lines = []
        with open('../data/'+ file +'/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        for line in lines:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = '../data/'+ file +'/IMG/' + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)


    X_all = np.array(images)

    y_all = np.array(measurements)

    with open('x_train.pickle','wb') as output_file:
        pickle.dump(X_all, output_file)

    with open('y_train.pickle','wb') as output_file:
        pickle.dump(y_all, output_file)



'''
1 track1
2 track1 reverse
3 offroad 1
4 offroad 2
5 offroad 3
6 track 2
7 track 2 reverse
'''
#consolidate training data
consolidate_input(['1'])


#load the training data
with open('x_train.pickle', 'rb') as input_file:
    X_all = pickle.load(input_file)

with open('y_train.pickle', 'rb') as input_file:
    y_all = pickle.load(input_file)


from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
import sklearn
from sklearn.model_selection import train_test_split

# split the data for training set and test set
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.25)

batch_s=32
def generator(X,Y, batch_size=batch_s):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        #shuffle first
        sklearn.utils.shuffle(X,Y)

        for offset in range(0, num_samples, batch_size):
            X_samples = X[offset:offset + batch_size]
            y_samples = Y[offset:offset + batch_size]

            images = []
            angles = []
            for x_sample,y_angle in zip(X_samples, y_samples):
                images.append(x_sample)
                angles.append(y_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=batch_s)
validation_generator = generator(X_valid, y_valid, batch_size=batch_s)

#define the model
def train_model(learningRate=0.01):

    model = Sequential()

    #pre-processing the image

    #cropping the image to retain the center part
    #input:160x320x3, output: 90x320x3
    model.add(Cropping2D(cropping=((55, 25), (0, 0)), input_shape=(160, 320, 3)))

    #normalize, and center the image
    #input: 80x320x3
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    #Implement NVIDIA End-to-End model, https://arxiv.org/abs/1604.07316
    model.add(Conv2D(24, (5, 5), activation="relu"))
    model.add(Conv2D(36, (5, 5), activation="relu"))
    model.add(Conv2D(48, (5, 5), activation="relu"))
    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))


    model.compile(loss = 'mse', optimizer=Adam(lr = learningRate))
    model.fit_generator(train_generator,
                        steps_per_epoch= len(X_train),
                        validation_data = validation_generator,
                        validation_steps = len(X_valid),
                        epochs = 3)
    model.save('model.h5')


train_model()

