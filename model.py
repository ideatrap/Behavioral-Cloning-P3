import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
import random

def consolidate_input (file_list):
    images = []
    measurements = []

    for file in file_list:
        lines = []
        with open('../data/'+ file +'/driving_log.csv') as csvfile:
            print('Transforming: ','../data/'+ file +'/driving_log.csv')
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        for line in lines:
            center_path = line[0]
            filename = center_path.split('/')[-1]
            current_path = '../data/'+ file +'/IMG/' + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

            measurement = float(line[3])
            measurements.append(measurement)

            #randomly flip the image
            factor = random.randint(1, 2)
            if factor == 1:
                images.append(np.fliplr(image))
                measurements.append(-measurement)


            #use the side camera
            steering_adjustment = 0.229

            # left camera
            l_chance = random.randint(1, 2)
            if l_chance == 1:
                left_path = line[1]
                filename = left_path.split('/')[-1]
                current_path = '../data/' + file + '/IMG/' + filename
                image = cv2.imread(current_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

                measurement = float(line[3]) + steering_adjustment
                measurements.append(measurement)

            # right camera
            l_chance = random.randint(1, 2)
            if l_chance == 1:
                right_path = line[2]
                filename = right_path.split('/')[-1]
                current_path = '../data/' + file + '/IMG/' + filename
                image = cv2.imread(current_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

                measurement = float(line[3]) - steering_adjustment
                measurements.append(measurement)


    X_all = np.array(images)

    y_all = np.array(measurements)

    return sklearn.utils.shuffle(X_all,y_all)

    print("Data are successfully consolidated.")



'''
0 downloaded data
1 track1
2 track1 reverse
3 offroad 1
4 offroad 2
5 offroad 3
6 track 2
7 track 2 reverse
8 track 2 without break
'''
#consolidate training data
X_all, y_all = consolidate_input(['0'])

#load the training data
print("Data are suceessfully loaded.")


# split the data for training set and test set
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.20)

batch_s=32
def generator(X,Y, batch_size=batch_s):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            X_samples = X[offset:offset + batch_size]
            y_samples = Y[offset:offset + batch_size]

            images = []
            angles = []
            for x_sample,y_angle in zip(X_samples, y_samples):
                images.append(x_sample)
                angles.append(y_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=batch_s)
validation_generator = generator(X_valid, y_valid, batch_size=batch_s)

#define the model
def train_model(learningRate=1e-4):

    model = Sequential()

    #pre-processing the image


    # normalize, and center the image
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # cropping the image to retain the center part
    model.add(Cropping2D(cropping=((55, 25), (0, 0))))


    #Implement NVIDIA End-to-End model, https://arxiv.org/abs/1604.07316
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3,activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(p=0.5))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer=Adam(lr = learningRate))



    model.fit_generator(train_generator,
                        samples_per_epoch= X_train.shape[0],
                        validation_data = validation_generator,
                        nb_val_samples = X_valid.shape[0],
                        nb_epoch = 4)
    model.save('model_1.h5')


train_model()

