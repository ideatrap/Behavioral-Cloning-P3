import csv
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


###
### for testing ###
# should be deleted at submission
###

'''
with open('test_img.pickle', 'rb') as input_file:
    img = pickle.load(input_file)
'''


def display(img):
    plt.imshow(img.squeeze(), cmap='gray')
    plt.show()



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


    X_train = np.array(images)

    y_train = np.array(measurements)

    with open('x_train.pickle','wb') as output_file:
        pickle.dump(X_train, output_file)

    with open('y_train.pickle','wb') as output_file:
        pickle.dump(y_train, output_file)



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
#consolidate_input(['3','4'])




#load the training data
with open('x_train.pickle', 'rb') as input_file:
    X_train = pickle.load(input_file)

with open('y_train.pickle', 'rb') as input_file:
    y_train = pickle.load(input_file)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

def train_model():

    model = Sequential()

    #pre-processing the image

    #cropping the image to retain the center part
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    #normalize, and center the image
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

    #TODO send data to gcloud
    #TODO build a network with Kera and Generators, and feature extraction
    #TODO collect more data
    #TODO use generator to process and train the image on the fly

    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
    model.save('model.h5')


train_model()