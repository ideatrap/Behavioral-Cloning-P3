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




