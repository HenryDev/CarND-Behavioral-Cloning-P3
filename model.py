import csv

import cv2
import numpy
from keras.layers import Flatten, Dense, Lambda, convolutional
from keras.models import Sequential


def augment_data(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement)
        augmented_measurements.append(measurement * -1)
    return augmented_images, augmented_measurements


def read_data():
    lines = []
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        for camera in range(3):
            source_path = line[camera]
            file_name = source_path.split('/')[-1]
            current_path = './data/img/' + file_name
            image = cv2.imread(current_path)
            images.append(image)
            steering_angle = float(line[3])
            measurements.append(steering_angle)
    augmented_images, augmented_measurements = augment_data(images, measurements)
    training_set = numpy.array(augmented_images)
    training_label = numpy.array(augmented_measurements)
    return training_set, training_label


def make_model():
    network = Sequential()
    network.add(Lambda(lambda pixel: pixel / 255 - 0.5, input_shape=(160, 320, 3)))
    network.add(convolutional.Cropping2D(cropping=((70, 25), (0, 0))))
    network.add(convolutional.Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    network.add(convolutional.Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    network.add(convolutional.Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    network.add(convolutional.Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2)))
    network.add(convolutional.Convolution2D(64, 2, 2, activation='relu', subsample=(2, 2)))
    network.add(Flatten())
    network.add(Dense(100))
    network.add(Dense(50))
    network.add(Dense(1))
    return network


model = make_model()
model.compile('adam', 'mse')
x, y = read_data()
model.fit(x, y, nb_epoch=1, validation_split=0.2)
model.save('model.h5')
