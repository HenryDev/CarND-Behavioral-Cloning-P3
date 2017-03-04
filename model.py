import csv
import cv2
import numpy
from keras.layers import Flatten, Dense, Lambda, convolutional, pooling
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
        source_path = line[0]
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
    network.add(convolutional.Convolution2D(6, 5, 5, activation='relu'))
    network.add(pooling.MaxPooling2D)
    network.add(convolutional.Convolution2D(6, 5, 5, activation='relu'))
    network.add(pooling.MaxPooling2D)
    network.add(Flatten())
    network.add(Dense(120))
    network.add(Dense(84))
    network.add(Dense(1))
    return network


model = make_model()
model.compile('adam', 'mse')
x, y = read_data()
model.fit(x, y, nb_epoch=2, validation_split=0.2)
model.save('model.h5')
