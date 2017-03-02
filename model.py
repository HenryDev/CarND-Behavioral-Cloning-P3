import csv
import cv2
import numpy
from keras.layers import Flatten, Dense, Input
from keras.models import Model

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
    measurement = float(line[3])
    measurements.append(measurement)
training_set = numpy.array(images)
training_label = numpy.array(measurements)

incoming = Input((160, 320, 3))
flattened = Flatten()(incoming)
fully_connected = Dense(1)(flattened)
model = Model(incoming, fully_connected)
model.compile('adam', 'mse')
model.fit(training_set, training_label, nb_epoch=3, validation_split=0.2)
model.save('model.h5')
