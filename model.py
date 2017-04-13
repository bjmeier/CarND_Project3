import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers import Dropout

correction_factor = 0.9

lines = []

cwd = os.getcwd()

with open(cwd+'/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
measurement = 0.0
current_path = ''

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = cwd+'/data/IMG/'+filename
    image = cv2.imread(current_path)
    images.append(image)

    # center
    if i == 0:
        measurement = float(line[3])
    # left
    if i == 1:
        measurement = float(line[3]) + correction_factor
    # right
    if i == 2:
        measurement = float(line[3]) - correction_factor

    measurements.append(measurement)


augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(-1 * measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(X_train, y_train,
                 validation_split=0.05,
                 shuffle=True,
                 batch_size=32,
                 epochs=2,
                 verbose=1)

model.save('model.h5')

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
