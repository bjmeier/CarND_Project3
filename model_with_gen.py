import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers import Cropping2D

correction_factor = 0.4
new_cf = False

write_to_file = False

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


def filelist(lines, write_to_file=False):
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = cwd+'/data/IMG/'+filename
            image = cv2.imread(current_path)

            # center
            if i == 0:
                measurement = float(line[3])
            # left
            if i == 1:
                measurement = float(line[3]) + correction_factor
            # right
            if i == 2:
                measurement = float(line[3]) - correction_factor

            paths.append(current_path)
            meas.append(measurement)

            imfl = np.fliplr(image)
            pa = current_path[:-4]+'_fl.jpg'
            if write_to_file:
                print(pa)
                cv2.imwrite(pa,imfl)
            paths.append(pa)
            meas.append(-1 * measurement)
    return (paths, meas)

paths = []
meas = []

if new_cf:
    paths, meas = filelist(lines, write_to_file)
    pickle.dump(paths, open("paths.p", "wb"))
    pickle.dump(meas,  open("meas.p",  "wb"))
else:
    paths = pickle.load( open("paths.p", "rb" ))
    meas  = pickle.load( open("meas.p",  "rb" ))


def gen(paths, meas, batch_size):
    W = 320
    H = 160
    N = len(meas)
    nbatch = int(N/batch_size)

    while 1:
        for b in range(nbatch):
            Xlist = []
            ylist = []
            for i in range(batch_size):
                Xlist.append(cv2.imread(paths[i + batch_size * b]))
                ylist.append(meas[i + batch_size * b])
            X = np.array(Xlist)
            y = np.array(ylist)
            yield X, y


def val_gen(paths, meas, batch_size):
    W = 320
    H = 160
    N = len(meas)
    print('valid')

    nbatch = int(N/batch_size)
    while 1:
        for b in range(nbatch):
            Xlist = []
            ylist = []
            for i in range(batch_size):
                Xlist.append(cv2.imread(paths[i + batch_size * b]))
                ylist.append(meas[i + batch_size * b])
            X = np.array(Xlist)
            y = np.array(ylist)
            print(b + 1, 'of', nbatch, 'valid', X.shape)
            yield X, y

plt.figure(0)
plt.hist(meas, bins=50)
X_train_files, X_valid_files, y_train, y_valid = train_test_split(paths, meas, test_size=0.2)

print(len(X_train_files))
print(len(X_valid_files))

bs = 32
spetrain = int(len(y_train) / bs)
spevalid = int(len(y_valid) / bs)

model = Sequential()
model.add(Cropping2D(cropping=((50, 25), (0, 0)), input_shape=(160, 320, 3)))
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

hist = model.fit_generator(
    generator=gen(X_train_files, y_train, bs),
    epochs=5,
    samples_per_epoch=spetrain,
    verbose=1,
    validation_steps=spevalid,
    validation_data=val_gen(X_valid_files, y_valid, bs),
    )

model.save('model1.h5')


plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
