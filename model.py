import csv
import os
import cv2
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, Dropout
import sklearn
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt

driving_log_path = os.path.join("data", "driving_log.csv")
img_dir_path = os.path.join("data", "IMG")
ch, row, col = 3, 160, 320

# needed for knowing whether to split on '\' or '/' as Windows folders/files are separated by '\'
recorded_in_windows = True
split_char = "/"
if recorded_in_windows:
    split_char = "\\"


def get_samples(log_path):
    samples = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skips first line
        for line in reader:
            samples.append(line)
    return samples

def generator(samples, steering_correction=0.0, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(os.path.join(img_dir_path, batch_sample[0].split(split_char)[-1]))

                center_angle = float(batch_sample[3])

                images.append(center_image)
                angles.append(center_angle)

                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle*-1.0)

                # if steering correction is very small, don't use left and right camera images in training
                # this is useful for easily testing
                if abs(steering_correction) > 0.001:
                    left_image = cv2.imread(os.path.join(img_dir_path, batch_sample[1].split(split_char)[-1]))
                    right_image = cv2.imread(os.path.join(img_dir_path, batch_sample[2].split(split_char)[-1]))
                    images.append(left_image)
                    angles.append(center_angle + steering_correction)

                    images.append(right_image)
                    angles.append(center_angle + steering_correction)

            X_train = np.reshape(images, (len(images), row, col, ch))
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def train_model(nb_epoch, dropout_prob, steering_correction, output_dir):
    train_samples, validation_samples = sklearn.model_selection.train_test_split(get_samples(driving_log_path), test_size=0.2)
    # X_train, y_train = read_data(driving_log_path, img_dir_path)

    train_generator = generator(train_samples, steering_correction, batch_size=32)
    validation_generator = generator(validation_samples, steering_correction, batch_size=32)

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch)))  # standardize
    model.add(Cropping2D(cropping=((50, 15), (0, 0))))  # crop upper and lower part of image
    model.add(Dropout(dropout_prob))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(dropout_prob))
    model.add(Dense(1))

    filepath = os.path.join(output_dir, "weights-{epoch:02d}-{val_loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]

    model.compile(loss="mse", optimizer="adam")

    # generated_data_multiplier is used in order to feet correct number of samples per epoch to the fit_generator(), as generator synthesizes new data
    generated_data_multiplier = 2
    if abs(steering_correction) > 0.001:
        generated_data_multiplier = 4

    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*generated_data_multiplier,
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                         nb_epoch=nb_epoch, verbose=1, callbacks=callbacks_list)

    return model, history_object


keras_model, history_object = train_model(40, 0.3, 0, "weights")
keras_model.save("model.h5")

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('Epochs')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(os.path.join("examples", "MSE_loss.png"))

