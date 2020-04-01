from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
import os

TRAIN_PATH = 'larger_set'
TEST_PATH = 'Test'
N_CLASSES = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHANNELS = 1


def read_images():

    training_images, testing_images = np.empty((0, IMG_HEIGHT, IMG_WIDTH, CHANNELS)), np.empty((0, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    training_labels, testing_labels = np.zeros((0, N_CLASSES), dtype=int), np.zeros((0, N_CLASSES), dtype=int)

    label = 0
    classes = sorted(os.walk(TRAIN_PATH).__next__()[1])

    for c in classes:
        c_train_dir = os.path.join(TRAIN_PATH, c)
        train_walk = os.walk(c_train_dir).__next__()
        c_test_dir = os.path.join(TEST_PATH, c)
        test_walk = os.walk(c_test_dir).__next__()
        arr = np.zeros((N_CLASSES,), dtype=int)
        arr[label] = 1

        for sample in train_walk[2]:
            if sample.endswith('.jpg'):
                img = Image.open(os.path.join(c_train_dir, sample))
                training_images = np.append(training_images, np.array(img).reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)), axis=0)
                training_labels = np.append(training_labels, arr.reshape((1, N_CLASSES)), axis=0)

        for sample in test_walk[2]:
            if sample.endswith('.jpg'):
                img = Image.open(os.path.join(c_test_dir, sample))
                testing_images = np.append(testing_images, np.array(img).reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)), axis=0)
                testing_labels = np.append(testing_labels, arr.reshape((1, N_CLASSES)), axis=0)

        label += 1

    training_images, training_labels = shuffle(training_images, training_labels, random_state=0)
    testing_images, testing_labels = shuffle(testing_images, testing_labels, random_state=0)

    return (training_images, training_labels), (testing_images, testing_labels)


def build_network():
    network = models.Sequential()
    network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Conv2D(64, (3, 3), activation='relu'))
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Conv2D(64, (3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    return network


network = build_network()
network.summary()
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

(train_images, train_labels), (test_images, test_labels) = read_images()

train_images = train_images.reshape(train_images.shape).astype('float32') / 255
test_images = test_images.reshape(test_images.shape).astype('float32') / 255

network.fit(train_images, train_labels, epochs=5, batch_size=32)
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print('test accuracy: ', test_accuracy)
