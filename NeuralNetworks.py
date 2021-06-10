import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn import metrics, svm, datasets, preprocessing
from sklearn.metrics import confusion_matrix
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPooling2D
from sys import exit

def read_data(file):
    f = open(file, "r")
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.rstrip("\n")            # iau liniile din fisier si sterg endline-ul
        img, label = line.split(",")        # separ id-ul imaginii de eticheta
        data.append((img, label))           # adaug in lista tuplul (imagine, eticheta)
    f.close()
    return data


def read_images(data, forTrain=True, imgsNumber=30):             # functie pentru citire a imaginilor
    noOfImgs = 0
    images = []
    labels = []
    for img, label in data:
        if noOfImgs == imgsNumber:
            break
        if forTrain:                 # verific daca imaginile sunt pentru train sau pentru validare
            filepath = "train/"
        else:
            filepath = "validation/"
        image = Image.open(filepath + img)       # iau imaginea
        np_image = np.array(image.getdata())        # transform imaginea in nparray
        images.append(np_image)                  # adaug imaginea in lista de imagini
        labels.append(int(label))            # adaug eticheta imaginii in lista de etichete
        noOfImgs += 1
    return images, labels        # returnez cele doua liste


def read_test_data(file):        # functie pentru citirea id-urilor imaginilor de test
    f = open(file, "r")
    lines = f.readlines()
    test_data = []
    for line in lines:
        line = line.rstrip("\n")
        test_data.append(line)
    f.close()
    return test_data


def read_test_images(test_data, imgsNumber=30): # functie pentru citire a imaginilor
    noOfImgs = 0
    test_images = []
    for img in test_data:            # parcurg fiecare id al imaginilor
        if noOfImgs == imgsNumber:
            break
        image = Image.open("test/" + img)        # iau imaginea
        np_image = np.array(image.getdata())    # incarc datele imaginii intr-un np array
        test_images.append(np_image)  #  salvez array-ul intr-o lista
        noOfImgs += 1
    return test_images


def write_data(images, predicted_labels):
    submission = "id,label\n"
    i = 0
    for image in images:
        submission += image + "," + str(int(predicted_labels[i])) + "\n"        # concatenez rezultatele intr-un string
        i += 1
    return submission


if __name__ == "__main__":

    # citire date train
    train_data = read_data("train.txt")
    train_images, train_labels = read_images(train_data, True, 55500)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # citire date validare
    validation_data = read_data("validation.txt")
    validation_images, validation_labels = read_images(validation_data, False, 10000)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    # citire date test
    test_data = read_test_data("test.txt")
    test_images = read_test_images(test_data, 10000)
    test_images = np.array(test_images)

    # normalizare a datelor
    train_images = train_images / 255
    validation_images = validation_images / 255
    test_images = test_images / 255


    # transformam imaginile in format 32x32
    train_images = np.reshape(train_images, (-1, 32, 32, 1))
    validation_images = np.reshape(validation_images, (-1, 32, 32, 1))
    test_images = np.reshape(test_images, (-1, 32, 32, 1))

    cnnModel = models.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),     # adaug layer convolutional
        BatchNormalization(),   #  layer de normalizare
        Dropout(0.2),   # layer de regularizare

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),  # layer de pooling pentru a reduce dimensiunea spatiala
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),      # layer complet
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(9, activation="softmax")  # layer de output cu activare softmax
    ])

    cnnModel.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])  # compilare model

    modelDetails = cnnModel.fit(train_images, train_labels, epochs=100, validation_data=(validation_images, validation_labels))     # antrenare pe datele de train

    validation_loss, validation_accuracy = cnnModel.evaluate(validation_images, validation_labels, verbose=2)       # evaluare date de validare
    print(modelDetails.history.keys())

    # accuracy
    plt.figure(1)
    plt.plot(modelDetails.history["accuracy"])
    plt.plot(modelDetails.history["val_accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "test"])
    plt.title("Overall accuracy")

    # loss
    plt.figure(2)
    plt.plot(modelDetails.history['loss'])
    plt.plot(modelDetails.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
    plt.title('Overall loss')

    plt.show()

    predicted_labels = cnnModel.predict(test_images)
    predicted_labels = [np.argmax(x) for x in predicted_labels]
    print("Acc",  validation_accuracy)

    val_predicted_labels = cnnModel.predict(validation_images)
    val_predicted_labels = [np.argmax(x) for x in val_predicted_labels] # luam label-ul maxim

    print("Matricea de confuzie: ")
    print(metrics.confusion_matrix(validation_labels, val_predicted_labels))

    submission = write_data(test_data, predicted_labels)        # incarcare date in fisier
    g = open("mysubmissionCNN.txt", "w")
    g.write(submission)
    g.close()

    # exit(1)
