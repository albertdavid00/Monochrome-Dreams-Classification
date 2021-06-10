import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn import metrics, svm, datasets, preprocessing
from PIL import Image
from sys import exit


def read_data(file):
    f = open(file, "r")
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.rstrip("\n")       # iau liniile din fisier si sterg endline-ul
        img, label = line.split(",")    # separ id-ul imaginii de eticheta
        data.append((img, label))       # adaug in lista tuplul (imagine, eticheta)
    f.close()
    return data


def read_images(data, forTrain=True, imgsNumber=30001):     # functie pentru citire a imaginilor
    noOfImgs = 0
    images = []
    labels = []
    for img, label in data:
        if noOfImgs == imgsNumber:
            break
        if forTrain:        # verific daca imaginile sunt pentru train sau pentru validare
            filepath = "train/"
        else:
            filepath = "validation/"
        image = Image.open(filepath + img)      # iau imaginea
        np_image = np.array(image.getdata())    # transform imaginea in nparray
        images.append(np_image)             # adaug imaginea in lista de imagini
        labels.append(int(label))           # adaug eticheta imaginii in lista de etichete
        noOfImgs += 1
    return images, labels   # returnez cele doua liste

def read_test_data(file):   # functie pentru citirea id-urilor imaginilor de test
    f = open(file, "r")
    lines = f.readlines()
    test_data = []
    for line in lines:
        line = line.rstrip("\n")
        test_data.append(line)
    f.close()
    return test_data

def read_test_images(test_data, imgsNumber= 5000):  # functie pentru citire a imaginilor
    noOfImgs = 0
    test_images = []
    for img in test_data:       # parcurg fiecare id al imaginilor
        if noOfImgs == imgsNumber:
            break
        image = Image.open("test/" + img)   # iau imaginea
        np_image = np.array(image.getdata())    # incarc datele imaginii intr-un np array
        test_images.append(np_image)    #  salvez array-ul intr-o lista
        noOfImgs += 1
    return test_images

def write_data(images, predicted_labels):
    submission = "id,label\n"
    i = 0
    for image in images:
        submission += image +  "," + str(int(predicted_labels[i])) + "\n"   # concatenez rezultatele intr-un string
        i += 1
    return submission

if __name__ == "__main__":

    # citire date train
    train_data = read_data("train.txt")
    train_images, train_labels = read_images(train_data, True, 30001)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # citire date validare
    validation_data = read_data("validation.txt")
    validation_images, validation_labels = read_images(validation_data, False, 5000)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    # citire date test
    test_data = read_test_data("test.txt")
    test_images = read_test_images(test_data, 5000)
    test_images = np.array(test_images)

    # normalizare a datelor
    train_images = train_images / 255
    validation_images = validation_images / 255
    test_images = test_images / 255

    pickedKernel = "poly"   # alegem valoare pentru kernel
    pickedC = 15            # alegem valoare pentru C
    print(pickedKernel)

    clf = svm.SVC(kernel = pickedKernel, C = pickedC, verbose= True)    # model
    clf.fit(train_images, train_labels)     # antrenare pe datele de training
    val_predicted_labels = clf.predict(validation_images)   # prezicere valori pe datele de validare
    test_predicted_labels = clf.predict(test_images)   # # prezicere valori pe datele de test

    print("Acuratete: ", metrics.accuracy_score(validation_labels, val_predicted_labels))  # acuratete pentru validare

    print("Confussion matrix: ")
    print(metrics.confusion_matrix(validation_labels, val_predicted_labels))    # matricea de confuzie

    submission = write_data(test_data, test_predicted_labels)  # salvez intr-un string predictiile
    g = open("mysubmissionSVM.txt", "w")
    g.write(submission)   # salvez in fisier
    g.close()

    exit(1)