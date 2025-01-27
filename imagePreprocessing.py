# imports
from __future__ import print_function
import numpy as np
import cv2
import os
import random
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import sklearn.metrics as skmetrics
import imagePreprocessingUtils as ipu

train_labels = []
test_labels = []


def preprocess_all_images():
    images_labels = []
    train_img_disc = []
    test_img_disc = []
    all_train_dis = []
    label_value = 0

    for (dirpath, dirnames, filenames) in os.walk(ipu.PATH):
        dirnames.sort()
        for label in dirnames:
            if label != '.DS_Store':
                for (subdirpath, subdirnames, images) in os.walk(os.path.join(ipu.PATH, label)):
                    count = 0
                    for image in images:
                        imagePath = os.path.join(ipu.PATH, label, image)
                        img = cv2.imread(imagePath)
                        if img is not None:
                            img_canny = ipu.get_canny_edge(img)[0]
                            surf_disc = ipu.get_SURF_descriptors(img_canny)

                            if count < (ipu.TOTAL_IMAGES * ipu.TRAIN_FACTOR * 0.01):
                                print('Train: Label is {} and Count is {}'.format(label, count))
                                train_img_disc.append(surf_disc)
                                all_train_dis.extend(surf_disc)
                                train_labels.append(label_value)
                            elif count < ipu.TOTAL_IMAGES:
                                print('Test: Label is {} and Count is {}'.format(label, count))
                                test_img_disc.append(surf_disc)
                                test_labels.append(label_value)
                            count += 1

                label_value += 1

    print('Length of train features:', len(train_img_disc))
    print('Length of test features:', len(test_img_disc))
    print('Length of all train descriptors:', len(all_train_dis))
    return all_train_dis, train_img_disc, test_img_disc


def kmeans(k, descriptor_list):
    print('K-Means started.')
    print('{} descriptors before clustering'.format(descriptor_list.shape[0]))
    kmeans_model = KMeans(k)
    kmeans_model.fit(descriptor_list)
    visual_words = kmeans_model.cluster_centers_
    return visual_words, kmeans_model


def mini_kmeans(k, descriptor_list):
    print('Mini Batch K-Means started.')
    print('{} descriptors before clustering'.format(descriptor_list.shape[0]))
    kmeans_model = MiniBatchKMeans(k)
    kmeans_model.fit(descriptor_list)
    print('Mini Batch K-Means trained to get visual words.')
    filename = 'mini_kmeans_model.sav'
    pickle.dump(kmeans_model, open(filename, 'wb'))
    return kmeans_model


def get_histograms(descriptors_by_class, visual_words, cluster_model):
    histograms_by_class = {}
    for label, images_descriptors in descriptors_by_class.items():
        print('Label: {}'.format(label))
        histograms = []
        for each_image_descriptors in images_descriptors:
            raw_words = cluster_model.predict(each_image_descriptors)
            hist = np.bincount(raw_words, minlength=len(visual_words))
            histograms.append(hist)
        histograms_by_class[label] = histograms
    print('Histograms successfully created for {} classes.'.format(len(histograms_by_class)))
    return histograms_by_class


def data_split(data_dictionary):
    X = []
    Y = []
    for key, values in data_dictionary.items():
        for value in values:
            X.append(value)
            Y.append(key)
    return X, Y


def predict_svm(X_train, X_test, y_train, y_test):
    svc = SVC(kernel='linear')
    print("Support Vector Machine started.")
    svc.fit(X_train, y_train)
    filename = 'svm_model.sav'
    pickle.dump(svc, open(filename, 'wb'))
    y_pred = svc.predict(X_test)
    np.savetxt('submission_svm.csv', np.c_[range(1, len(y_test) + 1), y_pred, y_test], delimiter=',',
               header='ImageId,PredictedLabel,TrueLabel', comments='', fmt='%d')
    calculate_metrics("SVM", y_test, y_pred)


def calculate_metrics(method, label_test, label_pred):
    # Calculating and printing metrics with 'weighted' average for precision, recall, and F1 to get varied values
    print("Accuracy score for ", method, skmetrics.accuracy_score(label_test, label_pred))
    print("Precision score for ", method, skmetrics.precision_score(label_test, label_pred, average='weighted'))
    print("F1 score for ", method, skmetrics.f1_score(label_test, label_pred, average='weighted'))
    print("Recall score for ", method, skmetrics.recall_score(label_test, label_pred, average='weighted'))



### STEP:1 Surf descriptors for all train and test images with class separation
all_train_dis, train_img_disc, test_img_disc = preprocess_all_images()

### STEP:2 MINI K-MEANS
mini_kmeans_model = mini_kmeans(ipu.N_CLASSES * ipu.CLUSTER_FACTOR, np.array(all_train_dis))

### Collecting VISUAL WORDS for all images (train, test)
print('Collecting visual words for train .....')
train_images_visual_words = [mini_kmeans_model.predict(visual_words) for visual_words in train_img_disc]
print('Visual words for train data collected. Length is {}'.format(len(train_images_visual_words)))

print('Collecting visual words for test .....')
test_images_visual_words = [mini_kmeans_model.predict(visual_words) for visual_words in test_img_disc]
print('Visual words for test data collected. Length is {}'.format(len(test_images_visual_words)))

### STEP:3 HISTOGRAMS (finding the occurrence of each visual word of images in total words)
print('Calculating Histograms for train...')
bovw_train_histograms = np.array(
    [np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR) for visual_words in
     train_images_visual_words])
print('Train histograms are collected. Length: {} '.format(len(bovw_train_histograms)))

print('Calculating Histograms for test...')
bovw_test_histograms = np.array(
    [np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR) for visual_words in
     test_images_visual_words])
print('Test histograms are collected. Length: {} '.format(len(bovw_test_histograms)))

print('Each histogram length is: {}'.format(len(bovw_train_histograms[0])))

# Preparing for training SVM
X_train = bovw_train_histograms
X_test = bovw_test_histograms
Y_train = train_labels
Y_test = test_labels

### Shuffling
buffer = list(zip(X_train, Y_train))
random.shuffle(buffer)
X_train, Y_train = zip(*buffer)

buffer = list(zip(X_test, Y_test))
random.shuffle(buffer)
X_test, Y_test = zip(*buffer)

print('Length of X-train: {} '.format(len(X_train)))
print('Length of Y-train: {} '.format(len(Y_train)))
print('Length of X-test: {} '.format(len(X_test)))
print('Length of Y-test: {} '.format(len(Y_test)))

predict_svm(X_train, X_test, Y_train, Y_test)
