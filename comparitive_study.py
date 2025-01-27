# Compatibility for Python 2.7
from __future__ import division
import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm  # Import tqdm for progress bars
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as skmetrics
import random
import imagePreprocessingUtils as ipu  # This refers to the pre-processing file you've shared

# Global lists to store labels
train_labels = []
test_labels = []

# Function to preprocess all images and extract features (same as before)
def preprocess_all_images():
    print("Preprocessing all images...")
    images_labels = []
    train_img_disc = []
    test_img_disc = []
    all_train_dis = []
    label_value = 0

    # Using tqdm to show progress for each label directory
    for (dirpath, dirnames, filenames) in os.walk(ipu.PATH):
        dirnames.sort()
        for label in tqdm(dirnames, desc="Processing Directories"):
            if label != '.DS_Store':
                for (subdirpath, subdirnames, images) in os.walk(os.path.join(ipu.PATH, label)):
                    count = 0
                    # Adding a tqdm progress bar for image processing
                    for image in tqdm(images, desc="Processing Images in Label {}".format(label), leave=False):
                        imagePath = os.path.join(ipu.PATH, label, image)
                        img = cv2.imread(imagePath)
                        if img is not None:
                            img_canny = ipu.get_canny_edge(img)[0]
                            surf_disc = ipu.get_SURF_descriptors(img_canny)

                            if count < (ipu.TOTAL_IMAGES * ipu.TRAIN_FACTOR * 0.01):
                                train_img_disc.append(surf_disc)
                                all_train_dis.extend(surf_disc)
                                train_labels.append(label_value)
                            elif count < ipu.TOTAL_IMAGES:
                                test_img_disc.append(surf_disc)
                                test_labels.append(label_value)
                            count += 1
                label_value += 1

    print("Preprocessing complete.")
    return all_train_dis, train_img_disc, test_img_disc


# K-means clustering
def mini_kmeans(k, descriptor_list):
    print("Performing MiniBatchKMeans clustering...")
    kmeans_model = MiniBatchKMeans(k)
    kmeans_model.fit(descriptor_list)
    filename = 'mini_kmeans_model.sav'
    pickle.dump(kmeans_model, open(filename, 'wb'))
    print("K-means clustering complete.")
    return kmeans_model


# Evaluation metrics
def calculate_metrics(method, label_test, label_pred):
    # 'weighted' average used for precision, recall, and F1 to better reflect individual class metrics
    accuracy = skmetrics.accuracy_score(label_test, label_pred)
    precision = skmetrics.precision_score(label_test, label_pred, average='weighted')
    recall = skmetrics.recall_score(label_test, label_pred, average='weighted')
    f1 = skmetrics.f1_score(label_test, label_pred, average='weighted')
    print("Metrics for {}: Accuracy={}, Precision={}, Recall={}, F1={}".format(method, accuracy, precision, recall, f1))
    return accuracy, precision, recall, f1


# Train models and evaluate them
def evaluate_all_models(X_train, X_test, y_train, y_test):
    print("Evaluating all models...")
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    # Using tqdm to track model evaluation progress
    for model_name, model in tqdm(models.iteritems(), desc="Training Models"):
        print("Training {}...".format(model_name))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save the model
        filename = "{}_model.sav".format(model_name)
        pickle.dump(model, open(filename, 'wb'))

        # Evaluate the model
        results[model_name] = calculate_metrics(model_name, y_test, y_pred)

    print("Model evaluation complete.")
    return results


if __name__ == "__main__":
    print("Starting the process...")

    # Step 1: Preprocess images
    all_train_dis, train_img_disc, test_img_disc = preprocess_all_images()

    # Step 2: Perform K-means clustering
    k = ipu.N_CLASSES * ipu.CLUSTER_FACTOR
    print("Clustering {} classes with factor {}.".format(ipu.N_CLASSES, ipu.CLUSTER_FACTOR))
    kmeans_model = mini_kmeans(k, np.array(all_train_dis))

    # Step 3: Visual words for train and test images
    print("Creating visual words for train and test images...")
    train_images_visual_words = [kmeans_model.predict(visual_words) for visual_words in tqdm(train_img_disc, desc="Train Images")]
    test_images_visual_words = [kmeans_model.predict(visual_words) for visual_words in tqdm(test_img_disc, desc="Test Images")]

    # Step 4: Create BOVW histograms
    print("Creating BOVW histograms...")
    bovw_train_histograms = np.array([np.bincount(visual_words, minlength=k) for visual_words in train_images_visual_words])
    bovw_test_histograms = np.array([np.bincount(visual_words, minlength=k) for visual_words in test_images_visual_words])

    # Preparing data for training
    X_train = bovw_train_histograms
    X_test = bovw_test_histograms
    Y_train = train_labels
    Y_test = test_labels

    # Shuffle data
    print("Shuffling training data...")
    buffer = list(zip(X_train, Y_train))
    random.shuffle(buffer)
    X_train, Y_train = zip(*buffer)  # Explicitly converting to lists after shuffling
    X_train = list(X_train)
    Y_train = list(Y_train)

    print("Shuffling test data...")
    buffer = list(zip(X_test, Y_test))
    random.shuffle(buffer)
    X_test, Y_test = zip(*buffer)  # Explicitly converting to lists after shuffling
    X_test = list(X_test)
    Y_test = list(Y_test)

    # Step 5: Evaluate all models and save results
    results = evaluate_all_models(X_train, X_test, Y_train, Y_test)

    # Save the comparative results
    print("Saving results...")
    with open('model_comparison_results.txt', 'w') as file:
        for model_name, metrics in results.iteritems():  # Using iteritems for Python 2.7 compatibility
            accuracy, precision, recall, f1 = metrics
            file.write("Model: {}\n".format(model_name))
            file.write("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}\n\n".format(accuracy, precision, recall, f1))

    print("Process completed successfully!")
