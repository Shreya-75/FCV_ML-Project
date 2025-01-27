import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set backend to TkAgg to avoid MacOSX framework dependency
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import imagePreprocessingUtils as ipu

# Read file name for CSV
filename = raw_input('Enter the csv file name to read: ')
sub = pd.read_csv(filename)
y_pred = np.array(sub.pop('PredictedLabel'))
y_test = np.array(sub.pop('TrueLabel'))

# Load class labels
class_labels = ipu.get_labels()


def plot_confusion_matrix(cm, labels,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Reds):
    """
    Prints and plots the confusion matrix with enhanced readability.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 8))  # Adjust the figure size for better readability
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))

    # Set font size for labels and adjust rotation for readability
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize each row
        cm_text = "Confusion Matrix"
    else:
        cm_text = "Confusion Matrix"

    print(cm_text)
    print(cm)

    # Define the threshold for text color contrast and display the matrix values
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]) if normalize else int(cm[i, j]),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.tight_layout(pad=2)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix with improved visibility
plot_confusion_matrix(cnf_matrix, labels=class_labels, title='Confusion Matrix')
plt.show()
