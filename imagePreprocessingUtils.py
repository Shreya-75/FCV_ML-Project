import numpy as np
import cv2
import os
import random
from imutils import paths

# PATH or folder name of dataset
PATH = 'data00'  # Updated to point to the new augmented dataset
# Train and test factor. 75% is used for training. 25% for testing.
TRAIN_FACTOR = 80
# Total number of images to be processed from each folder
TOTAL_IMAGES = 1000  # Reduced to 1000
# Total number of classes to be classified
N_CLASSES = 35
# Clustering factor
CLUSTER_FACTOR = 5

# START and END are rectangle coordinates (ROI) which is displayed on camera frame. Please use them accordingly based on your system.
# mac
START = (450, 75)
END = (800, 425)

IMG_SIZE = 128


def get_canny_edge(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert from RGB to HSV
    HSVImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Finding pixels with intensity of skin
    # Finding pixels with intensity of skin
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(HSVImage, lowerBoundary, upperBoundary)

    # Blurring of grayscale using medianBlur
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 3)
    skin = cv2.bitwise_and(grayImage, grayImage, mask=skinMask)
    #cv2.imshow("masked2", skin)

    # Canny edge detection
    canny = cv2.Canny(skin, 60, 60)
    return canny, skin


def get_SURF_descriptors(canny):
    # Initialising SURF
    surf = cv2.xfeatures2d.SURF_create()
    canny = cv2.resize(canny, (256, 256))
    # Computing SURF descriptors
    kp, des = surf.detectAndCompute(canny, None)
    surf_features_image = cv2.drawKeypoints(canny, kp, None, (0, 0, 255), 4)
    return des


# Find the index of the closest central point to the each SURF descriptor.
def find_index(image, centers):
    # Calculate all distances from the image to the centers
    distances = distance.cdist([image], centers, 'euclidean')
    # Get the index of the closest center
    index = np.argmin(distances)
    return index


def get_labels():
    class_labels = []
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            print(label)
            if not (label == '.DS_Store'):
                class_labels.append(label)

    return class_labels


def get_all_gestures():
    gestures = []
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            if not (label == '.DS_Store'):
                for (subdirpath, subdirnames, images) in os.walk(os.path.join(PATH, label)):
                    # Randomly shuffle the images in each folder
                    random.shuffle(images)
                    print(label)
                    # Limit to the first 1000 images for processing
                    images = images[:TOTAL_IMAGES]
                    for image_file in images:
                        imagePath = os.path.join(PATH, label, image_file)
                        img = cv2.imread(imagePath)
                        img = cv2.resize(img, (int(IMG_SIZE * 3 / 4), int(IMG_SIZE * 3 / 4)))
                        img = cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                          cv2.LINE_AA)
                        gestures.append(img)

    print('Length of gestures: {}'.format(len(gestures)))
    im_tile = concat_tile(gestures, (5, 7))
    return im_tile


def concat_tile(im_list_2d, size):
    count = 0
    all_imgs = []
    for row in range(size[1]):
        imgs = []
        for col in range(size[0]):
            imgs.append(im_list_2d[count])
            count += 1
        all_imgs.append(imgs)
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in all_imgs])


# Example usage (if you want to visualize the gestures)
if __name__ == "__main__":
    all_gestures_image = get_all_gestures()
    cv2.imshow("All Gestures", all_gestures_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
