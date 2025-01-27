import cv2
import numpy as np

def get_canny_edge(image):
    # Convert image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", grayImage)

    # Convert from RGB to HSV
    HSVImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Image", HSVImage)

    # Finding pixels with intensity of skin
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(HSVImage, lowerBoundary, upperBoundary)
    cv2.imshow("Skin Mask", skinMask)

    # Blurring and applying mask to grayscale image
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 3)
    skin = cv2.bitwise_and(grayImage, grayImage, mask=skinMask)
    cv2.imshow("Masked Grayscale (Skin Region)", skin)

    # Canny edge detection
    canny = cv2.Canny(skin, 60, 60)
    cv2.imshow("Canny Edges", canny)

    return canny, skin

def get_SURF_descriptors(canny):
    # Initialize SURF (requires OpenCV with xfeatures2d module)
    surf = cv2.xfeatures2d.SURF_create()

    # Resize canny output for SURF feature extraction
    canny = cv2.resize(canny, (256, 256))
    kp, des = surf.detectAndCompute(canny, None)

    # Draw keypoints on the canny image for visualization
    surf_features_image = cv2.drawKeypoints(canny, kp, None, (0, 0, 255), 2)
    cv2.imshow("SURF Keypoints", surf_features_image)

    return des

# Example usage:
# Load an example image
image = cv2.imread('/Users/prakashghatage/PycharmProject/FCV_ML/Indian-sign-language-recognition-master/data1/T/39.jpg')

# Process the image to get each stage output
canny, skin = get_canny_edge(image)
descriptors = get_SURF_descriptors(canny)

# Display all windows and wait for a key press to close
cv2.waitKey(0)
cv2.destroyAllWindows()
