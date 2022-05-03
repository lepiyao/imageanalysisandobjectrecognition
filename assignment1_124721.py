"""
Assignment 1
Master Program - Gruppe 02
Levi Hanny Santoso - 124721
Mohammad Seyedalizadeh - 124744
Sarvesh Kumar Singh - 124773
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def starting():
    """
    Preparing all the needed properties for the program
    """
    image = cv2.imread("./input_sat_image.jpg", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enhancedImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    minv = np.amin(enhancedImage)
    maxv = np.amax(enhancedImage)

    """
    Enhanced
    by using min-max value that has been taken 
    with amin and amax function from numpy
    then calculate every single "box" in the picture using the min-max calculation
    https://www.youtube.com/watch?v=pb5qEoEGy6E
    """
    for i in range(0,enhancedImage.shape[0]-1, 1):
        for j in range(0,enhancedImage.shape[1]-1, 1):
            enhancedImage[i,j] = ((enhancedImage[i,j] - minv  ) / (maxv - minv)) * 255

    """
    Binarization
    By replacing number that above 100 with 0 
    and lower than 100 with 1*255
    """
    binarymask = enhancedImage.copy()
    firstArray = np.where(binarymask < 100, 1 , binarymask)
    secondArray = np.where(firstArray > 100, 0 * 255, firstArray)
    thirdArray = np.where(secondArray == 1, 1 * 255, secondArray)
    # print("an_array3= ",an_array3)  

    """
    Create a checker array to be used to check 3x3 cube on the picture
    """
    checker = np.ones((5,5), np.uint8)
    """
    Morphological Operator 
    Opening, Closing, and Opening + Closing done with the built-in function from cv2
    https://www.youtube.com/watch?v=WQK_oOWW5Zo
    """
    openingMorph = cv2.morphologyEx(thirdArray, cv2.MORPH_OPEN, checker)
    closingMorph = cv2.morphologyEx(thirdArray, cv2.MORPH_CLOSE, checker)
    openingandclosingMorph = cv2.morphologyEx(openingMorph, cv2.MORPH_CLOSE, checker)

    """
    Overlay For Ehanced Picture with 0.5 alpha + Filtered/ Opening & Closing Image with 0.8 alpha
    """
    overlay = cv2.addWeighted(enhancedImage, 0.5, openingandclosingMorph, 0.8, 0)

    """
    Simple function to print out the images using pyplot
    """
    listTitle = ["Input", "Enhanced", "Binarization", "Filtered", "Overlay"]
    listImage = [image, enhancedImage, thirdArray, openingandclosingMorph, overlay]
    showImages(listTitle, listImage)

    # Saving the result images
    cv2.imwrite("Enhanced.jpg", enhancedImage)
    cv2.imwrite("Binarization.jpg", thirdArray)
    cv2.imwrite("Opening.jpg", openingMorph)
    cv2.imwrite("Closing.jpg", closingMorph)
    cv2.imwrite("Filtered.jpg", openingandclosingMorph)
    cv2.imwrite("Overlay.jpg", overlay)

    plt.show()

def showImages(listTitle, listImage):
    for i in range (len(listTitle)):
        if i == 0:
            plt.subplot(1,len(listTitle),i+1)
            plt.imshow(listImage[i])
            plt.title(listTitle[i])
        else:
            plt.subplot(1,len(listTitle),i+1)
            plt.imshow(listImage[i], cmap='gray')
            plt.title(listTitle[i])

if __name__ == "__main__" :
    starting()
