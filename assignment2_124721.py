import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import ndimage

image = cv2.imread("./ampelmaennchen.png", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

enhancedImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

arrayImage = cv2.normalize(enhancedImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
print(arrayImage)

def gogFilter(sigma, xSize, ySize):
    xInput = xSize
    yInput = ySize
    
    xTemp = np.arange(-xInput/2+0.5,xInput/2)
    print("X arange = ", xTemp)
    yTemp = np.arange(-yInput/2+0.5,yInput/2)
    print("y arange = ", yTemp)
    
    XGrid,YGrid = np.meshgrid(xTemp, yTemp, sparse=True)
    
    gogKernel = np.exp(-((XGrid**2 + YGrid**2)/(2.0*sigma**2)))
    
    return gogKernel/gogKernel.sum()

def deriveImage():
    derArray = np.array([2, 1, 0, -1, -2])
    gauss = gogFilter(0.5, 1, 5)
    filterX = derArray * gauss
    print(filterX)
    """
    Transpose the filterX Value using .T
    https://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T
    """
    filterY = filterX.T
    print(filterY)
    return filterX,filterY

filterX, filterY = deriveImage()

imageX = ndimage.convolve(arrayImage, filterX)
imageY = ndimage.convolve(arrayImage, filterY)

gradMagnitude = np.sqrt(imageX**2 + imageY**2)

#https://stackoverflow.com/questions/10571874/opencv-imwrite-saving-complete-black-jpeg
arrayImageNormed = 255 * (arrayImage - arrayImage.min()) / (arrayImage.max() - arrayImage.min())
arrayImageNormed = np.array(arrayImageNormed, np.int32)
# cv2.imwrite("default.jpg", arrayImageNormed)

#croppedImage because the array size is 355,255 so we change it to 255,255
croppedImage = arrayImage[50:305, 0:255]
croppedImageX = imageX[50:305, 0:255]
croppedImageY = imageY[50:305, 0:255]

array5x5 = np.ones((5, 5))

imageXX = np.multiply(croppedImageX,croppedImageX)
imageXY = np.multiply(croppedImageX,croppedImageY)
imageYY = np.multiply(croppedImageY,croppedImageY)

imageXX = ndimage.convolve(imageXX, array5x5);
imageXY = ndimage.convolve(imageXY, array5x5);
imageYY = ndimage.convolve(imageYY, array5x5);

sizeArray = np.size(croppedImage,1),np.size(croppedImage,0)
W = np.zeros(sizeArray).astype(np.float64)
Q = np.zeros(sizeArray).astype(np.float64)

for i in range(0,croppedImage.shape[1], 1):
    for j in range(0,croppedImage.shape[0], 1):
        print("i=", i, " j=", j)
        M = ([imageXX[i,j], imageXY[i,j]],[imageXY[i,j], imageYY[i,j]])
        t = (((np.trace(M)/2)**2) - (np.linalg.det(M)))
        print("M= ", M)
        print("t= ", t)
        if t > 0:
            #Calc Cornerness
            W[i,j] = np.linalg.det(M)/np.trace(M)
            print(W[i,j])
            #Calc Roundness
            Q[i,j] = 4 * np.linalg.det(M)/(np.trace(M)**2)
            print(Q[i,j])


W_final = (W > 0.004)
Q_final = (Q > 0.5)
R = W_final*Q_final;
R = 0.5*255*R 
with np.printoptions(threshold=np.inf):
    print(W_final)

plt.subplot(2,5,1)
plt.imshow(image)
plt.title("default")

plt.subplot(2,5,2)
plt.imshow(arrayImage, cmap='gray')
plt.title("normalize")

plt.subplot(2,5,3)
plt.imshow(imageX, cmap='gray')
plt.title("Image X")

plt.subplot(2,5,4)
plt.imshow(imageY, cmap='gray')
plt.title("Image Y")

plt.subplot(2,5,5)
plt.imshow(gradMagnitude, cmap='gray')
plt.title("Gradient Magnitude")

plt.subplot(2,5,6)
plt.imshow(W, cmap='jet')
plt.title("W")

plt.subplot(2,5,7)
plt.imshow(Q, cmap='jet')
plt.title("Q")

plt.subplot(2,5,8)
plt.imshow(R, cmap='jet')
plt.title("R")

plt.show()
