import numpy as np
import cv2
import matplotlib.pyplot as plt

def img_read(filename, mode='color'):
    if mode == 'color':
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    elif mode in ('grayscale', 'greyscale', 'gray', 'grey'):
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        return None
    

def display_image_and_histogram_plot(img_gray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_gray, cmap = "gray")
    ax1.set_title('Image')
    histogram, bins = np.histogram(img_gray.ravel(), 256, [0, 256], density= True)
    ax2.plot(histogram)
    ax2.set_title('Histogram')
    ax2.set_xlim([0, 256])
    plt.show()

def display_image_and_color_histogram_plot(img_color):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_color)
    ax1.set_title('Image')
    histogram_red, bins = np.histogram(img_color[:,:,0].ravel(), 256, [0, 256], density = True)
    histogram_green, bins = np.histogram(img_color[:,:,1].ravel(), 256, [0, 256], density = True)
    histogram_blue, bins = np.histogram(img_color[:,:,2].ravel(), 256, [0, 256], density = True)
    ax2.plot(histogram_red, color='r')
    ax2.plot(histogram_green, color='g')
    ax2.plot(histogram_blue, color='b')
    ax2.set_title('Color Histogram')
    ax2.set_xlim([0, 256])
    plt.show()

