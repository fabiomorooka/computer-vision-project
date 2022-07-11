# encoding: utf-8
import cv2
import matplotlib.pyplot as plt

# Plot some examples
def plot(x, y, label_list, path):
    img_numbers = [0, 300, 2053, 6523, 10753, 13253, 17568 , 23584, len(x)-1]
    img_list = []
    labels = []

    for img_number in img_numbers:
        img_list.append(x[img_number])
        img_label_number = y[img_number]
        labels.append(label_list[img_label_number])

    plt.figure(figsize=(15, 15))
     
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(cv2.cvtColor(img_list[i], cv2.COLOR_RGB2GRAY))
        plt.title(f'Label: {labels[i]}')

    plt.savefig(path)

def plot_gray(x, y, label_list, path):
    img_numbers = [0, 300, 2053, 6523, 10753, 13253, 17568 , 23584, len(x)-1]
    img_list = []
    labels = []

    for img_number in img_numbers:
        img_list.append(x[img_number])
        img_label_number = y[img_number]
        labels.append(label_list[img_label_number])

    plt.figure(figsize=(15, 15))
     
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(img_list[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')

    plt.savefig(path)
