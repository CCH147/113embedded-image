import cv2
import numpy as np
from matplotlib import pyplot as plt

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def plot_histogram(hist):
    plt.plot(hist)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def find_top_three(hist):
    top_three_indices = np.argsort(hist, axis=0)[-3:][::-1].flatten() # 找到前三大值的索引並展平 
    # 將結果轉換為可讀格式 
    top_three = [int(idx) for idx in top_three_indices] 
    return top_three
