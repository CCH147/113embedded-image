import cv2
import numpy as np

def label_hsv_areas(image,mask):
    
    labeled_img = np.zeros_like(image)
    labeled_img[image == 255] = 255
    road_colored = mask.copy()
    road_colored[labeled_img == 255] = [0, 255, 0]     # 將馬路區域著色為綠色
    return road_colored


def label_similar_areas(image, mask):
    fixed_color = (0, 0, 255) 
    
    # 進行連通區域標記
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # 創建輸出圖像
    colored_image = image.copy()
    
    # 為每個標記區域著色
    for label in range(1, num_labels):  # 從1開始以跳過背景(0)
        # 獲取當前標記的遮罩
        current_mask = labels == label
        colored_image[current_mask] = fixed_color
    
    return colored_image

