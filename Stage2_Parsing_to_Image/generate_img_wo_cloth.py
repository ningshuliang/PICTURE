import os
import sys
import numpy as np
import cv2


source_path = '/data4/Shuliang/xiaoice/Update/Stage2_Parsing_to_Image/experiments/Stage2/visualization/'
source_path2 = '/data4/Shuliang/CVPR_2024/Fashion_Design/sleeveless_top_skirt2/Img_Wo_Cloth1/'
save_path = '/data4/Shuliang/CVPR_2024/Fashion_Design/sleeveless_top_skirt2/Img_Wo_Cloth/'
files = os.listdir(source_path)
for name in files:
    img1 = cv2.imread(source_path+name)
    img2 = cv2.imread(source_path2+name)
    mask = (img2[:,:,0]==40)*(img2[:,:,1]==40)*(img2[:,:,2]==40)
    mask = np.stack([mask]*3,2)
    new_img = mask*40+(1-mask)*img1
    cv2.imwrite(save_path+name,new_img)