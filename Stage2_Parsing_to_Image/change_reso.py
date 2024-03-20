import numpy as np
import os
import cv2




img = cv2.imread('/data4/Shuliang/CVPR_2024/Fashion_Design/SHHQ_Parsing_to_Image/white_800.jpg')

new_img = img[:400,:400]

cv2.imwrite('jk.png',new_img)