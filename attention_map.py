import os
import numpy as np
import cv2

root_dir = os.getcwd()

img_path = 'test.jpg'

q_value = 255

score_1 = 0.7
score_2 = 1
score_3 = 0.5
img_attention_mask_array = np.zeros((240, 320), dtype=np.float16)
img_attention_mask_array[95:145, 135:185] = img_attention_mask_array[95:145, 135:185] + score_1
img_attention_mask_array[10:30, 200:220] = img_attention_mask_array[10:30, 200:220] + score_2
img_attention_mask_array[110:165, 155:205] = img_attention_mask_array[110:165, 155:205] + score_3
img_attention_mask_array = np.where(img_attention_mask_array > 1.0, 1, img_attention_mask_array)

img_attention_array = (img_attention_mask_array * q_value).astype(np.uint8)

# img_attention_array = np.zeros((240, 320), dtype=np.uint8)
# img_attention_array[95:145, 135:185] = img_attention_array[95:145, 135:185] + score_1
# img_attention_array[10:30, 200:220] = img_attention_array[10:30, 200:220] + score_2
# img_attention_array[110:165, 155:205] = img_attention_array[110:165, 155:205] + score_3

img_attention_array_color_map = cv2.applyColorMap(img_attention_array, cv2.COLORMAP_JET)

cv2.imshow("img_attention_array_color_map", img_attention_array_color_map)

img = cv2.imread(img_path)

img_color_map = cv2.applyColorMap(img, cv2.COLORMAP_JET)

attention_map = img * 0.6 + img_attention_array_color_map * 0.4
attention_map = attention_map.astype(np.uint8)

cv2.imshow("attention_map", attention_map)
cv2.imwrite('attention_map.jpg', attention_map)
# cv2.waitKey(0)
pass
