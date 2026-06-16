import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread(r"C:\fluttersrc\Strawberry-RUL-prediction\sample_data\raw_data\18-03-2026\cropped\frame-1_12-26-28.jpg")

# convert the image to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# extract the L, A, and B channels
l, a, b = cv2.split(lab)

# Khởi tạo CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,      # limit for contrast clipping (2.0 is a common default)
    tileGridSize=(8, 8) # size of the grid for histogram equalization (8x8 tiles)
)

# apply CLAHE on the L channel
l_clahe = clahe.apply(l)

# merge the CLAHE enhanced L channel back with A and B channels
lab_clahe = cv2.merge((l_clahe, a, b))

# convert to BGR
result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# save the result
cv2.imwrite(r"C:\fluttersrc\Strawberry-RUL-prediction\sample_data\filtered_images\output_clahe.jpg", result)

cv2.imshow("Original", img)
cv2.imshow("CLAHE", result)
cv2.waitKey(0)
cv2.destroyAllWindows()