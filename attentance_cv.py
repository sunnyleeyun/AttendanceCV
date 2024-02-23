from PIL import Image
from pytesseract import Output
import pytesseract
# import numpy as np
import cv2
# import matplotlib.pyplot as plt
filename = './images/img_name_1.jpg'
image = cv2.imread(filename)

text = pytesseract.image_to_string(image)
print(text)
