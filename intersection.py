import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import slic, join_segmentations, watershed
from skimage.color import label2rgb
from skimage import data
import PIL
from PIL import Image
import cv2
from numpy import asarray
size = 256,256

image1 = np.array(Image.open("/home/raman/Downloads/UNetpluswithefficientnet/data/masks/cju2y8s56ymqr083541ggdsml.jpg").resize((400,400)).convert('L'))
image3 = np.array(Image.open("/home/raman/Downloads/UNetpluswithefficientnet/data/masks/cju2y26c588bo07993ksd8eoz.jpg").resize((400,400)).convert('L'))
image2= np.array(Image.open("/home/raman/Downloads/UNetpluswithefficientnet/data/masks/cju2y40d8ulqo0993q0adtgtb.jpg").resize((400,400)).convert('L'))
print(image1.shape, image2.shape)

img_bwo = cv2.bitwise_or(image1, image2, image3)
#
cv2.imshow("aay",img_bwo)
cv2.waitKey(0)
cv2.destroyWindows()
