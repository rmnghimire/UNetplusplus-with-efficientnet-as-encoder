from test import model, test_loader
import torch
import cv2 as cv
from torchvision import transforms
from model import NestedUNet
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
import matplotlib.pyplot as plt
import torch
from dataloader import random_seed, PolypDataset
from torchvision import transforms
import skimage.io as io
import numpy as np



transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128), Image.BICUBIC),
            transforms.ToTensor()
    ])

transform1 =  transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((128,128), Image.BICUBIC),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
)
image ="/home/raman/Downloads/UNetpluswithefficientnet/data/images/cju1cvkfwqrec0993wbp1jlzm.jpg"
img = io.imread(image)

img = transform(img)
img = torch.unsqueeze(img,0)




# image = PolypDataset(image,image_mask, input_size)
# img_t =DataLoader(image, batch_size = 1)
#
# # input, label, idx = next(iter(test_loader))
#
# # batch_t = torch.unsqueeze(imge_t,0)
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = NestedUNet(num_classes= 1, input_channels = 3, bilinear = False).to('cuda')
model_path = "/home/raman/Downloads/UNetpluswithefficientnet/experiment_test/nodeepsupervision.pth"
model.load_state_dict(torch.load(model_path, map_location = 'cuda'))
model.eval()


label = "/home/raman/Downloads/UNetpluswithefficientnet/data/masks/cju0roawvklrq0799vmjorwfv.jpg"
label = io.imread(label)

label = transform1(label)
# label = torch.unsqueeze(label,0)

def predict_mask(input, threshold):
    output = model(input.to('cuda'))
    output = torch.sigmoid(output).detach().cpu().numpy()
    pred = output > threshold

    return pred

# Threshold for prediction
threshold = 0.6

pred = predict_mask(img, threshold)
reat = pred[0,0] * 255
reat = cv.convertScaleAbs(reat)
cv.imwrite("pred.jpg",reat)

# rat = torch.squeeze(torch.from_numpy(pred),0).permute(1,2,0)
# print(rat.shape, "rat")
# rat = np.array(pred)
# print(rat.shape, "rat2")
# rat= rat*255
# print(rat.shape, "sjap[e")
# cv.imshow("aayp", rat)
cv.waitKey(0)
# rat= cv.convertScaleAbs(rat)
# cv.imwrite("filename.jpg",rat)
def visualize(input, pred):
    img = pred.squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=80, sharex=True, sharey=True)
    titles = ['Input', 'Prediction']
    image_sets = [input, pred, label]
    for i, axis in enumerate(axes):
        if (i == 0):
            img = image_sets[i].squeeze(0).permute(1, 2, 0)
            # print(img.shape, "img")

        else:
            img = image_sets[i].squeeze()
            # a = np.array((img*255))
            # a = cv.convertScaleAbs((a))
            # cv.imshow("arr",a)
        axis.imshow(img, cmap = 'gray')

        axis.set_title(titles[i])
        # plt.savefig("aaup1.jpg")


    plt.show()

# Visualise Prediction
visualize(img, pred)

