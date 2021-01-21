import matplotlib.pyplot as plt
import torch
from test import test_loader, model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_mask(input, threshold):
    output = model(input.to(device))
    output = torch.sigmoid(output).detach().cpu().numpy()
    pred = output > threshold

    return pred

# Threshold for prediction
threshold = 0.5

# Get test image
input, label, idx = next(iter(test_loader))

pred = predict_mask(input, threshold)


def visualize(input, pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=80, sharex=True, sharey=True)
    titles = ['Input', 'Prediction', 'GT Mask']
    image_sets = [input, pred, label]
    for i, axis in enumerate(axes):
        if (i == 0):
            img = image_sets[i].squeeze(0).permute(1, 2, 0)
        else:
            img = image_sets[i].squeeze()
        axis.imshow(img, cmap = 'gray')

        axis.set_title(titles[i])
    plt.show()
# Visualise Prediction
visualize(input, pred)