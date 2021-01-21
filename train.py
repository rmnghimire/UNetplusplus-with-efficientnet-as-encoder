import gc
import os
from model import NestedUNet
import torch
from torch import optim
from tqdm import tqdm
from dataloader import random_seed, train_loader, val_loader, batch_size
from loss import DiceBCELoss
from model import NestedUNet

def train(model, train_dataloader, val_dataloader, batch_size, num_epochs, learning_rate, patience, model_path, device):
    """
    Function to train a u-net model for segmentation.
    model: U-Net model
    Args:
        train_dataloader: training set
        val_dataloader: validation set
        batch_size: batch_size for training.
        num_epochs: number of epochs to train.
        learning_rate: learning rate for the optimiser
        patience: number of epochs for early stopping.
        model_path: checkpoint path to store the model.
        device: CPU or GPU to train the model.

    Returns: A dictionary containing the training and validation losses.
    """

    # Loss Collection
    train_losses = []
    val_losses = []

    # Loss function
    criterion = DiceBCELoss().to(device)

    # Optimiser
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min' if model.n_classes > 1 else 'max',
    #                                                  patience=patience)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min',
                                                     patience=patience)

    # Patience count
    count = 0

    for epoch in tqdm(range(1, num_epochs + 1)):
        current_train_loss = 0.0
        current_val_loss = 0.0

        # Train model
        model.train()
        for features, labels, idx in train_dataloader:
            optimiser.zero_grad()
            features, labels = features.to(device), labels.to(device)
            output = model.forward(features)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            current_train_loss += loss.item()

            del features, labels
            gc.collect()
            torch.cuda.empty_cache()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            for features, labels, idx in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                output = model.forward(features)
                loss = criterion(output, labels)
                current_val_loss += loss.item()

                del features, labels
                gc.collect()
                torch.cuda.empty_cache()

        # Store Losses
        current_train_loss /= len(train_dataloader)
        train_losses.append(current_train_loss)

        current_val_loss /= len(val_dataloader)
        val_losses.append(current_val_loss)

        print("Epoch: {0:d} -> Train Loss: {1:0.8f} Val Loss: {2:0.8f} ".format(epoch, current_train_loss,
                                                                                current_val_loss))

        if ((epoch == 1) or (current_val_loss < best_val_loss)):

            best_val_loss = current_val_loss
            eq_train_loss = current_train_loss
            best_epoch = epoch
            count = 0

            # Save best model
            torch.save(model.state_dict(), model_path)

        # Check for patience level
        if (current_val_loss > best_val_loss):
            count += 1
            if (count == patience):
                break

    # Save best parameters
    best_model_params = {'train_losses': train_losses,
                         'val_losses': val_losses,
                         'best_val_loss': best_val_loss,
                         'eq_train_loss': eq_train_loss,
                         'best_epoch': best_epoch}

    return best_model_params

def main():
    output_dir = "experiment_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save_path = os.path.join(output_dir, "polyp_unet.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = os.path.join(output_dir, "polyp_unet1.pth")
    # Initiliase Model
    torch.manual_seed(random_seed)
    model = NestedUNet(num_classes=1,input_channels= 3,bilinear=True).to('cuda')

    # Hyperparameters
    num_epochs = 200
    learning_rate = 0.0001
    patience = 10

    # Initiliase Model
    torch.manual_seed(random_seed)
    # model = UNet(n_channels = 3, n_classes = 1, bilinear = False).to(device)
    model = NestedUNet(num_classes=1,input_channels=3,deep_supervision=True).to(device)
    # Train model
    best_model_params = train(model, train_loader, val_loader, batch_size, num_epochs,
                              learning_rate, patience, save_path, device)

    print("Training complete.")

    # Delete model to free memory
    del model, best_model_params
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()