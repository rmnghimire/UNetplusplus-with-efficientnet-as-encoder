import os
import random


def create_text_file(data_dict, data_type, img_txt_path, mask_txt_path=None):
    """
    Function to create text files containing image paths and mask paths
    Args:
    data_dict: Dictionary containing image paths and mask paths
    data_type: "train" or "test" data
    img_text_path: Path of text file containing image paths.
    img_mask_path: Path of text file containing mask paths.

    Example:
    create_text_file(train_dict, "train", "train_images.txt", "train_masks.txt")
    """
    with open(img_txt_path, "w") as f:
        with open(mask_txt_path, "w") as i:
            for key, value in data_dict.items():
                f.write(value[0])
                i.write(value[1])

                f.write("\n")
                i.write("\n")

    print(
        "\n\n{0} files created.\nImages: {1}\nMasks: {2}".format(
            data_type.capitalize(), img_txt_path, mask_txt_path
        )
    )
    return


def split_data(data_dir, seed, split_ratio=[0.80, 0.10, 0.10]):
    img_dir = os.path.join("data", "images")
    mask_dir = os.path.join("data", "masks")

    data_dict = {}
    for image in os.listdir(img_dir):
        img_path = os.path.join(img_dir, image)
        mask_path = os.path.join(mask_dir, image)
        data_dict[image] = [img_path, mask_path]

    # Random Seed
    random.seed(random_seed)
    data_list = list(data_dict.items())
    random.shuffle(data_list)

    # Data split
    train_size = int(len(data_list) * split_ratio[0])
    val_size = int(len(data_list) * split_ratio[1])
    test_size = len(data_list) - train_size - val_size

    train_dict = dict(data_list[:train_size])
    val_dict = dict(data_list[train_size : train_size + val_size])
    test_dict = dict(data_list[train_size + val_size :])

    return train_dict, val_dict, test_dict


data_dir = "data"
random_seed = 7
split_ratio = [0.80, 0.10, 0.10]

train_dict, val_dict, test_dict = split_data(data_dir, random_seed, split_ratio)

train_img_txt = "data/train_images.txt"
train_mask_txt = "data/train_masks.txt"
val_img_txt = "data/val_images.txt"
val_mask_txt = "data/val_masks.txt"
test_img_txt = "data/test_images.txt"
test_mask_txt = "data/test_masks.txt"

create_text_file(train_dict, "train", train_img_txt, train_mask_txt)
create_text_file(val_dict, "val", val_img_txt, val_mask_txt)
create_text_file(test_dict, "test", test_img_txt, test_mask_txt)
