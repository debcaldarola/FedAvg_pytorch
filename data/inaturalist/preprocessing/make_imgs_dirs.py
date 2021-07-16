import shutil
import pandas as pd
import glob
import os

TARGET_TRAIN_DIR = '../data/raw/train/'
TARGET_TEST_DIR = '../data/raw/test/'

def copy_images(path, imgs, train=True):
    for i, img in enumerate(imgs):
        file_path = glob.glob(path + img + '.jpg', recursive=True)
        if os.path.exists(file_path[0]):
            print("Skipping ", file_path[0])
            continue
        if not file_path:
            print("Image not found: ", img)
            continue
        print("Copying file ", img, file_path)
        if train:
            shutil.copy(file_path[0], TARGET_TRAIN_DIR + img + '.jpg')
        else:
            shutil.copy(file_path[0], TARGET_TEST_DIR + img + '.jpg')


print("--- Extract training images ---")
f = pd.read_csv('../data/raw/inaturalist-user-120k/federated_train_user_120k.csv')
imgs = f['image_id'].tolist()
copy_images('../data/raw/train_val_images/**/', imgs)

print("--- Extract test images ---")
f = pd.read_csv('../data/raw/inaturalist-user-120k/test.csv')
imgs = f['image_id'].tolist()
copy_images('../data/raw/train_val_images/**/', imgs, train=False)





