import shutil
import pandas as pd
import glob
import os

TARGET_TRAIN_DIR = '../data/raw/train/'
TARGET_TEST_DIR = '../data/raw/test/'


### NB: move all files from subdirectories to parent (current) dir: find . -mindepth 2 -type f -print -exec mv {} . \;

def copy_images(path, imgs, train=True):
    cnt = 0
    cnt_skipped = 0
    for i, img in enumerate(imgs):
        file_path = glob.glob(path + img + '.jpg', recursive=True)
        target_path = os.path.join(TARGET_TRAIN_DIR, img + '.jpg')
        if os.path.exists(target_path):
            print("Skipping ", target_path, cnt_skipped)
            cnt_skipped += 1
            continue
        if not file_path:
            print("Image not found: ", img)
            continue
        print("Copying file ", img, cnt)
        if train:
            shutil.copy(file_path[0], TARGET_TRAIN_DIR + img + '.jpg')
        else:
            shutil.copy(file_path[0], TARGET_TEST_DIR + img + '.jpg')
        cnt += 1


print("--- Extract training images ---")
f = pd.read_csv('../data/raw/inaturalist-user-120k/federated_train_user_120k.csv')
imgs = f['image_id'].tolist()
copy_images('../data/raw/train_val_images/**/', imgs)

print("--- Extract test images ---")
f = pd.read_csv('../data/raw/inaturalist-user-120k/test.csv')
imgs = f['image_id'].tolist()
copy_images('../data/raw/train_val_images/**/', imgs, train=False)
