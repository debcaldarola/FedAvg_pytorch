import pandas as pd
import os

alpha0 = pd.read_csv('cifar10_with_name/federated_train_alpha_0.00.csv')
other_files = []
filenames = []

for f in os.listdir('./cifar10/'):
    if f == 'test.csv' or f == 'federated_train_alpha_0.00.csv':
        continue
    print(f)
    csv_file = pd.read_csv('cifar10/' + f)
    csv_file['filename'] = ""
    filenames.append(f)
    other_files.append(csv_file)

print("Assigning filenames...")
for index, row in alpha0.iterrows():
    img_id = row['image_id']
    filename = row['filename']
    for file in other_files:
        file.loc[file['image_id'] == img_id, 'filename'] = filename

print("Saving new files...")
for name, new_csv in zip(filenames, other_files):
    new_csv.to_csv('cifar10_with_name/' + name)