import pandas as pd
import os

alpha0 = pd.read_csv('federated_train_alpha_0.00_with_name.csv')
other_files = []
filenames = []

for f in os.listdir('./cifar100/'):
    if f == 'test.csv' or f == 'federated_train_alpha_0.00.csv':
        continue
    print(f)
    csv_file = pd.read_csv('cifar100/' + f)
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
    file = name[:-4]
    new_csv.to_csv(file + '_with_name.csv')
