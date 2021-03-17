import json
import pandas as pd

with open('./data/other/federated_train_alpha_0.00.json') as f:
    train_json = json.load(f)

with open('federated_train_alpha_0.00.json') as f2:
    train_json2 = json.load(f2)

mapping = {}
data1 = train_json["user_data"]
data2 = train_json2["user_data"]

for (u1, u1_data), (u2, u2_data) in zip(data1.items(), data2.items()):
    filenames = u1_data['x']
    img_ids = u2_data['x']
    mapping.update(zip(img_ids, filenames))

print("Length:", len(mapping.keys()))


# cambiare nome per altro file
with open('./data/train/federated_train_alpha_100.00.json') as f:
    file = json.load(f)

for u in file["user_data"].keys():
    for i in range(len(file["user_data"][u]['x'])):
        id = file["user_data"][u]['x'][i]
        file["user_data"][u]['x'][i] = mapping[id]

with open('federated_train_alpha_100.00.json', 'w+') as f:
    json.dump(file, f)