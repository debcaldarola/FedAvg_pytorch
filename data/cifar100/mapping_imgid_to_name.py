import pandas as pd

cifar100_imgs = pd.read_csv('./data/raw/orderedcifar100_labelname.csv')

# save the filenames for each class
trainfiles_for_class = {k: [] for k in range(100)}
testfiles_for_class = {k: [] for k in range(100)}

for index, row in cifar100_imgs.iterrows():
    label = int(row['2'])
    filename = row['0']
    if '/test/' in filename:
        testfiles_for_class[label].append(filename)
    else:
        trainfiles_for_class[label].append(filename)

fed_cifar100_train = pd.read_csv('cifar100/federated_train_alpha_0.00.csv')
fed_cifar100_test = pd.read_csv('cifar100/test.csv')

fed_cifar100_train['filename'] = ""
fed_cifar100_test['filename'] = ""

for index, row in fed_cifar100_train.iterrows():
    label = row['class']
    filename = trainfiles_for_class[label].pop()
    # print(label, filename)
    fed_cifar100_train.loc[index, 'filename'] = filename

print(trainfiles_for_class)  # check empty

fed_cifar100_train.to_csv('federated_train_alpha_0.00_with_name.csv', header='None')

# for index, row in fed_cifar100_test.iterrows():
#     label = row['class']
#     filename = testfiles_for_class[label].pop()
#     # print(label, filename)
#     fed_cifar100_test.loc[index, 'filename'] = filename
#
# print(testfiles_for_class)
#
# fed_cifar100_test.to_csv('test_with_name.csv', header='None')
