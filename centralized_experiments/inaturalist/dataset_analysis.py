import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE_DIR = os.path.join('..', '..', 'data', 'inaturalist', 'data', 'train')


def get_clients_info(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        nclasses_per_user = dict.fromkeys(data['users'])
        class_list_per_user = dict.fromkeys(data['users'])
        nimgs_per_user = dict.fromkeys(data['users'])
        userdata = data['user_data']

    tot_classes = []
    for client, client_data in userdata.items():
        nimgs_per_user[client] = len(client_data['x'])
        client_classes = set(client_data['y'])
        nclasses_per_user[client] = len(client_classes)
        class_list_per_user[client] = list(client_classes)
        tot_classes.extend(client_classes)

    tot_classes = len(set(tot_classes))
    return nclasses_per_user, class_list_per_user, tot_classes, nimgs_per_user


def compute_overlapping_classes(classlist1, classlist2):
    if len(classlist1) > len(classlist2):
        shortlist = classlist2
        longlist = classlist1
    else:
        shortlist = classlist1
        longlist = classlist2

    cnt = 0
    for c in shortlist:
        if c in longlist:
            cnt += 1

    return cnt


def get_distance_matrix(nclients, class_list_per_user):
    overlapping_classes = np.zeros(shape=(nclients, nclients))
    normalized_overlapping_classes = np.zeros(shape=(nclients, nclients))

    for diag, c1_id in enumerate(class_list_per_user.keys()):
        for row in range(0, nclients - diag):
            col = row + diag
            if row == col:
                normalized_overlapping_classes[row][row] = 1
                overlapping_classes[row][row] = len(class_list_per_user[c1_id])
                continue
            c2_id = list(class_list_per_user.keys())[col]
            c1_classes = class_list_per_user[c1_id]
            c2_classes = class_list_per_user[c2_id]
            n = compute_overlapping_classes(c1_classes, c2_classes)
            overlapping_classes[row][col] = n
            overlapping_classes[col][row] = n
            # print(c1_classes, c2_classes, n, n / len(set(c1_classes + c2_classes)))
            normalized_overlapping_classes[row][col] = n / len(
                set(c1_classes + c2_classes))  # number of overlapping classes over the clients' possible classes
            normalized_overlapping_classes[col][row] = normalized_overlapping_classes[row][col]

    return overlapping_classes, normalized_overlapping_classes


def plot_histogram(labels, values, title):
    x = np.arange(len(labels))
    width = 0.8
    fig, ax = plt.subplots()
    ax.bar(x=x-width, height=values, color='g')
    plt.figure(figsize=(300, 300))
    # plt.rcParams["figure.figsize"] =
    ax.set_xlabel("Clients IDs")
    ax.set_ylabel("Number of classes per client")
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(title)
    fig.tight_layout()

    # save image
    figpath = os.path.join('plots', 'dataset_analysis')
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    figname = os.path.join(figpath, 'classes_per_client_with_ticks4.png')
    plt.savefig(figname)
    plt.close()


def plot_distance_matrix(distance_matrix, labels, title='Distance among clients\' tasks according to overlapping classes'):
    # plt.matshow(distance_matrix)
    fig, ax = plt.subplots()
    im = ax.matshow(distance_matrix)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    plt.title(title)
    plt.figure(figsize=(60, 60))
    # plt.rcParams["figure.figsize"] = (40, 40)
    # save image
    figpath = os.path.join('plots', 'dataset_analysis')
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    figname = os.path.join(figpath, 'clients_distance_matrix_large4.png')
    plt.savefig(figname)
    plt.close()


def main():
    # 1 apri json con split per clients
    # 2 dict client_id - # classi -> istogramma
    # 3 dict client_id - lista classi (= set(y)) -> distance matrix tra i diversi clients contando le classi uguali = quanto si sovrappongono i clients
    # -> valore della dist matrix da 0 a 1  [(# classi in overlap)/(tot classi possibili dei due clients)]

    json_file = os.path.join(FILE_DIR, 'federated_train_user_120k.json')
    nclasses_per_user, class_list_per_user, tot_classes, nimgs_per_user = get_clients_info(json_file)
    print("--- DATASET STATISTICS ---")
    print("- Total classes: ", tot_classes)
    print("- Total clients: ", len(class_list_per_user.keys()))
    print("- Samples per client:")
    print("\t Average:", np.mean(list(nimgs_per_user.values())))
    print("\t Std deviation:", np.std(list(nimgs_per_user.values())))
    print("-------------------------")

    # Plot histogram of classes per client
    print("Plotting histogram...")
    plot_histogram(nclasses_per_user.keys(), nclasses_per_user.values(), title='iNaturalist - Number of classes per '
                                                                               'client')
    cvs_path = os.path.join('dataset_description')
    if not os.path.exists(cvs_path):
        os.makedirs(cvs_path)
    csv_name = 'nclasses_per_user.csv'
    df1 = pd.DataFrame(list(nclasses_per_user.items()), index=nclasses_per_user.keys(), columns=['user_id', 'n_classes'])
    df1.to_csv(os.path.join(cvs_path, csv_name))

    # Plot distance matrix between clients according to overlapping classes
    print("Computing distance matrix...")
    nclients = len(class_list_per_user.keys())
    distance_matrix, normalized_distance_matrix = get_distance_matrix(nclients, class_list_per_user)
    print(normalized_distance_matrix)
    plot_distance_matrix(normalized_distance_matrix, class_list_per_user.keys())

    csv_name = 'clients_distance_matrix.csv'
    df2 = pd.DataFrame(normalized_distance_matrix, index=class_list_per_user.keys(),
                       columns=class_list_per_user.keys())
    df2.to_csv(os.path.join(cvs_path, csv_name))

    csv_name = 'noverlapping_classes_matrix.csv'
    df3 = pd.DataFrame(distance_matrix, index=nclasses_per_user.keys(),
                       columns=nclasses_per_user.keys())
    df3.to_csv(os.path.join(cvs_path, csv_name))

    print("All done")


if __name__ == '__main__':
    main()
