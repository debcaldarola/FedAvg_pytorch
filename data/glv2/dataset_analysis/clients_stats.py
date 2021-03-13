import pandas as pd
import os
import statistics

def main():
    df = pd.read_csv(os.path.join('..', 'landmarks-user-160k', 'federated_train.csv'))
    users = set(df['user_id'].tolist())
    n_users = len(users)
    print("Total users: {:d}".format(n_users))

    imgs_per_user = dict.fromkeys(users)

    for u in users:
        n = len(df[df['user_id'] == u]['image_id'])
        imgs_per_user[u] = n

    print("Avg of images per user:", statistics.mean(imgs_per_user.values()))
    print("Stdev of images per user:", statistics.stdev(imgs_per_user.values()))

if __name__ == '__main__':
    main()