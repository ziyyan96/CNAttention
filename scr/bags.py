import numpy as np
import random
from collections import defaultdict




def most_common(lst):
    return max(set(lst), key=lst.count)




def create_train_bags(X_df, y, bag_count, bag_size):
    X = np.array(X_df)
    idx = np.arange(X.shape[0])
    labels = np.array(y)
    num_classes = len(np.unique(labels))


    bags, bag_labels, bag_ids, inst_labels = [], [], [], []
    for _ in range(bag_count):
    # choose a majority class for this bag
        maj = int(np.random.choice(np.unique(labels), 1, replace=False))
        a = random.randint(3, min(bag_size, (labels == maj).sum()))
        idx_pos = np.random.choice(idx[labels == maj], a, replace=False)
        idx_neg = np.random.choice(idx[labels != maj], bag_size - a, replace=False)
        sel = np.concatenate([idx_pos, idx_neg])


        instances = X[sel]
        inst_y = labels[sel]


        # soft label by class fraction in bag
        bag_y = [0] * num_classes
        for c in np.unique(inst_y):
            bag_y[int(c)] = (inst_y == c).sum() / float(bag_size)


        bags.append(instances)
        bag_labels.append(bag_y)
        bag_ids.append(list(X_df.index[sel]))
        inst_labels.append(inst_y)


    # Keras expects list-of-arrays per instance position
    return list(np.swapaxes(bags, 0, 1)), np.array(bag_labels), np.array(bag_ids), np.array(inst_labels)