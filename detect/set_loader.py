import os

import facenet

import numpy as np


def load_my_pairs(object_dir):

    pairs = []
    names = [0, 0]
    pairs_filename = os.path.join(object_dir, 'pairs.txt')
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[:]:
            pair = line.strip().split()
            names[0] = pair[0]
            names[1] = pair[2]
            pairs.append(pair)
    pairs = np.array(pairs)
    names = np.array(names)

    def add_extension(path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    path_list = []
    issame_list = []
    nrof_skipped_pairs = 0
    for pair in pairs:
        issame = False
        path0 = add_extension(os.path.join(object_dir, names[0]))
        path1 = add_extension(os.path.join(object_dir, pair[2]))
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    paths = path_list
    labels = issame_list
    labels = np.asarray(labels)

    paths_batch_1 = []
    paths_batch_2 = []
    paths_batch_1.append(paths[0])
    paths_batch_2.append(paths[1])
    paths_batch_1 = np.asarray(paths_batch_1)
    paths_batch_2 = np.asarray(paths_batch_2)
    faces1 = facenet.load_data(paths_batch_1, False, False, 160)
    faces2 = facenet.load_data(paths_batch_2, False, False, 160)

    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

    onehot_labels = []
    for index in range(len(labels)):
        if labels[index]:
            onehot_labels.append([1, 0])
        else:
            onehot_labels.append([0, 1])

    return faces1, faces2, np.array(onehot_labels), names, paths_batch_1, paths_batch_2
