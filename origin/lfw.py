"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                              distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far


def get_paths(lfw_dir,
              pairs):  # lfw_dir = testset_path = "C:/Project/Face/cleverhans/cleverhans/datasets/lfw_mtcnnpy_160"
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:  # 6000个pair
        if len(pair) == 3:  # ['Anders_Fogh_Rasmussen', '1', '4'] '/lfw_mtcnnpy_160'
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(
                pair[1])))  # '\lfw_mtcnnpy_160\Zhu_Rongji\Zhu_Rongji_0001.png'
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(
                pair[2])))  # '\lfw_mtcnnpy_160\Zhu_Rongji\Zhu_Rongji_0003.png'
            issame = True
        elif len(pair) == 4:  # ['Abel_Pacheco', '2', 'Jean-Francois_Lemounier', '1']
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(
                pair[1])))  # '\lfw_mtcnnpy_160\Slobodan_Milosevic\Slobodan_Milosevic_0002.png'
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(
                pair[3])))  # '\lfw_mtcnnpy_160\Sok_An\Sok_An_0001.png'
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            # np.shape(path_list) = (12000,)
            issame_list.append(issame)
            # np.shape(issame_list) = (6000,)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list
    # path_list:['/lfw_mtcnnpy_160/Elaine/Elaine_0001.png','/lfw_mtcnnpy_160/Elaine/Elaine_0003.png'] (12000,)
    # issame_list: [True, True, True, ...] (6000,)


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:  # line = 'Slobodan_Milosevic      2       Sok_An  1'
            pair = line.strip().split()  # pair = ['Slobodan_Milosevic', '2', 'Sok_An', '1'] 或者这种形式 ['Wang_Yi', '1', '2']
            pairs.append(
                pair)  # np.shape(pairs) = (6000,) [...['Shane_Loux', '1', 'Val_Ackerman', '1'], ['Shawn_Marion', '1', 'Shirley_Jones', '1'], ['Slobodan_Milosevic', '2', 'Sok_An', '1']]
    return np.array(
        pairs)  # np.shape(np.array(pairs)) = (6000,) [list(['Abel_Pacheco', '1', '4']) list(['Akhmed_Zakayev', '1', '3']) ...]
    # list(['Abel_Pacheco', '1', '4']) = ['Abel_Pacheco', '1', '4']
    # 使用array函数将tuple和list转为array