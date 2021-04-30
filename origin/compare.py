"""Performs face alignment and calculates L2 distance between the embeddings of images."""

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

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import set_loader
import matplotlib.pyplot as plt


class Face():
    def __init__(self, object=[], name=[]):
        self.object = object
        self.name = name


def main(args):
    image_files = []
    read_dir = 'demo/origin'
    pairs_file = os.path.join(read_dir, 'pairs.txt')
    with open(pairs_file, 'r') as f:
        for line in f.readlines()[:]:
            pair = line.strip().split()  # ['Bear_Grylls', '45', 'Aamir_Khan', '40']
            attacker_image = pair[0] + '_' + pair[1].zfill(4) + '.png'
            victim_image = pair[2] + '_' + pair[3].zfill(4) + '.png'
            attacker_path = os.path.join(read_dir, attacker_image)
            victim_path = os.path.join(read_dir, victim_image)
            image_files.append(attacker_path)
            image_files.append(victim_path)

    Faces1, Faces2, labels, names, path1, path2 = set_loader.load_my_pairs(read_dir)
    faces1 = Face()
    faces1.object = Faces1
    faces1.name = names[0]
    faces2 = Face()
    faces2.object = Faces2
    faces2.name = names[1]

    images = load_and_align_data(image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    # ['demo/origin/Aamir_Khan_0040.png', 'demo/origin/Bear_Grylls_0045.png']
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            # facenet.load_model(args.model)
            facenet.load_model('model/20180402-114759')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))

            recognize_result = []
            clo = []
            if dist > 1.06:
                recognize_result.append('Fail')
                clo.append('R')
            elif dist < 1.06:
                recognize_result.append('Pass')
                clo.append('G')

            dist_file = os.path.join(read_dir, 'dist.txt')
            with open(dist_file, "w") as dist_txt:
                dist_txt.write(str(round(dist, 3)) + str(recognize_result[0]))

            plt.figure()
            ax = plt.subplot(2, 2, 1)
            plt.subplot(2, 2, 1)  # victim
            ax.set_title(str(names[1]))  # victim
            plt.text(80, 180, 'victim ', ha='center', fontsize=10, rotation=0, wrap=True)
            plt.axis('off')
            plt.imshow(Faces2[0])

            ax = plt.subplot(2, 2, 2)
            plt.subplot(2, 2, 2)  # attacker
            ax.set_title(str(names[0]))  # attacker
            plt.text(80, 180, 'facenet: ' + str(recognize_result[0]) + ' (' + str(round(dist, 3)) + ')', ha='center',
                     fontsize=10, rotation=0, wrap=True, color=str(clo[0]))
            plt.axis('off')
            plt.imshow(Faces1[0])

            plt.savefig(os.path.join(read_dir, 'origin_facenet.png'))
            plt.show()

            nrof_images = len(image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('model', type=str,
    #                     help='a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # save_path = 'log/compare_two_people1.log'
    # stdout_backup = sys.stdout
    # log_file = open(save_path, "w")
    # sys.stdout = log_file
    main(parse_arguments(sys.argv[1:]))
    # log_file.close()
    # sys.stdout = stdout_backup
    # print("Now this will be presented on screen")
