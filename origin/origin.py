# MIT License
#
# Copyright (c) 2020 Elaine Zhong
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

from cleverhans.model import Model

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import random
from time import sleep
import matplotlib.pyplot as plt
import set_loader
import warnings

warnings.filterwarnings('ignore')
os.system('conda activate astri')


class ImageClass():
    """Stores the paths to images for a given class"""

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path) # 把path中包含的"~"和"~user"转换成用户目录
    classes = [path.split('.')[0][:-5] for path in os.listdir(path_exp) if os.path.isfile(os.path.join(path_exp, path))] # 如果'图片'是'文件'
    # ['Elon_Musk', 'Jeff_Bezos']
    nrof_classes = len(classes) # 人脸个数
    for i in range(nrof_classes):
        class_name = classes[i]
        image_paths = []
        if os.path.isdir(path_exp):
            images = os.listdir(path_exp) # ['Elon_Musk_0001.png', 'Jeff_Bezos_0001.jpeg']
            image_paths = [os.path.join(path_exp, img) for img in images]
        dataset.append(ImageClass(class_name, image_paths))
        # Class1: name = 'Elon_Musk', image_paths = ['Elon_Musk_0001.png'] | Class2: name = 'Jeff_Bezos', image_paths = ['Jeff_Bezos_0001.jpeg']
    return dataset


class InceptionResnetV1Model(Model):
    model_path = "model/20180402-114759/20180402-114759.pb"

    def __init__(self):
        super(InceptionResnetV1Model, self).__init__(scope='model')

        self.victim_embedding_input = tf.placeholder(
            tf.float32,
            shape=(None, 512))
        facenet.load_model(self.model_path)
        graph = tf.get_default_graph()
        self.face_input = graph.get_tensor_by_name("input:0")
        self.embedding_output = graph.get_tensor_by_name("embeddings:0")

    def convert_to_classifier(self):
        distance = tf.reduce_sum(
            tf.square(self.embedding_output - self.victim_embedding_input),
            axis=1)

        threshold = 0.99
        score = tf.where(
            distance > threshold,
            0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * distance / threshold)

        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))
        self.layers = []
        self.layers.append(self.softmax_output)
        self.layer_names = []
        self.layer_names.append('logits')


class Face():
    def __init__(self, object=[], name=[]):
        self.object = object
        self.name = name


def main(args):
    victim_name     = args.victim_name
    attacker_name   = args.attacker_name
    print('********************************************')
    print('attacker:', attacker_name, ' vs. victim:', victim_name)
    print('********************************************')
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            sleep(random.random())
            output_dir = os.path.expanduser(args.output_dir) # demo/origin
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            src_path, _ = os.path.split(os.path.realpath(__file__))
            # C:\Project\demo\origin | C:\Project\demo\origin\origin.py
            dataset = get_dataset(args.input_dir) # demo/photo

            minsize = 20
            threshold = [0.6, 0.7, 0.7]
            factor = 0.709

            nrof_images_total = 0
            nrof_successfully_aligned = 0
            if args.random_order:
                random.shuffle(dataset)
            for cls in dataset: # Class1: name = 'Elon_Musk', image_paths = ['Elon_Musk_0001.png'] | Class2: name = 'Jeff_Bezos', image_paths = ['Jeff_Bezos_0001.jpeg']
                output_class_dir = output_dir # demo/origin
                for image_path in cls.image_paths: # ['demo/photo\\Elon_Musk_0001.png', 'demo/photo\\Jeff_Bezos_0001.jpeg']
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0] # 'demo/photo', 'Elon_Musk_0001.png' | ('Elon_Musk_0001', '.png')
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    if args.detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                             (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                                        det_arr.append(det[index, :])
                                else:
                                    det_arr.append(np.squeeze(det))

                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                                    bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                                    bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                    nrof_successfully_aligned += 1
                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    if args.detect_multiple_faces:
                                        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                    else:
                                        output_filename_n = "{}{}".format(filename_base, file_extension)
                                    misc.imsave(output_filename_n, scaled)
                            else:
                                print('Unable to align "%s"' % image_path)

            print('Total number of images: %s' % str(divmod(nrof_images_total, 2)[0]))
            print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

            attacker_index = []
            victim_index = []
            for image in os.listdir(args.input_dir): # ['Elon_Musk_0001.png', 'Jeff_Bezos_0001.jpeg']
                if attacker_name in image:
                    attacker_index.append(str(int(image.split('.')[0][-4:])))
                elif victim_name in image:
                    victim_index.append(str(int(image.split('.')[0][-4:])))

            pairs_path  = os.path.join(output_dir, 'pairs.txt')
            origin_path = output_dir
            with open(pairs_path, "w") as names_txt:
                txt = attacker_name + ' ' + str(attacker_index[0]) + ' ' + victim_name + ' ' + str(victim_index[0])
                names_txt.write(txt)

            Faces1, Faces2, labels, names, path1, path2 = set_loader.load_my_pairs(origin_path)
            image_files = []
            image_files.append(path1[0].replace('\\', '/'))
            image_files.append(path2[0].replace('\\', '/'))

            faces1 = Face()
            faces1.object = Faces1
            faces1.name = names[0]
            faces2 = Face()
            faces2.object = Faces2
            faces2.name = names[1]

            plt.figure()
            ax = plt.subplot(2, 2, 1)
            plt.subplot(2, 2, 1)  # victim
            ax.set_title(str(names[1]))  # victim
            plt.text(80, 180, 'victim ', ha='center', fontsize=10, rotation=0, wrap=True)
            plt.axis('off')
            plt.imshow(Faces2[0])

            model = InceptionResnetV1Model()
            model.convert_to_classifier()
            real_accuracy = []

            graph = tf.get_default_graph()
            phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
            feed_dict = {model.face_input: faces2.object,
                         phase_train_placeholder: False}
            victims_embeddings = sess.run(
                model.embedding_output, feed_dict=feed_dict)

            batch_size = graph.get_tensor_by_name("batch_size:0")
            feed_dict = {model.face_input: faces1.object,
                         model.victim_embedding_input: victims_embeddings,
                         phase_train_placeholder: False,
                         batch_size: 64}
            real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
            accuracy = np.mean(
                (np.argmax(labels, axis=-1)) == (np.argmax(real_labels, axis=-1))
            )

            reco_result = []
            col = []
            if real_labels[0][0] > real_labels[0][1]:
                reco_result.append('Pass')
                col.append('G')
            elif real_labels[0][0] < real_labels[0][1]:
                reco_result.append('Fail')
                col.append('R')

            with open('demo/origin/prob.txt', "w") as names_txt:
                txt = str(real_labels[0]) + str(reco_result[0])
                names_txt.write(txt)

            ax = plt.subplot(2, 2, 2)
            plt.subplot(2, 2, 2)  # attacker
            ax.set_title(str(names[0]))  # attacker
            plt.text(80, 180, 'facenet: ' + str(reco_result[0]) + ' (' + str(real_labels[0][0]) + ')', ha='center',
                     fontsize=10, rotation=0, wrap=True, color=str(col[0]))
            plt.axis('off')
            plt.imshow(Faces1[0])

            plt.savefig(os.path.join(output_dir, 'origin.png'))
            # plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        help='Directory with unaligned images.', default='demo/photo')
    parser.add_argument('--output_dir', type=str,
                        help='Directory with aligned face thumbnails.', default='demo/origin')
    parser.add_argument('attacker_name', type=str, help='The name of the face which is to be validated.', default='Bear_Grylls')
    parser.add_argument('victim_name', type=str, help='The name of the face which has registered and is also a victim.',
                        default='Aamir_Khan')
    parser.add_argument('--model', type=str,
                        help='a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='model/20180402-114759')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))





