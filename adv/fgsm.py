import facenet

import tensorflow as tf
import numpy as np
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod

import set_loader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import argparse

from scipy import misc
import shutil
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))


class Face():
    def __init__(self, object=[], name=[]):
        self.object = object
        self.name = name


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = InceptionResnetV1Model()
            model.convert_to_classifier()

            object_dir = 'demo/origin'
            Faces1, Faces2, labels, names, path1, path2 = set_loader.load_my_pairs(object_dir)

            if not os.path.exists('demo/adv/'):
                os.makedirs('demo/adv/')
            pairs_filename = os.path.join(object_dir, 'pairs.txt')
            shutil.copyfile('demo/origin/pairs.txt', 'demo/adv/pairs.txt')

            faces1 = Face()
            faces1.object = Faces1
            faces1.name = names[0]
            faces2 = Face()
            faces2.object = Faces2
            faces2.name = names[1]

            dist = []
            reco_result = []
            dist_file = os.path.join(object_dir, 'prob.txt')
            shutil.copyfile(dist_file, 'demo/adv/prob.txt')

            with open(dist_file, 'r') as f:
                for line in f.readlines()[:]:
                    pair = line
                    dist.append(pair[:-4])
                    reco_result.append(pair[-4:])
            distance = dist[0]
            recognize = reco_result[0]

            disco = []
            if recognize == 'Fail':
                disco.append('R')
            elif recognize == 'Pass':
                disco.append('G')

            plt.figure()
            plt.subplot(2, 2, 1)
            ax = plt.subplot(2, 2, 1)
            ax.set_title(str(names[1]))
            plt.text(80, 180, 'victim ', ha='center', fontsize=10, rotation=0, wrap=True)
            plt.imshow(Faces2[0])
            plt.axis('off')

            plt.subplot(2, 2, 2)
            ax = plt.subplot(2, 2, 2)
            ax.set_title(str(names[0]))
            plt.text(80, 180, 'facenet: ' + str(recognize) + ' (' + str(distance[1:]).split(" ")[0] + ')', ha='center',
                     fontsize=10, rotation=0, wrap=True, color=str(disco[0]))
            plt.imshow(Faces1[0])
            plt.axis('off')

            graph = tf.get_default_graph()
            phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
            feed_dict = {model.face_input: faces2.object,
                         phase_train_placeholder: False}
            victims_embeddings = sess.run(
                model.embedding_output, feed_dict=feed_dict)

            steps = 1
            eps = 0.01
            alpha = eps / steps
            fgsm = FastGradientMethod(model)
            fgsm_params = {'eps': alpha,
                           'clip_min': 0.,
                           'clip_max': 1.}
            adv_x = fgsm.generate(model.face_input, **fgsm_params)

            adv = Face()
            adv.object = faces1.object
            adv.name = 'adversary'
            for i in range(steps):
                print("FGSM step " + str(i + 1))
                feed_dict = {model.face_input: adv.object,
                             model.victim_embedding_input: victims_embeddings,
                             phase_train_placeholder: False}
                adv.object = sess.run(adv_x, feed_dict=feed_dict)
            set_loader.save_images(adv, faces1, faces2)

            noise_path = 'adv/fgsm/images/noise/0.png'
            noise_image = mpimg.imread(noise_path)
            plt.subplot(2, 2, 3)
            plt.text(80, 180, 'noise ', ha='center', fontsize=10, rotation=0, wrap=True)
            plt.imshow(noise_image)
            plt.axis('off')

            batch_size = graph.get_tensor_by_name("batch_size:0")
            feed_dict = {model.face_input: faces1.object,
                         model.victim_embedding_input: victims_embeddings,
                         phase_train_placeholder: False,
                         batch_size: 64}
            real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
            print('real probability is: ', labels)
            print('facenet probability is: ', real_labels[0])

            feed_dict = {model.face_input: adv.object,
                         model.victim_embedding_input: victims_embeddings,
                         phase_train_placeholder: False,
                         batch_size: 64}
            adversarial_labels = sess.run(
                model.softmax_output, feed_dict=feed_dict)
            print('adversarial probability is : ', adversarial_labels)

            different_faces_index = np.where((np.argmax(labels, axis=-1) == 1))
            different_faces_number_accordance = (np.argmax(labels[different_faces_index], axis=-1)) == (
                np.argmax(adversarial_labels[different_faces_index], axis=-1))

            clo = []
            adv_reco_result = []
            if not different_faces_number_accordance[0]:
                adv_reco_result.append('Pass')
                clo.append('G')
            else:
                adv_reco_result.append('Fail')
                clo.append('R')

            print('attack result is: ', adv_reco_result[0])
            plt.subplot(2, 2, 4)
            plt.text(80, 180, 'attack: ' + str(adv_reco_result[0]) + ' (' + str(adversarial_labels[different_faces_index][0][0]) + ')', ha='center',
                     fontsize=10, rotation=0, wrap=True, color=str(clo[0]))
            plt.imshow(adv.object[0])
            plt.axis('off')
            with open('demo/adv/adv_prob.txt', "w") as names_txt:
                txt = str(adversarial_labels[different_faces_index][0]) + str(adv_reco_result[0])
                names_txt.write(txt)

            plt.savefig(os.path.join('demo/adv', 'adv.png'))
            # plt.show()


if __name__ == '__main__':
    main()
