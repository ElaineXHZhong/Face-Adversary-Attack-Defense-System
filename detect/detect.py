import sys
sys.path.insert(0, 'C:/Users/ACS1/caffe/python') # 在import caffe前务必加上caffe/python的路径
import caffe
import cv2
import numpy as np
import os
from utils import get_data, get_layers, get_prob, get_identity, read_list
import set_loader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings

warnings.filterwarnings('ignore')

class Face:
    def __init__(self, object=[], name=[]):
        self.object = object
        self.name = name


def get_witnesses(dir_path, layers):
    witnesses = {}
    for f in os.listdir(dir_path):
        for line in open(dir_path + f, 'r'):
            line = line.strip().split(',')
            layer = layers.index(line.pop(0))
            if layer in witnesses:
                witnesses[layer].extend(map(int, line))
            else:
                witnesses[layer] = list(map(int, line))
    return witnesses


def weaken(x):
    return np.exp(-x / 100)


def strengthen(x):
    return 2.15 - np.exp(-x / 60)


def attribute_model(net, img_path):
    pre = 'data'
    net.blobs[pre].data[...] = get_data(img_path)

    for idx in witnesses:
        curr = vgg_layers[idx]
        post = vgg_layers[idx + 1]

        if pre == 'data':
            net.forward(end=post)
        else:
            net.forward(start=pre, end=post)

        neurons = witnesses[idx]
        attri_data = []
        for i in neurons:
            attri_data.append(np.sum(net.blobs[curr].data[0][i]))

        attri_mean = np.mean(attri_data)
        attri_std = np.std(attri_data)

        for i in range(len(net.blobs[curr].data[0])):
            if i not in neurons:
                other_data = np.sum(net.blobs[curr].data[0][i])

                if other_data > attri_mean:
                    deviation = 0
                    if attri_std != 0:
                        deviation = (other_data - attri_mean) / attri_std

                    if 'pool3' in curr:
                        tmp = net.blobs[curr].data[0][i]
                        h, w = tmp.shape
                        tmp = tmp[2:h - 2, 2:w - 2]
                        tmp = cv2.resize(tmp, (h, w), interpolation=cv2.INTER_CUBIC)
                        net.blobs[curr].data[0][i] = tmp

                    net.blobs[curr].data[0][i] *= weaken(deviation)

        for i in neurons:
            deviation = 0
            if attri_std != 0:
                deviation = abs(np.sum(net.blobs[curr].data[0][i]) - np.min(attri_data)) / attri_std
            net.blobs[curr].data[0][i] *= strengthen(deviation)

        pre = post

        net.forward(start=pre)
    return net.blobs['prob'].data[0].copy()


attack_path = 'demo/adv/'
caffe.set_mode_cpu()
vgg_root = 'model/vgg_face_caffe/'
vgg_deploy = vgg_root + 'VGG_FACE_deploy.prototxt'
vgg_weight = vgg_root + 'VGG_FACE.caffemodel'
vgg_net = caffe.Net(vgg_deploy, vgg_weight, caffe.TEST)
vgg_names = read_list(vgg_root + 'names.txt')
vgg_layers = get_layers(vgg_net)
witnesses = get_witnesses('detect/witnesses/', vgg_layers)
adv_count = 0
pairs_filename = os.path.join(attack_path, 'pairs.txt')
pairs = []
names = [0, 0]
with open(pairs_filename, 'r') as f:
    for line in f.readlines()[:]:
        pair = line.strip().split()
        names[0] = pair[0]
        names[1] = pair[2]
        pairs.append(pair)
pairs = pairs[0]

victim_path = attack_path + names[1] + '.png'
attacker_path = attack_path + 'adversary.png'
print('victim path: ', victim_path)
print('attacker path: ', attacker_path)

victim_id = []
attacker_id = []

victim_prob_ori = get_prob(vgg_net, victim_path)
victim_prob_a = attribute_model(vgg_net, victim_path)
victim_id_ori = np.argmax(victim_prob_ori)
victim_id_a = np.argmax(victim_prob_a)
victim_id.append(victim_id_ori)
victim_id.append(victim_id_a)

attacker_prob_ori = get_prob(vgg_net, attacker_path)
attacker_prob_a = attribute_model(vgg_net, attacker_path)
attacker_id_ori = np.argmax(attacker_prob_ori)
attacker_id_a = np.argmax(attacker_prob_a)
attacker_id.append(attacker_id_ori)
attacker_id.append(attacker_id_a)

col = []
label = []
if victim_id_a == attacker_id_a:
    label.append('Real')
    col.append('G')
elif victim_id_a != attacker_id_a:
    label.append('Adversary')
    col.append('R')
Faces1, Faces2, labels, names, path1, path2 = set_loader.load_my_pairs(attack_path)
faces1 = Face()
faces1.object = Faces1
faces1.name = names[0]
faces2 = Face()
faces2.object = Faces2
faces2.name = names[1]

plt.figure()
plt.subplot(2, 2, 1)
ax = plt.subplot(2, 2, 1)
ax.set_title(str(names[1]))
plt.text(80, 180, 'victim ', ha='center', fontsize=10, rotation=0, wrap=True)
plt.imshow(Faces2[0])
plt.axis('off')

dist = []
reco_result = []
with open('demo/adv/prob.txt', 'r') as f:
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

plt.subplot(2, 2, 2)
ax = plt.subplot(2, 2, 2)
ax.set_title(str(names[0]))
plt.text(80, 180, 'facenet: ' + str(recognize) + ' (' + str(distance[1:]).split(" ")[0] + ')', ha='center',
         fontsize=10, rotation=0, wrap=True, color=str(disco[0]))
plt.imshow(Faces1[0])
plt.axis('off')

noise_path = 'demo/adv/0.png'
noise_image = mpimg.imread(noise_path)
plt.subplot(2, 2, 3)
plt.text(80, 180, 'noise ', ha='center', fontsize=10, rotation=0, wrap=True)
plt.imshow(noise_image)
plt.axis('off')

adversary_path = 'demo/adv/adversary.png'
adversary_image = mpimg.imread(adversary_path)
plt.subplot(2, 2, 4)
plt.text(80, 180, 'detect: ' + str(label[0]), ha='center', fontsize=10, rotation=0, wrap=True, color=str(col[0]))
plt.imshow(adversary_image)
plt.axis('off')

if not os.path.exists('demo/detect/'):
    os.makedirs('demo/detect/')

with open('demo/detect/detect_result.txt', "w") as names_txt:
    txt1 = 'a ' + str(attacker_id_ori) + ' ' + str(attacker_id_a)
    txt2 = 'v ' + str(victim_id_ori) + ' ' + str(victim_id_a)
    txt = txt1 + ' ' + txt2
    names_txt.write(txt)

plt.savefig(os.path.join('demo/detect', 'detect.png'))
# plt.show()
