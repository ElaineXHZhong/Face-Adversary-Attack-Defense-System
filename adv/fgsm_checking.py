import os

checking_path = 'demo/adv'
adversary_path = checking_path + '/adversary.png'
adv_path = checking_path + '/adv.png'
noise_path = checking_path + '/0.png'

if not os.path.exists(adversary_path):
    exit('Adversary has not been generated yet!')
if not os.path.exists(adv_path):
    exit('Adversary display has not been generated yet!')
if not os.path.exists(noise_path):
    exit('FGSM perturbation has not been generated yet!')

print('Adversary attack is successfully implemented!')

