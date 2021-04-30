import os

checking_path = 'demo/detect'
detection_path = checking_path + '/detect.png'

if not os.path.exists(detection_path):
    exit('Adversary detection display not been generated yet!')

print('Adversary detection is successfully implemented!')

