import os

path_exp = os.path.expanduser('demo/photo') # 把path中包含的"~"和"~user"转换成用户目录
images = os.listdir(path_exp)
print(images)
# ['Elon_Musk', 'Jeff_Bezos'] | demo\photo\Elon_Musk_0001.png demo\photo\Jeff_Bezos_0001.jpeg