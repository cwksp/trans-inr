import os
import argparse
from os.path import join
from PIL import Image

import torch
from tqdm import tqdm
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--a', '-a')
parser.add_argument('--b', '-b')
parser.add_argument('--o', '-o')
args = parser.parse_args()

if os.path.exists(args.o):
    print('path exists')
    exit()
os.makedirs(args.o)

classes = [_ for _ in os.listdir(args.a) if not _.endswith('.txt')]
for c in classes:
    n = len(os.listdir(join(args.a, c))) // 3
    os.mkdir(join(args.o, c))
    for i in tqdm(range(n), desc=c):
        inp = join(args.a, c, f'{i}_input.png')
        gt = join(args.a, c, f'{i}_gt.png')
        a = join(args.a, c, f'{i}_pred.png')
        b = join(args.b, c, f'{i}_pred.png')

        inp, a, b, gt = [transforms.ToTensor()(Image.open(_)) for _ in [inp, a, b, gt]]
        mer = torch.cat([inp, a, b, gt], dim=2)

        transforms.ToPILImage()(mer).save(join(args.o, c, f'{i}.png'))
