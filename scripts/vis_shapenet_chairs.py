import argparse
import os

import yaml
import torch
import einops
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from utils import Averager, poses_to_rays, volume_rendering, batched_volume_rendering
from datasets.pixelnerf_shapenet import PixelnerfShapenet


train_points_per_ray = 128


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='../../data/trans-nf/pixelnerf/srn_chairs')
    parser.add_argument('--n-support', type=int, default=1)
    parser.add_argument('--n-query', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model', '-m')
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--outdir', '-o')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset = PixelnerfShapenet(args.dataset_root, 'chairs', 'test', args.n_support, args.n_query, support_lst=[64], repeat=args.repeat)

    model = models.make(torch.load(args.model, map_location='cpu')['model'], load_sd=True)
    model.cuda()
    model.eval()

    if os.path.exists(args.outdir):
        print('outdir exists!')
        exit()
    os.makedirs(args.outdir)

    lst = list(range(len(dataset)))
    lst = lst[:100]
    for index in tqdm(lst):
        _data = dataset[index]

        data = dict()
        for k, v in _data.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor([v])
            else:
                v = v.unsqueeze(0)
            data[k] = v.cuda()
        query_imgs = data.pop('query_imgs')
        query_poses = data.pop('query_poses')

        hyponet = model(data)

        B = query_imgs.shape[0]
        H, W = query_imgs.shape[-2:]
        rays_o, rays_d = poses_to_rays(query_poses, H, W, data['focal'][0]) # rays_o / rays_d: b n h w c

        # sup_rays_o, sup_rays_d = poses_to_rays(data['support_poses'], H, W, data['focal'][0])
        # rays_o = torch.cat([sup_rays_o, rays_o], dim=1)
        # rays_d = torch.cat([sup_rays_d, rays_d], dim=1)

        with torch.no_grad():
            pred = batched_volume_rendering(
                hyponet, rays_o, rays_d,
                near=data['near'][0],
                far=data['far'][0],
                points_per_ray=train_points_per_ray,
                use_viewdirs=hyponet.use_viewdirs,
                rand=False,
                batch_size=4096,
            )

        pred = einops.rearrange(pred[0][0], 'h w c -> c h w').clamp(0, 1).cpu()

        inp = data['support_imgs'][0][0].cpu()
        gt = query_imgs[0][0].cpu()

        # ave_mse.add((pred - gt).pow(2).mean())
        merged = torch.cat([inp, pred, gt], dim=2)

        transforms.ToPILImage()(merged).save(os.path.join(args.outdir, f'{index}.png'))
