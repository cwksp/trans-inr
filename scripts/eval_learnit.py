import argparse
import os

import yaml
import torch
import einops
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import models
from utils import Averager, poses_to_rays, volume_rendering, batched_volume_rendering
from datasets.learnit_shapenet import LearnitShapenet


render_batch_size = 4096
train_points_per_ray = 128


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='../../data/trans-nf/shapenet')
    parser.add_argument('--category', default='chairs')
    parser.add_argument('--n-support', type=int, default=1)
    parser.add_argument('--n-query', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model', '-m')
    parser.add_argument('--gpu', '-g')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset = LearnitShapenet(args.dataset_root, args.category, 'test', args.n_support, args.n_query, repeat=args.repeat)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    model = models.make(torch.load(args.model, map_location='cpu')['model'], load_sd=True)
    model.cuda()
    model.eval()

    all_mean_psnr = Averager()

    with torch.no_grad():
        for data in tqdm(loader):
            data = {k: v.cuda() for k, v in data.items()}
            query_imgs = data.pop('query_imgs')
            query_poses = data.pop('query_poses')

            hyponet = model(data)

            B = query_imgs.shape[0]
            H, W = query_imgs.shape[-2:]
            rays_o, rays_d = poses_to_rays(query_poses, H, W, data['focal'][0])

            gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')
            rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
            rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')

            pred = batched_volume_rendering(
                hyponet, rays_o, rays_d,
                near=data['near'][0],
                far=data['far'][0],
                points_per_ray=train_points_per_ray,
                use_viewdirs=hyponet.use_viewdirs,
                rand=False,
                batch_size=render_batch_size,
            )
            mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
            psnr = -10 * torch.log10(mses)
            all_mean_psnr.add(psnr.mean(), n=len(psnr))

    print(f'mean: {all_mean_psnr.item():.2f}')
