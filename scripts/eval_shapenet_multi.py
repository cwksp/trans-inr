import argparse
import os

import yaml
import torch
import einops
import skimage.metrics
import lpips
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import models
from utils import Averager, poses_to_rays, volume_rendering, batched_volume_rendering
from datasets.pixelnerf_dvr import PixelnerfDvr


train_points_per_ray = 128


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='../../data/trans-nf/pixelnerf/NMR_Dataset')
    parser.add_argument('--n-support', type=int, default=1)
    parser.add_argument('--n-query', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model', '-m')
    parser.add_argument('--gpu', '-g')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(os.path.join(args.dataset_root, 'metadata.yaml'), 'r') as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    dataset = PixelnerfDvr('shapenet', args.dataset_root, 'test', args.n_support, args.n_query, repeat=args.repeat, retcat=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    model = models.make(torch.load(args.model, map_location='cpu')['model'], load_sd=True)
    model.cuda()
    model.eval()

    C = len(dataset.cats)
    cats_psnr = {c: Averager() for c in range(C)}
    all_mean_psnr = Averager()
    cats_ssim = {c: Averager() for c in range(C)}
    all_mean_ssim = Averager()
    cats_lpips = {c: Averager() for c in range(C)}
    all_mean_lpips = Averager()

    lpips_vgg = lpips.LPIPS(net="vgg").cuda()

    with torch.no_grad():
        pbar = tqdm(loader)
        for data in pbar:
            data = {k: v.cuda() for k, v in data.items()}
            query_imgs = data.pop('query_imgs')
            query_poses = data.pop('query_poses')
            cats = data.pop('cat')

            hyponet = model(data)

            B, N = query_imgs.shape[:2]
            H, W = query_imgs.shape[-2:]
            rays_o, rays_d = poses_to_rays(query_poses, H, W, data['query_focals'])

            gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')
            rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
            rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')

            pred = volume_rendering(
                hyponet, rays_o, rays_d,
                near=data['near'][0],
                far=data['far'][0],
                points_per_ray=train_points_per_ray,
                use_viewdirs=hyponet.use_viewdirs,
                rand=False,
            )

            pred = pred.view(B * N, H, W, 3)
            gt = gt.view(B * N, H, W, 3)

            v = lpips_vgg(pred.permute(0, 3, 1, 2) * 2 - 1,
                          gt.permute(0, 3, 1, 2) * 2 - 1)
            v = v.view(v.shape[0], -1).mean(dim=-1)

            all_mean_lpips.add(v.mean().item(), n=B*N)
            for i in range(B * N):
                cats_lpips[cats[i // N].item()].add(v[i].item())

            imgs_pred = pred.cpu().numpy()
            imgs_gt = gt.cpu().numpy()
            for i in range(B * N):
                psnr = skimage.metrics.peak_signal_noise_ratio(
                    imgs_pred[i], imgs_gt[i], data_range=1
                )
                all_mean_psnr.add(psnr)
                cats_psnr[cats[i // N].item()].add(psnr)

                ssim = skimage.metrics.structural_similarity(
                    imgs_pred[i], imgs_gt[i], multichannel=True, data_range=1
                )
                all_mean_ssim.add(ssim)
                cats_ssim[cats[i // N].item()].add(ssim)

            pbar.set_description(desc=''
                f'psnr: {all_mean_psnr.item():.2f} '
                f'ssim: {all_mean_ssim.item():.3f} '
                f'lpips: {all_mean_lpips.item():.3f}')

    print(f'psnr: {all_mean_psnr.item():.2f}')
    lst = []
    for i in range(C):
        str_id = dataset.cats[i]
        name = metadata[str_id]['name']
        psnr = cats_psnr[i].item()
        lst.append(psnr)
        print(f'{name}', end=' ')
    print()
    for v in lst:
        print(f'{v:.2f}', end=' ')
    print()

    print(f'ssim: {all_mean_ssim.item():.3f}')
    lst = []
    for i in range(C):
        str_id = dataset.cats[i]
        name = metadata[str_id]['name']
        ssim = cats_ssim[i].item()
        lst.append(ssim)
        print(f'{name}', end=' ')
    print()
    for v in lst:
        print(f'{v:.3f}', end=' ')
    print()

    print(f'lpips: {all_mean_lpips.item():.3f}')
    lst = []
    for i in range(C):
        str_id = dataset.cats[i]
        name = metadata[str_id]['name']
        v = cats_lpips[i].item()
        lst.append(v)
        print(f'{name}', end=' ')
    print()
    for v in lst:
        print(f'{v:.3f}', end=' ')
    print()
