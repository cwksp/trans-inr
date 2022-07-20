import argparse
import os

import yaml
import torch
import einops
import numpy as np
import skimage.metrics
import lpips
from tqdm import tqdm
from torch.utils.data import DataLoader

import models
from utils import Averager, poses_to_rays, volume_rendering, batched_volume_rendering
from datasets.pixelnerf_shapenet import PixelnerfShapenet


render_batch_size = 4096
train_points_per_ray = 128


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='../../data/trans-nf/pixelnerf/srn_chairs')
    parser.add_argument('--category', default='chairs')
    parser.add_argument('--n-support', type=int, default=1)
    parser.add_argument('--n-query', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model', '-m')
    parser.add_argument('--gpu', '-g')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.n_support == 1:
        support_lst = [64]
    elif args.n_support == 2:
        support_lst = [64, 128]
    dataset = PixelnerfShapenet(args.dataset_root, args.category, 'test', args.n_support, args.n_query,
                                support_lst=support_lst, repeat=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    model = models.make(torch.load(args.model, map_location='cpu')['model'], load_sd=True)
    model.cuda()
    model.eval()

    all_mean_psnr = Averager()
    all_mean_ssim = Averager()
    all_mean_lpips = Averager()
    lpips_vgg = lpips.LPIPS(net="vgg").cuda()

    with torch.no_grad():
        pbar = tqdm(loader)
        for data in pbar:
            data = {k: v.cuda() for k, v in data.items()}
            query_imgs = data.pop('query_imgs')
            query_poses = data.pop('query_poses')

            hyponet = model(data)

            B, N = query_imgs.shape[:2]
            H, W = query_imgs.shape[-2:]
            rays_o, rays_d = poses_to_rays(query_poses, H, W, data['query_focals'])

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

            pred = pred.view(B * N, H, W, 3)
            gt = gt.view(B * N, H, W, 3)

            v = lpips_vgg(pred.permute(0, 3, 1, 2) * 2 - 1,
                          gt.permute(0, 3, 1, 2) * 2 - 1)
            all_mean_lpips.add(v.mean().item(), n=B*N)

            imgs_pred = pred.cpu().numpy()
            imgs_gt = gt.cpu().numpy()
            for i in range(B * N):
                psnr = skimage.metrics.peak_signal_noise_ratio(
                    imgs_pred[i], imgs_gt[i], data_range=1
                )
                all_mean_psnr.add(psnr)
                ssim = skimage.metrics.structural_similarity(
                    imgs_pred[i], imgs_gt[i], multichannel=True, data_range=1
                )
                all_mean_ssim.add(ssim)

            pbar.set_description(desc=''
                f'psnr: {all_mean_psnr.item():.2f} '
                f'ssim: {all_mean_ssim.item():.3f} '
                f'lpips: {all_mean_lpips.item():.3f}')
