import os

import torch
import numpy as np

from .co3d_dataset import Co3dDataset
from datasets import register


def make_co3d_dataset(root_path='/data/cyb/data/co3d', category='bowl', split='train_known'):
    if isinstance(split, str):
        split = [split]
    return Co3dDataset(
        frame_annotations_file=os.path.join(root_path, category, 'frame_annotations.jgz'),
        sequence_annotations_file=os.path.join(root_path, category, 'sequence_annotations.jgz'),
        subset_lists_file=os.path.join(root_path, category, 'set_lists.json'),
        subsets=split,
        dataset_root=root_path,
        # box_crop=True,
        # box_crop_context=0.3,
        # image_width=800,
        # image_height=800,
        # remove_empty_masks=True,
    )


@register('co3d_nvs')
class Co3dNvs(torch.utils.data.Dataset):

    def __init__(self, n_support, n_query, repeat=1, **kwargs):
        ds = make_co3d_dataset(**kwargs)
        self.ds = ds
        self.n_support = n_support
        self.n_query = n_query

        seqs = []
        seqs_name = []
        cur_seq = []
        last_name = None
        for i in range(len(ds.frame_annots)):
            cur_name = ds.frame_annots[i]['frame_annotation'].sequence_name
            if last_name is None or cur_name == last_name:
                cur_seq.append(i)
            else:
                seqs.append(cur_seq)
                seqs_name.append(last_name)
                cur_seq = [i]
            last_name = cur_name
        seqs.append(cur_seq)
        seqs_name.append(last_name)

        self.seqs = []
        self.seqs_name = []
        for i in range(len(seqs)):
            if len(seqs[i]) >= self.n_support + self.n_query:
                self.seqs.append(seqs[i])
                self.seqs_name.append(seqs_name[i])
        print(f'{len(self.seqs)} sequences.')

        self.z_near = 0.2
        self.z_far = 16
        self.repeat = repeat

    def __len__(self):
        return len(self.seqs) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.seqs)
        seq = self.seqs[idx]
        seq = np.random.choice(seq, self.n_support + self.n_query, replace=False)
        seq = [self.ds[i] for i in seq]
        imgs = []
        poses = []
        focals = []
        for x in seq:
            imgs.append(x['image'])
            r, t = x['R'], x['T']
            t = -r.t() @ t
            r = r.t()
            r[:, 0] *= -1
            r[:, 1] *= -1
            r[:, 2] *= -1
            poses.append(torch.cat([r, t.unsqueeze(-1)], dim=1))
            focals.append(x['focal_length'])

        imgs = torch.stack(imgs)
        poses = torch.stack(poses)
        focals = torch.stack(focals)
        t = self.n_support
        return {
            'support_imgs': imgs[:t],
            'support_poses': poses[:t],
            'support_focals': focals[:t],
            'query_imgs': imgs[t:],
            'query_poses': poses[t:],
            'query_focals': focals[t:],
            'near': self.z_near,
            'far': self.z_far,
        }
