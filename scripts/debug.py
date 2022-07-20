import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from IPython import embed

d = torch.load('debug.pth')

plt.figure()
ax = plt.axes(projection='3d')

for i, x in enumerate(d['query_imgs']):
    transforms.ToPILImage()(x).save(f'vis/{i}.png')

poses = d['query_poses']
print(poses.shape)
# for i in range(4):
#     p = poses[i]
#     tx, ty, tz = p[:, -1]
#     cols = ['red', 'green', 'blue']
#     for j in range(3):
#         x, y, z = p[:, j]
#         ax.plot3D([tx, tx + x], [ty, ty + y], [tz, tz + z], cols[j])
for i in range(0, len(poses)):
    # ax.plot3D(poses[i: i + 2, 0, -1], poses[i: i + 2, 1, -1], poses[i: i + 2, 2, -1], color=(i / len(poses), 0, 0))
    sgn = -1
    pR, pT = poses[i][:3, :3], poses[i][:3, -1]
    a, b, c = pT
    a *= sgn

    x, y, z = pR[:, 0] / 5
    x *= sgn
    ax.quiver([a], [b], [c], [x], [y], [z], color=(i / len(poses), 0, 0))

    x, y, z = pR[:, 1] / 5
    x *= sgn
    ax.quiver([a], [b], [c], [x], [y], [z], color=(0, 0.8, 0))

    x, y, z = -pR[:, 2] * 5
    x *= sgn
    ax.quiver([a], [b], [c], [x], [y], [z], color=(0, 0, 0.8))

plt.show()
