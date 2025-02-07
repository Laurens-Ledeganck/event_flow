"""
WORK IN PROGRESS
"""

# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
...


# settings
model = 'test_model.pth'  # indicate where to find the model


# load & inspect the model
model = torch.load(model, map_location='cpu')
model.eval()

# test a random input
with torch.no_grad():
    # inp_voxel = torch.randn((2, 2, 128, 128), dtype=torch.float32)
    # inp_cnt = torch.randint(0, 3, (2, 2, 128, 128), dtype=torch.float32)
    inp_voxel = torch.load("test_voxel_2.pth")
    inp_cnt = torch.load("test_cnt_2.pth")
    print(inp_voxel.shape, inp_cnt.shape)
    pred = model(inp_voxel, inp_cnt)

    print(pred["flow"][0].shape)
    flow = np.array(pred["flow"][0])
    u = flow[:, 0]
    v =  flow[:, 1]
    x = np.arange(0, u.shape[2])
    y = np.arange(0, u.shape[1])
    x, y = np.meshgrid(x, y)
    fig, ax = plt.subplots(u.shape[0], 1, figsize=(2, 5))
    for i, a in enumerate(ax):
        a.quiver(x, y, u[i], v[i], color='r')
    plt.show()

print("\nDone.\n")
