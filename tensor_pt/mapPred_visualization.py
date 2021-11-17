import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
from sklearn.metrics import mean_squared_error


data = torch.load('results_tensors.pt')
train_yt = data['train_yt'] 

train_yt = data['train_yt'] # list 229 integers [50, 51, ..., 278]
train_ypr = data['train_ypr'] # list of 229 arrays of size (45,), preditionc
train_ygt = data["train_ygt"] # list of 229 arrays of size (45,), groud truth

test_yt = data['test_yt'] # list of 121 ints [279, ..., 399]
test_ypr = data['test_ypr'] # list of 121 arrays of size (45,)
test_ygt = data['test_ygt'] # list of 121 arrays of size (45,)

cmap = 'Oranges'
cmap_error = 'Greys'
interpolation='nearest'

filenames = []

time_series = train_yt
ground_truth = train_ygt
prediction = train_ypr

mse_err = np.zeros(shape=(len(time_series)))


for i, t in enumerate(time_series):
    heatmap1 = ground_truth[i].reshape(5,9) # (45,) array
    heatmap2 = prediction[i].reshape(5,9) # (45,) array

    mse_err[i] = mean_squared_error(heatmap1, heatmap2) / (i+1)

    # max_heat = int(np.amax(np.amax(heatmap1), np.amax(heatmap2)))
    # mim_heat = int(np.amin(np.amin(heatmap1), np.amin(heatmap2)))
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3.5), ncols=3)
    
    # pos1 = ax1.imshow(heatmap1, cmap=cmap, interpolation=interpolation)
    pos1 = ax1.imshow(heatmap1, cmap=cmap, interpolation=interpolation, vmin=-8, vmax=10)

    ax1.set_title(f'Groud Truth\n t={t}')
    fig.colorbar(pos1, ax=ax1)

    # pos2 = ax2.imshow(heatmap1, cmap=cmap, interpolation=interpolation)
    pos2 = ax2.imshow(heatmap1, cmap=cmap, interpolation=interpolation, vmin=-8, vmax=10)

    ax2.set_title(f'Prediction\n t={t}')
    fig.colorbar(pos2, ax=ax2)

    mse_plot = mse_err[:i]
    pos3 = ax3.plot(mse_plot)
    ax3.set_title(f'Mean square error \n t={t}')

    fig_path = './figs/frame{}.png'.format(t)
    filenames.append(fig_path)
    fig.savefig(fig_path)
    plt.cla()


# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


# remove files
for filename in set(filenames):
    os.remove(filename)

print('animation created!')