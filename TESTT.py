
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = sns.diverging_palette(0, 20, as_cmap=True)
new_cmap = truncate_colormap(cmap, 0.51, 1)


# Broad
mat = np.abs(np.random.randn(8, 8))*0.8
mat[1:3,:] +=2.5
mat[2:, 1:3] +=3
mat[6, 5] = 4.3
mat[6, 5] = 3.9
np.fill_diagonal(mat, [3.2, 3.7, 4.9, 1.5, 3, 3.9, 4.1, 2])

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
xtick_labels = ['LF', 'LC', 'LP', 'LO', 'RF', 'RC', 'RP', 'RO']
ytick_labels = xtick_labels
grid_kws = {"height_ratios": (.93, .04), "hspace": .25}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 4), gridspec_kw=grid_kws)

sns.heatmap(mat, mask=mask, cmap=new_cmap, square=True, linewidths=.9, ax=ax, cbar=True, cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"}, xticklabels=xtick_labels, yticklabels=ytick_labels, vmin=0, vmax=4.8)



# Delta
mat = np.abs(np.random.randn(8, 8))*0.8
mat[1:3,:] +=1.5
#mat[2:, 1:3] +=3
#mat[6, 5] = 4.3
#mat[6, 5] = 3.9
np.fill_diagonal(mat, [2.7, 3.2, 2.2, 1.5, 3, 3.1, 2.6, 2])

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
xtick_labels = ['LF', 'LC', 'LP', 'LO', 'RF', 'RC', 'RP', 'RO']
ytick_labels = xtick_labels
grid_kws = {"height_ratios": (.93, .04), "hspace": .25}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 4), gridspec_kw=grid_kws)

sns.heatmap(mat, mask=mask, cmap=new_cmap, square=True, linewidths=.9, ax=ax, cbar=True, cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"}, xticklabels=xtick_labels, yticklabels=ytick_labels, vmin=0, vmax=4.8)


# Theta
mat = np.abs(np.random.randn(8, 8))*0.8
mat[1:3,:] +=1.5
#mat[2:, 1:3] +=3
#mat[6, 5] = 4.3
#mat[6, 5] = 3.9
np.fill_diagonal(mat, [3.1, 1.2, 1.2, 1.5, 3, 2.1, 2.0, 2])

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
xtick_labels = ['LF', 'LC', 'LP', 'LO', 'RF', 'RC', 'RP', 'RO']
ytick_labels = xtick_labels
grid_kws = {"height_ratios": (.93, .04), "hspace": .25}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 4), gridspec_kw=grid_kws)

sns.heatmap(mat, mask=mask, cmap=new_cmap, square=True, linewidths=.9, ax=ax, cbar=True, cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"}, xticklabels=xtick_labels, yticklabels=ytick_labels, vmin=0, vmax=4.8)




# Alpha
mat = np.abs(np.random.randn(8, 8))*0.8
mat[1:3,:] +=2.5
mat[2:, 1:3] +=3.2
mat[6, 5] = 4.1
mat[6, 5] = 4
np.fill_diagonal(mat, [2.2, 3.8, 5, 3.5, 2.7, 3.9, 4.1, 2.9])
mat[1:3, 0] -= 1.5
mat[4,1:3] = [1.3, 1.7]
mat[:, 3] +=1

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
xtick_labels = ['LF', 'LC', 'LP', 'LO', 'RF', 'RC', 'RP', 'RO']
ytick_labels = xtick_labels
grid_kws = {"height_ratios": (.93, .04), "hspace": .25}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 4), gridspec_kw=grid_kws)

sns.heatmap(mat, mask=mask, cmap=new_cmap, square=True, linewidths=.9, ax=ax, cbar=True, cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"}, xticklabels=xtick_labels, yticklabels=ytick_labels, vmin=0, vmax=4.8)



# Beta
mat = np.abs(np.random.randn(8, 8))*0.9
np.fill_diagonal(mat, [4.0, 2.8, 3, 2.5, 3.0, 3.9, 2.1, 2.9])
mat[:, 0] += 1.5
mat[1, 0] = 3.6
mat[5, 4] = 4.1
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
xtick_labels = ['LF', 'LC', 'LP', 'LO', 'RF', 'RC', 'RP', 'RO']
ytick_labels = xtick_labels
grid_kws = {"height_ratios": (.93, .04), "hspace": .25}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(5, 4), gridspec_kw=grid_kws)

sns.heatmap(mat, mask=mask, cmap=new_cmap, square=True, linewidths=.9, ax=ax, cbar=True, cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"}, xticklabels=xtick_labels, yticklabels=ytick_labels, vmin=0, vmax=4.8)




mat = np.random.randn(4*8, 4*8)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
xtick_labels = ['LF', 'LC', 'LP', 'LO', 'RF', 'RC', 'RP', 'RO'] * 4
ytick_labels = xtick_labels
fig, ax = plt.subplots(1, 1, figsize=(11, 8))
sns.heatmap(mat, mask=mask, cmap="vlag", square=True, linewidths=.7, cbar=False, xticklabels=xtick_labels, yticklabels=ytick_labels, vmin=-0.8, vmax=0.8)