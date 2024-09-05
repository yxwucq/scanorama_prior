import numpy as np
from scanorama_prior import *
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import sys
from time import time

from process import load_names

NAMESPACE = 'different'

data_names = [
    'data/293t_jurkat/293t',
    'data/brain/neuron_9k',
    'data/hsc/hsc_mars',
    'data/macrophage/uninfected',
    'data/pancreas/pancreas_inDrop',
    'data/pbmc/10x/68k_pbmc',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)

    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)
    
    visualize(datasets_dimred, labels, NAMESPACE, data_names)
