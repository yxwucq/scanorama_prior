from annoy import AnnoyIndex
from intervaltree import IntervalTree
from itertools import cycle, islice
import anndata as ad
import numpy as np
import operator
import random
import time
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import sys
import warnings
from typing import Optional
from scipy.spatial.distance import cdist

from .utils import plt, dispersion, reduce_dimensionality
from .utils import visualize_cluster, visualize_expr, visualize_dropout
from .utils import handle_zeros_in_scale

print('scanorama.scanorama loaded')
# Default parameters.
ALPHA = 0.10
APPROX = True
BATCH_SIZE = 5000
DIMRED = 100
HVG = None
KNN = 30
N_ITER = 500
PERPLEXITY = 1200
SIGMA = 15
VERBOSE = 2
SEARCH_FACTOR = 5
BIAS_SCALE = 0
OVERBIAS = 0

def compute_center_vectors(datasets, cell_types):
    center_vectors = []
    for dataset, types in zip(datasets, cell_types):
        unique_types = np.unique(types)
        centers = {cell_type: np.mean(dataset[types == cell_type], axis=0) for cell_type in unique_types}
        
        vectors = np.zeros_like(dataset)
        for i, cell_type in enumerate(types):
            vectors[i] = dataset[i] - centers[cell_type]
        
        center_vectors.append(vectors)
    return center_vectors

# Do batch correction on a list of data sets.
def correct(datasets_full, genes_list, return_dimred=False,
            batch_size=BATCH_SIZE, verbose=VERBOSE, ds_names=None,
            dimred=DIMRED, approx=APPROX, sigma=SIGMA, alpha=ALPHA, knn=KNN,
            return_dense=False, hvg=None, union=False, seed=0):
    """Integrate and batch correct a list of data sets.

    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    return_dimred: `bool`, optional (default: `False`)
        In addition to returning batch corrected matrices, also returns
        integrated low-dimesional embeddings.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    return_dense: `bool`, optional (default: `False`)
        Return `numpy.ndarray` matrices instead of `scipy.sparse.csr_matrix`.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.
    seed: `int`, optional (default: 0)
        Random seed to use.

    Returns
    -------
    corrected, genes
        By default (`return_dimred=False`), returns a two-tuple containing a
        list of `scipy.sparse.csr_matrix` each with batch corrected values,
        and a single list of genes containing the intersection of inputted
        genes.

    integrated, corrected, genes
        When `return_dimred=True`, returns a three-tuple containing a list
        of `numpy.ndarray` with integrated low dimensional embeddings, a list
        of `scipy.sparse.csr_matrix` each with batch corrected values, and a
        a single list of genes containing the intersection of inputted genes.
    """
    np.random.seed(seed)
    random.seed(seed)

    datasets_full = check_datasets(datasets_full)

    datasets, genes = merge_datasets(datasets_full, genes_list,
                                     ds_names=ds_names, union=union)
    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg,
                                          dimred=dimred)

    datasets_dimred = assemble(
        datasets_dimred, # Assemble in low dimensional space.
        expr_datasets=datasets, # Modified in place.
        verbose=verbose, knn=knn, sigma=sigma, approx=approx,
        alpha=alpha, ds_names=ds_names, batch_size=batch_size,
    )

    if return_dense:
        datasets = [ ds.toarray() for ds in datasets ]

    if return_dimred:
        return datasets_dimred, datasets, genes

    return datasets, genes

# Integrate a list of data sets.
def integrate(datasets_full, genes_list, cell_types_list, batch_size=BATCH_SIZE,
              verbose=VERBOSE, ds_names=None, dimred=DIMRED, approx=APPROX,
              sigma=SIGMA, alpha=ALPHA, knn=KNN, union=False, hvg=None, seed=0,
              sketch=False, sketch_method='geosketch', sketch_max=10000,
              type_similarity_matrix=None, batch_key=Optional[str]):
    """Integrate a list of data sets.

    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.
    seed: `int`, optional (default: 0)
        Random seed to use.
    sketch: `bool`, optional (default: False)
        Apply sketching-based acceleration by first downsampling the datasets.
        See Hie et al., Cell Systems (2019).
    sketch_method: {'geosketch', 'uniform'}, optional (default: `geosketch`)
        Apply the given sketching method to the data. Only used if
        `sketch=True`.
    sketch_max: `int`, optional (default: 10000)
        If a dataset has more cells than `sketch_max`, downsample to
        `sketch_max` using the method provided in `sketch_method`. Only used
        if `sketch=True`.

    Returns
    -------
    integrated, genes
        Returns a two-tuple containing a list of `numpy.ndarray` with
        integrated low dimensional embeddings and a single list of genes
        containing the intersection of inputted genes.
    """
    np.random.seed(seed)
    random.seed(seed)

    if len(datasets_full) != len(cell_types_list):
        raise ValueError("Number of datasets must match number of cell type lists")

    if type_similarity_matrix is None:
        print("Warning: No type similarity matrix provided. Using default equal weights.")
        unique_types = set(type for types in cell_types_list for type in types)
        type_similarity_matrix = np.ones((len(unique_types), len(unique_types)))

    datasets_full = check_datasets(datasets_full)

    print('Checking datasets...')
    datasets, genes = merge_datasets(datasets_full, genes_list,
                                    ds_names=ds_names, union=union)
    cell_types = [np.array(types) for types in cell_types_list]

    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg,
                                        dimred=dimred)

    print('Computing center vectors...')
    time_start = time.time()
    center_vectors = compute_center_vectors(datasets_dimred, cell_types)
    print('Computing center vectors...done')
    print('Time: {:.2f}s'.format(time.time() - time_start))
    
    if sketch:
        print('Applying sketching-based acceleration...')
        datasets_dimred = integrate_sketch(
            datasets_dimred, cell_types, type_similarity_matrix,
            sketch_method=sketch_method, N=sketch_max,
            integration_fn=assemble, integration_fn_args={
                'verbose': verbose, 'knn': knn, 'sigma': sigma,
                'approx': approx, 'alpha': alpha, 'ds_names': ds_names,
                'batch_size': batch_size,
            }
        )
    else:
        datasets_dimred = assemble(
            datasets_dimred,
            cell_types,
            center_vectors,
            type_similarity_matrix,
            verbose=verbose, knn=knn, sigma=sigma, approx=approx,
            alpha=alpha, ds_names=ds_names, batch_size=batch_size,
        )

    return datasets_dimred, genes

# Batch correction with scanpy's AnnData object.
def correct_scanpy(adatas, **kwargs):
    """Batch correct a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate and/or correct.
        `adata.var_names` must be set to the list of genes.
    return_dimred : `bool`, optional (default=`False`)
        When `True`, the returned `adatas` are each modified to
        also have the integrated low-dimensional embeddings in
        `adata.obsm['X_scanorama']`.
    kwargs : `dict`
        See documentation for the `correct()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    corrected
        By default (`return_dimred=False`), returns a list of new
        `scanpy.api.AnnData`.
        When `return_dimred=True`, `corrected` also includes the
        integrated low-dimensional embeddings in
        `adata.obsm['X_scanorama']`.
    """
    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        datasets_dimred, datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )
    else:
        datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )

    from anndata import AnnData

    new_adatas = []
    for i in range(len((adatas))):
        adata = AnnData(datasets[i])
        adata.obs = adatas[i].obs
        adata.obsm = adatas[i].obsm

        # Ensure that variables are in the right order,
        # as Scanorama rearranges genes to be in alphabetical
        # order and as the intersection (or union) of the
        # original gene sets.
        adata.var_names = genes
        gene2idx = { gene: idx for idx, gene in
                     zip(adatas[i].var.index,
                         adatas[i].var_names.values) }
        var_idx = [ gene2idx[gene] for gene in genes ]
        adata.var = adatas[i].var.loc[var_idx]

        adata.uns = adatas[i].uns
        new_adatas.append(adata)

    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        for adata, X_dimred in zip(new_adatas, datasets_dimred):
            adata.obsm['X_scanorama'] = X_dimred

    return new_adatas

# Integration with scanpy's AnnData object.
def integrate_scanpy(adatas, **kwargs):
    """Integrate a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate.
    kwargs : `dict`
        See documentation for the `integrate()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    None
    """
    print('Integrating {} datasets...'.format(len(adatas)))
    cell_types_list = [adata.obs['cell_type'].values for adata in adatas]
    
    datasets_dimred, genes = integrate(
        [adata.X for adata in adatas],
        [adata.var_names.values for adata in adatas],
        cell_types_list,
        **kwargs
    )

    for adata, X_dimred in zip(adatas, datasets_dimred):
        adata.obsm['X_scanorama'] = X_dimred

# Visualize a scatter plot with cluster labels in the
# `cluster' variable.
def plot_clusters(coords, clusters, s=1, colors=None):
    if coords.shape[0] != clusters.shape[0]:
        sys.stderr.write(
            'Error: mismatch, {} cells, {} labels\n'
            .format(coords.shape[0], clusters.shape[0])
        )
        sys.exit(1)

    if colors is None:
        colors = np.array(
            list(islice(cycle([
                '#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00',
                '#ffe119', '#e6194b', '#ffbea3',
                '#911eb4', '#46f0f0', '#f032e6',
                '#d2f53c', '#008080', '#e6beff',
                '#aa6e28', '#800000', '#aaffc3',
                '#808000', '#ffd8b1', '#000080',
                '#808080', '#fabebe', '#a3f4ff'
            ]), int(max(clusters) + 1)))
        )

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1],
                c=colors[clusters], s=s)

# Put datasets into a single matrix with the intersection of all genes.
def merge_datasets(datasets, genes, ds_names=None, verbose=True,
                   union=False):
    if union:
        sys.stderr.write(
            'WARNING: Integrating based on the union of genes is '
            'highly discouraged, consider taking the intersection '
            'or requantifying gene expression.\n'
        )

    # Find genes in common.
    keep_genes = set()
    for idx, gene_list in enumerate(genes):
        if len(keep_genes) == 0:
            keep_genes = set(gene_list)
        elif union:
            keep_genes |= set(gene_list)
        else:
            keep_genes &= set(gene_list)
        if not union and not ds_names is None and verbose:
            print('After {}: {} genes'.format(ds_names[idx], len(keep_genes)))
        if len(keep_genes) == 0:
            print('Error: No genes found in all datasets, exiting...')
            sys.exit(1)
    if verbose:
        print('Found {} genes among all datasets'
              .format(len(keep_genes)))

    if union:
        union_genes = sorted(keep_genes)
        for i in range(len(datasets)):
            if verbose:
                print('Processing data set {}'.format(i))
            X_new = np.zeros((datasets[i].shape[0], len(union_genes)))
            X_old = csc_matrix(datasets[i])
            gene_to_idx = { gene: idx for idx, gene in enumerate(genes[i]) }
            for j, gene in enumerate(union_genes):
                if gene in gene_to_idx:
                    X_new[:, j] = X_old[:, gene_to_idx[gene]].toarray().flatten()
            datasets[i] = csr_matrix(X_new)
        ret_genes = np.array(union_genes)
    else:
        # Only keep genes in common.
        ret_genes = np.array(sorted(keep_genes))
        for i in range(len(datasets)):
            # Remove duplicate genes.
            uniq_genes, uniq_idx = np.unique(genes[i], return_index=True)
            datasets[i] = datasets[i][:, uniq_idx]

            # Do gene filtering.
            gene_sort_idx = np.argsort(uniq_genes)
            gene_idx = [ idx for idx in gene_sort_idx
                         if uniq_genes[idx] in keep_genes ]
            datasets[i] = datasets[i][:, gene_idx]
            assert(np.array_equal(uniq_genes[gene_idx], ret_genes))

    return datasets, ret_genes

def check_datasets(datasets_full):
    datasets_new = []
    for i, ds in enumerate(datasets_full):
        if issubclass(type(ds), np.ndarray):
            datasets_new.append(csr_matrix(ds))
        elif issubclass(type(ds), csr_matrix):
            datasets_new.append(ds)
        else:
            sys.stderr.write('ERROR: Data sets must be numpy array or '
                             'scipy.sparse.csr_matrix, received type '
                             '{}.\n'.format(type(ds)))
            sys.exit(1)
    return datasets_new

# Randomized SVD.
def dimensionality_reduce(datasets, dimred=DIMRED):
    X = vstack(datasets)
    X = reduce_dimensionality(X, dim_red_k=dimred)
    datasets_dimred = []
    base = 0
    for ds in datasets:
        datasets_dimred.append(X[base:(base + ds.shape[0]), :])
        base += ds.shape[0]
    return datasets_dimred

# Normalize and reduce dimensionality.
def process_data(datasets, genes, hvg=HVG, dimred=DIMRED, verbose=False):
    # Only keep highly variable genes
    if not hvg is None and hvg > 0 and hvg < len(genes):
        if verbose:
            print('Highly variable filter...')
        X = vstack(datasets)
        disp = dispersion(X)
        highest_disp_idx = np.argsort(disp[0])[::-1]
        top_genes = set(genes[highest_disp_idx[range(hvg)]])
        for i in range(len(datasets)):
            gene_idx = [ idx for idx, g_i in enumerate(genes)
                         if g_i in top_genes ]
            datasets[i] = datasets[i][:, gene_idx]
        genes = np.array(sorted(top_genes))

    # Normalize.
    if verbose:
        print('Normalizing...')
    for i, ds in enumerate(datasets):
        datasets[i] = normalize(ds, axis=1)

    # Compute compressed embedding.
    if dimred > 0:
        if verbose:
            print('Reducing dimension...')
        datasets_dimred = dimensionality_reduce(datasets, dimred=dimred)
        if verbose:
            print('Done processing.')
        return datasets_dimred, genes

    if verbose:
        print('Done processing.')

    return datasets, genes

# Plot t-SNE visualization.
def visualize(assembled, labels, namespace, data_names,
              gene_names=None, gene_expr=None, genes=None,
              n_iter=N_ITER, perplexity=PERPLEXITY, verbose=VERBOSE,
              learn_rate=200., early_exag=12., embedding=None,
              shuffle_ds=False, size=1, multicore_tsne=True,
              image_suffix='.svg', viz_cluster=False, colors=None,
              random_state=None,):
    # Fit t-SNE.
    if embedding is None:
        try:
            from MulticoreTSNE import MulticoreTSNE
            tsne = MulticoreTSNE(
                n_iter=n_iter, perplexity=perplexity,
                verbose=verbose, random_state=random_state,
                learning_rate=learn_rate,
                early_exaggeration=early_exag,
                n_jobs=40
            )
        except ImportError:
            multicore_tsne = False

        if not multicore_tsne:
            tsne = TSNE(
                n_iter=n_iter, perplexity=perplexity,
                verbose=verbose, random_state=random_state,
                learning_rate=learn_rate,
                early_exaggeration=early_exag
            )

        tsne.fit(np.concatenate(assembled))
        embedding = tsne.embedding_

    if shuffle_ds:
        rand_idx = range(embedding.shape[0])
        random.shuffle(list(rand_idx))
        embedding = embedding[rand_idx, :]
        labels = labels[rand_idx]

    # Plot clusters together.
    plot_clusters(embedding, labels, s=size, colors=colors)
    plt.title(('Panorama ({} iter, perplexity: {}, sigma: {}, ' +
               'knn: {}, hvg: {}, dimred: {}, approx: {})')
              .format(n_iter, perplexity, SIGMA, KNN, HVG,
                      DIMRED, APPROX))
    plt.savefig(namespace + image_suffix, dpi=500)

    # Plot clusters individually.
    if viz_cluster and not shuffle_ds:
        for i in range(len(data_names)):
            visualize_cluster(embedding, i, labels,
                              cluster_name=data_names[i], size=size,
                              viz_prefix=namespace,
                              image_suffix=image_suffix)

    # Plot gene expression levels.
    if (not gene_names is None) and \
       (not gene_expr is None) and \
       (not genes is None):
        if shuffle_ds:
            gene_expr = gene_expr[rand_idx, :]
        for gene_name in gene_names:
            visualize_expr(gene_expr, embedding,
                           genes, gene_name, size=size,
                           viz_prefix=namespace,
                           image_suffix=image_suffix)

    return embedding

def nn_with_type(ds1, ds2, ds1_types, ds2_types, type_similarity_matrix, knn=KNN):
    def weighted_distance_matrix(X1, X2, types1, types2):
        euclidean_dist = cdist(X1, X2, metric='euclidean')

        type_sim_dict = {t1: {t2: type_similarity_matrix.loc[t1, t2] for t2 in type_similarity_matrix.columns} for t1 in type_similarity_matrix.index}
        type_sim = np.array([[type_sim_dict[t1][t2] for t2 in types2] for t1 in types1])

        weighted_dist = euclidean_dist / type_sim
        return weighted_dist
    
    dist_matrix = weighted_distance_matrix(ds1, ds2, ds1_types, ds2_types)
    
    nearest_neighbors = np.argsort(dist_matrix, axis=1)[:, :knn]
    match = set()
    for d, neighbors in enumerate(nearest_neighbors):
        for r in neighbors:
            match.add((d, r))
    return match

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, knn=KNN, metric='manhattan', n_trees=10):
    # Build index.
    warnings.warn('Approximate nearest neighbors does not incorporate cell type information in building the graph.')
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

def nn_with_type_approx(ds1, ds2, ds1_types, ds2_types, type_similarity_matrix, knn=KNN, metric='angular', n_trees=10, search_k=-1):
    # Build index
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    type_sim_dict = {t1: {t2: type_similarity_matrix.loc[t1, t2] for t2 in type_similarity_matrix.columns} for t1 in type_similarity_matrix.index}
    type_sim = np.array([[type_sim_dict[t1][t2] for t2 in ds2_types] for t1 in ds1_types])

    # Search index and refine results
    match = set()
    for i in range(ds1.shape[0]):
        # Get 5 * knn approximate nearest neighbors with distances
        approx_nn, distances = a.get_nns_by_vector(ds1[i, :], SEARCH_FACTOR * knn, search_k=search_k, include_distances=True)
        distances = np.array(distances)
        approx_type_sim = np.array([type_sim[i][j] for j in approx_nn])
        # Compute type-weighted distances
        weighted_distances = distances / approx_type_sim
        
        # Get top knn
        top_knn = sorted(zip(approx_nn, weighted_distances), key=lambda x: x[1])[:knn]          

        # Add to match set
        for j, _ in top_knn:
            match.add((i, j))
            
    return match

def mnn(ds1, ds2, ds1_types, ds2_types, type_similarity_matrix, knn=KNN, approx=APPROX):
    if approx:
        match1 = nn_with_type_approx(ds1, ds2, ds1_types, ds2_types, type_similarity_matrix=type_similarity_matrix, knn=knn)
    else:
        match1 = nn_with_type(ds1, ds2, ds1_types, ds2_types, type_similarity_matrix=type_similarity_matrix, knn=knn)
    
    if approx:
        match2 = nn_with_type_approx(ds2, ds1, ds2_types, ds1_types, type_similarity_matrix=type_similarity_matrix, knn=knn)
    else:
        match2 = nn_with_type(ds2, ds1, ds2_types, ds1_types, type_similarity_matrix=type_similarity_matrix, knn=knn)
    
    mutual = match1 & set([(b, a) for a, b in match2])
    return mutual

# Visualize alignment between two datasets.
def plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind):
    tsne = TSNE(n_iter=400, verbose=VERBOSE, random_state=69)

    tsne.fit(curr_ds)
    plt.figure()
    coords_ds = tsne.embedding_[:, :]
    coords_ds[:, 1] += 100
    plt.scatter(coords_ds[:, 0], coords_ds[:, 1])

    tsne.fit(curr_ref)
    coords_ref = tsne.embedding_[:, :]
    plt.scatter(coords_ref[:, 0], coords_ref[:, 1])

    x_list, y_list = [], []
    for r_i, c_i in zip(ds_ind, ref_ind):
        x_list.append(coords_ds[r_i, 0])
        x_list.append(coords_ref[c_i, 0])
        x_list.append(None)
        y_list.append(coords_ds[r_i, 1])
        y_list.append(coords_ref[c_i, 1])
        y_list.append(None)
    plt.plot(x_list, y_list, 'b-', alpha=0.3)
    plt.show()


# Populate a table (in place) that stores mutual nearest neighbors
# between datasets.
def fill_table(table, i, curr_ds, datasets, curr_types, datasets_types, type_similarity_matrix,
               base_ds=0, knn=KNN, approx=APPROX):
    curr_ref = np.concatenate(datasets)
    ref_types = np.concatenate(datasets_types)
    if approx:
        match = nn_with_type_approx(curr_ds, curr_ref, curr_types, ref_types, type_similarity_matrix=type_similarity_matrix, knn=knn)
    else:
        match = nn_with_type(curr_ds, curr_ref, curr_types, ref_types, type_similarity_matrix=type_similarity_matrix, knn=knn)

    # Build interval tree.
    itree_ds_idx = IntervalTree()
    itree_pos_base = IntervalTree()
    pos = 0
    for j in range(len(datasets)):
        n_cells = datasets[j].shape[0]
        itree_ds_idx[pos:(pos + n_cells)] = base_ds + j
        itree_pos_base[pos:(pos + n_cells)] = pos
        pos += n_cells

    # Store all mutual nearest neighbors between datasets.
    for d, r in match:
        interval = itree_ds_idx[r]
        assert(len(interval) == 1)
        j = interval.pop().data
        interval = itree_pos_base[r]
        assert(len(interval) == 1)
        base = interval.pop().data
        if not (i, j) in table:
            table[(i, j)] = set()
        table[(i, j)].add((d, r - base))
        assert(r - base >= 0)

gs_idxs = None

# Fill table of alignment scores.
def find_alignments_table(datasets, cell_types, type_similarity_matrix, knn=KNN, approx=APPROX, verbose=VERBOSE,
                          prenormalized=False):
    if not prenormalized:
        datasets = [ normalize(ds, axis=1) for ds in datasets ]

    table = {}
    for i in range(len(datasets)):
        if len(datasets[:i]) > 0:
            fill_table(table, i, datasets[i], datasets[:i], cell_types[i], cell_types[:i],
                       type_similarity_matrix, knn=knn, approx=approx)
        if len(datasets[i+1:]) > 0:
            fill_table(table, i, datasets[i], datasets[i+1:], cell_types[i], cell_types[i+1:],
                       type_similarity_matrix, knn=knn, base_ds=i+1, approx=approx)
    # Count all mutual nearest neighbors between datasets.
    matches = {}
    table1 = {}
    if verbose > 1:
        table_print = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i >= j:
                continue
            if not (i, j) in table or not (j, i) in table:
                continue
            match_ij = table[(i, j)]
            match_ji = set([ (b, a) for a, b in table[(j, i)] ])
            matches[(i, j)] = match_ij & match_ji

            table1[(i, j)] = (max(
                float(len(set([ idx for idx, _ in matches[(i, j)] ]))) /
                datasets[i].shape[0],
                float(len(set([ idx for _, idx in matches[(i, j)] ]))) /
                datasets[j].shape[0]
            ))
            if verbose > 1:
                table_print[i, j] += table1[(i, j)]

    if verbose > 1:
        print(table_print)
        return table1, table_print, matches
    else:
        return table1, None, matches

# Find the matching pairs of cells between datasets.
def find_alignments(datasets, cell_types, type_similarity_matrix, knn=KNN, approx=APPROX, verbose=VERBOSE,
                    alpha=ALPHA, prenormalized=False):
    table1, _, matches = find_alignments_table(
        datasets, cell_types, type_similarity_matrix,
        knn=knn, approx=approx, verbose=verbose,
        prenormalized=prenormalized,
    )

    alignments = [ (i, j) for (i, j), val in reversed(
        sorted(table1.items(), key=operator.itemgetter(1))
    ) if val > alpha ]

    return alignments, matches

# Find connections between datasets to identify panoramas.
def connect(datasets, knn=KNN, approx=APPROX, alpha=ALPHA,
            verbose=VERBOSE):
    # Find alignments.
    alignments, _ = find_alignments(
        datasets, knn=knn, approx=approx, alpha=alpha,
        verbose=verbose
    )
    if verbose:
        print(alignments)

    panoramas = []
    connected = set()
    for i, j in alignments:
        # See if datasets are involved in any current panoramas.
        panoramas_i = [ panoramas[p] for p in range(len(panoramas))
                        if i in panoramas[p] ]
        assert(len(panoramas_i) <= 1)
        panoramas_j = [ panoramas[p] for p in range(len(panoramas))
                        if j in panoramas[p] ]
        assert(len(panoramas_j) <= 1)

        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            panoramas.append([ i ])
            panoramas_i = [ panoramas[-1] ]

        if len(panoramas_i) == 0:
            panoramas_j[0].append(i)
        elif len(panoramas_j) == 0:
            panoramas_i[0].append(j)
        elif panoramas_i[0] != panoramas_j[0]:
            panoramas_i[0] += panoramas_j[0]
            panoramas.remove(panoramas_j[0])

        connected.add(i)
        connected.add(j)

    for i in range(len(datasets)):
        if not i in connected:
            panoramas.append([ i ])

    return panoramas

# To reduce memory usage, split bias computation into batches.
def batch_bias(curr_ds, match_ds, bias, curr_types, match_cell_types, type_similarity_matrix, sigma=SIGMA, batch_size=None, alpha=0.1):
    if batch_size is None:
        weights = rbf_kernel(curr_ds, match_ds, gamma=0.5*sigma)
        type_weights = compute_type_weights(curr_types, match_cell_types, type_similarity_matrix)
        weights = weights * type_weights**2 # curr*match
        weights = normalize(weights, axis=1, norm='l1')
        avg_bias = np.dot(weights, bias)
        return avg_bias

    base = 0
    avg_bias = np.zeros(curr_ds.shape)
    denom = np.zeros(curr_ds.shape[0])
    while base < match_ds.shape[0]:
        batch_idx = range(
            base, min(base + batch_size, match_ds.shape[0])
        )
        weights = rbf_kernel(curr_ds, match_ds[batch_idx, :],
                             gamma=0.5*sigma)
        type_weights = compute_type_weights(curr_types, match_cell_types[batch_idx], type_similarity_matrix)
        weights = weights * type_weights**2
        avg_bias += np.dot(weights, bias[batch_idx, :])
        denom += np.sum(weights, axis=1)
        base += batch_size

    denom = handle_zeros_in_scale(denom, copy=False)
    avg_bias /= denom[:, np.newaxis]
    # random perturbation to increase robustness
    avg_bias *= (1+OVERBIAS)
    avg_bias *= (1+np.random.normal(0, BIAS_SCALE, avg_bias.shape))
    # avg_bias += np.random.normal(0, 0.1, avg_bias.shape)

    # correction_strength = compute_correction_strength(cell_types, type_similarity_matrix)
    # avg_bias *= correction_strength[:, np.newaxis]

    return avg_bias

def compute_type_weights(curr_types, match_cell_types, type_similarity_matrix):    
    type_sim_dict = {t1: {t2: type_similarity_matrix.loc[t1, t2] for t2 in type_similarity_matrix.columns} for t1 in type_similarity_matrix.index}
    type_weights = np.array([[type_sim_dict[t1][t2] for t2 in match_cell_types] for t1 in curr_types])
    return type_weights

# def compute_correction_strength(cell_types, type_similarity_matrix):
#     unique_types = np.unique(cell_types)
#     type_stability = np.array([np.mean(type_similarity_matrix.loc[t, :]) for t in unique_types])
#     type_to_stability = dict(zip(unique_types, type_stability))
    
#     correction_strength = np.array([type_to_stability[t] for t in cell_types])
    
#     # Normalize correction strength
#     correction_strength = (correction_strength - np.min(correction_strength)) / (np.max(correction_strength) - np.min(correction_strength))
    
#     # Adjust the range of correction strength (e.g., from 0.5 to 1)
#     correction_strength = 0.5 + 0.5 * correction_strength
    
#     return correction_strength

# Compute nonlinear translation vectors between dataset
# and a reference.
def transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix, sigma=SIGMA, cn=False,
              batch_size=None, alpha=0.1):
    # Compute the matching.
    match_ds = curr_ds[ds_ind, :]
    match_ref = curr_ref[ref_ind, :]
    match_cell_types = curr_types[ds_ind]
    match_cell_types_ref = ref_types[ref_ind]
    
    match_curr_center_vec = curr_center_vec[ds_ind, :]
    match_ref_center_vec = ref_center_vec[ref_ind, :]

    type_sim_dict = {t1: {t2: type_similarity_matrix.loc[t1, t2] for t2 in type_similarity_matrix.columns} for t1 in type_similarity_matrix.index}
    center_adjusted_weight = np.array([type_sim_dict[t1][t2] for t1, t2 in zip(match_cell_types, match_cell_types_ref)])

    center_adjusted_bias = (match_ref - match_ds) + center_adjusted_weight[:, np.newaxis]**2 *(match_curr_center_vec - match_ref_center_vec)
    
    if cn:
        match_ds = match_ds.toarray()
        curr_ds = curr_ds.toarray()
        center_adjusted_bias = center_adjusted_bias.toarray()

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            avg_bias = batch_bias(curr_ds, match_ds, center_adjusted_bias, curr_types, match_cell_types, type_similarity_matrix, 
                                  sigma=sigma, batch_size=batch_size, alpha=alpha)  
        except RuntimeWarning:
            sys.stderr.write('WARNING: Oversmoothing detected, refusing to batch '
                             'correct, consider lowering sigma value.\n')
            return csr_matrix(curr_ds.shape, dtype=float)
        except MemoryError:
            if batch_size is None:
                sys.stderr.write('WARNING: Out of memory, consider turning on '
                                 'batched computation with batch_size parameter.\n')
            else:
                sys.stderr.write('WARNING: Out of memory, consider lowering '
                                 'the batch_size parameter.\n')
            return csr_matrix(curr_ds.shape, dtype=float)

    if cn:
        avg_bias = csr_matrix(avg_bias)

    return avg_bias

# Finds alignments between datasets and uses them to construct
# panoramas. "Merges" datasets by correcting gene expression
# values.
def assemble(datasets, cell_types, center_vectors, type_similarity_matrix, verbose=VERBOSE, view_match=False, knn=KNN,
             sigma=SIGMA, approx=APPROX, alpha=ALPHA, expr_datasets=None,
             ds_names=None, batch_size=None,
             alignments=None, matches=None):
    if len(datasets) == 1:
        return datasets

    if alignments is None and matches is None:
        alignments, matches = find_alignments(
            datasets, cell_types, type_similarity_matrix,
            knn=knn, approx=approx, alpha=alpha, verbose=verbose,
        )

    ds_assembled = {}
    panoramas = []
    for i, j in alignments:
        if verbose:
            if ds_names is None:
                print('Processing datasets {}'.format((i, j)))
            else:
                print('Processing datasets {} <=> {}'.
                      format(ds_names[i], ds_names[j]))

        # Only consider a dataset a fixed amount of times.
        if not i in ds_assembled:
            ds_assembled[i] = 0
        ds_assembled[i] += 1
        if not j in ds_assembled:
            ds_assembled[j] = 0
        ds_assembled[j] += 1
        if ds_assembled[i] > 3 and ds_assembled[j] > 3:
            continue

        # See if datasets are involved in any current panoramas.
        panoramas_i = [ panoramas[p] for p in range(len(panoramas))
                        if i in panoramas[p] ]
        assert(len(panoramas_i) <= 1)
        panoramas_j = [ panoramas[p] for p in range(len(panoramas))
                        if j in panoramas[p] ]
        assert(len(panoramas_j) <= 1)

        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            if datasets[i].shape[0] < datasets[j].shape[0]:
                i, j = j, i
            panoramas.append([ i ])
            panoramas_i = [ panoramas[-1] ]

        # Map dataset i to panorama j.
        
        if len(panoramas_i) == 0:
            curr_ds = datasets[i]
            curr_ref = np.concatenate([datasets[p] for p in panoramas_j[0]])
            curr_types = cell_types[i]
            ref_types = np.concatenate([cell_types[p] for p in panoramas_j[0]])
            curr_center_vec = center_vectors[i]
            ref_center_vec = np.concatenate([center_vectors[p] for p in panoramas_j[0]])

            match = []
            base = 0
            for p in panoramas_j[0]:
                if i < p and (i, p) in matches:
                    match.extend([(a, b + base) for a, b in matches[(i, p)]])
                elif i > p and (p, i) in matches:
                    match.extend([(b, a + base) for a, b in matches[(p, i)]])
                base += datasets[p].shape[0]
                

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]

            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix,
                             sigma=sigma, batch_size=batch_size)
            datasets[i] = curr_ds + bias

            if expr_datasets:
                curr_ds = expr_datasets[i]
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_j[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix,
                                 sigma=sigma, cn=True, batch_size=batch_size)
                expr_datasets[i] = curr_ds + bias

            panoramas_j[0].append(i)

        # Map dataset j to panorama i.
        elif len(panoramas_j) == 0:
            curr_ds = datasets[j]
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_i[0] ])
            curr_types = cell_types[j]
            ref_types = np.concatenate([ cell_types[p] for p in panoramas_i[0] ])
            curr_center_vec = center_vectors[j]
            ref_center_vec = np.concatenate([center_vectors[p] for p in panoramas_i[0]])
            
            match = []
            base = 0
            for p in panoramas_i[0]:
                if j < p and (j, p) in matches:
                    match.extend([ (a, b + base) for a, b in matches[(j, p)] ])
                elif j > p and (p, j) in matches:
                    match.extend([ (b, a + base) for a, b in matches[(p, j)] ])
                base += datasets[p].shape[0]

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]

            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix,
                             sigma=sigma, batch_size=batch_size)
            datasets[j] = curr_ds + bias

            if expr_datasets:
                curr_ds = expr_datasets[j]
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_i[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix,
                                 sigma=sigma, cn=True, batch_size=batch_size)
                expr_datasets[j] = curr_ds + bias

            panoramas_i[0].append(j)

        # Merge two panoramas together.
        else:
            curr_ds = np.concatenate([ datasets[p] for p in panoramas_i[0] ])
            curr_ref = np.concatenate([ datasets[p] for p in panoramas_j[0] ])
            curr_types = np.concatenate([ cell_types[p] for p in panoramas_i[0] ])
            ref_types = np.concatenate([ cell_types[p] for p in panoramas_j[0] ])
            curr_center_vec = np.concatenate([center_vectors[p] for p in panoramas_i[0]])
            ref_center_vec = np.concatenate([center_vectors[p] for p in panoramas_j[0]])

            # Find base indices into each panorama.
            base_i = 0
            for p in panoramas_i[0]:
                if p == i: break
                base_i += datasets[p].shape[0]
            base_j = 0
            for p in panoramas_j[0]:
                if p == j: break
                base_j += datasets[p].shape[0]

            # Find matching indices.
            match = []
            base = 0
            for p in panoramas_i[0]:
                if p == i and j < p and (j, p) in matches:
                    match.extend([ (b + base, a + base_j)
                                   for a, b in matches[(j, p)] ])
                elif p == i and j > p and (p, j) in matches:
                    match.extend([ (a + base, b + base_j)
                                   for a, b in matches[(p, j)] ])
                base += datasets[p].shape[0]
            base = 0
            for p in panoramas_j[0]:
                if p == j and i < p and (i, p) in matches:
                    match.extend([ (a + base_i, b + base)
                                   for a, b in matches[(i, p)] ])
                elif p == j and i > p and (p, i) in matches:
                    match.extend([ (b + base_i, a + base)
                                   for a, b in matches[(p, i)] ])
                base += datasets[p].shape[0]

            ds_ind = [ a for a, _ in match ]
            ref_ind = [ b for _, b in match ]

            # Apply transformation to entire panorama.
            bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix,
                             sigma=sigma, batch_size=batch_size)
            curr_ds += bias
            base = 0
            for p in panoramas_i[0]:
                n_cells = datasets[p].shape[0]
                datasets[p] = curr_ds[base:(base + n_cells), :]
                base += n_cells

            if not expr_datasets is None:
                curr_ds = vstack([ expr_datasets[p]
                                   for p in panoramas_i[0] ])
                curr_ref = vstack([ expr_datasets[p]
                                    for p in panoramas_j[0] ])
                bias = transform(curr_ds, curr_ref, ds_ind, ref_ind, curr_types, ref_types, curr_center_vec, ref_center_vec, type_similarity_matrix,
                                 sigma=sigma, cn=True, batch_size=batch_size)
                curr_ds += bias
                base = 0
                for p in panoramas_i[0]:
                    n_cells = expr_datasets[p].shape[0]
                    expr_datasets[p] = curr_ds[base:(base + n_cells), :]
                    base += n_cells

            # Merge panoramas i and j and delete one.
            if panoramas_i[0] != panoramas_j[0]:
                panoramas_i[0] += panoramas_j[0]
                panoramas.remove(panoramas_j[0])

        # Visualize.
        if view_match:
            plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind)

    return datasets

# Sketch-based acceleration of integration.
def integrate_sketch(datasets_dimred, cell_types, type_similarity_matrix, 
                     sketch_method='geosketch', N=10000, batch_key=None, orginal_datasets=None,
                     integration_fn=assemble, integration_fn_args={}):

    from geosketch import gs, uniform

    if sketch_method.lower() == 'geosketch' or sketch_method.lower() == 'gs':
        sampling_fn = gs
    if sketch_method.lower() == 'by_cell_type':
        print('Using by_cell_type sketch method')
        if batch_key is None or orginal_datasets is None:
            raise ValueError('batch_key and orginal_datasets must be provided for by_cell_type sketch method')
        sampling_fn = uniform_sample_by_cell_type
    else:
        sampling_fn = uniform

    
    # Sketch each dataset.
    if sketch_method.lower() != 'by_cell_type':
        sketch_idxs = [
            sorted(set(sampling_fn(X, N, replace=False)))
            if X.shape[0] > N else list(range(X.shape[0]))
            for X in datasets_dimred
        ]
    else:
        sketch_idxs = [
            sorted(set(sampling_fn(X, N, adata, batch_key)))
            if X.shape[0] > N else list(range(X.shape[0]))
            for X, adata in zip(datasets_dimred, orginal_datasets)
        ]
    datasets_sketch = [ X[idx] for X, idx in zip(datasets_dimred, sketch_idxs) ]

    # Integrate the dataset sketches.
    datasets_int = integration_fn(
        datasets_sketch[:],
        [cell_types[i][idx] for i, idx in enumerate(sketch_idxs)],
        type_similarity_matrix,
        **integration_fn_args
    )

    # Apply integrated coordinates back to full data.
    for i, (X_dimred, X_sketch) in enumerate(zip(datasets_dimred, datasets_sketch)):
        X_int = datasets_int[i]

        neigh = NearestNeighbors(n_neighbors=3).fit(X_dimred)
        _, neigh_idx = neigh.kneighbors(X_sketch)

        ds_idxs, ref_idxs = [], []
        for ref_idx in range(neigh_idx.shape[0]):
            for k_idx in range(neigh_idx.shape[1]):
                ds_idxs.append(neigh_idx[ref_idx, k_idx])
                ref_idxs.append(ref_idx)

        bias = transform(X_dimred, X_int, ds_idxs, ref_idxs, 15, batch_size=1000)

        datasets_int[i] = X_dimred + bias

    return datasets_int

# Non-optimal dataset assembly. Simply accumulate datasets into a
# reference.
def assemble_accum(datasets, verbose=VERBOSE, knn=KNN, sigma=SIGMA,
                   approx=APPROX, batch_size=None):
    if len(datasets) == 1:
        return datasets

    for i in range(len(datasets) - 1):
        j = i + 1

        if verbose:
            print('Processing datasets {}'.format((i, j)))

        ds1 = datasets[j]
        ds2 = np.concatenate(datasets[:i+1])
        match = mnn(ds1, ds2, knn=knn, approx=approx)

        ds_ind = [ a for a, _ in match ]
        ref_ind = [ b for _, b in match ]

        bias = transform(ds1, ds2, ds_ind, ref_ind, sigma=sigma,
                         batch_size=batch_size)
        datasets[j] = ds1 + bias

    return datasets

def uniform_sample_by_cell_type(X, N, adata, batch_key):
    """
    According to the cell type, uniformly sample N cells from adata.
    
    Parameters
    ----------
    X: np.ndarray
        The data matrix.
    N: int
        The number of cells to sample.
    adata: AnnData
        The AnnData object. 
    batch_key: str
        The batch key in adata.obs.
    """
    # Check if batch_key is in adata.obs
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    # Get the unique cell types
    batches = adata.obs[batch_key].unique()
    batch_counts = adata.obs[batch_key].value_counts()
    sorted_batches = batch_counts.sort_values().index
    sampled_indices = []

    total_batches = len(sorted_batches)
    samples_per_batch = N // total_batches
    remainder = N % total_batches

    deficit = 0

    for i, batch in enumerate(sorted_batches):
        batch_indices = adata.obs[adata.obs[batch_key] == batch].reset_index().index
        if i < remainder:
            batch_samples = samples_per_batch + 1 + deficit // (total_batches - i)
        else:
            batch_samples = samples_per_batch + deficit // (total_batches - i)

        if len(batch_indices) < batch_samples:
            sampled_indices.extend(batch_indices)
            deficit += batch_samples - len(batch_indices)
        else:
            sampled_indices.extend(np.random.choice(batch_indices, size=batch_samples, replace=False))
            deficit = 0

    if len(sampled_indices) > N:
        sampled_indices = np.random.choice(sampled_indices, size=N, replace=False)

    return sampled_indices