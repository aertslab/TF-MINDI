"""Create region embeddings from annotated regions."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
)

"""
    Create region embeddings using seqlets
     - count = count vector of seqlet cluster occurence per region
     - pca/vae = mean pool of pca/vae reduced seqlet embeddings per region

    region_adata = tm.tl.embed_regions(seqlet_adata, embedding)
"""


EPSILON = 10**-10
DEF_DICT = {
    'pca':50,
    'vae':10,
    'count':"predicted_5.0_predicted_family",
}


#### Main function --------------------------------------------------------------------------------------------

def embed_regions(
        adata: AnnData,
        embedding: Literal["count","pca","vae"] = "pca",
        secondary: int | str | None = None,
        class_col: str = "cell_type",
        weighted: bool = True,
        normalised: bool = True,
        noise_factor: float = 0.0,
        tsne: bool = True,
        save_path: str = None,
        TSNE_kwargs: dict | None = None,
) -> AnnData:
    """
    Aggregate per-seqlet embeddings into per-region embeddings and return a region AnnData.

    For each unique region (example_idx), seqlet-level embeddings are aggregated
    (optionally weighted and normalised) into a single vector using the specified
    embedding strategy. The result is returned as an AnnData object with one row
    per region, optionally with a t-SNE embedding computed and stored in obsm['X_tsne'].

    Parameters
    ----------
    adata : AnnData
        Input seqlet-level AnnData object. Must contain 'example_idx' and class_col
        in obs, and embeddings in obsm or layers depending on the embedding strategy.
    embedding : {"pca", "vae", "count"}, optional
        Aggregation strategy:
        - "pca"   : mean-aggregate PCA latent vectors (weighted/normalised).
        - "vae"   : mean-aggregate VAE latent vectors (weighted/normalised).
        - "count" : count-based aggregation with optional noise injection.
        Default is "pca".
    secondary : int, str or None, optional
        Secondary dimensionality parameter passed to the aggregation function.
        Interpretation depends on the embedding strategy. Default is None.
    class_col : str, optional
        Column in adata.obs containing cell type or class labels, carried over
        into the region AnnData obs. Default is "cell_type".
    weighted : bool, optional
        Whether to weight seqlets during aggregation. Default is True.
    normalised : bool, optional
        Whether to normalise embeddings during aggregation. Default is True.
    noise_factor : float, optional
        Standard deviation of Gaussian noise added during count aggregation.
        Only used when embedding="count". Default is 0.0 (no noise).
    tsne : bool, optional
        Whether to compute a t-SNE embedding of the region vectors and store
        it in region_adata.obsm['X_tsne']. Default is True.
    save_path : str or None, optional
        If provided, the region AnnData is written to this path as an .h5ad file.
        Default is None.
    TSNE_kwargs : dict or None, optional
        Additional keyword arguments forwarded to sklearn.manifold.TSNE.
        Only used when tsne=True. Default is None.

    Returns
    -------
    AnnData
        Region-level AnnData object with:
        - .X           : aggregated embedding matrix, shape (n_regions, n_dims)
        - .obs         : region metadata including example_idx and class_col
        - .obsm['X_tsne']: t-SNE embedding (only if tsne=True)
    """
    # Check and clean input
    embedding, secondary, w, n = _sanity_checks_and_fixes(adata, embedding, secondary, weighted, normalised)

    # Calculate weights
    weight_df = _calc_weights(adata, weighted)

    # Calculate embeddings
    print(f" [embed] Calculating {embedding} embeddings ", end="")
    latent = None
    example_idx = None
    match embedding:

        case "pca" | "vae":
            latent, example_idx = _mean_aggregate(adata, embedding, secondary, w, n, weight_df)

        case "count":
            latent, example_idx, _ = _count_aggragate(secondary, w, n, weight_df, noise_factor)

    # Create region adata object
    region_adata = AnnData(
        X=latent,
        obs=pd.DataFrame(index=example_idx).merge(
                adata.obs[['example_idx',class_col]].drop_duplicates(),
                left_index=True,
                right_on='example_idx',
                how='left'),
    )

    # Calculate TSNE reduction if asked
    if tsne:
        calculate_embedding_tsne(region_adata, TSNE_kwargs)

    # Save on disk
    if save_path:
        region_adata.write_h5ad(save_path)

    return region_adata


def calculate_embedding_tsne(region_adata: AnnData, TSNE_kwargs: dict) -> np.ndarray:
    """
    Compute a t-SNE embedding of region_adata.X and store it in obsm['X_tsne'].

    Runs scikit-learn's TSNE on the feature matrix of the input AnnData object.
    Default parameters (n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    can be overridden via TSNE_kwargs.

    Parameters
    ----------
    region_adata : AnnData
        AnnData object whose .X matrix will be embedded. The resulting
        embedding is stored in-place under region_adata.obsm['X_tsne'].
    TSNE_kwargs : dict
        Additional keyword arguments passed to sklearn.manifold.TSNE.
        Any keys provided will override the defaults.

    Returns
    -------
    None
        The embedding is written to region_adata.obsm['X_tsne'] in-place.
        Shape: (n_obs, n_components).
    """
    _kw: dict = {
        'n_components': 2,
        'perplexity': 30,
        'random_state': 42,
        'n_jobs': -1,
    }

    if TSNE_kwargs:
        _kw.update(TSNE_kwargs)
    tsne_obj = TSNE(**_kw)
    print(" [embed] Calculating TSNE reduction...")
    region_adata.obsm['X_tsne'] = tsne_obj.fit_transform(region_adata.X)


def leiden_clustering(region_adata: AnnData, resolution: float = 5.0, use_rep: str = 'X') -> None:
    """
    Run Leiden clustering on a region AnnData object and store results in obs['leiden'].

    Builds a KNN graph (k=12, Euclidean) from the specified embedding, runs Leiden
    community detection, and prefixes cluster labels with 'l' (e.g. '0' -> 'l0').

    Parameters
    ----------
    region_adata : AnnData
        AnnData object to cluster. The KNN graph and clustering results are
        stored in-place.
    resolution : float, optional
        Leiden resolution parameter. Higher values produce more clusters.
        Default is 5.0.
    use_rep : str, optional
        Key in obsm to use as the embedding for KNN graph construction.
        Use 'X' to use the raw feature matrix. Default is 'X'.

    Returns
    -------
    None
        Results are written in-place:
        - region_adata.obsp['connectivities'] and ['distances'] (KNN graph)
        - region_adata.obs['leiden'] (cluster labels prefixed with 'l')
    """
    sc.pp.neighbors(region_adata, use_rep=use_rep, n_neighbors=12, metric="euclidean")
    sc.tl.leiden(region_adata, resolution=resolution, key_added='leiden', flavor='igraph')
    region_adata.obs['leiden'] = [f"l{c}" for c in list(region_adata.obs['leiden'])]


def optimal_hierarchical_clustering(
        region_adata: AnnData,
        metric: Literal["ARI","AMI","FMI","homogeneity","completeness","V_measure"] = "ARI",
        class_col: str = 'cell_type',
        precomped_cluster: str = None,
        cluster_name: str = 'region_cluster',
        lower_cut: float = 0.75,
        seed_resolution: float = 5.0,
        override_cut: float = None,
) -> ():

    # Seed the embedding space with tiny leiden clusters
    if precomped_cluster is not None:
        print(f" [clustering] Using precomputed cluster column '{precomped_cluster}'")
        region_adata.obs['leiden'] = region_adata.obs[precomped_cluster]
    else:
        print(" [clustering] Seeding embedding space with leiden clusters...")
        leiden_clustering(region_adata, resolution=seed_resolution)

    # Calculate mean vectors for each leiden seed
    print(" [clustering] Build hierarchical tree...")
    region_vectors = pd.DataFrame(region_adata.X)
    region_vectors['leiden'] = list(region_adata.obs['leiden'])
    region_clusters = region_vectors.groupby('leiden').mean()

    # Build hierarchical tree of leiden seeds
    dist = pdist(region_clusters.to_numpy(), metric="cosine")
    Z = linkage(dist, method="average")

    if override_cut is not None:
        print(f" [clustering] Using override cut height {override_cut}")
        optimal_res = override_cut
        optimal_ari = None
        resolutions = [override_cut]
        results = []

        clusters = fcluster(Z, t=override_cut, criterion="distance")
        new_clusters = dict(zip(region_clusters.index, [f"H{c-1}" for c in clusters], strict=False))
        region_adata.obs[cluster_name] = [new_clusters[c] for c in region_adata.obs['leiden']]
        print(f" [clustering] Saved clusters in .obs {cluster_name}")

    else:
        # Try 100 cutting heights for the hierarchical tree
        resolutions = [i/100 for i in range(100)]
        results = []
        for res in resolutions:
            print(f"\r [clustering] Cutting at height {res}", end="", flush=True)

            clusters = fcluster(Z, t=res, criterion="distance")
            new_clusters = dict(zip(region_clusters.index, [f"H{c-1}" for c in clusters], strict=False))
            region_adata.obs[cluster_name] = [new_clusters[c] for c in region_adata.obs['leiden']]

            labels_pred = region_adata.obs[cluster_name]
            labels_true = region_adata.obs[class_col]
            n_clusters = labels_pred.nunique()
            ari   = adjusted_rand_score(labels_true, labels_pred)
            ami   = adjusted_mutual_info_score(labels_true, labels_pred)
            fmi   = fowlkes_mallows_score(labels_true, labels_pred)
            h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)

            results.append({
                'resolution': res, 'n_clusters': n_clusters,
                'ARI': ari, 'AMI': ami, 'FMI': fmi,
                'homogeneity': h, 'completeness': c, 'V_measure': v
            })

        resolutions = [r['resolution'] for r in results]
        optimal_res = resolutions[np.argmax([r['ARI'] for r in results])]
        optimal_ari = np.max([r['ARI'] for r in results])
        print(f"\n [clustering] The optimal resolution is {optimal_res} with an ARI of {optimal_ari}")

        clusters = fcluster(Z, t=optimal_res * lower_cut, criterion="distance")
        new_clusters = dict(zip(region_clusters.index, [f"H{c-1}" for c in clusters], strict=False))
        region_adata.obs[cluster_name] = [new_clusters[c] for c in region_adata.obs['leiden']]
        print(f" [clustering] Saved clusters in .obs {cluster_name}")

    # Plot
    if results:
        print(" [clustering] Plotting...")
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(resolutions, [r['ARI']          for r in results], label='ARI', color='k')
        ax1.plot(resolutions, [r['homogeneity']  for r in results], label='homogeneity')
        ax1.plot(resolutions, [r['completeness'] for r in results], label='completeness')
        ax1.plot(resolutions, [r['V_measure']    for r in results], label='V_measure')
        ax1.axvline(optimal_res, color='k', linestyle='--', label='max ARI')
        ax1.axvline(optimal_res * lower_cut, color='gray', linestyle='--', label='max ARI x lower cut')
        ax1.set_ylabel('Cluster pureness metrics')
        ax1.set_xlabel('Cutting heights')
        ax1.set_title('ARI score with respect to cut height for region clusters')

        ax2 = ax1.twinx()
        ax2.plot(resolutions, [r['n_clusters'] for r in results], label='N', color='black', linestyle=':')
        ax2.set_ylabel('N clusters')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        plt.tight_layout()
        plt.show()
    else:
        print(" [clustering] Skipping plot (override_cut was used, no sweep results to display)")

    return (optimal_res, optimal_ari)


def get_region_profiles(
        seqlet_adata: AnnData,
        region_adata: AnnData,
        annotation_col: str = 'seqlet_cluster',
        weight_col: str = 'attribution',
        region_cluster_col: str = 'region_cluster'
) -> DataFrame:
    """
    Compute row-max-normalised motif contribution profiles per region cluster.

    For each region cluster, sums the per-seqlet attribution scores across all
    seqlets belonging to that cluster, grouped by motif annotation. The resulting
    summed scores are row-max-normalised (each row divided by its maximum value)
    so that values lie in [0, 1] and clusters are comparable regardless of size.

    As a side effect, the list of annotation labels is stored in
    region_adata.uns[annotation_col] for downstream use.

    Parameters
    ----------
    seqlet_adata : AnnData
        Seqlet-level AnnData object. Must contain 'example_idx', weight_col,
        and annotation_col in obs.
    region_adata : AnnData
        Region-level AnnData object. Must contain 'example_idx' and
        region_cluster_col in obs. The annotation column names are stored
        in region_adata.uns[annotation_col] as a side effect.
    annotation_col : str, optional
        Column in seqlet_adata.obs containing motif/seqlet cluster labels
        (e.g. TF family annotations). Used as columns in the output profile.
        Default is 'seqlet_cluster'.
    weight_col : str, optional
        Column in seqlet_adata.obs containing the per-seqlet contribution
        scores to sum and normalise. Default is 'attribution'.
    region_cluster_col : str, optional
        Column in region_adata.obs containing region cluster labels.
        Used to group regions for aggregation. Default is 'region_cluster'.

    Returns
    -------
    DataFrame
        Row-max-normalised motif profile matrix of shape
        (n_region_clusters, n_annotation_labels), where each value represents
        the normalised summed attribution of a motif annotation within a
        region cluster. Index is region cluster labels, columns are annotation
        labels.

    Side effects
    ------------
    - region_adata.uns[annotation_col] : list of annotation column names.
    - region_adata.uns['normalised_weighted_sum'] : raw numpy array of the
      normalised profiles before DataFrame conversion.
    """
    profile = (seqlet_adata
        .obs[['example_idx',weight_col,annotation_col]]
        .merge(region_adata.obs, on='example_idx',how='left')
        .pivot_table(index='example_idx',columns=annotation_col,values=weight_col,aggfunc='sum'))

    region_adata.uns[annotation_col] = list(profile.columns)

    region_adata.obs = region_adata.obs.merge(
        region_adata.obs[['example_idx']].merge(
            profile, on='example_idx', how='left').fillna(0),on='example_idx',how='left')

    region_adata.uns['normalised_weighted_sum'] = \
        region_adata.obs.groupby(region_cluster_col)[region_adata.uns[annotation_col]].sum().to_numpy() / \
            np.max(region_adata.obs.groupby(region_cluster_col)[region_adata.uns[annotation_col]].sum().to_numpy(), axis=1)[:,None]

    index = region_adata.obs.groupby(region_cluster_col)[region_adata.uns[annotation_col]].sum().index

    region_adata.obs.drop(columns=region_adata.uns[annotation_col], inplace=True)

    return pd.DataFrame(region_adata.uns['normalised_weighted_sum'], index=index, columns=region_adata.uns[annotation_col])

#### Helper functions -----------------------------------------------------------------------------------------

def _sanity_checks_and_fixes(
        adata: AnnData,
        embedding: Literal["count","pca","vae"] = "pca",
        secondary: int | str | None = None,
        weighted: bool = True,
        normalised: bool = True,
):
    w = 'weighted' if weighted else 'unweighted'
    n = 'normalised' if normalised else 'unnormalised'

    vae_keys = [k for k in adata.obsm.keys() if len(k.split('_')) > 1 and k.split('_')[1] == 'vae']

    # --- pca checks ---------------------------------------------------------
    if embedding == "pca":
        assert "X_pca" in adata.obsm, (
            "Must first reduce seqlet space using pca before aggregating using pca.\n"
            "tm.tl.reduce_seqlet_space(adata)"
        )
        if secondary is None:
            secondary = DEF_DICT['pca']

    # --- vae checks ---------------------------------------------------------
    elif embedding == "vae":
        assert len(vae_keys) > 0, (
            "Must first reduce seqlet space using 'vae' before aggregating VAE reduced seqlet embeddings.\n"
            "tm.tl.reduce_seqlet_space(adata, embedding='vae')"
        )
        if secondary is None:
            if len(vae_keys) == 1:
                secondary = int(vae_keys[0].split('_')[2])
            elif f"X_vae_{DEF_DICT['vae']}" in adata.obsm:
                secondary = DEF_DICT['vae']
            else:
                raise ValueError(
                    f"Multiple VAE reductions found: {vae_keys}. "
                    f"Please specify which to use via 'secondary'."
                )
        assert f"X_vae_{secondary}" in adata.obsm, (
            f"VAE reduction with {secondary} latents not found.\n"
            f"Available: {vae_keys}\n"
            f"tm.tl.reduce_seqlet_space(adata, embedding='vae', latents={secondary})"
        )

    # --- count checks -------------------------------------------------------
    elif embedding == "count":
        if secondary is None:
            secondary = DEF_DICT['count']
            print(f" [embed] No annotation column specified, defaulting to '{secondary}'")
        if secondary not in adata.obs.columns:
            raise KeyError(
                f"Annotation column '{secondary}' not found in adata.obs.\n"
                f"Available columns: {list(adata.obs.columns)}"
            )

    print(f" [embed] embedding='{embedding}', secondary='{secondary}', {w}, {n}")

    return embedding, secondary, w, n


def _calc_weights(adata: AnnData, weighted: bool) -> np.ndarray:

    weight_df = adata.obs.copy()
    weight_df['weight'] = np.ones((adata.obs.shape[0],1))

    if weighted:
        print(" [embed] Calculating weights")
        weight_df['attribution'] = weight_df['attribution'].fillna(0)
        weight_df['att_abs'] = weight_df['attribution'].abs()
        weight_df = weight_df.merge(
            weight_df[['example_idx','att_abs']].groupby('example_idx').agg('sum').rename(
                columns={'att_abs':'att_sum'})['att_sum'], on='example_idx')
        weight_df['att_softmax'] = weight_df.apply(lambda r: (r['attribution']+EPSILON)/(r['att_sum']+EPSILON), axis=1)
        weight_df['weight'] = np.asarray(weight_df['att_softmax'])[:, None]

    return weight_df


def _mean_aggregate(adata: AnnData, reduction: str, latent: int, w: str, n:str, weight_df: pd.DataFrame) -> dict:

    reduction_key = 'X_pca' if reduction == 'pca' else f'X_vae_{latent}'

    print(f"on obsm.{reduction_key}")

    ### Grab n_seqlets x first 'latent' latents from reduction and multiply seqlets with weights
    region_df = pd.DataFrame(data=adata.obsm[reduction_key][:,:latent] * weight_df['weight'].values[:, None],
                            index=adata.obs['example_idx']).groupby('example_idx').agg('mean')

    return ((region_df.to_numpy() / np.max(np.abs(region_df.to_numpy()), axis=1)[:,None] if n == 'normalised' else region_df.to_numpy()), list(region_df.index))


def _count_aggragate(annotation_column: str, w: str, n:str, weight_df: pd.DataFrame, noise_factor: float) -> dict:

    print(f"by constructing count vectors at adata.obs['{annotation_column}'] resolution")

    region_df = (weight_df.groupby(["example_idx", annotation_column],
                observed=True)["weight"].sum().unstack(fill_value=0))

    noise = np.random.uniform(0,np.mean(region_df.to_numpy()),region_df.to_numpy().shape)

    return (region_df.to_numpy() / np.max(np.abs(region_df.to_numpy()) + noise * noise_factor, axis=1)[:,None] if n == 'normalised' else region_df.to_numpy(), list(region_df.index), list(region_df.columns))
