import numpy as np
import pandas as pd
import scipy.sparse as sp
from scanpy.pp import neighbors

def ct_by_ct_obs_matrix(
    region_adata,
    label_1,
    label_2,
    ct_key="cell_type",
    model_key="model",
    conn_key="connectivities",
):
    """
    Compute the observed label_1 label_2 cross-cell-type connectivity matrix
    from a region-level neighbor graph.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing:
        - adata.obs[ct_key]: cell type annotation per region
        - adata.obs[model_key]: model annotation per region
        - adata.obsp[conn_key]: neighbor graph (sparse, typically CSR)

    ct_key : str
        Column in adata.obs defining cell type labels.

    model_key : str
        Column in adata.obs defining model labels.

    conn_key : str
        Key in adata.obsp storing the neighbor graph.

    label_1 : str
        Label used for first collection of cell type regions in adata.obs[model_key].

    label_2 : str
        Label used for second collection of cell type regions in adata.obs[model_key].

    Returns
    -------
    obs_df : pandas.DataFrame (n_ct × n_ct)
        Observed cross-model neighbor fractions.
        Rows = label 1 cell types
        Columns = label 2 cell types

        Entry (h, m) is:
            (# edges from label 1 CT h → label 2 CT m)
            ------------------------------------------------
            (total # edges from label 1 CT h → all label 2 regions)

        Rows may contain NaNs if label 1 CT h has zero label_1 label_2 edges.

    cross : numpy.ndarray (n_ct × n_ct)
        Raw weighted edge counts from label 1 CTs to label 2 CTs.
        Same layout as obs_df, but unnormalized.

    denom : numpy.ndarray (n_ct,)
        Denominator per label 1 cell type:
        total number (or weight) of edges from label 1 CT h to *any* label 2 region.

    ct_list : list[str]
        Ordered list of cell types defining the row/column order.

    h_idx : numpy.ndarray
        Indices of label 1 regions in adata.obs.

    m_idx : numpy.ndarray
        Indices of label 2 regions in adata.obs.

    C_hm : scipy.sparse.csr_matrix
        Submatrix of the neighbor graph containing only
        label 1 rows × label 2 columns.

    H : scipy.sparse.csr_matrix
        One-hot membership matrix for label 1 regions:
        shape = (n_label 1_regions × n_ct)

    M : scipy.sparse.csr_matrix
        One-hot membership matrix for label 2 regions:
        shape = (n_label 2_regions × n_ct)
    """

    if conn_key not in region_adata.obsp:
        neighbors(region_adata)

    C = region_adata.obsp[conn_key].tocsr()

    obs = region_adata.obs
    ct_cat = pd.Categorical(obs[ct_key])
    ct_list = list(ct_cat.categories)
    n_ct = len(ct_list)

    is_h = (obs[model_key].values == label_1)
    is_m = (obs[model_key].values == label_2)

    h_idx = np.where(is_h)[0]
    m_idx = np.where(is_m)[0]

    # Label_1->Label_2 block once
    C_hm = C[h_idx, :][:, m_idx].tocsr()

    # Map ct to 0..n_ct-1
    ct_codes = ct_cat.codes  # -1 if missing, but should not happen
    h_codes = ct_codes[h_idx]
    m_codes = ct_codes[m_idx]

    # Build one-hot membership matrices (sparse)
    H = sp.csr_matrix(
        (np.ones_like(h_codes, dtype=np.float32),
         (np.arange(len(h_codes)), h_codes)),
        shape=(len(h_codes), n_ct)
    )
    M = sp.csr_matrix(
        (np.ones_like(m_codes, dtype=np.float32),
         (np.arange(len(m_codes)), m_codes)),
        shape=(len(m_codes), n_ct)
    )

    # cross edges: (ct_h x ct_m)
    cross = (H.T @ (C_hm @ M)).toarray()

    # denom: total label 1->label 2 weight per label 1 ct
    denom_vec = (H.T @ (C_hm @ np.ones((len(m_codes), 1), dtype=np.float32)))
    denom = np.asarray(denom_vec).ravel()

    # Avoid divide-by-zero for rare CTs
    obs_mat = np.divide(cross, denom[:, None], out=np.full_like(cross, np.nan), where=(denom[:, None] > 0))

    obs_df = pd.DataFrame(obs_mat, index=ct_list, columns=ct_list)
    return obs_df, cross, denom, ct_list, h_idx, m_idx, C_hm, H, M

def ct_by_ct_nes(
    region_adata,
    label_1,
    label_2,
    n_perm=200,
    seed=0,
    ct_key="cell_type",
    model_key="model",
    conn_key="connectivities"
):
    """
    Compute normalized enrichment scores (NES) for cell-type connectivity
    cell-type connectivity using permutation testing for two labels (model systems or model).

    This function compares the observed neighbor connectivity
    to a null distribution obtained by permuting label_1 cell-type labels
    across label_1 regions.

    Parameters
    ----------
    adata : AnnData
        AnnData object with region-level annotations and neighbor graph.

    n_perm : int
        Number of permutations used to estimate the null distribution.

    seed : int
        Random seed for reproducibility.

    ct_key, model_key, conn_key, label_1, label_2
        Same meaning as in ct_by_ct_obs_matrix.

    Returns
    -------
    obs_df : pandas.DataFrame (n_ct × n_ct)
        Observed label_1 label_2 connectivity fractions.
        Same as returned by ct_by_ct_obs_matrix.

    nes_df : pandas.DataFrame (n_ct × n_ct)
        Normalized enrichment scores (z-scores):

            NES(h, m) = (obs(h, m) - mean_null(h, m)) / sd_null(h, m)

        NaN if:
        - no label_1 label_2 edges for CT h
        - null standard deviation is zero

    mu_df : pandas.DataFrame (n_ct × n_ct)
        Mean of the permutation null distribution for each CT pair.

    sd_df : pandas.DataFrame (n_ct × n_ct)
        Standard deviation of the permutation null distribution
        for each CT pair.
    """

    obs_df, cross, denom, ct_list, h_idx, m_idx, C_hm, H, M = ct_by_ct_obs_matrix(
        region_adata,
        ct_key=ct_key,
        model_key=model_key,
        conn_key=conn_key,
        label_1=label_1,
        label_2=label_2,
    )

    # Observed fractions as numpy
    obs_mat = obs_df.values.astype(np.float64)

    rng = np.random.default_rng(seed)

    n_ct = len(ct_list)
    n_m = len(m_idx)

    # label 2 ct codes (0..n_ct-1), permuted each round
    ct_cat = pd.Categorical(region_adata.obs[ct_key])
    m_codes0 = ct_cat.codes[m_idx].copy()

    # Online mean/var (Welford)
    mean = np.zeros((n_ct, n_ct), dtype=np.float64)
    M2   = np.zeros((n_ct, n_ct), dtype=np.float64)

    ones_m = np.ones((n_m, 1), dtype=np.float32)
    denom_safe = denom.astype(np.float64)
    denom_safe[denom_safe == 0] = np.nan  # keep divisions sane

    for t in range(1, n_perm + 1):
        perm_codes = rng.permutation(m_codes0)

        # Build permuted label 2 one-hot (sparse)
        M_perm = sp.csr_matrix(
            (np.ones_like(perm_codes, dtype=np.float32),
             (np.arange(n_m), perm_codes)),
            shape=(n_m, n_ct)
        )

        cross_p = (H.T @ (C_hm @ M_perm)).toarray().astype(np.float64)
        frac_p = cross_p / denom_safe[:, None]   # (ct_h x ct_m)

        # update online mean/var
        delta = frac_p - mean
        mean += delta / t
        delta2 = frac_p - mean
        M2 += delta * delta2

    var = M2 / max(n_perm - 1, 1)
    sd = np.sqrt(var)

    nes = (obs_mat - mean) / sd
    nes[sd == 0] = np.nan

    nes_df = pd.DataFrame(nes, index=ct_list, columns=ct_list)
    mu_df  = pd.DataFrame(mean, index=ct_list, columns=ct_list)
    sd_df  = pd.DataFrame(sd, index=ct_list, columns=ct_list)

    region_adata.uns['ct_by_ct'] = {
        'obs': obs_df,
        'nes': nes_df,
        'mu': mu_df,
        'sd': sd_df,
        'label_1': label_1,
        'label_2': label_2
    }
    return

def cell_type_obs_matrix(
    region_adata,
    ct_key="cell_type",
    conn_key="connectivities",
):
    """
    Compute the observed cell-type by cell-type neighbor connectivity
    matrix across *all* regions in the dataset (no model/label split).

    This is the all-vs-all counterpart of `ct_by_ct_obs_matrix`: instead
    of comparing two model/label groups (e.g. label 1 vs. label 2), every
    region is compared against every other region using its own
    cell type annotation.

    Parameters
    ----------
    region_adata : AnnData
        AnnData object containing:
        - region_adata.obs[ct_key]: cell type annotation per region
        - region_adata.obsp[conn_key]: neighbor graph (sparse, typically CSR)

    ct_key : str
        Column in region_adata.obs defining cell type labels.

    conn_key : str
        Key in region_adata.obsp storing the neighbor graph.

    Returns
    -------
    obs_df : pandas.DataFrame (n_ct × n_ct)
        Observed cross-cell-type neighbor fractions.

        Entry (i, j) is:
            (# edges from CT i → CT j)
            ------------------------------------------
            (total # edges from CT i → all regions)

        Rows may contain NaNs if CT i has zero edges.

    cross : numpy.ndarray (n_ct × n_ct)
        Raw weighted edge counts from CT i to CT j.
        Same layout as obs_df, but unnormalized.

    denom : numpy.ndarray (n_ct,)
        Denominator per cell type: total number (or weight) of
        edges from CT i to any region.

    ct_list : list[str]
        Ordered list of cell types defining the row/column order.

    ct_codes : numpy.ndarray
        Cell type code (0..n_ct-1) for every region in region_adata.obs.

    C : scipy.sparse.csr_matrix
        Full neighbor graph (n_regions × n_regions).

    G : scipy.sparse.csr_matrix
        One-hot membership matrix for all regions:
        shape = (n_regions × n_ct)
    """

    if conn_key not in region_adata.obsp:
        neighbors(region_adata)

    C = region_adata.obsp[conn_key].tocsr()

    obs = region_adata.obs
    ct_cat = pd.Categorical(obs[ct_key])
    ct_list = list(ct_cat.categories)
    n_ct = len(ct_list)

    ct_codes = ct_cat.codes  # -1 if missing, but should not happen
    n = len(ct_codes)

    # One-hot membership matrix for all regions
    G = sp.csr_matrix(
        (np.ones_like(ct_codes, dtype=np.float32),
         (np.arange(n), ct_codes)),
        shape=(n, n_ct)
    )

    # cross edges: (ct_i x ct_j), over all regions
    cross = (G.T @ (C @ G)).toarray()

    # denom: total weight from CT i to any region
    denom_vec = (G.T @ (C @ np.ones((n, 1), dtype=np.float32)))
    denom = np.asarray(denom_vec).ravel()

    # Avoid divide-by-zero for rare CTs
    obs_mat = np.divide(cross, denom[:, None], out=np.full_like(cross, np.nan), where=(denom[:, None] > 0))

    obs_df = pd.DataFrame(obs_mat, index=ct_list, columns=ct_list)
    return obs_df, cross, denom, ct_list, ct_codes, C, G

def cell_type_nes(
    region_adata,
    n_perm=200,
    seed=0,
    ct_key="cell_type",
    conn_key="connectivities"
):
    """
    Compute normalized enrichment scores (NES) for all-vs-all
    cell-type connectivity using permutation testing.

    This is the all-vs-all counterpart of `ct_by_ct_nes`: instead of
    comparing two model/label groups, it evaluates, for every cell
    type pair (i, j), whether regions of cell type i are more (or
    less) connected to regions of cell type j than expected by
    chance, using the full neighbor graph of `region_adata`.

    The null distribution is built by permuting cell type labels
    (across all regions) on the "target" side of the connectivity,
    while keeping the "source" cell type grouping fixed - mirroring
    the permutation scheme used in `ct_by_ct_nes`.

    Parameters
    ----------
    region_adata : AnnData
        AnnData object with region-level annotations and neighbor graph.

    n_perm : int
        Number of permutations used to estimate the null distribution.

    seed : int
        Random seed for reproducibility.

    ct_key, conn_key
        Same meaning as in `cell_type_obs_matrix`.

    Returns
    -------
    obs_df : pandas.DataFrame (n_ct × n_ct)
        Observed all-vs-all cell-type connectivity fractions.
        Same as returned by `cell_type_obs_matrix`.

    nes_df : pandas.DataFrame (n_ct × n_ct)
        Normalized enrichment scores (z-scores):

            NES(i, j) = (obs(i, j) - mean_null(i, j)) / sd_null(i, j)

        NaN if:
        - no edges for CT i
        - null standard deviation is zero

    mu_df : pandas.DataFrame (n_ct × n_ct)
        Mean of the permutation null distribution for each CT pair.

    sd_df : pandas.DataFrame (n_ct × n_ct)
        Standard deviation of the permutation null distribution
        for each CT pair.
    """

    obs_df, cross, denom, ct_list, ct_codes0, C, G = cell_type_obs_matrix(
        region_adata,
        ct_key=ct_key,
        conn_key=conn_key,
    )

    # Observed fractions as numpy
    obs_mat = obs_df.values.astype(np.float64)

    rng = np.random.default_rng(seed)

    n_ct = len(ct_list)
    n = len(ct_codes0)

    # Online mean/var (Welford)
    mean = np.zeros((n_ct, n_ct), dtype=np.float64)
    M2   = np.zeros((n_ct, n_ct), dtype=np.float64)

    denom_safe = denom.astype(np.float64)
    denom_safe[denom_safe == 0] = np.nan  # keep divisions sane

    for t in range(1, n_perm + 1):
        perm_codes = rng.permutation(ct_codes0)

        # Build permuted "target"-side one-hot membership (sparse);
        # the "source" side (G) is kept fixed to the real labels.
        G_perm = sp.csr_matrix(
            (np.ones_like(perm_codes, dtype=np.float32),
             (np.arange(n), perm_codes)),
            shape=(n, n_ct)
        )

        cross_p = (G.T @ (C @ G_perm)).toarray().astype(np.float64)
        frac_p = cross_p / denom_safe[:, None]   # (ct_i x ct_j)

        # update online mean/var
        delta = frac_p - mean
        mean += delta / t
        delta2 = frac_p - mean
        M2 += delta * delta2

    var = M2 / max(n_perm - 1, 1)
    sd = np.sqrt(var)

    nes = (obs_mat - mean) / sd
    nes[sd == 0] = np.nan

    nes_df = pd.DataFrame(nes, index=ct_list, columns=ct_list)
    mu_df  = pd.DataFrame(mean, index=ct_list, columns=ct_list)
    sd_df  = pd.DataFrame(sd, index=ct_list, columns=ct_list)

    region_adata.uns['cell_type_nes'] = {
        'obs': obs_df,
        'nes': nes_df,
        'mu': mu_df,
        'sd': sd_df,
    }
    return

