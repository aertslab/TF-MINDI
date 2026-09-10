import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tfmindi.pl._utils import render_plot

def ct_by_ct_heatmap(
    region_adata,
    label_1,
    label_2,
    row_order=None,
    col_order = None,
    cmap='Reds',
    **kwargs,
    ):
    """
    Plot the normalized enrichment score for the local connectivites of cell types 
    of label_1 and label_2 based on a region-level neighbor graph.

    Parameters
    ----------
    region_adata : AnnData
        AnnData object containing:
        - region_adata.obs[ct_key]: cell type annotation per region
        - region_adata.obs[model_key]: model annotation per region
        - region_adata.obsp[conn_key]: neighbor graph (sparse, typically CSR)
        - region_adata.uns['ct_by_ct']: dictionary containing the following keys:
            - 'obs': observed cross-model neighbor fractions (pandas.DataFrame)
            - 'nes': normalized enrichment scores (pandas.DataFrame)
            - 'mu': mean values (pandas.DataFrame)
            - 'sd': standard deviation values (pandas.DataFrame)    
    
    label_1 : str
        Label used for first collection of cell type regions in region_adata.obs[model_key].
    
    label_2 : str
        Label used for second collection of cell type regions in region_adata.obs[model_key].

    row_order : list of strings, optional
        Order of rows (cell types of label_1) to display in the heatmap.
    
    col_order : list of strings, optional
        Order of columns (cell types of label_2) to display in the heatmap.
    
    cmap : str, optional
        Colormap for the heatmap. Default is 'Reds'.
    """

    ## Filter out invalid rows and columns (all NaN) from the NES dataframe
    nes_df = region_adata.uns['ct_by_ct']['nes']
    nes_df = nes_df[~nes_df.isna().all(axis=1)]
    nes_df = nes_df.loc[:,~nes_df.isna().all()]

    if col_order is not None:
        nes_df = nes_df[col_order]
    else:
        column_order = np.argsort(np.argmax(nes_df, 0))
        nes_df = nes_df[nes_df.columns[column_order]]
    
    if row_order is not None:
        row_order = np.where(row_order)[0]
        nes_df = nes_df.iloc[row_order, :]
    else:
        row_order = np.argsort(np.argmax(nes_df, 1))
        nes_df = nes_df.iloc[row_order, :]

    fig = plt.figure()
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(20, 1),
        width_ratios=(3, 1),
        hspace=0.3,
    )
    ax = fig.add_subplot(grid[0, :])
    masked_nes = np.ma.masked_invalid(nes_df.to_numpy())
    image = ax.imshow(masked_nes, cmap=cmap, vmin=0, aspect="auto")

    ax.set_xticks(np.arange(nes_df.shape[1]))
    ax.set_xticklabels(nes_df.columns)
    ax.set_yticks(np.arange(nes_df.shape[0]))
    ax.set_yticklabels(nes_df.index)
    ax.set_xlabel(f"{label_2} cell type")
    ax.set_ylabel(f"{label_1} cell type")

    cbar_ax = fig.add_subplot(grid[1, 1])
    fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar_ax.xaxis.set_ticks_position("bottom")
    cbar_ax.set_title("Normalized\nEnrichment Score (NES)", pad=8)
    
    render_kwargs = {
        "title": f"Region connectivity between {label_1} and {label_2} cell types",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)