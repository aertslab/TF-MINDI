"""Utility functions for plotting."""

from __future__ import annotations

import random
import warnings
from collections.abc import Mapping

import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import is_numeric_dtype


def _generate_random_colors(n_colors: int, seed: int = 42) -> list[str]:
    """
    Generate n distinct random hex colors.

    Parameters
    ----------
    n_colors
        Number of colors to generate
    seed
        Random seed for reproducibility

    Returns
    -------
    list[str]
        List of hex color strings
    """
    # A private Random instance: seeding the module-level one would reset the caller's
    # global random stream as a side effect of drawing a palette.
    rng = random.Random(seed)
    return [f"#{rng.randint(0, 0xFFFFFF):06x}" for _ in range(n_colors)]


_NA_COLOR = "#D3D3D3"  # lightgray, used for NaN ("Unknown") values


def _categories(values: pd.Series) -> list:
    """Category order used for storing colors, matching scanpy (sorted unique values for non-categoricals)."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    return list(pd.Categorical(values.dropna()).categories)


def _legacy_colors_to_list(adata: AnnData, column: str) -> None:
    """Convert a legacy ``{category: color}`` dict in ``adata.uns[f'{column}_colors']`` to scanpy's list format."""
    color_key = f"{column}_colors"
    colors = adata.uns.get(color_key)
    if not isinstance(colors, Mapping) or column not in adata.obs.columns:
        return
    # Keys read back from HDF5 are strings in lexicographic order, so map by key rather than position
    colors = {str(k): v for k, v in colors.items()}
    categories = _categories(adata.obs[column])
    if all(str(c) in colors for c in categories):
        adata.uns[color_key] = [colors[str(c)] for c in categories]
    else:
        # Incomplete legacy palette: drop it so it gets regenerated
        del adata.uns[color_key]


_NA_COLOR = "#D3D3D3"  # lightgray, used for NaN ("Unknown") values


def _categories(values: pd.Series) -> list:
    """Category order used for storing colors, matching scanpy (sorted unique values for non-categoricals)."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    return list(pd.Categorical(values.dropna()).categories)


def _legacy_colors_to_list(adata: AnnData, column: str) -> None:
    """Convert a legacy ``{category: color}`` dict in ``adata.uns[f'{column}_colors']`` to scanpy's list format."""
    color_key = f"{column}_colors"
    colors = adata.uns.get(color_key)
    if not isinstance(colors, Mapping) or column not in adata.obs.columns:
        return
    # Keys read back from HDF5 are strings in lexicographic order, so map by key rather than position
    colors = {str(k): v for k, v in colors.items()}
    categories = _categories(adata.obs[column])
    if all(str(c) in colors for c in categories):
        adata.uns[color_key] = [colors[str(c)] for c in categories]
    else:
        # Incomplete legacy palette: drop it so it gets regenerated
        del adata.uns[color_key]


def render_plot(
    fig: plt.Figure,
    width: int = 8,
    height: int = 8,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    supxlabel: str | None = None,
    supylabel: str | None = None,
    tight_rect: tuple | None = None,
    title_fontsize: int = 16,
    x_label_fontsize: int = 14,
    y_label_fontsize: int = 14,
    x_tick_fontsize: int = 12,
    y_tick_fontsize: int = 12,
    x_label_rotation: int = 0,
    y_label_rotation: int = 0,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    **kwargs,
) -> None | plt.Figure:
    """
    Render a plot with customization options.

    Note
    ----
    This function should never be called directly. Rather, the other plotting functions call this function.

    Parameters
    ----------
    fig
        The figure object to render.
    width
        Width of the plot (inches).
    height
        Height of the plot (inches).
    title
        Title of the plot.
    xlabel
        Label for the X-axis.
    ylabel
        Label for the Y-axis.
    supxlabel
        Suplabel for the X-axis.
    supylabel
        Suplabel for the Y-axis.
    tight_rect
        Normalized coordinates in which subplots will fit.
    title_fontsize
        Font size for the title.
    x_label_fontsize
        Font size for the X-axis labels.
    y_label_fontsize
        Font size for the Y-axis labels.
    x_tick_fontsize
        Font size for the X-axis ticks.
    y_tick_fontsize
        Font size for the Y-axis ticks
    x_label_rotation
        Rotation of the X-axis labels in degrees.
    y_label_rotation
        Rotation of the Y-axis labels in degrees.
    show
        Whether to display the plot. Set this to False if you want to return the figure object to customize it further.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    dpi
        Resolution for saving figures.
    **kwargs
        Additional arguments passed to plt.savefig().

    Returns
    -------
    None or matplotlib.Figure
        Returns figure object if show=False and save_path=None, otherwise None.
    """
    fig.set_size_inches(width, height)

    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)

    for ax in fig.axes:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=x_label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=y_label_fontsize)

        for label in ax.get_xticklabels():
            label.set_rotation(x_label_rotation)
            label.set_fontsize(x_tick_fontsize)
        for label in ax.get_yticklabels():
            label.set_fontsize(y_tick_fontsize)
            label.set_rotation(y_label_rotation)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "This figure includes Axes that are not compatible with tight_layout")
        if tight_rect:
            fig.tight_layout(rect=tight_rect)
        else:
            fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, **kwargs)

    if show:
        plt.show()

    if not show and not save_path:
        return fig

    return None


def ensure_colors(
    adata: AnnData,
    column: str,
    cmap: str = "tab20",
    force_regenerate: bool = False,
) -> dict[str, str]:
    """
    Ensure colors exist for a categorical column, generating them if needed.

    Colors are stored in adata.uns[f'{column}_colors'] following scanpy conventions: a list of colors
    aligned with the column's categories (sorted unique values for non-categorical columns).
    NaN values are returned as "Unknown" (lightgray) but are not stored.

    Parameters
    ----------
    adata
        AnnData object containing the data
    column
        Column name in adata.obs to generate/retrieve colors for
    cmap
        Matplotlib colormap name to use for color generation
    force_regenerate
        If True, regenerate colors even if they already exist

    Returns
    -------
    dict[str, str]
        Dictionary mapping category values to hex color codes

    Raises
    ------
    ValueError
        If column doesn't exist in adata.obs
    """
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found in adata.obs")

    color_key = f"{column}_colors"
    _legacy_colors_to_list(adata, column)
    values = adata.obs[column]
    unique_values = _categories(values)
    stored = adata.uns.get(color_key)

    if stored is None or force_regenerate or len(stored) < len(unique_values):
        # Generate colors
        colormap = plt.get_cmap(cmap)
        if len(unique_values) <= colormap.N:
            # Use discrete colors from colormap
            colors = [colormap(i) for i in range(len(unique_values))]
        else:
            # Too many categories for the discrete colormap - use fallback chain
            if cmap == "tab10":
                # Upgrade to tab20 if more than 10 categories
                colormap = plt.get_cmap("tab20")
                if len(unique_values) <= 20:
                    colors = [colormap(i) for i in range(len(unique_values))]
                else:
                    # More than 20 categories - use random colors
                    hex_colors = _generate_random_colors(len(unique_values))
                    colors = [mcol.to_rgba(color) for color in hex_colors]
            elif cmap == "tab20" and len(unique_values) > 20:
                # Use random colors for more than 20 categories
                hex_colors = _generate_random_colors(len(unique_values))
                colors = [mcol.to_rgba(color) for color in hex_colors]
            else:
                # For other colormaps, use them as continuous
                colormap = plt.get_cmap(cmap)
                colors = colormap(np.linspace(0, 1, len(unique_values)))

        # Convert to hex colors
        hex_colors = [mcol.to_hex(color) for color in colors]

        # A real "Unknown" category keeps the lightgray convention
        hex_colors = [_NA_COLOR if v == "Unknown" else c for v, c in zip(unique_values, hex_colors, strict=True)]
        # Store in AnnData using scanpy convention: list aligned with the categories
        adata.uns[color_key] = hex_colors
        stored = hex_colors

    # Extra stored colors are ignored, as in scanpy
    color_map = {v: str(c) for v, c in zip(unique_values, stored, strict=False)}
    if values.isnull().any():
        color_map["Unknown"] = _NA_COLOR

    return color_map


def get_colors(
    adata: AnnData,
    column: str,
    cmap: str = "tab20",
) -> dict[str, str]:
    """
    Get colors for a categorical column, generating them if they don't exist.

    Parameters
    ----------
    adata
        AnnData object containing the data
    column
        Column name in adata.obs to get colors for
    cmap
        Matplotlib colormap name to use if colors need to be generated

    Returns
    -------
    dict[str, str]
        Dictionary mapping category values to hex color codes
    """
    return ensure_colors(adata, column, cmap, force_regenerate=False)


def set_colors(
    adata: AnnData,
    column: str,
    color_dict: dict[str, str],
) -> None:
    """
    Manually set colors for a categorical column.

    Parameters
    ----------
    adata
        AnnData object to store colors in
    column
        Column name in adata.obs to set colors for
    color_dict
        Dictionary mapping category values to color codes (hex, named, or RGB).
        Categories not in the dictionary keep their existing (or newly generated) colors.
        Colors for NaN values cannot be stored and are ignored.
    """
    color_map = ensure_colors(adata, column)
    categories = _categories(adata.obs[column])
    unknown = [value for value in color_dict if value not in categories]
    if unknown:
        warnings.warn(f"Ignoring colors for values not among the categories of '{column}': {unknown}", stacklevel=2)

    for value, color in color_dict.items():
        try:
            color_map[value] = mcol.to_hex(color)
        except ValueError:
            # If conversion fails, keep original (might be a valid color name)
            color_map[value] = color

    # Store using scanpy convention: list aligned with the categories
    adata.uns[f"{column}_colors"] = [color_map[c] for c in categories]


def reset_colors(
    adata: AnnData,
    column: str | None = None,
) -> None:
    """
    Reset stored colors, either for a specific column or all columns.

    Parameters
    ----------
    adata
        AnnData object containing stored colors
    column
        Column name to reset colors for. If None, reset all colors.
    """
    if column is None:
        # Reset all colors (find all keys ending with '_colors')
        color_keys = [key for key in adata.uns.keys() if key.endswith("_colors")]
        for key in color_keys:
            del adata.uns[key]
    else:
        # Reset specific column
        color_key = f"{column}_colors"
        if color_key in adata.uns:
            del adata.uns[color_key]


def get_point_colors(
    adata: AnnData,
    column: str,
    cmap: str = "tab20",
    use_stored_colors: bool = True,
) -> tuple[list[str], dict[str, str] | None]:
    """
    Get point colors for plotting based on a categorical column.

    Parameters
    ----------
    adata
        AnnData object containing the data
    column
        Column name in adata.obs to color by
    cmap
        Matplotlib colormap name (used if not using stored colors)
    use_stored_colors
        Whether to use/generate stored colors or create fresh ones

    Returns
    -------
    tuple[list[str], dict[str, str] | None]
        - List of colors for each point
        - Color mapping dictionary (None for continuous data)
    """
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found in adata.obs")

    color_values = adata.obs[column]

    # Handle categorical data. Treat any non-numeric column (category, object, or
    # the pandas string dtype used by newer pandas versions) as discrete/categorical.
    if not is_numeric_dtype(color_values):
        if use_stored_colors:
            color_map = get_colors(adata, column, cmap)
        else:
            # Generate fresh colors without storing
            if color_values.dtype == "category":
                if "Unknown" not in color_values.cat.categories:
                    color_values = color_values.cat.add_categories(["Unknown"])
                color_values = color_values.fillna("Unknown")
            else:
                color_values = color_values.fillna("Unknown")

            unique_values = color_values.unique()
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(unique_values)))
            color_map = dict(zip(unique_values, colors, strict=False))
            if "Unknown" in color_map:
                color_map["Unknown"] = "lightgray"

        # Create a copy for color mapping and convert NaN to "Unknown"
        color_values_for_mapping = color_values.copy()
        if color_values.dtype == "category":
            if "Unknown" not in color_values_for_mapping.cat.categories:
                color_values_for_mapping = color_values_for_mapping.cat.add_categories(["Unknown"])
            color_values_for_mapping = color_values_for_mapping.fillna("Unknown")
        else:
            color_values_for_mapping = color_values_for_mapping.fillna("Unknown")

        point_colors = [color_map[val] for val in color_values_for_mapping]
        return point_colors, color_map
    else:
        # Continuous data - return values directly
        return color_values, None
