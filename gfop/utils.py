# Standard library imports
import os
import pkg_resources
from importlib import resources
from typing import List, Optional

# Third-party imports
import pandas as pd


def _load_food_metadata() -> pd.DataFrame:
    """
    Reads Global FoodOmics ontology and metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing Global FoodOmics ontology and metadata.
    """
    # Use importlib.resources if possible; fall back to pkg_resources
    try:
        with resources.open_text(
            "data", "foodomics_multiproject_metadata.txt"
        ) as stream:
            gfop_metadata = pd.read_csv(stream, sep="\t")
    except (ModuleNotFoundError, ImportError):
        stream = pkg_resources.resource_stream(
            __name__, "data/foodomics_multiproject_metadata.txt"
        )
        gfop_metadata = pd.read_csv(stream, sep="\t")
    return gfop_metadata


def _load_sample_types(
    gfop_metadata: pd.DataFrame, simple_complex: str = "all"
) -> pd.DataFrame:
    """
    Filters Global FoodOmics metadata by simple, complex, or all types of foods.

    Parameters
    ----------
    gfop_metadata : pd.DataFrame
        The metadata DataFrame to filter.
    simple_complex : str, optional
        One of 'simple', 'complex', or 'all'.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame of Global FoodOmics ontology.
    """
    if simple_complex != "all":
        gfop_metadata = gfop_metadata[gfop_metadata["simple_complex"] == simple_complex]

    col_sample_types = ["sample_name"] + [f"sample_type_group{i}" for i in range(1, 7)]
    return gfop_metadata[["filename", *col_sample_types]].set_index("filename")


def _validate_groups(gnps_network: pd.DataFrame, groups_included: List[str]) -> None:
    """
    Validates that the provided group names exist in the GNPS network data.

    Parameters
    ----------
    gnps_network : pd.DataFrame
        The GNPS network DataFrame to validate against.
    groups_included : List[str]
        The groups to validate.

    Raises
    ------
    ValueError
        If any of the group names in `groups_included` are invalid.
    """
    valid_groups = set(gnps_network["DefaultGroups"].unique())

    # Check included groups
    invalid_included_groups = set(groups_included) - valid_groups
    if len(invalid_included_groups) > 0:
        raise ValueError(
            f"The following groups in groups_included are invalid: {invalid_included_groups}"
        )


def food_counts_to_wide(food_counts: pd.DataFrame, level: int = None) -> pd.DataFrame:
    """
    Convert the food counts dataframe from long to wide format for a specific
    ontology level, with 'group' as part of the columns. If the data is already
    filtered by level, no level needs to be passed.

    Parameters
    ----------
    food_counts : pd.DataFrame
        A long-format DataFrame with columns ['filename', 'food_type', 'count',
        'level', 'group'] representing food counts across different ontology
        levels and groups.
    level : int, optional
        The ontology level to filter by before converting to wide format. If
        None, the level is inferred from the data. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame where each combination of 'food_type' and
        'group' becomes a column, and rows are indexed by 'filename'.

    Raises
    ------
    ValueError
        If multiple levels are found in the data and `level` is not specified,
        or if no data is available for the specified level.
    """
    # If level is not specified, infer the level from the data
    if level is None:
        levels_in_data = food_counts["level"].unique()
        if len(levels_in_data) > 1:
            raise ValueError(
                "Multiple levels found in the data. Please specify a level to convert to wide format."
            )
        level = levels_in_data[0]  # If the data is already filtered, use that level

    # Filter the food counts dataframe by the specified level, if not already filtered
    filtered_food_counts = food_counts[food_counts["level"] == level]

    if filtered_food_counts.empty:
        raise ValueError(f"No data available for level {level}")

    # Pivot the filtered dataframe to wide format with 'food_type' and 'group' as columns
    food_counts_wide = filtered_food_counts.pivot_table(
        index="filename", columns="food_type", values="count", fill_value=0
    )
    group_df = (
        filtered_food_counts[["filename", "group"]]
        .drop_duplicates()
        .set_index("filename")
    )
    wide_format_food_counts = food_counts_wide.join(group_df)

    return wide_format_food_counts


def calculate_proportions(food_counts: pd.DataFrame, level: int = None) -> pd.DataFrame:
    """
    Calculate the proportion of each food type within each sample for a given
    level.

    Parameters
    ----------
    food_counts : pd.DataFrame
        A long-format DataFrame with columns ['filename', 'food_type', 'count',
        'level', 'group'] representing food counts across different ontology
        levels and groups.
    level : int, optional
        The ontology level to filter by before calculating proportions. If None,
        the level is inferred from the data. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame where each food type column contains the
        proportion of that food type within each sample (row). Rows are indexed
        by 'filename', and proportions sum to 1 for each sample.

    Raises
    ------
    ValueError
        If multiple levels are found in the data and `level` is not specified.
    """
    # If level is not specified, infer the level from the data
    if level is None:
        levels_in_data = food_counts["level"].unique()
        if len(levels_in_data) > 1:
            raise ValueError(
                "Multiple levels found in the data. Please specify a level to calculate proportions."
            )
        level = levels_in_data[0]  # If the data is already filtered, use that level

    # Use the existing function to convert to wide format
    df_wide = food_counts_to_wide(food_counts, level)

    # Identify numeric columns (food type counts)
    numeric_cols = df_wide.select_dtypes(include=[float, int]).columns

    # Calculate proportions across food types (columns) for each sample (row)
    df_proportions = df_wide.copy()
    df_proportions[numeric_cols] = (
        df_proportions[numeric_cols]
        .div(df_proportions[numeric_cols].sum(axis=1), axis=0)
        .fillna(0)
    )

    return df_proportions
