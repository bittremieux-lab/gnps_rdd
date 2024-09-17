import pandas as pd


def food_counts_to_wide(food_counts: pd.DataFrame, level: int = None) -> pd.DataFrame:
    """
    Convert the food counts dataframe from long to wide format for a specific ontology level,
    with 'group' as part of the columns. If the data is already filtered by level, no level needs to be passed.

    Args:
        food_counts (pd.DataFrame): A long-format dataframe with columns ['filename', 'food_type', 'count', 'level', 'group'].
        level (int, optional): The ontology level to filter by before converting to wide format. If the data is already filtered, this can be None.

    Returns:
        pd.DataFrame: A wide-format dataframe where each combination of 'food_type' and 'group' is a column.
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
    Calculate the proportion of each food type within each sample for a given level.

    Args:
        food_counts (pd.DataFrame): A long-format DataFrame with columns ['filename', 'food_type', 'count', 'level', 'group'].
        level (int): The ontology level to filter by.

    Returns:
        pd.DataFrame: A wide-format DataFrame with proportions calculated per sample (row), per food type (columns).
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
