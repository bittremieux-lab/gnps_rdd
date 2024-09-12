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
        levels_in_data = food_counts['level'].unique()
        if len(levels_in_data) > 1:
            raise ValueError("Multiple levels found in the data. Please specify a level to convert to wide format.")
        level = levels_in_data[0]  # If the data is already filtered, use that level

    # Filter the food counts dataframe by the specified level, if not already filtered
    filtered_food_counts = food_counts[food_counts['level'] == level]

    if filtered_food_counts.empty:
        raise ValueError(f"No data available for level {level}")

    # Pivot the filtered dataframe to wide format with 'food_type' and 'group' as columns
    wide_df = filtered_food_counts.pivot_table(
        index='filename', 
        columns=['food_type', 'group'], 
        values='count', 
        aggfunc='first', 
        fill_value=0
    ).reset_index()

    wide_df.columns.name = None
    return wide_df
