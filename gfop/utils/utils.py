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
    food_counts_wide = food_counts.pivot_table(index='filename', columns='food_type', values='count', fill_value=0)
    group_df = food_counts[['filename', 'group']].drop_duplicates().set_index('filename')
    wide_format_food_counts = food_counts_wide.join(group_df)
    return wide_format_food_counts

import pandas as pd
import os

def update_group_with_metadata_column(food_counts: pd.DataFrame, metadata_file: str, merge_column: str) -> pd.DataFrame:
    """
    Update the groups in the food counts dataframe based on user-uploaded metadata,
    allowing the user to specify which column to use for merging.
    
    Args:
        food_counts (pd.DataFrame): The original food counts dataframe with group information.
        metadata_file (str): Path to the user-uploaded metadata file (CSV or TSV) with updated information.
        merge_column (str): The column in the metadata file to use for updating (e.g., 'group', 'diet').
    
    Returns:
        pd.DataFrame: The food counts dataframe with updated information from the specified column.
    """
    # Detect file extension to determine the separator
    file_extension = os.path.splitext(metadata_file)[1]
    
    if file_extension == '.csv':
        metadata = pd.read_csv(metadata_file)  # Load CSV file
    elif file_extension == '.tsv':
        metadata = pd.read_csv(metadata_file, sep='\t')  # Load TSV file
    else:
        raise ValueError("Metadata file must be either a CSV or TSV.")
    
    # Ensure that the metadata contains the necessary columns, such as 'filename' and the specified column
    if not {'filename', merge_column}.issubset(metadata.columns):
        raise ValueError(f"Metadata file must contain 'filename' and '{merge_column}' columns.")
    
    # Merge the metadata with the original food counts to update the group information
    updated_food_counts = pd.merge(
        food_counts,
        metadata[['filename', merge_column]],  # We only need the filename and the specified merge column
        on='filename',
        how='left',  # Left join to retain all rows in food_counts
        suffixes=('', '_updated')  # Prevent column name collision
    )
    
    # Replace the original group with the updated column values
    updated_food_counts['group'] = updated_food_counts[merge_column].fillna(updated_food_counts['group'])
    
    # Drop the helper column from the metadata
    updated_food_counts = updated_food_counts.drop(columns=[merge_column])
    
    return updated_food_counts

