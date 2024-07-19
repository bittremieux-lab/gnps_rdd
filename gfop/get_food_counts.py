import pkg_resources
import numpy as np
import pandas as pd
from typing import List

def load_food_metadata() -> pd.DataFrame:
    """
    Read Global FoodOmics ontology and metadata.
    Return: a dataframe containing Global FoodOmics ontology and metadata.
    """
    stream = pkg_resources.resource_stream(__name__, 'data/foodomics_multiproject_metadata.txt')
    gfop_metadata = pd.read_csv(stream, sep='\t')
    # Remove trailing whitespace
    gfop_metadata = gfop_metadata.apply(lambda col: col.str.strip()
                                        if col.dtype == 'object' else col)
    return gfop_metadata

def get_sample_types(simple_complex: str = 'all') -> pd.DataFrame:
    """
    Filter Global FoodOmics metadata by simple, complex or all type of foods.
    Return:
        Global FoodOmics ontology containing only simple, only complex, or all foods.
    Args:
        simple_complex (string): one of 'simple', 'complex', or 'all'.
                                 Simple foods are single ingredients while complex foods contain multiple ingredients.
                                 'all' will return both simple and complex foods.
    """
    gfop_metadata = load_food_metadata()
    if simple_complex != 'all':
        gfop_metadata = gfop_metadata[gfop_metadata['simple_complex'] == simple_complex]
    col_sample_types = ['sample_name'] + [f'sample_type_group{i}' for i in range(1, 7)]
    return (gfop_metadata[['filename', *col_sample_types]]
            .set_index('filename'))

def get_sample_metadata(gnps_network: pd.DataFrame, all_groups: List[str]) -> pd.DataFrame:
    """
    Extract filenames and group of the study group(all_groups) from the GNPS network dataframe

    Return:
        Dataframe with all the filenames in the study spectrum files and the group they belong to.
    Args:
        gnps_network(dataframe): Dataframe generated from classical molecular networking
                                  with study dataset(s) and reference dataset.
        all_groups(list): can contain 'G1', 'G2' to denote study spectrum files.
    """
    df_filtered = gnps_network[~gnps_network['DefaultGroups'].str.contains(',')]
    df_selected = df_filtered[df_filtered['DefaultGroups'].isin(all_groups)]
    df_exploded_files = df_selected.assign(UniqueFileSources=df_selected['UniqueFileSources'].str.split('|')).explode('UniqueFileSources')
    # Create the final dataframe with the selected groups and filenames
    filenames_df = df_exploded_files[['DefaultGroups', 'UniqueFileSources']].rename(columns={'DefaultGroups': 'group', 'UniqueFileSources': 'filename'})
    filenames_df = filenames_df.drop_duplicates().reset_index(drop=True)
    
    return filenames_df

def get_file_food_counts(gnps_network: pd.DataFrame, sample_types: pd.DataFrame, all_groups: List[str], some_groups: List[str],
                         filename: str, level: int) -> pd.Series:
    """
    Generate food counts for an individual sample in a study dataset.
    A count indicates a spectral match between a reference food and the study sample.

    Args:
        gnps_network (dataframe): tsv file generated from classical molecular networking
                                  with study dataset(s) and reference dataset.
        sample_types (dataframe): obtained using get_sample_types().
        all_groups (list): can contain 'G1', 'G2' to denote study spectrum files.
        some_groups (list): can contain 'G3', 'G4' to denote reference spectrum files.
        filename (string): name of study sample mzXML file.
        level (integer): indicates the level of the food ontology to use.
                         One of 1, 2, 3, 4, 5, 6, or 0.
                         0 will return counts for individual reference spectrum files, rather than food categories.
    Return:
        A vector
    Examples:
        get_file_food_counts(gnps_network = gnps_network,
                             sample_types = sample_types,
                             all_groups = ['G1'],
                             some_groups = ['G4'],
                             filename = 'sample1.mzXML',
                             level = 5)
    """
    # Select GNPS job groups.
    groups = {f'G{i}' for i in range(1, 7)}
    groups_excluded = list(groups - set([*all_groups, *some_groups]))
    df_selected = gnps_network[
        (gnps_network[all_groups] > 0).all(axis=1) &
        (gnps_network[some_groups] > 0).any(axis=1) &
        (gnps_network[groups_excluded] == 0).all(axis=1)].copy()
    df_selected = df_selected[
        df_selected['UniqueFileSources'].apply(lambda cluster_fn:
            any(fn in cluster_fn for fn in filename))]
    filenames = (df_selected['UniqueFileSources'].str.split('|')
                 .explode())
    # Select food hierarchy levels.
    sample_types = sample_types[f'sample_type_group{level}' if level > 0 else 'sample_name']
    # Match the GNPS job results to the food sample types.
    sample_types_selected = sample_types.reindex(filenames)
    sample_types_selected = sample_types_selected.dropna()
    # Discard samples that occur less frequent than water (blank).
    if level > 0:
        water_count = (sample_types_selected == 'water').sum()
    else:
        water_count = 0 # TO-DO implement filtering for file-level counts
    sample_counts = sample_types_selected.value_counts()
    sample_counts_valid = sample_counts.index[sample_counts > water_count]
    sample_types_selected = sample_types_selected[
        sample_types_selected.isin(sample_counts_valid)]
    # Get sample counts at the specified level.
    return sample_types_selected.value_counts()

def get_dataset_food_counts(gnps_network: str,
                            sample_types: str, 
                            all_groups: List[str], 
                            some_groups: List[str],
                            level: int) -> pd.DataFrame:
    """
    Generate a table of food counts for a study dataset.

    Args:
        gnps_network (string): path to tsv file generated from classical molecular.
                               networking with study dataset(s) and reference dataset.
        sample_types (string): one of 'simple', 'complex', or 'all'.
                               Simple foods are single ingredients while complex foods contain multiple ingredients.
                               'all' will return both simple and complex foods.
        all_groups (list): can contain 'G1', 'G2' to denote study spectrum files.
        some_groups (list): can contain 'G3', 'G4' to denote reference spectrum files.
        level (integer): indicates the level of the food ontology to use.
                         One of 1, 2, 3, 4, 5, 6, or 0.
                         0 will return counts for individual reference spectrum files, rather than food categories.
    Return:
        A data frame
    Examples:
        get_dataset_food_counts(gnps_network = 'METABOLOMICS-SNETS-V2-07f85565-view_all_clusters_withID_beta-main.tsv',
                                sample_types = 'simple',
                                all_groups = ['G1'],
                                some_groups = ['G4'],
                                level = 5)
    """
    food_counts, filenames = [], []
    gnps_network = pd.read_csv(gnps_network, sep='\t')
    sample_types = get_sample_types(sample_types)
    metadata = get_sample_metadata(gnps_network, all_groups)
    for filename in metadata['filename']:
        file_food_counts = get_file_food_counts(gnps_network, sample_types, all_groups, some_groups, [filename], level)
        if len(file_food_counts) > 0:
            food_counts.append(file_food_counts)
            filenames.append(filename)
    food_counts = (pd.concat(food_counts, axis=1, sort=True).fillna(0).astype(int).T)
    food_counts.index = pd.Index(filenames, name='filename')
    return food_counts

def get_dataset_food_counts_all(gnps_network: str, 
                                sample_types: str, 
                                all_groups: List[str], 
                                some_groups: List[str], 
                                levels: int = 6) -> pd.DataFrame:
    """
    Generate a table of food counts for a study dataset for all levels at once in long format.

    Args:
        gnps_network (string): Path to tsv file generated from classical molecular networking
                               with study dataset(s) and reference dataset.
        sample_types (string): One of 'simple', 'complex', or 'all'.
                               Simple foods are single ingredients while complex foods contain multiple ingredients.
                               'all' will return both simple and complex foods.
        all_groups (list): List of study spectrum file groups.
        some_groups (list): List of reference spectrum file groups.
        levels (integer): Number of levels to calculate food counts for.
    Return:
        A long format dataframe with columns: filename, food_type, level, count, group.
    """
    gnps_network_df = pd.read_csv(gnps_network, sep='\t')
    metadata = get_sample_metadata(gnps_network_df, all_groups)
    
    all_data = []
    for level in range(levels + 1):
        food_counts = get_dataset_food_counts(gnps_network, sample_types, all_groups, some_groups, level)
        food_counts_long = food_counts.reset_index().melt(id_vars='filename', var_name='food_type', value_name='count')
        food_counts_long['level'] = level
        all_data.append(food_counts_long)
        
    result_df = pd.concat(all_data, ignore_index=True)
    result_df['group'] = result_df['filename'].map(metadata.set_index('filename')['group'])
    
    return result_df
