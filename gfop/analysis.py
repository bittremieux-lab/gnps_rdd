from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import clr
from foodcounts import FoodCounts
from utils import food_counts_to_wide
import pandas as pd
import numpy as np


def perform_pca_food_counts(
    food_counts_instance,
    level: int = 3,
    n_components: int = 3,
    apply_clr: bool = True,
    food_types: list = None,
):
    """
    Perform PCA on food counts data using the FoodCounts instance, with an option for CLR transformation.

    Args:
        food_counts_instance (FoodCounts): The instance of the FoodCounts class.
        level (int): Ontology level to filter food types. Defaults to 3.
        n_components (int): Number of principal components to calculate. Defaults to 3.
        apply_clr (bool): Whether to apply the CLR transformation before PCA. Defaults to True.
        food_types (list): List of specific food types to include in the analysis. Defaults to all food types.

    Returns:
        pd.DataFrame: A dataframe containing PCA scores, explained variance, and sample metadata.
        list: Explained variance percentages.
    """
    # Step 1: Filter counts by level
    food_counts_filtered = food_counts_instance.filter_counts(level=level)

    # Step 2: Filter food types if specified by the user
    if food_types:
        food_counts_filtered = food_counts_filtered[
            food_counts_filtered["food_type"].isin(food_types)
        ]

    # Step 3: Convert to wide format for PCA
    food_counts_wide = food_counts_to_wide(food_counts_filtered, level=level)

    # Step 4: Extract numeric columns (food types) for PCA
    food_type_columns = food_counts_wide.select_dtypes(include=[np.number]).columns
    X = food_counts_wide[food_type_columns].values

    # Step 5: Apply CLR transformation if selected
    if apply_clr:
        X = clr(X + 1)  # Add 1 to avoid issues with zeros, then apply CLR

    # Step 6: Standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # Step 7: Perform PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Step 8: Create a dataframe with PCA results
    pca_df = pd.DataFrame(pca_scores, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["filename"] = food_counts_wide.index

    # Merge with sample metadata
    sample_metadata = food_counts_instance.sample_metadata
    pca_df = pd.merge(pca_df, sample_metadata, on="filename", how="left")

    return pca_df, explained_variance
