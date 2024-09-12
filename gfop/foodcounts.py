import pkg_resources
import numpy as np
import pandas as pd
from typing import List

class FoodCounts:
    def __init__(self, gnps_network: str, sample_types: str, all_groups: List[str], some_groups: List[str], levels: int = 6):
        """
        Initializes the FoodCounts object and automatically creates the food counts for all levels and all types.
        
        Args:
            gnps_network (str): Path to the tsv file generated from classical molecular networking.
            sample_types (str): One of 'simple', 'complex', or 'all'.
            all_groups (List[str]): List of study spectrum file groups.
            some_groups (List[str]): List of reference spectrum file groups.
            levels (int): Number of levels to calculate food counts for.
        """
        self.gnps_network = pd.read_csv(gnps_network, sep="\t")
        self.sample_types = self.load_sample_types(sample_types)
        self.all_groups = all_groups
        self.some_groups = some_groups
        self.levels = levels
        self.food_metadata = self.load_food_metadata()
        self.sample_metadata = self.get_sample_metadata(self.gnps_network, self.all_groups)
        self.counts = self.create()

    def load_food_metadata(self) -> pd.DataFrame:
        """
        Read Global FoodOmics ontology and metadata.
        Return: a dataframe containing Global FoodOmics ontology and metadata.
        """
        stream = pkg_resources.resource_stream(
            __name__, "data/foodomics_multiproject_metadata.txt"
        )
        gfop_metadata = pd.read_csv(stream, sep="\t")
        # Remove trailing whitespace
        gfop_metadata = gfop_metadata.apply(
            lambda col: col.str.strip() if col.dtype == "object" else col
        )
        return gfop_metadata

    def load_sample_types(self, simple_complex: str = "all") -> pd.DataFrame:
        """
        Filter Global FoodOmics metadata by simple, complex or all type of foods.
        
        Args:
            simple_complex (str): One of 'simple', 'complex', or 'all'.
        Return:
            Filtered Global FoodOmics ontology.
        """
        gfop_metadata = self.load_food_metadata()
        if simple_complex != "all":
            gfop_metadata = gfop_metadata[gfop_metadata["simple_complex"] == simple_complex]
        col_sample_types = ["sample_name"] + [f"sample_type_group{i}" for i in range(1, 7)]
        return gfop_metadata[["filename", *col_sample_types]].set_index("filename")

    def get_sample_metadata(self, gnps_network: pd.DataFrame, all_groups: List[str]) -> pd.DataFrame:
        """
        Extract filenames and group of the study group(all_groups) from the GNPS network dataframe.

        Args:
            gnps_network (pd.DataFrame): Dataframe generated from classical molecular networking.
            all_groups (List[str]): List of study groups.
        Return:
            Dataframe with filenames and group associations.
        """
        df_filtered = gnps_network[~gnps_network["DefaultGroups"].str.contains(",")]
        df_selected = df_filtered[df_filtered["DefaultGroups"].isin(all_groups)]
        df_exploded_files = df_selected.assign(
            UniqueFileSources=df_selected["UniqueFileSources"].str.split("|")
        ).explode("UniqueFileSources")
        filenames_df = df_exploded_files[["DefaultGroups", "UniqueFileSources"]].rename(
            columns={"DefaultGroups": "group", "UniqueFileSources": "filename"}
        )
        return filenames_df.drop_duplicates().reset_index(drop=True)

    def get_level_food_counts(self, level: int) -> pd.DataFrame:
        """
        Generate a table of food counts for a specific ontology level.
        
        Args:
            level (int): The ontology level to use for filtering food types.
        Return:
            A dataframe containing food counts for a specific level.
        """
        food_counts, filenames = [], []
        metadata = self.sample_metadata

        for filename in metadata["filename"]:
            file_food_counts = self.get_file_food_counts(level, [filename])
            if len(file_food_counts) > 0:
                food_counts.append(file_food_counts)
                filenames.append(filename)
        if not food_counts:
            return pd.DataFrame()  # Return an empty dataframe if no data to concatenate     
        food_counts = pd.concat(food_counts, axis=1, sort=True).fillna(0).astype(int).T
        food_counts.index = pd.Index(filenames, name="filename")
        return food_counts

    def get_file_food_counts(
        self, level: int, filename: str
    ) -> pd.Series:
        """
        Generate food counts for an individual sample in a study dataset.
        
        Args:
            level (int): The ontology level to use for filtering food types.
            filename (str): Filename of the sample.
        Return:
            A vector of food counts for the sample.
        """
        groups = {f"G{i}" for i in range(1, 7)}
        groups_excluded = list(groups - set([*self.all_groups, *self.some_groups]))
        df_selected = self.gnps_network[
            (self.gnps_network[self.all_groups] > 0).all(axis=1)
            & (self.gnps_network[self.some_groups] > 0).any(axis=1)
            & (self.gnps_network[groups_excluded] == 0).all(axis=1)
        ].copy()
        df_selected = df_selected[
            df_selected["UniqueFileSources"].apply(
                lambda cluster_fn: any(fn in cluster_fn for fn in filename)
            )
        ]
        filenames = df_selected["UniqueFileSources"].str.split("|").explode()

            # Check if the requested level exists in the sample types dataframe
        column_name = f"sample_type_group{level}" if level > 0 else "sample_name"
        if column_name not in self.sample_types.columns:
            return pd.Series(dtype=int)  # Return an empty Series if the column doesn't exist
        
        sample_types = self.sample_types[f"sample_type_group{level}" if level > 0 else "sample_name"]
        sample_types_selected = sample_types.reindex(filenames).dropna()
        
        if level > 0:
            water_count = (sample_types_selected == "water").sum()
        else:
            water_count = 0

        sample_counts = sample_types_selected.value_counts()
        sample_counts_valid = sample_counts.index[sample_counts > water_count]
        sample_types_selected = sample_types_selected[
            sample_types_selected.isin(sample_counts_valid)
        ]
        return sample_types_selected.value_counts()
    
    def create(self) -> pd.DataFrame:
        """
        Generate a table of food counts for a study dataset for all levels at once in long format.
        
        Return:
            A long format dataframe with columns: filename, food_type, level, count, group.
        """
        all_data = []
        for level in range(self.levels + 1):
            food_counts = self.get_level_food_counts(level)
            if food_counts.empty:
                continue  # Skip if no data for this level
            food_counts_long = food_counts.reset_index().melt(
                id_vars="filename", var_name="food_type", value_name="count"
            )
            food_counts_long["level"] = level
            all_data.append(food_counts_long)
        # If no data was found, return an empty DataFrame
        if not all_data:
            return pd.DataFrame(columns=["filename", "food_type", "count", "level", "group"])
        
        result_df = pd.concat(all_data, ignore_index=True)
        result_df["group"] = result_df["filename"].map(
            self.sample_metadata.set_index("filename")["group"]
        )
        return result_df
    
    def filter_counts(self, food_types: List[str] = None, level: int = 3) -> pd.DataFrame:
        """Filter the food counts by food types and ontology level."""
        if self.counts is None:
            raise ValueError("Food counts have not been created yet. Call create() first.")
        if food_types is None:
            filtered_df = self.counts[self.counts["level"] == level]
            return filtered_df
        filtered_df = self.counts[
            (self.counts["food_type"].isin(food_types)) & 
            (self.counts["level"] == level)
        ]
        return filtered_df
