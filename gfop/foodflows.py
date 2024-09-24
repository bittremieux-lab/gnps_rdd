import numpy as np
import pandas as pd
from typing import List, Optional
import os
from importlib import resources
import pkg_resources


class FoodFlows:
    def __init__(
        self,
        gnps_network: str,
        sample_types: str,
        groups_included: List[str],
        max_hierarchy_level: int = 6,
    ) -> None:
        """
        Initializes the FoodFlows object and creates the flows and processes dataframes needed to visualize the flow of foods in the dataset.

        Args:
            gnps_network (str): Path to the TSV file generated from classical molecular networking.
            sample_types (str): One of 'simple', 'complex', or 'all'.
            groups_included (List): Groups of interest in the molecular network to generate food flows
            max_hierarchy_level (int): Maximum level for food flows calculation.
        """
        # Load GNPS network data
        self.gnps_network = pd.read_csv(gnps_network, sep="\t")
        self.sample_types = self._load_sample_types(sample_types)
        self.groups_included = groups_included
        self.max_hierarchy_level = max_hierarchy_level
        self.food_metadata = self._load_food_metadata()
        # Validate group names
        self._validate_groups()
        # Generate flows and processes dataframes
        self.flows, self.processes = self.generate_foodflows()

    def _validate_groups(self) -> None:
        """
        Validates that the provided group names exist in the GNPS network data.
        Raises a ValueError if invalid group names are found.
        """
        valid_groups = set(self.gnps_network["DefaultGroups"].unique())

        # Check included groups
        invalid_included_groups = set(self.groups_included) - valid_groups
        if invalid_included_groups:
            raise ValueError(
                f"The following groups in all_groups are invalid: {invalid_included_groups}"
            )

    def _load_food_metadata(self) -> pd.DataFrame:
        """
        Reads Global FoodOmics ontology and metadata.

        Returns:
            pd.DataFrame: A dataframe containing Global FoodOmics ontology and metadata.
        """
        # Use importlib.resources if possible; fall back to pkg_resources
        try:
            with resources.open_text(
                "data", "foodomics_multiproject_metadata.txt"
            ) as stream:
                gfop_metadata = pd.read_csv(stream, sep="\t")
        except (ModuleNotFoundError, ImportError):
            # Fallback for older Python versions
            stream = pkg_resources.resource_stream(
                __name__, "data/foodomics_multiproject_metadata.txt"
            )
            gfop_metadata = pd.read_csv(stream, sep="\t")

        # Remove trailing whitespace
        gfop_metadata = gfop_metadata.apply(
            lambda col: col.str.strip() if col.dtype == "object" else col
        )
        return gfop_metadata

    def _load_sample_types(self, simple_complex: str = "all") -> pd.DataFrame:
        """
        Filters Global FoodOmics metadata by simple, complex, or all types of foods.

        Args:
            simple_complex (str): One of 'simple', 'complex', or 'all'.

        Returns:
            pd.DataFrame: Filtered Global FoodOmics ontology.
        """
        gfop_metadata = self._load_food_metadata()
        if simple_complex != "all":
            gfop_metadata = gfop_metadata[
                gfop_metadata["simple_complex"] == simple_complex
            ]
        col_sample_types = ["sample_name"] + [
            f"sample_type_group{i}" for i in range(1, 7)
        ]
        return gfop_metadata[["filename", *col_sample_types]].set_index("filename")

    def generate_foodflows(self):
        # Select GNPS job groups.
        groups = {f"G{i}" for i in range(1, 7)}
        groups_excluded = list(groups - set(self.groups_included))
        df_selected = self.gnps_network[
            (self.gnps_network[self.groups_included] > 0).all(axis=1)
            & (self.gnps_network[groups_excluded] == 0).all(axis=1)
        ].copy()
        filenames = (
            df_selected["UniqueFileSources"].str.split("|").explode()
        )  # .unique())
        # Select food hierarchy levels.
        sample_types = self.sample_types[
            [f"sample_type_group{i}" for i in range(1, self.max_hierarchy_level + 1)]
        ]
        # Match the GNPS job results to the food sample types.
        sample_types_selected = sample_types.reindex(filenames)
        sample_types_selected = sample_types_selected.dropna()
        # Discard samples that occur less frequent than water (blank).
        water_count = (sample_types_selected["sample_type_group1"] == "water").sum()
        sample_counts = sample_types_selected[
            f"sample_type_group{self.max_hierarchy_level}"
        ].value_counts()
        sample_counts_valid = sample_counts.index[sample_counts > water_count]
        sample_types_selected = sample_types_selected[
            sample_types_selected[f"sample_type_group{self.max_hierarchy_level}"].isin(
                sample_counts_valid
            )
        ]
        # Get the flows between consecutive food hierarchy levels.
        flows, processes = [], []
        for i in range(1, self.max_hierarchy_level):
            g1, g2 = f"sample_type_group{i}", f"sample_type_group{i + 1}"
            flow = (
                sample_types_selected.groupby([g1, g2])
                .size()
                .reset_index()
                .rename(columns={g1: "source", g2: "target", 0: "value"})
            )
            flow["source"] = flow["source"] + f"_{i}"
            flow["target"] = flow["target"] + f"_{i + 1}"
            flow["type"] = flow["target"]
            flows.append(flow)
            process = pd.concat(
                [flow["source"], flow["target"]], ignore_index=True
            ).to_frame()
            process["level"] = [
                *np.repeat(i, len(flow["source"])),
                *np.repeat(i + 1, len(flow["target"])),
            ]
            processes.append(process)
        return (
            pd.concat(flows, ignore_index=True),
            pd.concat(processes, ignore_index=True)
            .drop_duplicates()
            .rename(columns={0: "id"})
            .set_index("id"),
        )
