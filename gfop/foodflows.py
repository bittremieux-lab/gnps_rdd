# Standard library imports
import os
import pkg_resources
from importlib import resources
from typing import List, Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Internal imports
from utils import _load_food_metadata, _load_sample_types, _validate_groups


class FoodFlows:
    def __init__(
        self,
        gnps_network: str,
        sample_types: str,
        groups_included: List[str],
        max_hierarchy_level: int = 6,
    ) -> None:
        """
        Initializes the FoodFlows object and creates the flows and processes
        dataframes needed to visualize the flow of foods in the network.

        Parameters
        ----------
        gnps_network : str
            Path to the TSV file generated from classical molecular networking.
        sample_types : str
            One of 'simple', 'complex', or 'all'.
        groups_included : List[str]
            Groups of interest in the molecular network to generate food flows.
        max_hierarchy_level : int, optional
            Maximum level for food flows calculation, by default 6.
        """
        # Load GNPS network data
        self.gnps_network = pd.read_csv(gnps_network, sep="\t")
        self.food_metadata = _load_food_metadata()  # Call the utility function
        self.sample_types = _load_sample_types(
            self.food_metadata, sample_types
        )  # Call the utility function
        self.groups_included = groups_included
        self.max_hierarchy_level = max_hierarchy_level

        # Validate group names
        _validate_groups(
            self.gnps_network, self.groups_included
        )  # Call the utility function

        # Generate flows and processes dataframes
        self.flows, self.processes = self.generate_foodflows()

    def generate_foodflows(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates food flows and processes from the GNPS network data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - flows: DataFrame with source, target, and value columns.
            - processes: DataFrame with process details.
        """
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
