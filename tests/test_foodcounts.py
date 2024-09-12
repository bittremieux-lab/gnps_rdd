import sys
import os
import pytest
import pandas as pd

# Add the path to the 'gfop' directory to the system path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gfop'))
sys.path.append(project_path)

from foodcounts import FoodCounts  # Import after updating the path

@pytest.fixture
def create_foodcounts():
    """
    Fixture to create a FoodCounts object for testing using a GNPS file.
    """
    gnps_network_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gfop', 'data', 'sample_gnps_vegomn.tsv'))
    return FoodCounts(
        gnps_network=gnps_network_file_path,
        sample_types='all',
        all_groups=['G1'],
        some_groups=['G4'],
        levels=6
    )

def test_load_food_metadata(create_foodcounts):
    """
    Test that food metadata is loaded correctly.
    """
    foodcounts = create_foodcounts
    metadata = foodcounts.load_food_metadata()
    
    assert isinstance(metadata, pd.DataFrame)
    assert not metadata.empty

def test_generate_food_counts(create_foodcounts):
    """
    Test the food counts generation.
    """
    foodcounts = create_foodcounts
    counts = foodcounts.create()

    assert isinstance(counts, pd.DataFrame)
    assert not counts.empty

def test_filter_counts(create_foodcounts):
    """
    Test the filter_counts method.
    """
    foodcounts = create_foodcounts
    filtered_counts = foodcounts.filter_counts(level=3)

    assert isinstance(filtered_counts, pd.DataFrame)
    assert not filtered_counts.empty

def test_file_not_found():
    """
    Test that an invalid file path raises a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        FoodCounts(
            gnps_network='invalid_path.tsv',  # This file does not exist
            sample_types='all',
            all_groups=['G1'],
            some_groups=['G4'],
            levels=6
        )
def test_empty_gnps_network(tmp_path):
    """
    Test that an empty GNPS network file is handled correctly.
    """
    # Create an empty GNPS network file
    empty_file = tmp_path / "empty_gnps_network.tsv"
    empty_file.write_text("UniqueFileSources\tDefaultGroups\tG1\tG2\tG3\tG4\n")  # Just headers, no data

    foodcounts = FoodCounts(
        gnps_network=str(empty_file),
        sample_types='all',
        all_groups=['G1'],
        some_groups=['G4'],
        levels=6
    )

    counts = foodcounts.create()
    assert counts.empty, "Counts should be empty when there is no data"

@pytest.mark.parametrize("level", [0, 3, 6])
def test_generate_food_counts_for_different_levels(create_foodcounts, level):
    """
    Test food counts generation for different ontology levels.
    """
    foodcounts = create_foodcounts
    counts = foodcounts.get_level_food_counts(level)

    assert isinstance(counts, pd.DataFrame)
    assert not counts.empty, f"Counts should not be empty for level {level}"
    assert 'filename' in counts.index.names, f"'filename' should be part of the index for level {level}"

def test_filter_invalid_food_types(create_foodcounts):
    """
    Test that filtering by invalid food types returns an empty DataFrame.
    """
    foodcounts = create_foodcounts
    filtered_counts = foodcounts.filter_counts(food_types=['non_existent_food'], level=3)

    assert filtered_counts.empty, "Filtered counts should be empty for non-existent food types"

def test_filter_without_food_types(create_foodcounts):
    """
    Test that filtering without specifying food types returns all results for the given level.
    """
    foodcounts = create_foodcounts
    filtered_counts = foodcounts.filter_counts(level=3)

    assert isinstance(filtered_counts, pd.DataFrame)
    assert not filtered_counts.empty, "Filtered counts should not be empty"
    assert 'food_type' in filtered_counts.columns

def test_load_sample_metadata(create_foodcounts):
    """
    Test that sample metadata is loaded and processed correctly.
    """
    foodcounts = create_foodcounts
    sample_metadata = foodcounts.get_sample_metadata(foodcounts.gnps_network, foodcounts.all_groups)

    assert isinstance(sample_metadata, pd.DataFrame)
    assert 'filename' in sample_metadata.columns
    assert 'group' in sample_metadata.columns
    assert not sample_metadata.empty, "Sample metadata should not be empty"

@pytest.mark.parametrize("sample_type", ["simple", "complex", "all"])
def test_generate_food_counts_for_different_sample_types(tmp_path, sample_type):
    """
    Test food counts generation for different sample types.
    """
    gnps_network_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gfop', 'data', 'sample_gnps_vegomn.tsv'))
    foodcounts = FoodCounts(
        gnps_network=gnps_network_file_path,
        sample_types=sample_type,
        all_groups=['G1'],
        some_groups=['G4'],
        levels=6
    )

    counts = foodcounts.create()
    assert isinstance(counts, pd.DataFrame)
    assert not counts.empty, f"Counts should not be empty for sample type {sample_type}"

def test_exceeding_max_level(create_foodcounts):
    """
    Test that requesting a level higher than available in the data is handled gracefully.
    """
    foodcounts = create_foodcounts
    counts = foodcounts.get_level_food_counts(level=10)  # Assuming max level is less than 10

    assert counts.empty, "Counts should be empty when requested level exceeds available levels"

def test_invalid_group_names():
    """
    Test that providing invalid group names raises a KeyError or returns an empty DataFrame.
    """
    gnps_network_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gfop', 'data', 'sample_gnps_vegomn.tsv'))

    try:
        food_counts = FoodCounts(
            gnps_network=gnps_network_file_path,
            sample_types='all',
            all_groups=['InvalidGroup'],  # Invalid group
            some_groups=['G4'],
            levels=6
        )
        
        # Generate food counts
        counts = food_counts.create()

        # Check if the DataFrame is empty
        assert counts.empty, "DataFrame should be empty for invalid group names"
    
    except KeyError:
        # If a KeyError is raised, the test passes
        print("KeyError was raised as expected for invalid group names.")