import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from typing import List
from foodcounts import FoodCounts
from utils import food_counts_to_wide, calculate_proportions
import pandas as pd
import plotly.graph_objects as go


def plot_food_type_distribution(
    food_counts_instance,
    level: int = 3,
    food_types: list = None,
    group_by: bool = False,
    library: str = "sns",
    figsize=(10, 6),
):
    """
    Plot a bar chart showing the distribution of food types.

    Args:
        food_counts_instance (FoodCounts): An instance of the FoodCounts class.
        level (int): The ontology level to filter by.
        food_types (list, optional): Specific food types to include. Defaults to None.
        group_by (bool): If True, group by 'group'. Defaults to False.
        library (str): Visualization library to use ('sns' or 'plotly'). Defaults to 'sns'.
        figsize (tuple): Figure size for the plot. Defaults to (10, 6).

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The figure object for the plot.
    """
    # Internal data filtering
    filtered_counts = food_counts_instance.filter_counts(
        food_types=food_types, level=level
    )

    # Check if filtered_counts is empty
    if filtered_counts.empty:
        raise ValueError("No data available for the specified level and food types.")

    # Grouping logic
    if group_by:
        data = (
            filtered_counts.groupby(["food_type", "group"])["count"].sum().reset_index()
        )
    else:
        data = filtered_counts.groupby("food_type")["count"].sum().reset_index()

    # Seaborn/Matplotlib Visualization
    if library == "sns":
        fig, ax = plt.subplots(figsize=figsize)
        if group_by:
            sns.barplot(
                x="food_type",
                y="count",
                hue="group",
                data=data,
                palette="viridis",
                ax=ax,
            )
        else:
            sns.barplot(x="food_type", y="count", data=data, palette="viridis", ax=ax)
        ax.set_title("Food Type Distribution")
        ax.set_xlabel("Food Type")
        ax.set_ylabel("Total Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    # Plotly Visualization
    elif library == "plotly":
        if group_by:
            fig = px.bar(
                data,
                x="food_type",
                y="count",
                color="group",
                barmode="group",
                title="Food Type Distribution by Group",
            )
        else:
            fig = px.bar(data, x="food_type", y="count", title="Food Type Distribution")

        # Customize layout
        fig.update_layout(
            xaxis_title="Food Type",
            yaxis_title="Total Count",
            xaxis_tickangle=-45,
            template="plotly_white",
        )
        return fig

    else:
        raise ValueError("Invalid library selected. Choose 'sns' or 'plotly'.")


def box_plot_food_proportions(
    food_counts_instance,
    level: int = 3,
    food_types: list = None,
    group_by: bool = False,
    group_colors: dict = None,
    library: str = "plotly",
    figsize=(10, 6),
):
    """
    Plot box plots showing the distribution of food proportions.

    Args:
        food_counts_instance (FoodCounts): An instance of the FoodCounts class.
        level (int): The ontology level to filter by.
        food_types (list, optional): Specific food types to include. Defaults to None.
        group_by (bool): If True, group by 'group'. Defaults to False.
        group_colors (dict, optional): A dictionary mapping group names to colors.
        library (str): Visualization library to use ('plotly' or 'sns'). Defaults to 'plotly'.
        figsize (tuple): Figure size for matplotlib plots. Defaults to (10, 6).

    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: The figure object for the plot.
    """
    # Handle no food types selection
    if not food_types:
        # Return an empty figure for Plotly or Matplotlib
        if library == "plotly":
            fig = go.Figure()  # Create an empty Plotly figure
            fig.update_layout(
                title="No food types selected",
                xaxis_title="Food Type",
                yaxis_title="Proportion",
            )
            return fig
        elif library == "sns":
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("No food types selected")
            ax.set_xlabel("Food Type")
            ax.set_ylabel("Proportion")
            return fig
        else:
            raise ValueError("Invalid library selected. Choose 'plotly' or 'sns'.")

    # Access counts from the FoodCounts instance
    counts = food_counts_instance.counts

    # Calculate proportions
    df_proportions = calculate_proportions(counts, level=level)

    # Convert the data to long format
    df_long = df_proportions.reset_index().melt(
        id_vars=["filename", "group"], var_name="food_type", value_name="proportion"
    )

    # Filter by the provided food types
    df_long = df_long[df_long["food_type"].isin(food_types)]

    # If group_by is True, we need to include the 'group' column
    if group_by:
        # Box plot grouped by food type and group
        data = df_long[["food_type", "group", "proportion"]]
        if library == "plotly":
            fig = go.Figure()

            # Iterate through each group to add traces
            groups = df_long["group"].unique()
            for i, group in enumerate(groups):
                group_data = df_long[df_long["group"] == group]
                fig.add_trace(
                    go.Box(
                        x=group_data["food_type"],
                        y=group_data["proportion"],
                        name=group,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(
                            color=group_colors.get(group) if group_colors else None
                        ),
                        offsetgroup=i,
                    )
                )
            fig.update_layout(
                title="Grouped Proportion Distribution of Selected Food Types",
                xaxis_title="Food Type",
                yaxis_title="Proportion",
                boxmode="group",
            )
            return fig

        elif library == "sns":
            fig, ax = plt.subplots(figsize=figsize)

            # Plot the boxplot with white boxes and no color fill
            sns.boxplot(
                x="food_type",
                y="proportion",
                hue="group",
                data=data,
                ax=ax,
                boxprops=dict(facecolor="white", edgecolor="black"),  # White boxes
                whiskerprops=dict(color="black"),  # Black whiskers
                medianprops=dict(color="black"),  # Black median line
                capprops=dict(color="black"),  # Black caps on whiskers
                showfliers=False,  # Hide outliers (you can enable this if needed)
            )

            # Add a stripplot with colored points
            sns.stripplot(
                x="food_type",
                y="proportion",
                hue="group",
                data=data,
                dodge=True,
                jitter=True,
                palette=group_colors,  # Apply group colors only to points
                ax=ax,
                alpha=0.7,
                marker="o",
                edgecolor="black",  # Optional: Add a black edge around the points
                linewidth=0.6,
                legend=False,
            )

            ax.set_title("Grouped Proportion Distribution of Selected Food Types")
            ax.set_xlabel("Food Type")
            ax.set_ylabel("Proportion")

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")

            # Remove the legend created by stripplot to avoid duplicate legends
            ax.legend_.remove()

            # Add a single legend for the plot
            handles, labels = ax.get_legend_handles_labels()

            # Only slice handles and labels if group_colors is provided
            if group_colors:
                ax.legend(
                    handles[: len(group_colors)],
                    labels[: len(group_colors)],
                    title="Group",
                )
            else:
                ax.legend(handles, labels, title="Group")

            plt.tight_layout()
            return fig

    else:
        # No grouping by 'group', just create a boxplot for food types
        data = df_long[["food_type", "proportion"]]
        if library == "plotly":
            fig = go.Figure()

            # Add a single boxplot for each food type
            for food_type in df_long["food_type"].unique():
                food_data = df_long[df_long["food_type"] == food_type]
                fig.add_trace(
                    go.Box(
                        x=food_data["food_type"],
                        y=food_data["proportion"],
                        name=food_type,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(
                            color=group_colors.get(food_type) if group_colors else None
                        ),
                    )
                )

            fig.update_layout(
                title="Proportion Distribution of Selected Food Types",
                xaxis_title="Food Type",
                yaxis_title="Proportion",
            )
            return fig

        elif library == "sns":
            fig, ax = plt.subplots(figsize=figsize)

            # Plot the boxplot with white boxes and no color fill
            sns.boxplot(
                x="food_type",
                y="proportion",
                data=data,
                ax=ax,
                boxprops=dict(facecolor="white", edgecolor="black"),  # White boxes
                whiskerprops=dict(color="black"),  # Black whiskers
                medianprops=dict(color="black"),  # Black median line
                capprops=dict(color="black"),  # Black caps on whiskers
                showfliers=False,  # Hide outliers (you can enable this if needed)
            )

            # Add a stripplot with colored points
            sns.stripplot(
                x="food_type",
                y="proportion",
                data=data,
                dodge=True,
                jitter=True,
                palette=group_colors,  # Apply group colors only to points
                ax=ax,
                alpha=0.7,
                marker="o",
                edgecolor="black",  # Optional: Add a black edge around the points
                linewidth=0.6,
            )

            ax.set_title("Proportion Distribution of Selected Food Types")
            ax.set_xlabel("Food Type")
            ax.set_ylabel("Proportion")

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
            return fig

        else:
            raise ValueError("Invalid library selected. Choose 'plotly' or 'sns'.")


def plot_food_proportion_heatmap(
    food_counts_instance,
    level: int = 3,
    food_types: list = None,
    library: str = "sns",
    figsize=(12, 8),
):
    """
    Plot a heatmap of food proportions for given food types.

    Args:
        food_counts_instance (FoodCounts): An instance of the FoodCounts class.
        level (int): The ontology level to filter by.
        food_types (list, optional): Specific food types to include. Defaults to None (all food types).
        library (str): Visualization library to use ('sns' or 'plotly'). Defaults to 'sns'.
        figsize (tuple): Figure size for matplotlib plots. Defaults to (12, 8).

    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: The figure object for the heatmap.
    """
    # Access counts from the FoodCounts instance
    counts = food_counts_instance.counts

    # Calculate proportions
    df_proportions = calculate_proportions(counts, level=level)

    # Filter by the provided food types, or select all food types if None is provided
    if food_types is not None:
        df_proportions_filtered = df_proportions[
            df_proportions.columns.intersection(food_types)
        ]
    else:
        df_proportions_filtered = df_proportions

    food_type_columns = df_proportions_filtered.select_dtypes(
        include=["int", "float"]
    ).columns
    df_proportions_filtered = df_proportions_filtered[food_type_columns]
    # Transpose the dataframe so that food types are on the y-axis and samples are on the x-axis

    # Seaborn heatmap (matplotlib backend)
    if library == "sns":
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            df_proportions_filtered, cmap="viridis", annot=False, cbar=True
        )
        ax.set_title(f"Proportion Heatmap of Food Types (Level {level})")
        ax.set_xlabel("Food Types")
        ax.set_ylabel("Samples")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return ax.get_figure()

    # Plotly heatmap
    elif library == "plotly":
        fig = go.Figure(
            data=go.Heatmap(
                z=df_proportions_filtered.values,
                x=df_proportions_filtered.columns,
                y=df_proportions_filtered.index,
                colorscale="Viridis",
                colorbar=dict(title="Proportion"),
            )
        )
        fig.update_layout(
            title=f"Proportion Heatmap of Food Types (Level {level})",
            xaxis_title="Food Types",
            yaxis_title="Samples",
        )
        return fig

    else:
        raise ValueError("Invalid library selected. Choose 'sns' or 'plotly'.")


def plot_pca_results(
    pca_df: pd.DataFrame,
    explained_variance: list,
    group_by: bool = True,
    library: str = "plotly",
    figsize=(10, 6),
    group_colors: dict = None,
):
    """
    Plot the PCA results using either Plotly or Seaborn/Matplotlib.

    Args:
        pca_df (pd.DataFrame): DataFrame containing PCA scores and metadata.
        explained_variance (list): List of explained variance ratios for each component.
        group_by (bool): Whether to color the points by 'group'. Defaults to True.
        library (str): Visualization library to use ('plotly' or 'sns'). Defaults to 'plotly'.
        figsize (tuple): Figure size for Matplotlib plots. Defaults to (10, 6).
        group_colors (dict, optional): A dictionary mapping group names to colors.

    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: The figure object for the plot.
    """
    if library == "plotly":
        fig = go.Figure()

        # Add scatter plot for the first two principal components
        if group_by:
            groups = pca_df["group"].unique()
            for group in groups:
                group_data = pca_df[pca_df["group"] == group]
                fig.add_trace(
                    go.Scatter(
                        x=group_data["PC1"],
                        y=group_data["PC2"],
                        mode="markers",
                        name=group,
                        marker=dict(
                            color=group_colors.get(group) if group_colors else None
                        ),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=pca_df["PC1"], y=pca_df["PC2"], mode="markers", name="Samples"
                )
            )

        # Customize layout
        fig.update_layout(
            title="PCA Plot of Food Counts",
            xaxis_title=f"PC1 [{explained_variance[0] * 100:.1f}%]",
            yaxis_title=f"PC2 [{explained_variance[1] * 100:.1f}%]",
            template="plotly_white",
        )
        return fig

    elif library == "sns":
        fig, ax = plt.subplots(figsize=figsize)
        if group_by:
            sns.scatterplot(
                x="PC1", y="PC2", hue="group", data=pca_df, palette=group_colors, ax=ax
            )
        else:
            sns.scatterplot(x="PC1", y="PC2", data=pca_df, ax=ax)
        ax.set_title("PCA Plot of Food Counts")
        ax.set_xlabel(f"PC1 [{explained_variance[0] * 100:.1f}%]")
        ax.set_ylabel(f"PC2 [{explained_variance[1] * 100:.1f}%]")
        plt.tight_layout()
        return fig

    else:
        raise ValueError("Invalid library selected. Choose 'plotly' or 'sns'.")


def plot_explained_variance(
    explained_variance: list, library: str = "plotly", figsize=(10, 6)
):
    """
    Plot the explained variance of each principal component as a bar chart.

    Args:
        explained_variance (list): List of explained variance ratios for each component.
        library (str): Visualization library to use ('plotly' or 'sns'). Defaults to 'plotly'.
        figsize (tuple): Figure size for Matplotlib plots. Defaults to (10, 6).

    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: The figure object for the plot.
    """
    pc_labels = [f"PC{i+1}" for i in range(len(explained_variance))]
    variance_percentages = [var * 100 for var in explained_variance]

    if library == "plotly":
        fig = go.Figure(
            go.Bar(
                x=pc_labels,
                y=variance_percentages,
                text=[f"{var:.1f}%" for var in variance_percentages],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Explained Variance by Principal Component",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance (%)",
            template="plotly_white",
        )
        return fig

    elif library == "sns":
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=pc_labels, y=variance_percentages, ax=ax)
        ax.set_title("Explained Variance by Principal Component")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        for i, var in enumerate(variance_percentages):
            ax.text(i, var, f"{var:.1f}%", ha="center")
        plt.tight_layout()
        return fig

    else:
        raise ValueError("Invalid library selected. Choose 'plotly' or 'sns'.")


def visualize_sankey(food_flows, color_mapping_file):
    """
    Visualize the food flows as a Sankey diagram.

    Args:
        food_flows: FoodFlows object containing the flows and processes dataframes.
        color_mapping_file: CSV file mapping the sample types to their respective colors.

    Returns:
        fig: Plotly figure of the Sankey diagram.
    """

    # Load the color mapping
    color_df = pd.read_csv(color_mapping_file, sep="\;")
    color_df["color_code"] = color_df["color_code"].fillna(value="#D3D3D3")
    color_mapping = dict(zip(color_df["descriptor"], color_df["color_code"]))

    # Prepare the Sankey node and link data
    all_nodes = list(
        pd.concat([food_flows.flows["source"], food_flows.flows["target"]]).unique()
    )
    node_indices = {node: idx for idx, node in enumerate(all_nodes)}
    # Map the flows to indices and assign colors
    source_indices = food_flows.flows["source"].map(node_indices)
    target_indices = food_flows.flows["target"].map(node_indices)
    values = food_flows.flows["value"]

    # Assign colors to nodes
    node_colors = [color_mapping.get(node, "#D3D3D3") for node in all_nodes]
    link_colors = [
        color_mapping.get(node, "#D3D3D3") for node in food_flows.flows["source"]
    ]
    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color=link_colors,  # Flow color, adjust if needed
                ),
                textfont=dict(color="black", size=10, family="Arial black"),
            )
        ]
    )

    # Set the layout of the diagram
    fig.update_layout(title_text="Food Flows Sankey Diagram", font_size=10)

    return fig
