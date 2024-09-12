import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

def plot_food_type_distribution(filtered_food_counts: pd.DataFrame, group_by: bool = False, library: str = 'sns'):
    """
    Plot a bar chart showing the distribution of food types, with an option to group by 'group' and choose the visualization library.
    
    Args:
        filtered_food_counts (pd.DataFrame): A filtered dataframe with columns ['filename', 'food_type', 'count', 'level', 'group'].
        group_by (bool): If True, group by 'group' and 'food_type'. If False, aggregate food types across all groups.
        library (str): Visualization library to use ('sns' for seaborn/matplotlib, 'plotly' for plotly).
    
    Returns:
        None: Displays a bar chart using the specified visualization library.
    """
    # Grouping logic
    if group_by:
        food_type_counts = filtered_food_counts.groupby(['food_type', 'group'])['count'].sum().reset_index()
    else:
        food_type_counts = filtered_food_counts.groupby('food_type')['count'].sum().reset_index()

    # Seaborn/Matplotlib Visualization
    if library == 'sns':
        plt.figure(figsize=(10, 6))
        if group_by:
            sns.barplot(x='food_type', y='count', hue='group', data=food_type_counts, palette='viridis')
        else:
            sns.barplot(x='food_type', y='count', data=food_type_counts, palette='viridis')
        plt.title('Food Type Distribution')
        plt.xlabel('Food Type')
        plt.ylabel('Total Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Plotly Visualization
    elif library == 'plotly':
        if group_by:
            fig = px.bar(food_type_counts, x='food_type', y='count', color='group', barmode='group', title='Food Type Distribution by Group')
        else:
            fig = px.bar(food_type_counts, x='food_type', y='count', title='Food Type Distribution')
        
        # Customize layout for readability
        fig.update_layout(
            xaxis_title='Food Type',
            yaxis_title='Total Count',
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        fig.show()

    else:
        raise ValueError("Invalid library selected. Choose 'sns' or 'plotly'.")
