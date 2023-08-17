import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)  # Avoid wrapping in the terminal


def plot_subclasses_dice(data):
    unique_losses = data['loss'].unique()
    palette = sns.color_palette("husl", len(unique_losses))
    # Extracting subclass column names
    dice_subclass_columns = [col for col in data.columns if "dice_class" in col]

    # Melt the data to create a long-form dataset suitable for boxplots
    dice_subclass_data = data.melt(id_vars='loss', value_vars=dice_subclass_columns,
                                   var_name='Subclass', value_name='Dice Score')

    # Create a column to distinguish between "before" and "after" registration and to extract the class number
    dice_subclass_data['Registration'] = dice_subclass_data['Subclass'].apply(
        lambda x: 'Before Registration' if 'before' in x else 'After Registration')
    dice_subclass_data['Class'] = dice_subclass_data['Subclass'].str.extract('class_(\d)').astype(int)

    for i in range(1, 6):
        class_data = dice_subclass_data[dice_subclass_data['Class'] == i].copy()

        # Extracting the median value for "Before Registration" for clearer representation
        median_before_reg = class_data[class_data['Registration'] == 'Before Registration']['Dice Score'].median()

        plt.figure(figsize=(8, 6))
        plt.axhline(median_before_reg, color='red', linestyle='--', label=f'Dice Score for Class {i} Before Registration')

        sns.boxplot(x='loss', y='Dice Score', data=class_data[class_data['Registration'] == 'After Registration'],
                    palette=palette)
        plt.title(f'Dice Scores for Class {i} Before and After Registration')
        plt.xlabel('Loss Function')
        plt.ylim(0, 1)  # Set y-axis limits
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_mean_dice(data):
    mean_dice_columns = ['after_reg_mean_dice']

    mean_dice_data = data.melt(id_vars='loss', value_vars=mean_dice_columns,
                               var_name='Mean Dice Score', value_name='Dice Score')

    unique_losses = data['loss'].unique()
    palette = sns.color_palette("husl", len(unique_losses))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxplot(x='loss', y='Dice Score', data=mean_dice_data, ax=ax, palette=palette)

    # Adding a dotted red line for before_reg_mean_dice
    before_reg_mean_dice = data['before_reg_mean_dice'].iloc[0]
    ax.axhline(y=before_reg_mean_dice, color='red', linestyle='dotted', label='Mean Dice Before Registration')

    ax.set_title('Mean Dice Score After Registration for Each Loss')
    ax.set_xlabel('Loss Function')
    ax.set_ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_rmse(data):
    mean_rmse_columns = ['after_reg_rmse']

    mean_rmse_data = data.melt(id_vars='loss', value_vars=mean_rmse_columns,
                               var_name='Mean RMSE', value_name='RMSE')

    unique_losses = data['loss'].unique()
    palette = sns.color_palette("husl", len(unique_losses))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxplot(x='loss', y='RMSE', data=mean_rmse_data, ax=ax, palette=palette)

    before_reg_rmse = data['before_reg_rmse'].iloc[0]  # Assuming the value is the same for all rows
    ax.axhline(y=before_reg_rmse, color='red', linestyle='dotted', label='Before Registration RMSE')

    ax.set_title('RMSE After Registration for Each Loss')
    ax.set_xlabel('Loss Function')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metrics_per_loss(data, metric_column, ylabel):
    metric_data = data[['loss', metric_column]]

    unique_losses = data['loss'].unique()
    palette = sns.color_palette("husl", len(unique_losses))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxplot(x='loss', y=metric_column, data=metric_data, ax=ax, palette=palette)

    ax.set_title(f'{ylabel} After Registration for Each Loss')
    ax.set_xlabel('Loss Function')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_var_effect(data, x_axis, y_axis, x_label, y_label, title, legend_title):
    unique_losses = data['loss'].unique()
    palette = sns.color_palette("husl", len(unique_losses))

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x_axis, y=y_axis, hue='loss', data=data, palette=palette, width=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if y_axis == 'after_reg_mean_dice':
        plt.ylim(0, 1)
    plt.legend(title=legend_title)
    plt.show()


def plot_3d(data):
    # Extract data from the dictionary
    # Extract data from the dictionary
    lr = data["lr"]
    be = data["be"]
    reg = data["reg"]
    mean_dice = data["after_reg_mean_dice"]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.2,
                           hspace=0.2)  # Adjust wspace and hspace
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')

    # Plot the 3D scatter plot with alpha (transparency)
    scatter_3d = ax_3d.scatter(lr, be, reg, c=mean_dice, cmap='viridis', s=100, vmin=min(mean_dice),
                               vmax=max(mean_dice), alpha=0.7)  # Adjust alpha value
    ax_3d.set_xlabel('lr')
    ax_3d.set_ylabel('be')
    ax_3d.set_zlabel('reg')

    # Adjust rotation of the 3D plot (azimuth and elevation)
    ax_3d.view_init(elev=20, azim=30)  # Adjust these values

    # Create 2D scatter plots for the slides
    ax_lr_be = fig.add_subplot(gs[0, 0])
    sc_lr_be = ax_lr_be.scatter(lr, be, c=mean_dice, cmap='viridis', s=100)
    ax_lr_be.set_xlabel('lr')
    ax_lr_be.set_ylabel('be')

    ax_be_reg = fig.add_subplot(gs[1, 1])
    sc_be_reg = ax_be_reg.scatter(be, reg, c=mean_dice, cmap='viridis', s=100)
    ax_be_reg.set_xlabel('be')
    ax_be_reg.set_ylabel('reg')

    ax_mean_dice = fig.add_subplot(gs[0, 1])
    sc_mean_dice = ax_mean_dice.scatter(lr, reg, c=mean_dice, cmap='viridis', s=100)
    ax_mean_dice.set_xlabel('lr')
    ax_mean_dice.set_ylabel('reg')

    # Add colorbar with a different colormap
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sc_mean_dice, cax=cbar_ax, cmap='plasma')  # Change the colormap here
    cbar.set_label('Mean Dice')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    plt.show()


def top_n_hparams(data, target_metric, n=5, high=True):
    if target_metric not in data.columns:
        raise ValueError("Invalid target metric")

    grouped = data.groupby('loss')
    top_records = []

    for name, group in grouped:
        sorted_group = group.sort_values(by=target_metric, ascending=False)
        if high:
            top_n = sorted_group.nlargest(n, target_metric)
        else:
            top_n = sorted_group.nsmallest(n, target_metric)
        top_records.append(top_n)

    before_metric = target_metric.replace('after_reg', 'before_reg')
    result = []

    # result.append(f"loss,lr,be,reg,{before_metric},{target_metric}")
    for group_df in top_records:
        columns_to_select = ['loss', 'lr', 'be', 'reg', before_metric, target_metric]
        result.append(group_df[columns_to_select])# .to_string(index=False, header=False))

    return pd.concat(result, ignore_index=True)


if __name__== '__main__':
    data = pd.read_csv('imgs/grid_search_results.csv')
    # for target_metric, high in {'after_reg_mean_dice': True, 'after_reg_dice_class_1': True,
    #                             'after_reg_dice_class_2': True, 'after_reg_dice_class_3': True,
    #                             'after_reg_dice_class_4': True, 'after_reg_dice_class_5': True,
    #                             'after_reg_rmse': False,
    #                             }.items():
    #     print(120*'-')
    #     print(f"Top 5 hyperparameters for {target_metric}")
    #     print(120*'-')
    #     print(top_n_hparams(data, target_metric, n=5, high=high))
    plot_subclasses_dice(data)
    plot_mean_dice(data)
    plot_mean_rmse(data)
    plot_metrics_per_loss(data, 'after_reg_folding_ratio', 'Average Folding Ratio')
    plot_metrics_per_loss(data, 'after_reg_mag_det_jac_det', 'Average Mag Jac Det')
    # For lr vs Mean Dice Score plot
    plot_var_effect(data, x_axis='lr', y_axis='after_reg_mean_dice',
                    x_label='Learning Rate (LR)', y_label='Mean Dice Score',
                    title='Effect of Learning Rate on Mean Dice after Registration (per Loss Function)',
                    legend_title='Loss Function')
    # For lr vs RMSE Score plot
    plot_var_effect(data, x_axis='lr', y_axis='after_reg_rmse',
                    x_label='Learning Rate (LR)', y_label='RMSE Score',
                    title='Effect of Learning Rate on RMSE after Registration (per Loss Function)',
                    legend_title='Loss Function')
    # For be vs Mean Dice Score plot
    plot_var_effect(data, x_axis='be', y_axis='after_reg_mean_dice',
                    x_label='Bending Energy Weight (be)', y_label='Mean Dice Score',
                    title='Effect of Bending Energy on Mean Dice after Registration (per Loss Function)',
                    legend_title='Loss Function')
    # For be vs RMSE Score plot
    plot_var_effect(data, x_axis='lr', y_axis='after_reg_rmse',
                    x_label='Bending Energy Weight (be)', y_label='RMSE Score',
                    title='Effect of Bending Energy Weight on RMSE after Registration (per Loss Function)',
                    legend_title='Loss Function')
    # For l2_reg vs Mean Dice Score plot
    plot_var_effect(data, x_axis='reg', y_axis='after_reg_mean_dice',
                    x_label='L2 Reg (reg)', y_label='Mean Dice Score',
                    title='Effect of L2 Reg on Mean Dice after Registration (per Loss Function)',
                    legend_title='Loss Function')
    # For l2_reg vs RMSE Score plot
    plot_var_effect(data, x_axis='reg', y_axis='after_reg_rmse',
                    x_label='L2 Reg (reg)', y_label='RMSE Score',
                    title='Effect of L2 Reg on RMSE after Registration (per Loss Function)',
                    legend_title='Loss Function')
    plot_3d(data)
