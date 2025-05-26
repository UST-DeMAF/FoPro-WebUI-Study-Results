import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import Counter
from statistics import mode, StatisticsError
import os
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene, combine_pvalues
import ast
import re




def split_data_by_task_tool(df, t1_tool_col='T1_UQ_1', t2_tool_col='T2_UQ_1'):
    df_t1_demaf_t2_monokle = df[(df[t1_tool_col].str.lower() == 'demaf') & (df[t2_tool_col].str.lower() == 'monokle')]
    df_t1_monokle_t2_demaf = df[(df[t1_tool_col].str.lower() == 'monokle') & (df[t2_tool_col].str.lower() == 'demaf')]
    return df_t1_demaf_t2_monokle, df_t1_monokle_t2_demaf


def read_csv_to_df(file_path):
    """
    Reads a CSV file into a pandas DataFrame and splits string entries on ";" into arrays of strings.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the processed data.
    """
    try:
        df = pd.read_csv(file_path)

        # Process string entries, excluding specific columns
        excluded_columns = ['new_header', 'original_header']
        for col in df.select_dtypes(include=['object']).columns:
            if col not in excluded_columns:
                df[col] = df[col].apply(
                    lambda x: np.array([s.strip().lower() for s in x.split(";")])
                    if isinstance(x, str) and ";" in x
                    else x.lower() if isinstance(x, str)
                    else x
                )
                # Cast single-element lists to strings
                #df[col] = df[col].apply(
                #   lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
                #)
        print(f"Data loaded and processed successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


def plot_boxplot(df, columns, output_file=None, tile='', x_label='Columns', y_label='Values',
                 y_range=None, horizontal_lines=False, show_means=True, size=(10, 6), tight=True):
    try:
        plt.figure(figsize=size)
        df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
        if df[columns].empty:
            raise ValueError("No valid numerical data to plot with columns " + str(columns))
        # Determine whiskers
        whisker_val = 1.5  # Default matplotlib value

        # Create boxplot
        box = plt.boxplot(
            [df[col].dropna() for col in columns],
            patch_artist=True,
            showmeans=show_means,
            meanline=True,
            meanprops={"color": "black", "linestyle": "--", "linewidth": 0.75},
            medianprops={"color": "black", "linewidth": 0.75},
            whis=whisker_val
        )

        # Set colors for boxes
        if columns and columns[0].lower().startswith("demaf"):
            # Skip the first color if first entry is DeMAF
            colors = sns.color_palette("tab10", n_colors=len(columns) + 1)[1:]
        else:
            colors = sns.color_palette("tab10", n_colors=len(columns))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(tile)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(range(1, len(columns) + 1), columns)
        if y_range:
            plt.ylim(y_range)
        plt.grid(axis='y', visible=horizontal_lines)
        if tight:
            plt.tight_layout()
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, dpi=300)
            print(f"Box plot saved to {output_file}")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting box plot: {e}")
    finally:
        plt.close()

def plot_pie_chart(
    df,
    column,
    output_file=None,
    title="",
    labels_map=None,
    autopct='%1.1f%%',
    startangle=100,
    size=(12, 7),
    tight=True
):
    """
    Plots a colorblind-friendly pie chart for the value counts of a column.
    Flattens lists/arrays and converts NaN to 'None'.
    Shows a legend instead of labels on the pie.
    """
    # Flatten and clean data
    values = []
    for entry in df[column]:
        if isinstance(entry, (list, np.ndarray)):
            for v in entry:
                if pd.isna(v) or str(v).lower() == 'nan':
                    values.append('No answer')
                else:
                    values.append(str(v))
        elif pd.isna(entry) or str(entry).lower() == 'nan':
            values.append('No answer')
        else:
            values.append(str(entry))

    counts = pd.Series(values).value_counts(dropna=False)
    labels = counts.index.astype(str)
    if labels_map:
        labels = [labels_map.get(l, l) for l in labels]
    colors = sns.color_palette("tab20", n_colors=len(counts))

    plt.figure(figsize=size)
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=None,  # No labels on the pie
        colors=colors,
        autopct=autopct,
        startangle=startangle,
        counterclock=False,
        pctdistance=0.75
    )
    # plt.title(title)
    # Create legend with color patches and labels
    plt.legend(wedges, labels, title=title, loc="center left", bbox_to_anchor=(1, 0.5))
    if tight:
        plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_grouped_boxplot(
    df,
    group_labels,
    subgroup_labels,
    output_file=None,
    title='',
    x_label='',
    y_label='',
    y_range=None,
    show_means=True,
    size=(10, 6),
    tight=True,
    fontsize=16
):
    """
    Plots a grouped boxplot (e.g., two boxes per x-axis entry).

    Parameters:
        df: DataFrame where columns are in the order [x1_g1, x1_g2, x2_g1, x2_g2, ...]
        group_labels: List of x-axis group labels (e.g., knowledge levels)
        subgroup_labels: List of subgroup labels (e.g., ["DeMAF", "Monokle"])
        output_file: Path to save the plot
        title, x_label, y_label, y_range, show_means, size, tight: as in plot_boxplot
    """
    plt.figure(figsize=size)
    data = [df[col].dropna().values for col in df.columns]

    n_groups = len(group_labels)
    n_subgroups = len(subgroup_labels)
    positions = []
    labels = []
    for i in range(n_groups):
        for j in range(n_subgroups):
            positions.append(i * (n_subgroups + 1) + j)
            labels.append(subgroup_labels[j])
    # Plot each box at the correct position
    box = plt.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=show_means,
        meanline=True,
        meanprops={"color": "black", "linestyle": "--", "linewidth": 0.75},
        medianprops={"color": "black", "linewidth": 0.75},
        whis=1.5
    )

    # Set colors for subgroups
    colors = sns.color_palette("tab10", n_colors=n_subgroups)
    for patch, i in zip(box['boxes'], range(len(data))):
        patch.set_facecolor(colors[i % n_subgroups])

    # Set x-ticks in the center of each group
    group_centers = [i * (n_subgroups + 1) + (n_subgroups - 1) / 2 for i in range(n_groups)]
    plt.xticks(group_centers, group_labels, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    if y_range:
        plt.ylim(y_range)
    if tight:
        plt.tight_layout()        
        plt.subplots_adjust(left=0.10)  # Add this line to ensure y-label is visible

    # Create legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=8) for i in range(n_subgroups)]
    plt.legend(handles, subgroup_labels, title="Tool", loc="best")

    # Set tick label font size for y-axis
    plt.yticks(fontsize=fontsize)   

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Grouped box plot saved to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_histogram(data, bins, output_file, title="", x_label="", y_label="", swap_axes=True, figsize=(21, 9), text_size=16):
    """
    Plots a histogram and saves it to the specified output file.

    Parameters:
        data (list or array-like): The data to plot in the histogram.
        bins (int or sequence): Number of bins or bin edges for the histogram.
        output_file (str): Path to save the histogram image.
        title (str): Title of the histogram.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        swap_axes (bool): If True, swaps the x and y axes.
        :param figsize: Size of the figure.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Plot the histogram
    plt.figure(figsize=figsize)
    ax = plt.gca()

    counter = Counter(data)
    labels = list(counter.keys())
    counts = list(counter.values())

    cmap = plt.get_cmap("tab10")  # Farbpalette
    colors = [cmap(i % 10) for i in range(len(counts))]

    # Ensure the count axis has integer steps
    if swap_axes:
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.label.set_size(text_size)
        ax.xaxis.label.set_size(text_size)
        ax.tick_params(axis='y', labelsize=text_size)
        ax.tick_params(axis='x', labelsize=text_size)
    else:
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, counts, color=colors, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.label.set_size(text_size)
        ax.xaxis.label.set_size(text_size)
        ax.tick_params(axis='y', labelsize=text_size)
        ax.tick_params(axis='x', labelsize=text_size)

    ax.set_title(title, fontsize=text_size)
    
    plt.tight_layout()

    plt.savefig(output_file, dpi = 300)
    plt.close()

def plot_grouped_barchart(
    df,
    group_labels,
    subgroup_labels,
    output_file=None,
    title='',
    x_label='',
    y_label='',
    y_range=None,
    size=(10, 6),
    tight=True,
    subgroupname = "Group",
):
    """
    Plots a grouped vertical bar chart.

    Parameters:
        df: DataFrame where columns are in the order [x1_g1, x1_g2, x2_g1, x2_g2, ...]
        group_labels: List of x-axis group labels (e.g., knowledge levels)
        subgroup_labels: List of subgroup labels (e.g., ["DeMAF", "Monokle"])
        output_file: Path to save the plot
        title, x_label, y_label, y_range, size, tight: as in plot_boxplot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    n_groups = len(group_labels)
    n_subgroups = len(subgroup_labels)
    bar_width = 0.8 / n_subgroups
    index = np.arange(n_groups)

    # Prepare data: means for each bar
    means = np.zeros((n_groups, n_subgroups))
    for i in range(n_groups):
        for j in range(n_subgroups):
            col_idx = i * n_subgroups + j
            col = df.columns[col_idx]
            means[i, j] = np.nanmean(df[col])

    colors = sns.color_palette("tab10", n_colors=n_subgroups)

    plt.figure(figsize=size)
    for j in range(n_subgroups):
        plt.bar(
            index + j * bar_width,
            means[:, j],
            bar_width,
            label=subgroup_labels[j],
            color=colors[j]
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(index + bar_width * (n_subgroups - 1) / 2, group_labels)
    if y_range:
        plt.ylim(y_range)
    plt.legend(title=subgroupname)
    if tight:
        plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Grouped bar chart saved to {output_file}")
    else:
        plt.show()
    plt.close()


def plot_demographics(df, output_folder="plots/demographics"):
    """
    Plots the distribution of 'education', 'field', and 'gender' columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        output_folder (str): The folder where the plots will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Plot 'education' column
    plt.figure(figsize=(8, 6))
    sns.countplot(y='education', data=df, order=df['education'].value_counts().index, hue='education', palette='viridis', legend=False)
    plt.title('Distribution of Education Levels')
    plt.xlabel('Count')
    plt.ylabel('Education')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'education_distribution.png'), dpi = 300)
    plt.close()

    # Plot 'field' column
    plt.figure(figsize=(10, 8))
    sns.countplot(y='field', data=df, order=df['field'].value_counts().index, hue='field', palette='magma', legend=False)
    plt.title('Distribution of Fields of Study')
    plt.xlabel('Count')
    plt.ylabel('Field')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'field_distribution.png'), dpi = 300)
    plt.close()

    # Plot 'gender' column
    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', data=df, order=df['gender'].value_counts().index, hue='gender', palette='coolwarm', legend=False)
    plt.title('Distribution of Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'gender_distribution.png'), dpi = 300)
    plt.close()


def plot_task_times(df, y_range=None):
    """
    Plots task times split by the tool used for each task, and uses the y_range
    for all box plots if provided.

    :param df: Full DataFrame containing the data.
    :param y_range: Optional tuple (min, max) to set the y-axis limits.
    """
    # Create DataFrames for DeMAF and Monokle task times
    demaf_task_times = pd.DataFrame({
        'Task 1 (DeMAF)': df[df["T1_UQ_1"].apply(lambda x: "demaf" in x)]['T1_time'],
        'Task 2 (DeMAF)': df[df["T2_UQ_1"].apply(lambda x: "demaf" in x)]['T2_time']
    })

    monokle_task_times = pd.DataFrame({
        'Task 1 (Monokle)': df[df["T1_UQ_1"].apply(lambda x: "monokle" in x)]['T1_time'],
        'Task 2 (Monokle)': df[df["T2_UQ_1"].apply(lambda x: "monokle" in x)]['T2_time']
    })

    # Temporarily fill NaN values with 0 for conversion to int, then replace 0 with NA
    demaf_task_times = demaf_task_times.fillna(0).astype(int).replace(0, pd.NA)
    monokle_task_times = monokle_task_times.fillna(0).astype(int).replace(0, pd.NA)

    # Grouped boxplot for overall, DeMAF, and Monokle for T1 and T2 ---
    # Prepare data
    t1_overall = pd.to_numeric(df['T1_time'], errors='coerce').dropna().tolist()
    t1_demaf = pd.to_numeric(df[df["T1_UQ_1"].str.lower() == "demaf"]['T1_time'], errors='coerce').dropna().tolist()
    t1_monokle = pd.to_numeric(df[df["T1_UQ_1"].str.lower() == "monokle"]['T1_time'], errors='coerce').dropna().tolist()
    t2_overall = pd.to_numeric(df['T2_time'], errors='coerce').dropna().tolist()
    t2_demaf = pd.to_numeric(df[df["T2_UQ_1"].str.lower() == "demaf"]['T2_time'], errors='coerce').dropna().tolist()
    t2_monokle = pd.to_numeric(df[df["T2_UQ_1"].str.lower() == "monokle"]['T2_time'], errors='coerce').dropna().tolist()

    # Pad lists to same length
    max_len = max(len(t1_overall), len(t1_demaf), len(t1_monokle), len(t2_overall), len(t2_demaf), len(t2_monokle))
    t1_overall += [np.nan] * (max_len - len(t1_overall))
    t1_demaf += [np.nan] * (max_len - len(t1_demaf))
    t1_monokle += [np.nan] * (max_len - len(t1_monokle))
    t2_overall += [np.nan] * (max_len - len(t2_overall))
    t2_demaf += [np.nan] * (max_len - len(t2_demaf))
    t2_monokle += [np.nan] * (max_len - len(t2_monokle))

    # Combine Task 1 and Task 2 for "Both", "DeMAF", "Monokle"
    both_vals = t1_overall + t2_overall
    demaf_vals = t1_demaf + t2_demaf
    monokle_vals = t1_monokle + t2_monokle
    max_len = max(len(both_vals), len(demaf_vals), len(monokle_vals))
    both_vals += [np.nan] * (max_len - len(both_vals))
    demaf_vals += [np.nan] * (max_len - len(demaf_vals))
    monokle_vals += [np.nan] * (max_len - len(monokle_vals))

    # Find the maximum length among all lists
    all_lists = [both_vals, demaf_vals, monokle_vals, t1_overall, t1_demaf, t1_monokle, t2_overall, t2_demaf, t2_monokle]
    max_len = max(len(lst) for lst in all_lists)

    # Pad all lists to max_len with np.nan
    both_vals += [np.nan] * (max_len - len(both_vals))
    demaf_vals += [np.nan] * (max_len - len(demaf_vals))
    monokle_vals += [np.nan] * (max_len - len(monokle_vals))
    t1_overall += [np.nan] * (max_len - len(t1_overall))
    t1_demaf += [np.nan] * (max_len - len(t1_demaf))
    t1_monokle += [np.nan] * (max_len - len(t1_monokle))
    t2_overall += [np.nan] * (max_len - len(t2_overall))
    t2_demaf += [np.nan] * (max_len - len(t2_demaf))
    t2_monokle += [np.nan] * (max_len - len(t2_monokle))

    plot_df = pd.DataFrame({
        "Both_Both": both_vals,
        "Both_DeMAF": demaf_vals,
        "Both_Monokle": monokle_vals,
        "T1_Both": t1_overall,
        "T1_DeMAF": t1_demaf,
        "T1_Monokle": t1_monokle,
        "T2_Both": t2_overall,
        "T2_DeMAF": t2_demaf,
        "T2_Monokle": t2_monokle
    })

    group_labels = ["Both", "Task 1", "Task 2"]
    subgroup_labels = ["Both", "DeMAF", "Monokle"]
    ordered_columns = []
    for group in ["Both", "T1", "T2"]:
        for subgroup in ["Both", "DeMAF", "Monokle"]:
            ordered_columns.append(f"{group}_{subgroup}")

    xtick_labels = ["Both Tasks", "Task 1", "Task 2"]    

    plot_grouped_boxplot(
        plot_df[ordered_columns],
        group_labels=xtick_labels,
        subgroup_labels=subgroup_labels,
        output_file="plots/task-times/grouped_task_times.png",
        title="Task Completion Time by Task and Tool",
        x_label=None,
        y_label="Time (seconds)",
        y_range=y_range,
        tight=True
    )
    # General task times plot
    plot_boxplot(
        df,
        ['T1_time', 'T2_time'],
        output_file="plots/task-times/task_times_overall.png",
        tile="Task Times (Both)",
        x_label="Tasks",
        y_label="Time (seconds)",
        y_range=y_range,
        size=(5,6)
    )

    # Task times for DeMAF
    plot_boxplot(
        demaf_task_times,
        ['Task 1 (DeMAF)', 'Task 2 (DeMAF)'],
        output_file="plots/task-times/task_times_demaf_only.png",
        tile="Task Times for DeMAF",
        x_label="Tasks",
        y_label="Time (seconds)",
        y_range=y_range,
        size=(5,6)
    )

    # Task times for Monokle
    plot_boxplot(
        monokle_task_times,
        ['Task 1 (Monokle)', 'Task 2 (Monokle)'],
        output_file="plots/task-times/task_times_monokle_only.png",
        tile="Task Times for Monokle",
        x_label="Tasks",
        y_label="Time (seconds)",
        y_range=y_range,
        size=(5,6)
    )

    # Helper function to plot comparison for a given task (1 or 2)
    def plot_comparison(task, demaf_df, monokle_df):
        col_demaf = f"Task {task} (DeMAF)"
        col_monokle = f"Task {task} (Monokle)"
        comparison_df = pd.DataFrame({
            'DeMAF': demaf_df[col_demaf],
            'Monokle': monokle_df[col_monokle]
        })
        plot_boxplot(
            comparison_df,
            ['DeMAF', 'Monokle'],
            output_file=f"plots/task-times/task{task}_time_comparison.png",
            tile=f"Comparison of Task {task} Time by Tool",
            x_label="Tool",
            y_label="Time (seconds)",
            y_range=y_range,
            size=(5,6)
        )

    # Plot comparisons for Task 1 and Task 2 using the helper
    plot_comparison(1, demaf_task_times, monokle_task_times)
    plot_comparison(2, demaf_task_times, monokle_task_times)

    def get_task_independent_tool_times(df, tool):
        vals_t1 = pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == tool, "T1_time"], errors="coerce").dropna() if "T1_time" in df.columns else pd.Series(dtype=float)
        vals_t2 = pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == tool, "T2_time"], errors="coerce").dropna() if "T2_time" in df.columns else pd.Series(dtype=float)
        return pd.concat([vals_t1, vals_t2], ignore_index=True)

    # --- Task-independent, per-tool evaluation over all relevant columns (combined plot & stats) ---
    demaf_times = get_task_independent_tool_times(df, "demaf")
    monokle_times = get_task_independent_tool_times(df, "monokle")

    # Write stats to file
    stats_file = "eval/task_time_stats_task_independent.txt"
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w") as f:
        for tool, vals in [("DeMAF", demaf_times), ("Monokle", monokle_times)]:
            vals = vals.dropna()
            if not vals.empty:
                median = round(np.median(vals), 2)
                mean = round(np.mean(vals), 2)
                try:
                    mode_val = round(mode(vals), 2)
                except StatisticsError:
                    mode_val = np.nan
                std = round(np.std(vals, ddof=1), 2) if len(vals) > 1 else np.nan
                f.write(f"\n=== Task-independent Task Times for {tool} (all tasks) ===\n")
                f.write(f" Values: {vals.tolist()}\n")
                f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_val}, Std Dev: {std}\n")
            else:
                f.write(f"\n=== Task-independent Task Times for {tool} (all tasks) ===\nNo data.\n")

    # Plot combined boxplot
    if not demaf_times.empty or not monokle_times.empty:
        max_len = max(len(demaf_times), len(monokle_times))
        demaf_list = demaf_times.tolist() + [np.nan] * (max_len - len(demaf_times))
        monokle_list = monokle_times.tolist() + [np.nan] * (max_len - len(monokle_times))
        plot_df = pd.DataFrame({
            "DeMAF": demaf_list,
            "Monokle": monokle_list
        })
        plot_boxplot(
            plot_df,
            ["DeMAF", "Monokle"],
            output_file="plots/task-times/task_independent_demaf_monokle.png",
            tile="Task Completion Time by Tool (both tasks)",
            x_label="Tool",
            y_label="Time (seconds)",
            y_range=y_range,
            tight=True
        )


def plot_demographics_pie_charts(df, output_folder="plots/demographics_pie"):
    """
    Plots a separate colorblind-friendly pie chart for each demographic column:
    'gender', 'age', 'education', 'field', and 'KQ_3' (most used tool).
    Each chart is saved as a separate file in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    columns = {
        "gender": "Gender",
        "age": "Age",
        "education": "Education Level",
        "field": "Background",
        "KQ_3": "Tools Used Before",
        "KQ_5": "Most Used Tool"
    }
    for col, title in columns.items():
        output_file = os.path.join(output_folder, f"{col}_pie.png")
        plot_pie_chart(
            df,
            column=col,
            output_file=output_file,
            title=title,
            tight=True
        )

# Helper to compute stats for a given series
def summarize(series):
    values = series.dropna().tolist()
    median = round(np.median(values), 2) if values else np.nan
    mean = round(np.mean(values), 2) if values else np.nan
    try:
        mode_value = round(mode(values), 2) if values else np.nan
    except StatisticsError:
        mode_value = np.nan
    std = round(np.std(values, ddof=1), 2) if len(values) > 1 else np.nan
    return values, median, mean, mode_value, std

def compute_all_likert_stats_and_plots(df, mapping_df, output_file="eval/likert_stats.txt", plot_output_folder="plots/likert"):
    """
    Computes general statistics for the Likert scale columns of both tasks (T1 and T2) and
    writes the results to a text file, using original column names from mapping_df for output and plots.
    """
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def get_task_independent_nasa(df, tool):
        vals_t1 = pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == tool, "T1_LQ_agg"], errors="coerce").dropna() if "T1_LQ_agg" in df.columns else pd.Series(dtype=float)
        vals_t2 = pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == tool, "T2_LQ_agg"], errors="coerce").dropna() if "T2_LQ_agg" in df.columns else pd.Series(dtype=float)
        return pd.concat([vals_t1, vals_t2], ignore_index=True)

    # Create a dictionary from the mapping DataFrame
    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header']))

    tasks = ["T1", "T2"]
    with open(output_file, "w") as f:
        group_labels = ["Task 1", "Task 2"]
        subgroup_labels = ["Both", "DeMAF", "Monokle"]
        plot_data = {}

        for task in tasks:
            # Only aggregate the first five Likert scale columns (NASA TLX)
            likert_cols = [f"{task}_LQ_{i}" for i in range(1, 6)]
            df[likert_cols] = df[likert_cols].apply(pd.to_numeric, errors='coerce')

            # Define conditions based on the tool indicator column
            condition_demaf = df[f"{task}_UQ_1"].str.lower() == "demaf"
            condition_monokle = df[f"{task}_UQ_1"].str.lower() == "monokle"

            # Aggregate mean NASA Likert score per participant for this task
            likert_agg = df[likert_cols].mean(axis=1, skipna=True)
            overall_vals = likert_agg.tolist()
            demaf_vals = likert_agg[condition_demaf].tolist()
            monokle_vals = likert_agg[condition_monokle].tolist()

            # Pad to same length for DataFrame
            max_len = max(len(overall_vals), len(demaf_vals), len(monokle_vals), 1)
            overall_vals += [np.nan] * (max_len - len(overall_vals))
            demaf_vals += [np.nan] * (max_len - len(demaf_vals))
            monokle_vals += [np.nan] * (max_len - len(monokle_vals))
            plot_data[f"{task}_Both"] = overall_vals
            plot_data[f"{task}_DeMAF"] = demaf_vals
            plot_data[f"{task}_Monokle"] = monokle_vals

        # Combine Task 1 and Task 2 for "Both", "DeMAF", "Monokle"
        both_vals = plot_data["T1_Both"] + plot_data["T2_Both"]
        demaf_vals = plot_data["T1_DeMAF"] + plot_data["T2_DeMAF"]
        monokle_vals = plot_data["T1_Monokle"] + plot_data["T2_Monokle"]
        max_len = max(len(both_vals), len(demaf_vals), len(monokle_vals))
        both_vals += [np.nan] * (max_len - len(both_vals))
        demaf_vals += [np.nan] * (max_len - len(demaf_vals))
        monokle_vals += [np.nan] * (max_len - len(monokle_vals))
        plot_data["Both_Both"] = both_vals
        plot_data["Both_DeMAF"] = demaf_vals
        plot_data["Both_Monokle"] = monokle_vals

        

        # The columns must be in the order [T1_Both, T1_DeMAF, T1_Monokle, T2_Both, T2_DeMAF, T2_Monokle]
        group_labels = ["Both Tasks", "Task 1", "Task 2"]
        subgroup_labels = ["Both", "DeMAF", "Monokle"]
        ordered_columns = []
        for group in ["Both", "T1", "T2"]:
            for subgroup in ["Both", "DeMAF", "Monokle"]:
                ordered_columns.append(f"{group}_{subgroup}")        

        # Pad all lists to the same length
        max_len = max(len(lst) for lst in plot_data.values())
        for key in plot_data:
            plot_data[key] += [np.nan] * (max_len - len(plot_data[key]))
        plot_df = pd.DataFrame(plot_data)

        plot_grouped_boxplot(
            plot_df[ordered_columns],
            group_labels=group_labels,
            subgroup_labels=subgroup_labels,
            output_file=f"{plot_output_folder}/grouped_nasa_likert_by_task.png",
            title="Aggregated, Normalized NASA-TLX by Task and Tool",
            x_label=None,
            y_label="Aggregated, Normalized NASA-TLX",
            y_range=(0.5, 7.5),
            tight=True
        )

        for task in tasks:
            # Identify Likert columns for the current task
            likert_cols = [f"{task}_LQ_{i}" for i in range(1, 17)]
            if task == "T1":  # Add tool doc likert
                likert_cols += [f"{task}_UQ_16"]
            else:
                likert_cols += [f"{task}_UQ_14"]
            df[likert_cols] = df[likert_cols].apply(pd.to_numeric, errors='coerce')

            # Define conditions based on the tool indicator column
            condition_demaf = df[f"{task}_UQ_1"].str.lower() == "demaf"
            condition_monokle = df[f"{task}_UQ_1"].str.lower() == "monokle"           

            f.write(f"\nGeneral statistics for {task} Likert Scale Questions:\n")
            for col in likert_cols:
                overall_vals, overall_med, overall_mean, overall_mode, overall_std = summarize(df[col])
                demaf_vals, demaf_med, demaf_mean, demaf_mode, demaf_std = summarize(df.loc[condition_demaf, col])
                monokle_vals, monokle_med, monokle_mean, monokle_mode, monokle_std = summarize(df.loc[condition_monokle, col])

                # Use the mapped column name for output and plots
                mapped_col_name = mapping_dict.get(col, col)
                if task == "T1":
                    mapped_col_name = f"{mapped_col_name} [T1]"
                else:
                    mapped_col_name = mapped_col_name.replace(".1", " [T2]")

                f.write(f"\nStatistics for {mapped_col_name}:\n")
                f.write(f" Both - Values: {overall_vals}\n")
                f.write(f"           Median: {overall_med}, Mean: {overall_mean}, Mode: {overall_mode}, Std Dev: {overall_std}\n")
                f.write(f" DeMAF   - Values: {demaf_vals}\n")
                f.write(f"           Median: {demaf_med}, Mean: {demaf_mean}, Mode: {demaf_mode}, Std Dev: {demaf_std}\n")
                f.write(f" Monokle - Values: {monokle_vals}\n")
                f.write(f"           Median: {monokle_med}, Mean: {monokle_mean}, Mode: {monokle_mode}, Std Dev: {monokle_std}\n")

                # Create a DataFrame for plotting
                max_length = max(len(overall_vals), len(demaf_vals), len(monokle_vals))
                overall_vals += [np.nan] * (max_length - len(overall_vals))
                demaf_vals += [np.nan] * (max_length - len(demaf_vals))
                monokle_vals += [np.nan] * (max_length - len(monokle_vals))

                plot_data = pd.DataFrame({
                    'Both': overall_vals,
                    'DeMAF': demaf_vals,
                    'Monokle': monokle_vals
                })

                # Generate the plot
                y_range = (0.5, 7.5) if col in [f"{task}_LQ_{i}" for i in range(1, 6)] else (0.5, 5.5)
                plot_boxplot(
                    plot_data,
                    ['Both', 'DeMAF', 'Monokle'],
                    output_file=f"{plot_output_folder}/{col}_likert_plot.png",
                    tile=f"\"{mapped_col_name}\"",
                    x_label="",
                    y_label="Likert Scale Value",
                    y_range=y_range,
                    size=(10, 4),
                )

            # Additional aggregation for Likert cols 1 to 5
            agg_cols = [f"{task}_LQ_{i}" for i in range(1, 6)]
            df[task + '_LQ_agg'] = df[agg_cols].mean(axis=1, skipna=True)

            overall_agg = pd.to_numeric(df[task + '_LQ_agg'], errors='coerce')
            demaf_agg = pd.to_numeric(df.loc[condition_demaf, task + '_LQ_agg'], errors='coerce')
            monokle_agg = pd.to_numeric(df.loc[condition_monokle, task + '_LQ_agg'], errors='coerce')

            overall_vals, overall_med, overall_mean, overall_mode, overall_std = summarize(overall_agg)
            demaf_vals, demaf_med, demaf_mean, demaf_mode, demaf_std = summarize(demaf_agg)
            monokle_vals, monokle_med, monokle_mean, monokle_mode, monokle_std = summarize(monokle_agg)

            agg_col_name = mapping_dict.get(task + '_LQ_agg', task + '_LQ_agg')

            f.write(f"\nAggregated, Normalized statistics for {agg_col_name} (Columns 1 to 5):\n")
            f.write(f" Both - Values: {overall_vals}\n")
            f.write(f"           Median: {overall_med}, Mean: {overall_mean}, Mode: {overall_mode}, Std Dev: {overall_std}\n")
            f.write(f" DeMAF   - Values: {demaf_vals}\n")
            f.write(f"           Median: {demaf_med}, Mean: {demaf_mean}, Mode: {demaf_mode}, Std Dev: {demaf_std}\n")
            f.write(f" Monokle - Values: {monokle_vals}\n")
            f.write(f"           Median: {monokle_med}, Mean: {monokle_mean}, Mode: {monokle_mode}, Std Dev: {monokle_std}\n")

            # Prepare plotting aggregated data
            max_length = max(len(overall_vals), len(demaf_vals), len(monokle_vals))
            overall_vals += [np.nan] * (max_length - len(overall_vals))
            demaf_vals += [np.nan] * (max_length - len(demaf_vals))
            monokle_vals += [np.nan] * (max_length - len(monokle_vals))
            agg_plot_data = pd.DataFrame({
                'Both': overall_vals,
                'DeMAF': demaf_vals,
                'Monokle': monokle_vals
            })
            plot_boxplot(
                agg_plot_data,
                ['Both', 'DeMAF', 'Monokle'],
                output_file=f"{plot_output_folder}/{task}_agg_likert_plot.png",
                tile=f"Aggregated, Normalized NASA-TLX for Task {agg_col_name[1:2]}",
                x_label="",
                y_label="Likert Scale Value",
                y_range=(0.5, 7.5),
                size=(10,4)
            )
        # --- Aggregated NASA-TLX per tool, combined over both tasks ---
        demaf_agg_both = get_task_independent_nasa(df, "demaf")
        monokle_agg_both = get_task_independent_nasa(df, "monokle")

        if not demaf_agg_both.empty or not monokle_agg_both.empty:
            vals, median, mean, mode_val, std = summarize(demaf_agg_both)
            f.write(f"\n=== Aggregated NASA-TLX (cols 1-5) for DeMAF (both tasks) ===\n")
            f.write(f" Values: {vals}\n")
            f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_val}, Std Dev: {std}\n")

            vals, median, mean, mode_val, std = summarize(monokle_agg_both)
            f.write(f"\n=== Aggregated NASA-TLX (cols 1-5) for Monokle (both tasks) ===\n")
            f.write(f" Values: {vals}\n")
            f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_val}, Std Dev: {std}\n")

            # Plot boxplot for both
            max_len = max(len(demaf_agg_both), len(monokle_agg_both))
            demaf_list = demaf_agg_both.tolist() + [np.nan] * (max_len - len(demaf_agg_both))
            monokle_list = monokle_agg_both.tolist() + [np.nan] * (max_len - len(monokle_agg_both))
            plot_df = pd.DataFrame({
                "DeMAF": demaf_list,
                "Monokle": monokle_list
            })
            plot_boxplot(
                plot_df,
                ["DeMAF", "Monokle"],
                output_file=f"{plot_output_folder}/agg_nasa_both_tasks_demaf_monokle.png",
                tile="Aggregated, Normalized NASA-TLX by Tool (both tasks)",
                x_label="Tool",
                y_label="Aggregated, Normalized NASA-TLX",
                y_range=(0.5, 7.5)
            )

def compute_all_likert_stats_and_plots_combined(df, mapping_df, output_file="eval/likert_stats_tasks_combined.txt", plot_output_folder="plots/likert_task_combined"):
    """
    Computes general statistics for the Likert scale columns of both tasks (T1 and T2) combined
    (i.e., no split by task), writes the results to a text file, and creates plots.
    Uses original column names from mapping_df for output and plots.
    """
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header']))

    # All Likert columns (T1 and T2)
    likert_cols = [f"T1_LQ_{i}" for i in range(1, 17)] + [f"T2_LQ_{i}" for i in range(1, 17)]
    # Add tool doc likert
    likert_cols += ["T1_UQ_16", "T2_UQ_14"]
    likert_cols = [col for col in likert_cols if col in df.columns]

    # Combine T1 and T2 for each Likert question (e.g., T1_LQ_1 + T2_LQ_1)
    combined_likert = {}
    for i in range(1, 17):
        cols = [col for col in [f"T1_LQ_{i}", f"T2_LQ_{i}"] if col in df.columns]
        if cols:
            combined_likert[f"LQ_{i}"] = pd.concat([pd.to_numeric(df[col], errors='coerce') for col in cols], ignore_index=True)
    # Tool doc likert
    if "T1_UQ_16" in df.columns or "T2_UQ_14" in df.columns:
        cols = [col for col in ["T1_UQ_16", "T2_UQ_14"] if col in df.columns]
        combined_likert["LQ_tooldoc"] = pd.concat([pd.to_numeric(df[col], errors='coerce') for col in cols], ignore_index=True)

    # Helper to get tool masks for both tasks
    mask_demaf = (df["T1_UQ_1"].str.lower() == "demaf") | (df["T2_UQ_1"].str.lower() == "demaf")
    mask_monokle = (df["T1_UQ_1"].str.lower() == "monokle") | (df["T2_UQ_1"].str.lower() == "monokle")

    def get_tool_combined_values(cols, tool):
        if tool == "demaf":
            mask = (df["T1_UQ_1"].str.lower() == "demaf")
            vals_t1 = pd.to_numeric(df.loc[mask, cols[0]], errors="coerce") if cols[0] in df.columns else pd.Series(dtype=float)
            mask = (df["T2_UQ_1"].str.lower() == "demaf")
            vals_t2 = pd.to_numeric(df.loc[mask, cols[1]], errors="coerce") if len(cols) > 1 and cols[1] in df.columns else pd.Series(dtype=float)
            return pd.concat([vals_t1, vals_t2], ignore_index=True)
        elif tool == "monokle":
            mask = (df["T1_UQ_1"].str.lower() == "monokle")
            vals_t1 = pd.to_numeric(df.loc[mask, cols[0]], errors="coerce") if cols[0] in df.columns else pd.Series(dtype=float)
            mask = (df["T2_UQ_1"].str.lower() == "monokle")
            vals_t2 = pd.to_numeric(df.loc[mask, cols[1]], errors="coerce") if len(cols) > 1 and cols[1] in df.columns else pd.Series(dtype=float)
            return pd.concat([vals_t1, vals_t2], ignore_index=True)
        else:
            return pd.Series(dtype=float)

    with open(output_file, "w") as f:
        for i in range(1, 17):
            cols = [col for col in [f"T1_LQ_{i}", f"T2_LQ_{i}"] if col in df.columns]
            if not cols:
                continue
            key = f"LQ_{i}"
            mapped_col_name = mapping_dict.get(f"T1_LQ_{i}", key)
            # All participants
            both = pd.concat([pd.to_numeric(df[col], errors='coerce') for col in cols], ignore_index=True)
            # DeMAF only
            demaf = get_tool_combined_values(cols, "demaf")
            # Monokle only
            monokle = get_tool_combined_values(cols, "monokle")

            # Write stats
            f.write(f"\nStatistics for {mapped_col_name} (combined T1+T2):\n")
            for label, series in [("Both", both), ("DeMAF", demaf), ("Monokle", monokle)]:
                values = series.dropna().tolist()
                median = round(np.median(values), 2) if values else np.nan
                mean = round(np.mean(values), 2) if values else np.nan
                try:
                    mode_value = round(mode(values), 2) if values else np.nan
                except StatisticsError:
                    mode_value = np.nan
                std = round(np.std(values, ddof=1), 2) if len(values) > 1 else np.nan    
                f.write(f"[{label}]:\n")            
                f.write(f" Values: {values}\n")
                f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_value}, Std Dev: {std}\n")

            # Plot
            max_len = max(len(both), len(demaf), len(monokle))
            both_list = both.tolist() + [np.nan] * (max_len - len(both))
            demaf_list = demaf.tolist() + [np.nan] * (max_len - len(demaf))
            monokle_list = monokle.tolist() + [np.nan] * (max_len - len(monokle))
            plot_data = pd.DataFrame({
                "Both": both_list,
                "DeMAF": demaf_list,
                "Monokle": monokle_list
            })
            plot_boxplot(
                plot_data,
                ["Both", "DeMAF", "Monokle"],
                output_file=f"{plot_output_folder}/{key}_likert_plot.png",
                tile=f"\"{mapped_col_name}\" (combined T1+T2)",
                x_label="",
                y_label="Likert Scale Value",
                y_range=(0.5, 7.5),
            )

        # Tool doc likert split by tool
        both = pd.concat([
            pd.to_numeric(df["T1_UQ_16"], errors='coerce') if "T1_UQ_16" in df.columns else pd.Series(dtype=float),
            pd.to_numeric(df["T2_UQ_14"], errors='coerce') if "T2_UQ_14" in df.columns else pd.Series(dtype=float)
        ], ignore_index=True)

        demaf = pd.concat([
            pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == "demaf", "T1_UQ_16"], errors='coerce') if "T1_UQ_16" in df.columns else pd.Series(dtype=float),
            pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == "demaf", "T2_UQ_14"], errors='coerce') if "T2_UQ_14" in df.columns else pd.Series(dtype=float)
        ], ignore_index=True)

        monokle = pd.concat([
            pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == "monokle", "T1_UQ_16"], errors='coerce') if "T1_UQ_16" in df.columns else pd.Series(dtype=float),
            pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == "monokle", "T2_UQ_14"], errors='coerce') if "T2_UQ_14" in df.columns else pd.Series(dtype=float)
        ], ignore_index=True)

        f.write(f"\nStatistics for LQ_tooldoc (combined T1+T2) \n")
        for label, series in [("Both", both), ("DeMAF", demaf), ("Monokle", monokle)]:
            values = series.dropna().tolist()
            median = round(np.median(values), 2) if values else np.nan
            mean = round(np.mean(values), 2) if values else np.nan
            try:
                mode_value = round(mode(values), 2) if values else np.nan
            except StatisticsError:
                mode_value = np.nan
            std = round(np.std(values, ddof=1), 2) if len(values) > 1 else np.nan
            f.write(f"[{label}]:\n")
            f.write(f" Values: {values}\n")
            f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_value}, Std Dev: {std}\n")

        # Plot
        max_len = max(len(both), len(demaf), len(monokle))
        both_list = both.tolist() + [np.nan] * (max_len - len(both))
        demaf_list = demaf.tolist() + [np.nan] * (max_len - len(demaf))
        monokle_list = monokle.tolist() + [np.nan] * (max_len - len(monokle))
        plot_data = pd.DataFrame({
            "Both": both_list,
            "DeMAF": demaf_list,
            "Monokle": monokle_list
        })
        plot_boxplot(
            plot_data,
            ["Both", "DeMAF", "Monokle"],
            output_file=f"{plot_output_folder}/LQ_tooldoc_likert_plot.png",
            tile="\"If you used the documentation, how useful was it?\" (combined T1+T2)",
            x_label="",
            y_label="Likert Scale Value",
            y_range=(0.5, 5.5),
        )

        # --- Aggregated NASA-TLX (cols 1-5) per tool, combined over both tasks ---
        def get_task_independent_nasa(df, tool):
            vals_t1 = pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == tool, "T1_LQ_agg"], errors="coerce").dropna() if "T1_LQ_agg" in df.columns else pd.Series(dtype=float)
            vals_t2 = pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == tool, "T2_LQ_agg"], errors="coerce").dropna() if "T2_LQ_agg" in df.columns else pd.Series(dtype=float)
            return pd.concat([vals_t1, vals_t2], ignore_index=True)

        demaf_agg_both = get_task_independent_nasa(df, "demaf")
        monokle_agg_both = get_task_independent_nasa(df, "monokle")

        if not demaf_agg_both.empty or not monokle_agg_both.empty:
            for label, series in [("DeMAF", demaf_agg_both), ("Monokle", monokle_agg_both)]:
                values = series.dropna().tolist()
                median = round(np.median(values), 2) if values else np.nan
                mean = round(np.mean(values), 2) if values else np.nan
                try:
                    mode_value = round(mode(values), 2) if values else np.nan
                except StatisticsError:
                    mode_value = np.nan
                std = round(np.std(values, ddof=1), 2) if len(values) > 1 else np.nan
                f.write(f"\n=== Aggregated NASA-TLX (cols 1-5) for {label} (both tasks) ===\n")
                f.write(f" Values: {values}\n")
                f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_value}, Std Dev: {std}\n")

            # Plot boxplot for both
            max_len = max(len(demaf_agg_both), len(monokle_agg_both))
            demaf_list = demaf_agg_both.tolist() + [np.nan] * (max_len - len(demaf_agg_both))
            monokle_list = monokle_agg_both.tolist() + [np.nan] * (max_len - len(monokle_agg_both))
            plot_df = pd.DataFrame({
                "DeMAF": demaf_list,
                "Monokle": monokle_list
            })
            plot_boxplot(
                plot_df,
                ["DeMAF", "Monokle"],
                output_file=f"{plot_output_folder}/agg_nasa_both_tasks_demaf_monokle.png",
                tile="Aggregated, Normalized NASA-TLX by Tool (both tasks)",
                x_label="Tool",
                y_label="Aggregated, Normalized NASA-TLX",
                y_range=(0.5, 7.5)
            )
        
        # --- Find and write best/worst performing Likert columns for DeMAF and Monokle ---
        demaf_means = {}
        monokle_means = {}

        # Collect means for each Likert question (including tool doc)
        for i in range(1, 17):
            cols = [col for col in [f"T1_LQ_{i}", f"T2_LQ_{i}"] if col in df.columns]
            if not cols:
                continue
            key = f"LQ_{i}"
            mapped_col_name = mapping_dict.get(f"T1_LQ_{i}", key)
            demaf = get_tool_combined_values(cols, "demaf")
            monokle = get_tool_combined_values(cols, "monokle")
            demaf_means[mapped_col_name] = np.nanmean(demaf) if len(demaf) > 0 else np.nan
            monokle_means[mapped_col_name] = np.nanmean(monokle) if len(monokle) > 0 else np.nan

        # Tool doc
        if "T1_UQ_16" in df.columns or "T2_UQ_14" in df.columns:
            demaf = pd.concat([
                pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == "demaf", "T1_UQ_16"], errors='coerce') if "T1_UQ_16" in df.columns else pd.Series(dtype=float),
                pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == "demaf", "T2_UQ_14"], errors='coerce') if "T2_UQ_14" in df.columns else pd.Series(dtype=float)
            ], ignore_index=True)
            monokle = pd.concat([
                pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == "monokle", "T1_UQ_16"], errors='coerce') if "T1_UQ_16" in df.columns else pd.Series(dtype=float),
                pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == "monokle", "T2_UQ_14"], errors='coerce') if "T2_UQ_14" in df.columns else pd.Series(dtype=float)
            ], ignore_index=True)
            demaf_means["LQ_tooldoc"] = np.nanmean(demaf) if len(demaf) > 0 else np.nan
            monokle_means["LQ_tooldoc"] = np.nanmean(monokle) if len(monokle) > 0 else np.nan

        # Sort and select best/worst
        def get_best_worst(means_dict):
            nasa_keys = set()
            for i in range(1, 6):
                # Use mapped names if available, else fallback to LQ_{i}
                nasa_keys.add(mapping_dict.get(f"T1_LQ_{i}", f"LQ_{i}"))
            valid = [(col, val) for col, val in means_dict.items() if not np.isnan(val)]
            # NASA: lower is better, others: higher is better
            nasa = [(col, val) for col, val in valid if col in nasa_keys]
            non_nasa = [(col, val) for col, val in valid if col not in nasa_keys]
            # For NASA: best = lowest, worst = highest
            nasa_best = sorted(nasa, key=lambda x: x[1])[:2]
            nasa_worst = sorted(nasa, key=lambda x: -x[1])[:2]
            # For others: best = highest, worst = lowest
            non_nasa_best = sorted(non_nasa, key=lambda x: -x[1])[:2]
            non_nasa_worst = sorted(non_nasa, key=lambda x: x[1])[:2]
            # Combine for output: best = best NASA + best non-NASA, worst = worst NASA + worst non-NASA
            best = nasa_best + non_nasa_best
            worst = nasa_worst + non_nasa_worst
            return best, worst

        demaf_best, demaf_worst = get_best_worst(demaf_means)
        monokle_best, monokle_worst = get_best_worst(monokle_means)

        f.write("\n\n=== Top 2 NASA and none-NASA Best and Worst Performing Likert Columns (by mean) ===\n")
        f.write("DeMAF:\n")
        f.write("  Best:\n")
        for col, val in demaf_best:
            f.write(f"    {col}: {val:.2f}\n")
        f.write("  Worst:\n")
        for col, val in demaf_worst:
            f.write(f"    {col}: {val:.2f}\n")

        f.write("Monokle:\n")
        f.write("  Best:\n")
        for col, val in monokle_best:
            f.write(f"    {col}: {val:.2f}\n")
        f.write("  Worst:\n")
        for col, val in monokle_worst:
            f.write(f"    {col}: {val:.2f}\n")

def evaluate_task_performance(df, df_task_solutions):
    """
    Evaluates task performance based on the provided DataFrames and captures
    count of responses that are "don't know" or empty. For each task question,
    it adds a performance value and a corresponding _dontknow column to df.
    """

    tasks = ["T1", "T2"]
    for index, data in df.iterrows():
        for task in tasks:
            task_cols = [f"{task}_UQ_{i}" for i in (range(2, 14) if task == "T1" else range(2, 12))]
            current_tool = data[f"{task}_UQ_1"]
            for col in task_cols:
                row = df_task_solutions.loc[df_task_solutions['question'] == col.lower()]
                if not row.empty:
                    answer_column = f"answer_{current_tool}"
                    solution_answers = row[answer_column].iloc[0] if not pd.isna(row[answer_column].iloc[0]) else row['answer'].iloc[0]
                    if not isinstance(solution_answers, (list, np.ndarray)):
                        solution_answers = [solution_answers]

                    participant_answers = data[col]

                    if not isinstance(participant_answers, (list, np.ndarray)):
                        participant_answers = [participant_answers]

                    # Normalize all answers to string for comparison
                    def normalize_answer(ans):
                        # Try to convert to float, then to int if possible, then to string
                        try:
                            f = float(ans)
                            i = int(f)
                            # If float is integer-valued, use int
                            return str(i) if f == i else str(f)
                        except (ValueError, TypeError):
                            return str(ans).strip().lower()

                    solution_answers = [normalize_answer(ans) for ans in solution_answers]
                    participant_answers = [normalize_answer(ans) for ans in participant_answers]

                    # Count "don't know" or empty responses
                    dont_know_count = sum(
                        1 for ans in participant_answers
                        if (pd.isna(ans)) or
                        (isinstance(ans, str) and (ans.strip() == "" or ans.strip().lower() in ["don't know", "dont know"]))
                    )
                    # Add the count to a new column for this question
                    df.loc[index, col + "_dont-know"] = dont_know_count

                    correct_participant_answer = 0
                    if col not in ['T1_UQ_13', 'T2_UQ_9']:
                        correct_participant_answer = sum(1 for answer in participant_answers if answer in solution_answers)
                        wrong_participant_answer = sum(1 for answer in participant_answers if answer not in solution_answers)
                        correct_participant_answer -= wrong_participant_answer
                    else:
                        if participant_answers[0] in solution_answers:
                            correct_participant_answer = 2


                    performance = correct_participant_answer / len(solution_answers) * 100 if len(solution_answers) > 0 else 0
                    performance = max(0, performance)  # Ensure performance is not negative

                    # Save performance value for the question
                    df.loc[index, col + "_perf"] = performance
                else:
                    print(f"Column {col} does not exist in df_task_solutions['question'].")
    df = aggregate_perf_by_task(df)

    # Add NASA-TLX aggregation columns if not present
    for task in ["T1", "T2"]:
        nasa_cols = [f"{task}_LQ_{i}" for i in range(1, 6)]
        if not f"{task}_LQ_agg" in df.columns:
            df[f"{task}_LQ_agg"] = df[nasa_cols].mean(axis=1, skipna=True)

    # Clean all string columns of newlines before saving
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('\r', ' ', regex=False).str.replace('\n', ' ', regex=False)

    df = aggregate_perf_by_task(df)
    df.to_csv("prepared data/study_data_perf.csv", index=False, lineterminator="\n", quoting=1)
    return df


def aggregate_perf_by_task(df):
    """
    Aggregates performance columns by task for each participant.
    Assumes performance columns are named like 'T1_UQ_2_perf' etc.
    Overwrites existing aggregated columns to avoid duplicates.
    """
    tasks = ["T1", "T2"]
    for task in tasks:
        perf_cols = [col for col in df.columns if col.startswith(f"{task}_UQ") and col.endswith('_perf')]
        dont_know_cols = [col for col in df.columns if col.startswith(f"{task}_UQ") and col.endswith('_dont-know')]
        if perf_cols:
            df[f"{task}_perf_agg"] = df[perf_cols].mean(axis=1, skipna=True)
        if dont_know_cols:
            df[f"{task}_dont-know_agg"] = df[dont_know_cols].sum(axis=1, skipna=True)
    return df


def plot_and_save_performance(df, stats_output_file=r'eval/performance_stats.txt'):
    """
    Plots performance box plots for each performance column in the DataFrame,
    aggregates performance by task, and computes statistics (values, median, mean,
    mode, and standard deviation) for overall, DeMAF, and Monokle data for both
    individual and aggregated performance columns. The box plots are saved and stats
    are written to a file.
    """

    # Helper to compute statistics for a given series
    def compute_stats(series):
        vals = series.dropna().tolist()
        if vals:
            median_val = round(np.median(vals), 2)
            mean_val = round(np.mean(vals), 2)
            try:
                mode_val = round(mode(vals), 2)
            except StatisticsError:
                mode_val = np.nan
            std_val = round(np.std(vals, ddof=1), 2) if len(vals) > 1 else np.nan
        else:
            median_val = mean_val = mode_val = std_val = np.nan
        return vals, median_val, mean_val, mode_val, std_val
    
    def get_task_independent_tool_values(df, tool, perf_col_pattern="_perf"):
        """
        Returns a single Series combining all relevant columns (from both tasks) for the given tool.
        For example, for 'demaf', combines all T1 and T2 columns where the tool was 'demaf'.
        """
        # Find all columns that match the performance pattern (e.g., '_perf' but not 'agg')
        perf_cols = [col for col in df.columns if perf_col_pattern in col and "agg" not in col]
        values = []
        for col in perf_cols:
            if col.startswith("T1_"):
                mask = df["T1_UQ_1"].str.lower() == tool
            elif col.startswith("T2_"):
                mask = df["T2_UQ_1"].str.lower() == tool
            else:
                continue
            vals = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
            values.extend(vals.tolist())
        return pd.Series(values)

    # --- Grouped boxplot data collection for performance (aggregated by task) ---
    tasks = ["T1", "T2"]
    subgroup_labels = ["Both", "DeMAF", "Monokle"]
    plot_data = {}

    for task in tasks:
        perf_cols = [col for col in df.columns if col.startswith(f"{task}_UQ_") and col.endswith('_perf')]
        # Aggregate mean performance per participant for this task
        perf_agg = df[perf_cols].mean(axis=1, skipna=True)
        condition_demaf = df[f"{task}_UQ_1"].str.lower() == "demaf"
        condition_monokle = df[f"{task}_UQ_1"].str.lower() == "monokle"
        overall_vals = perf_agg.tolist()
        demaf_vals = perf_agg[condition_demaf].tolist()
        monokle_vals = perf_agg[condition_monokle].tolist()
        max_len = max(len(overall_vals), len(demaf_vals), len(monokle_vals), 1)
        overall_vals += [np.nan] * (max_len - len(overall_vals))
        demaf_vals += [np.nan] * (max_len - len(demaf_vals))
        monokle_vals += [np.nan] * (max_len - len(monokle_vals))
        plot_data[f"{task}_Both"] = overall_vals
        plot_data[f"{task}_DeMAF"] = demaf_vals
        plot_data[f"{task}_Monokle"] = monokle_vals
    # Combine Task 1 and Task 2 for "Both", "DeMAF", "Monokle"
    both_vals = plot_data["T1_Both"] + plot_data["T2_Both"]
    demaf_vals = plot_data["T1_DeMAF"] + plot_data["T2_DeMAF"]
    monokle_vals = plot_data["T1_Monokle"] + plot_data["T2_Monokle"]
    max_len = max(len(both_vals), len(demaf_vals), len(monokle_vals))
    both_vals += [np.nan] * (max_len - len(both_vals))
    demaf_vals += [np.nan] * (max_len - len(demaf_vals))
    monokle_vals += [np.nan] * (max_len - len(monokle_vals))
    plot_data["Both_Both"] = both_vals
    plot_data["Both_DeMAF"] = demaf_vals
    plot_data["Both_Monokle"] = monokle_vals   

    # After filling plot_data for both tasks:
    max_len = max(len(lst) for lst in plot_data.values())
    for key in plot_data:
        plot_data[key] += [np.nan] * (max_len - len(plot_data[key]))
    plot_df = pd.DataFrame(plot_data)    

    group_labels = ["Both", "T1", "T2"]
    ordered_columns = []
    for group in group_labels:
        for subgroup in subgroup_labels:
            ordered_columns.append(f"{group}_{subgroup}")

    xtick_labels = ["Both Tasks", "Task 1", "Task 2"]

    plot_grouped_boxplot(
        plot_df[ordered_columns],
        group_labels=xtick_labels,
        subgroup_labels=subgroup_labels,
        output_file="plots/task-performance/grouped_performance_by_task.png",
        title="Aggregated, Normalized Performance by Task and Tool",
        x_label=None,
        y_label="Performance (%)",
        y_range=(-0.5, 100.5),
        tight=True
    )

    with open(stats_output_file, "w") as f:
        # Process individual performance columns (those containing _perf but not agg)
        overall_means = {}
        demaf_means = {}
        monokle_means = {}
        overall_perfect_counts = {}
        demaf_perfect_counts = {}
        monokle_perfect_counts = {}

        for col in df.columns:
            if "_perf" in col and "agg" not in col:
                # Determine task from column name (e.g., T1 or T2)
                task = col.split('_')[0]
                condition_demaf = df[f"{task}_UQ_1"].str.lower() == "demaf"
                condition_monokle = df[f"{task}_UQ_1"].str.lower() == "monokle"

                # Prepare series for overall, DeMAF, and Monokle
                overall = pd.to_numeric(df[col], errors="coerce")
                demaf = pd.to_numeric(df.loc[condition_demaf, col], errors="coerce")
                monokle = pd.to_numeric(df.loc[condition_monokle, col], errors="coerce")

                # Pad lists to the same length for plotting
                overall_list = overall.dropna().tolist()
                demaf_list = demaf.dropna().tolist()
                monokle_list = monokle.dropna().tolist()
                max_length = max(len(overall_list), len(demaf_list), len(monokle_list))
                overall_list += [np.nan] * (max_length - len(overall_list))
                demaf_list += [np.nan] * (max_length - len(demaf_list))
                monokle_list += [np.nan] * (max_length - len(monokle_list))

                plot_df = pd.DataFrame({
                    "Both": overall_list,
                    "DeMAF": demaf_list,
                    "Monokle": monokle_list
                })

                output_path = f"plots/task-performance/{col}_performance.png"
                plot_boxplot(
                    plot_df,
                    ["Both", "DeMAF", "Monokle"],
                    output_file=output_path,
                    tile=f"Performance Distribution for {col}",
                    x_label="Tool",
                    y_label="Performance (%)",
                    y_range=(-0.5, 100.5),
                    tight=True
                )

                overall_stats = compute_stats(overall)
                demaf_stats = compute_stats(demaf)
                monokle_stats = compute_stats(monokle)

                # Count perfect answers (performance == 100)
                overall_perfect = sum(np.isclose(overall, 100))
                demaf_perfect = sum(np.isclose(demaf, 100))
                monokle_perfect = sum(np.isclose(monokle, 100))

                f.write(f"\nStatistics for {col}:\n")
                f.write(f" Both - Values: {overall_stats[0]}\n")
                f.write(f"           Median: {overall_stats[1]}, Mean: {overall_stats[2]}, Mode: {overall_stats[3]}, Std Dev: {overall_stats[4]}\n")
                f.write(f"           Perfect answers (100): {overall_perfect}\n")
                f.write(f" DeMAF   - Values: {demaf_stats[0]}\n")
                f.write(f"           Median: {demaf_stats[1]}, Mean: {demaf_stats[2]}, Mode: {demaf_stats[3]}, Std Dev: {demaf_stats[4]}\n")
                f.write(f"           Perfect answers (100): {demaf_perfect}\n")
                f.write(f" Monokle - Values: {monokle_stats[0]}\n")
                f.write(f"           Median: {monokle_stats[1]}, Mean: {monokle_stats[2]}, Mode: {monokle_stats[3]}, Std Dev: {monokle_stats[4]}\n")
                f.write(f"           Perfect answers (100): {monokle_perfect}\n")
                print(f"Stats computed and written for {col}")

                overall_means[col] = overall_stats[2]
                demaf_means[col] = demaf_stats[2]
                monokle_means[col] = monokle_stats[2]
                overall_perfect_counts[col] = overall_perfect
                demaf_perfect_counts[col] = demaf_perfect
                monokle_perfect_counts[col] = monokle_perfect

        tasks = ["T1", "T2"]
        for task in tasks:
            f.write(f"\n=== {task} Performance Columns Ranked by Mean (Both) ===\n")
            task_overall = {col: val for col, val in overall_means.items() if col.startswith(f"{task}_UQ_")}
            for col, val in sorted(task_overall.items(), key=lambda x: -x[1] if x[1] is not None else float('-inf')):
                perfect = overall_perfect_counts.get(col, 0)
                f.write(f"{col}: {val:.2f} (Perfect: {perfect})\n")

            f.write(f"\n=== {task} Performance Columns Ranked by Mean (DeMAF) ===\n")
            task_demaf = {col: val for col, val in demaf_means.items() if col.startswith(f"{task}_UQ_")}
            for col, val in sorted(task_demaf.items(), key=lambda x: -x[1] if x[1] is not None else float('-inf')):
                perfect = demaf_perfect_counts.get(col, 0)
                f.write(f"{col}: {val:.2f} (Perfect: {perfect})\n")

            f.write(f"\n=== {task} Performance Columns Ranked by Mean (Monokle) ===\n")
            task_monokle = {col: val for col, val in monokle_means.items() if col.startswith(f"{task}_UQ_")}
            for col, val in sorted(task_monokle.items(), key=lambda x: -x[1] if x[1] is not None else float('-inf')):
                perfect = monokle_perfect_counts.get(col, 0)
                f.write(f"{col}: {val:.2f} (Perfect: {perfect})\n")

        # Process aggregated performance columns (ending with _perf_agg)
        for col in [col for col in df.columns if col.endswith('_perf_agg')]:
            task = col.split('_')[0]
            condition_demaf = df[f"{task}_UQ_1"].str.lower() == "demaf"
            condition_monokle = df[f"{task}_UQ_1"].str.lower() == "monokle"

            overall = pd.to_numeric(df[col], errors="coerce")
            demaf = pd.to_numeric(df.loc[condition_demaf, col], errors="coerce")
            monokle = pd.to_numeric(df.loc[condition_monokle, col], errors="coerce")

            overall_list = overall.dropna().tolist()
            demaf_list = demaf.dropna().tolist()
            monokle_list = monokle.dropna().tolist()
            max_length = max(len(overall_list), len(demaf_list), len(monokle_list))
            overall_list += [np.nan] * (max_length - len(overall_list))
            demaf_list += [np.nan] * (max_length - len(demaf_list))
            monokle_list += [np.nan] * (max_length - len(monokle_list))

            plot_df = pd.DataFrame({
                "Both": overall_list,
                "DeMAF": demaf_list,
                "Monokle": monokle_list
            })

            output_path = f"plots/task-performance/{col}_performance.png"
            plot_boxplot(
                plot_df,
                ["Both", "DeMAF", "Monokle"],
                output_file=output_path,
                tile=f"Aggregated, Normalized Performance Distribution for {col}",
                x_label="Tool",
                y_label="Performance (%)",
                y_range=(-0.5, 100.5),
                tight=True
            )

            overall_stats = compute_stats(overall)
            demaf_stats = compute_stats(demaf)
            monokle_stats = compute_stats(monokle)

            f.write(f"\nStatistics for {col} (Aggregated, Normalized):\n")
            f.write(f" Both - Values: {overall_stats[0]}\n")
            f.write(f"           Median: {overall_stats[1]}, Mean: {overall_stats[2]}, Mode: {overall_stats[3]}, Std Dev: {overall_stats[4]}\n")
            f.write(f" DeMAF   - Values: {demaf_stats[0]}\n")
            f.write(f"           Median: {demaf_stats[1]}, Mean: {demaf_stats[2]}, Mode: {demaf_stats[3]}, Std Dev: {demaf_stats[4]}\n")
            f.write(f" Monokle - Values: {monokle_stats[0]}\n")
            f.write(f"           Median: {monokle_stats[1]}, Mean: {monokle_stats[2]}, Mode: {monokle_stats[3]}, Std Dev: {monokle_stats[4]}\n")
            print(f"Stats computed and written for aggregated column {col}")

            # --- Task-independent, per-tool evaluation over all relevant columns ---
            for tool in ["demaf", "monokle"]:
                combined_vals = get_task_independent_tool_values(df, tool)
                if not combined_vals.empty:
                    vals, median, mean, mode_val, std = compute_stats(combined_vals)
                    perfect = sum(np.isclose(combined_vals, 100))
                    # Write to file using the already-open file handle
                    f.write(f"\n=== Task-independent stats for {tool.capitalize()} (all tasks, all questions) ===\n")
                    f.write(f" Values: {vals}\n")
                    f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_val}, Std Dev: {std}\n")
                    f.write(f" Perfect answers (100): {perfect}\n")
                    # Plot
                    plot_boxplot(
                        pd.DataFrame({f"{tool.capitalize()} (all tasks)": combined_vals}),
                        [f"{tool.capitalize()} (all tasks)"],
                        output_file=f"plots/task-performance/{tool}_alltasks_performance.png",
                        tile=f"Performance Distribution for {tool.capitalize()} (all tasks, all questions)",
                        x_label="Tool",
                        y_label="Performance (%)",
                        y_range=(-0.5, 100.5),
                        tight=True
                    )
            # --- Task-independent, per-tool evaluation over all relevant columns (combined plot) ---
            demaf_vals = get_task_independent_tool_values(df, "demaf")
            monokle_vals = get_task_independent_tool_values(df, "monokle")

            if not demaf_vals.empty or not monokle_vals.empty:
                # Pad to same length for plotting
                max_len = max(len(demaf_vals), len(monokle_vals))
                demaf_list = demaf_vals.tolist() + [np.nan] * (max_len - len(demaf_vals))
                monokle_list = monokle_vals.tolist() + [np.nan] * (max_len - len(monokle_vals))
                plot_df = pd.DataFrame({
                    "DeMAF (all tasks)": demaf_list,
                    "Monokle (all tasks)": monokle_list
                })
                plot_boxplot(
                    plot_df,
                    ["DeMAF (all tasks)", "Monokle (all tasks)"],
                    output_file="plots/task-performance/task_independent_demaf_monokle.png",
                    tile="Task-independent Performance by Tool (all questions, all tasks)",
                    x_label="Tool",
                    y_label="Performance (%)",
                    y_range=(-0.5, 100.5),
                    tight=True
                )
            # --- Aggregated performance per tool, combined over both tasks ---
            def get_agg_perf_tool(df, tool):
                vals_t1 = pd.to_numeric(df.loc[df["T1_UQ_1"].str.lower() == tool, "T1_perf_agg"], errors="coerce").dropna() if "T1_perf_agg" in df.columns else pd.Series(dtype=float)
                vals_t2 = pd.to_numeric(df.loc[df["T2_UQ_1"].str.lower() == tool, "T2_perf_agg"], errors="coerce").dropna() if "T2_perf_agg" in df.columns else pd.Series(dtype=float)
                return pd.concat([vals_t1, vals_t2], ignore_index=True)

            demaf_agg_both = get_agg_perf_tool(df, "demaf")
            monokle_agg_both = get_agg_perf_tool(df, "monokle")

            # Write stats to file
            if not demaf_agg_both.empty or not monokle_agg_both.empty:
                vals, median, mean, mode_val, std = compute_stats(demaf_agg_both)
                perfect = sum(np.isclose(demaf_agg_both, 100))
                f.write(f"\n=== Aggregated Normalized Performance for DeMAF (both tasks) ===\n")
                f.write(f" Values: {vals}\n")
                f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_val}, Std Dev: {std}\n")
                f.write(f" Perfect answers (100): {perfect}\n")

                vals, median, mean, mode_val, std = compute_stats(monokle_agg_both)
                perfect = sum(np.isclose(monokle_agg_both, 100))
                f.write(f"\n=== Aggregated Normalized Performance for Monokle (both tasks) ===\n")
                f.write(f" Values: {vals}\n")
                f.write(f" Median: {median}, Mean: {mean}, Mode: {mode_val}, Std Dev: {std}\n")
                f.write(f" Perfect answers (100): {perfect}\n")

                # Plot boxplot for both
                max_len = max(len(demaf_agg_both), len(monokle_agg_both))
                demaf_list = demaf_agg_both.tolist() + [np.nan] * (max_len - len(demaf_agg_both))
                monokle_list = monokle_agg_both.tolist() + [np.nan] * (max_len - len(monokle_agg_both))
                plot_df = pd.DataFrame({
                    "DeMAF": demaf_list,
                    "Monokle": monokle_list
                })
                plot_boxplot(
                    plot_df,
                    ["DeMAF", "Monokle"],
                    output_file="plots/task-performance/agg_perf_both_tasks_demaf_monokle.png",
                    tile="Aggregated Normalized Performance by Tool (both tasks)",
                    x_label="Tool",
                    y_label="Performance (%)",
                    y_range=(-0.5, 100.5),
                    tight=True
                )
    return df


def compute_and_plot_kq_stats(df, header_mapping, output_file='eval/knowledge_stats.txt', plot_output_folder='plots/knowledge'):
    """
    Analyze and plot data for columns KQ_1 to KQ_12.
    Write statistical metrics to a file and generate histograms.
    Maps None or NaN values to "No answer" and includes original column names in the output.
    """
    # Define the columns to analyze
    kq_columns = [f"KQ_{i}" for i in range(1, 13)]

    # Create a mapping of new headers to original headers
    mapping_dict = dict(zip(header_mapping['new_header'], header_mapping['original_header']))

    # Prepare the output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open the output file for writing
    with open(output_file, "w") as f:
        f.write("KQ Column Statistics:\n")

        # Iterate over each column
        for col in kq_columns:
            if col in df.columns and col != "KQ_4":
                # Get the original column name
                original_col_name = mapping_dict.get(col, col)

                # Flatten the data: extract individual answers from lists
                flattened_data = []
                for entry in df[col]:
                    if isinstance(entry, list) or isinstance(entry, np.ndarray):
                        flattened_data.extend(entry)  # Add all elements in the list/array
                    elif pd.isna(entry) or entry is None:
                        #flattened_data.append("No answer")  # Map None/NaN to "No answer"
                        continue
                    else:
                        flattened_data.append(str(entry))  # Convert other scalar values to strings

                # Convert to a pandas Series for easier processing
                flattened_series = pd.Series(flattened_data)

                # Get histogram data
                hist_data = flattened_series.value_counts().sort_index()
                f.write(f"\nStatistics for {original_col_name}:\n")
                f.write(f"Values: {[str(x) for x in hist_data.index.tolist()]}\n")
                f.write(f"Counts: {hist_data.values.tolist()}\n")

                # Create a histogram plot
                plot_histogram(
                    flattened_series,
                    bins=len(hist_data),
                    output_file=f"{plot_output_folder}/{col}_histogram.png",
                    title=f"Histogram for \"{original_col_name}\"",
                    x_label="",
                    y_label="",
                    swap_axes=True,
                )
            elif col == "KQ_4" and col in df.columns:
                # Get the original column name
                original_col_name = mapping_dict.get(col, col)

                # Convert the column to a pandas Series for easier processing
                column_series = pd.to_numeric(df[col], errors='coerce')

                # Compute statistics
                mean_val = column_series.mean()
                median_val = column_series.median()
                std_dev = column_series.std()
                try:
                    mode_val = mode(column_series.dropna())
                except StatisticsError:
                    mode_val = "No unique mode"

                # Write statistics to the file
                f.write(f"\nStatistics for {original_col_name} (Boxplot):\n")
                f.write(f"Values: {column_series.dropna().tolist()}\n")
                f.write(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std Dev: {std_dev:.2f}, Mode: {mode_val}\n")

                # Create a boxplot
                plot_boxplot(
                    pd.DataFrame({original_col_name: column_series}),
                    [original_col_name],
                    output_file=f"{plot_output_folder}/{col}_boxplot.png",
                    tile=f"Boxplot for \"{original_col_name}\"",
                    x_label="",
                    y_label="Values",
                    y_range=(0.5, 5.5)
                )
            else:
                f.write(f"{col}: Column not found in the DataFrame.\n\n")

        # --- Grouped barchart for KQ_6 to KQ_10 ---
        barchart_cols = [f"KQ_{i}" for i in range(6, 11)]
        # Collect all unique answer options across these columns
        all_options = set()
        for col in barchart_cols:
            if col in df.columns:
                for entry in df[col]:
                    if isinstance(entry, (list, np.ndarray)):
                        all_options.update(str(v) for v in entry if not pd.isna(v))
                    elif not pd.isna(entry):
                        all_options.add(str(entry))
        all_options = sorted(all_options)
        # Prepare counts for each column and option
        counts_dict = {col: {opt: 0 for opt in all_options} for col in barchart_cols}
        for col in barchart_cols:
            if col in df.columns:
                for entry in df[col]:
                    if isinstance(entry, (list, np.ndarray)):
                        for v in entry:
                            if not pd.isna(v):
                                counts_dict[col][str(v)] += 1
                    elif not pd.isna(entry):
                        counts_dict[col][str(entry)] += 1
        # Build DataFrame for grouped barchart: columns are KQ_6...KQ_10, rows are options
        barchart_df = pd.DataFrame(counts_dict).T[all_options]
        # Transpose so each group is a KQ column, each subgroup is an answer option
        plot_grouped_barchart(
            barchart_df.T,  # transpose so columns are answer options
            group_labels=barchart_cols,  # KQ_6 to KQ_10
            subgroup_labels=all_options,  # answer options
            output_file=f"{plot_output_folder}/grouped_barchart_KQ6_KQ10.png",
            title="Grouped Barchart for KQ_6 to KQ_10",
            x_label="Knowledge Question",
            y_label="Count",
            tight=True
        )

def plot_knowledge_terms_grouped_barchart(df, mapping_df, output_file='plots/knowledge/grouped_barchart_terms.png'):
    """
    Plots a grouped bar chart for all columns that originally started with
    'Do you know the following terms? [term]', using the mapping_df to get the original names.
    Each term is a group, and bars are for 'Yes', 'Unsure', 'No' responses.
    """
    # Build mapping from mapped col name to original col name
    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header']))

    # Find all mapped columns whose original name starts with the desired prefix
    term_cols = [col for col in df.columns if col in mapping_dict and mapping_dict[col].startswith("Do you know the following terms? [")]
    term_labels = [re.search(r"\[(.*?)\]", mapping_dict[col]).group(1) if re.search(r"\[(.*?)\]", mapping_dict[col]) else mapping_dict[col] for col in term_cols]

    # Possible answers (ensure consistent order)
    answer_options = ["Yes", "Unsure", "No"]

    # Prepare data for grouped barchart: columns are [term1_Yes, term1_Unsure, term1_No, term2_Yes, ...]
    plot_data = {}
    for term_col, term_label in zip(term_cols, term_labels):
        counts = {ans: 0 for ans in answer_options}
        for val in df[term_col]:
            val_str = str(val).strip().capitalize()
            if val_str in answer_options:
                counts[val_str] += 1
            elif val_str.lower() == "unsure":
                counts["Unsure"] += 1
            elif val_str.lower() == "no":
                counts["No"] += 1
            elif val_str.lower() == "yes":
                counts["Yes"] += 1
        for ans in answer_options:
            plot_data[f"{term_label}_{ans}"] = [counts[ans]]

    # Build DataFrame: each column is term_answer, one row (counts)
    plot_df = pd.DataFrame(plot_data)

    # Reshape for grouped barchart: columns in order [term1_Yes, term1_Unsure, term1_No, term2_Yes, ...]
    ordered_columns = []
    for term_label in term_labels:
        for ans in answer_options:
            ordered_columns.append(f"{term_label}_{ans}")

    # group_labels: term names, subgroup_labels: answer options
    group_labels = term_labels
    subgroup_labels = answer_options

    xtick_labels = ["TADM", "TOSCA", "EDMM", "DeMAF", "TSDM"]

    # Repeat counts to have at least 1 row per group (for compatibility with your function)
    plot_grouped_barchart(
        plot_df[ordered_columns],
        group_labels=xtick_labels,
        subgroup_labels=subgroup_labels,
        output_file=output_file,
        title="Histogram of Terminology Knowledge",
        x_label=None,
        y_label="Count",
        tight=True,
        subgroupname="Answer"
    )

def compute_and_plot_dontknow_counts(df, output_file=r'eval/dontknow_stats.txt', plot_output_folder=r'plots/dont-know'):
    """
    For each participant, counts how often they answered with "don't know", "nan", or left the answer empty.
    Also, for each question column (for both tasks), computes the count distribution across participants.
    Results are written to a text file and box plots are generated.
    """
    tasks = ["T1", "T2"]
    participant_counts = []
    per_question_stats = {}

    def is_dontknow(val):
        if pd.isna(val):
            return True
        if isinstance(val, str):
            v = val.strip().lower()
            return v in ["don't know", "dont know", "", "nan"]
        return False

    for index, row in df.iterrows():
        total_count = 0
        for task in tasks:
            q_range = range(2, 14) if task == "T1" else range(2, 12)
            for i in q_range:
                col = f"{task}_UQ_{i}"
                count = 0
                ans = row[col]
                # Try to parse string representations of lists
                if isinstance(ans, str) and ans.startswith("[") and ans.endswith("]"):
                    try:
                        parsed = ast.literal_eval(ans)
                        if isinstance(parsed, (list, tuple)):
                            ans = parsed
                    except Exception:
                        pass
                # Now handle list/array
                if isinstance(ans, (list, np.ndarray)):
                    for item in ans:
                        if is_dontknow(item):
                            count += 1
                else:
                    if is_dontknow(ans):
                        count = 1
                total_count += count
                per_question_stats.setdefault(col, []).append(count)
        df.loc[index, "total_dontknow"] = total_count
        participant_counts.append(total_count)

    # Write aggregated statistics to file.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("Participant 'don't know' counts:\n")
        for idx, cnt in enumerate(participant_counts):
            uuid = df.iloc[idx]["UUID"] if "UUID" in df.columns else "N/A"
            f.write(f"Participant {idx +1} (UUID: {uuid}): {cnt}\n")
        f.write("\nPer question 'don\'t know' statistics:\n")
        for col, counts in per_question_stats.items():
            arr = np.array(counts)
            med = round(np.median(arr), 2) if arr.size else np.nan
            mean_val = round(np.mean(arr), 2) if arr.size else np.nan
            f.write(f"{col}: counts = {counts}, median = {med}, mean = {mean_val}\n")

    # Plot per-participant total counts. Separate by tool based on T1_UQ_1.
    overall = df["total_dontknow"].dropna().tolist()
    condition_demaf = df["T1_UQ_1"].str.lower() == "demaf"
    condition_monokle = df["T1_UQ_1"].str.lower() == "monokle"
    demaf_counts = df.loc[condition_demaf, "total_dontknow"].dropna().tolist()
    monokle_counts = df.loc[condition_monokle, "total_dontknow"].dropna().tolist()
    # Create DataFrame for plotting.
    max_length = max(len(overall), len(demaf_counts), len(monokle_counts))
    overall += [np.nan] * (max_length - len(overall))
    demaf_counts += [np.nan] * (max_length - len(demaf_counts))
    monokle_counts += [np.nan] * (max_length - len(monokle_counts))
    plot_df = pd.DataFrame({
        "Both": overall,
        "DeMAF": demaf_counts,
        "Monokle": monokle_counts
    })
    os.makedirs(os.path.dirname(plot_output_folder + "/participant_dontknow.png"), exist_ok=True)
    plot_boxplot(
        plot_df,
        ["Both", "DeMAF", "Monokle"],
        output_file=plot_output_folder + "/participant_dontknow.png",
        tile="Participant 'don\'t know' Counts",
        x_label="Category",
        y_label="Count"
    )

    # --- Grouped boxplot for T1 and T2 don't know counts (overall, DeMAF, Monokle) ---

    t1_overall = df["T1_dont-know_agg"].dropna().tolist() if "T1_dont-know_agg" in df.columns else []
    t1_demaf = df.loc[condition_demaf, "T1_dont-know_agg"].dropna().tolist() if "T1_dont-know_agg" in df.columns else []
    t1_monokle = df.loc[condition_monokle, "T1_dont-know_agg"].dropna().tolist() if "T1_dont-know_agg" in df.columns else []
    t2_overall = df["T2_dont-know_agg"].dropna().tolist() if "T2_dont-know_agg" in df.columns else []
    t2_demaf = df.loc[condition_demaf, "T2_dont-know_agg"].dropna().tolist() if "T2_dont-know_agg" in df.columns else []
    t2_monokle = df.loc[condition_monokle, "T2_dont-know_agg"].dropna().tolist() if "T2_dont-know_agg" in df.columns else []

    max_len = max(len(t1_overall), len(t1_demaf), len(t1_monokle), len(t2_overall), len(t2_demaf), len(t2_monokle), 1)
    t1_overall += [np.nan] * (max_len - len(t1_overall))
    t1_demaf += [np.nan] * (max_len - len(t1_demaf))
    t1_monokle += [np.nan] * (max_len - len(t1_monokle))
    t2_overall += [np.nan] * (max_len - len(t2_overall))
    t2_demaf += [np.nan] * (max_len - len(t2_demaf))
    t2_monokle += [np.nan] * (max_len - len(t2_monokle))

    grouped_df = pd.DataFrame({
        "T1_Both": t1_overall,
        "T1_DeMAF": t1_demaf,
        "T1_Monokle": t1_monokle,
        "T2_Both": t2_overall,
        "T2_DeMAF": t2_demaf,
        "T2_Monokle": t2_monokle
    })

    plot_grouped_boxplot(
        grouped_df[["T1_Both", "T1_DeMAF", "T1_Monokle", "T2_Both", "T2_DeMAF", "T2_Monokle"]],
        group_labels=["Task 1", "Task 2"],
        subgroup_labels=["Both", "DeMAF", "Monokle"],
        output_file=plot_output_folder + "/grouped_dontknow_counts.png",
        title="Don't Know Counts by Task and Tool",
        x_label="Task",
        y_label="Don't Know Count",
        tight=True
    )

    # Plot per question box plots.
    for task in tasks:
        q_range = range(2, 14) if task == "T1" else range(2, 12)
        for i in q_range:
            col = f"{task}_UQ_{i}"
            counts = per_question_stats.get(col, [])
            df_counts = pd.DataFrame({"Count": counts})
            output_path = f"{plot_output_folder}/{col}_dontknow_boxplot.png"
            plot_boxplot(
                df_counts,
                ["Count"],
                output_file=output_path,
                tile=f"{col} 'don\'t know' Counts",
                x_label="",
                y_label="Count"
            )
    return df


def evaluate_relative_performance(df, mapping_df=None, output_file=r'eval/performance_comparison.txt', plot_output_folder=r'plots/performance-comparison'):
    """
    Compares the performance of DeMAF and Monokle per task (T1 and T2).
    For each performance column, determines which tool performed better on average.
    Writes the evaluation to a file and creates boxplots for each question and aggregated performance.
     Optionally uses mapping_df to convert mapped column names to original names.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(plot_output_folder, exist_ok=True)

    # Prepare mapping dictionary if provided
    mapping_dict = None
    if mapping_df is not None:
        mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header']))

    tasks = ["T1", "T2"]
    with open(output_file, "w") as f:
        for task in tasks:
            perf_cols = [col for col in df.columns if col.startswith(f"{task}_UQ_") and col.endswith("_perf")]
            f.write(f"\n=== {task} Performance Comparison ===\n")
            for col in perf_cols:
                # Remove '_perf' to get the base column name for mapping
                base_col = col[:-5] if col.endswith('_perf') else col
                original_col_name = mapping_dict.get(base_col, col) if mapping_dict else col

                condition_demaf = df[f"{task}_UQ_1"] == "demaf"
                condition_monokle = df[f"{task}_UQ_1"] == "monokle"
                demaf_perf = pd.to_numeric(df.loc[condition_demaf, col], errors="coerce").dropna()
                monokle_perf = pd.to_numeric(df.loc[condition_monokle, col], errors="coerce").dropna()
                demaf_mean = demaf_perf.mean() if not demaf_perf.empty else float('nan')
                monokle_mean = monokle_perf.mean() if not monokle_perf.empty else float('nan')

                if np.isnan(demaf_mean) and np.isnan(monokle_mean):
                    verdict = "No data for either tool."
                elif np.isnan(demaf_mean):
                    verdict = "Only Monokle has data."
                elif np.isnan(monokle_mean):
                    verdict = "Only DeMAF has data."
                elif demaf_mean > monokle_mean:
                    verdict = f"DeMAF performed better by {demaf_mean - monokle_mean:.2f} percentage points."
                elif monokle_mean > demaf_mean:
                    verdict = f"Monokle performed better by {monokle_mean - demaf_mean:.2f} percentage points."
                else:
                    verdict = "Both tools performed equally."

                f.write(f"\n{col} - \"{original_col_name}\":\n")
                f.write(f"  DeMAF mean: {demaf_mean:.2f}\n")
                f.write(f"  Monokle mean: {monokle_mean:.2f}\n")
                f.write(f"  Verdict: {verdict}\n")

                # Plot boxplot for this question
                max_length = max(len(demaf_perf), len(monokle_perf))
                demaf_list = demaf_perf.tolist() + [np.nan] * (max_length - len(demaf_perf))
                monokle_list = monokle_perf.tolist() + [np.nan] * (max_length - len(monokle_perf))
                plot_df = pd.DataFrame({
                    "DeMAF": demaf_list,
                    "Monokle": monokle_list
                })
                plot_boxplot(
                    plot_df,
                    ["DeMAF", "Monokle"],
                    output_file=f"{plot_output_folder}/{col}_comparison.png",
                    tile=f"Performance Comparison for \"{original_col_name}\"",
                    x_label="Tool",
                    y_label="Performance (%)",
                    y_range=(-0.5, 100.5)
                )

            # --- Collect performance differences for ranking ---
            perf_diffs = []
            for col in perf_cols:
                base_col = col[:-5] if col.endswith('_perf') else col
                original_col_name = mapping_dict.get(base_col, col) if mapping_dict else col

                condition_demaf = df[f"{task}_UQ_1"] == "demaf"
                condition_monokle = df[f"{task}_UQ_1"] == "monokle"
                demaf_perf = pd.to_numeric(df.loc[condition_demaf, col], errors="coerce").dropna()
                monokle_perf = pd.to_numeric(df.loc[condition_monokle, col], errors="coerce").dropna()
                demaf_mean = demaf_perf.mean() if not demaf_perf.empty else float('nan')
                monokle_mean = monokle_perf.mean() if not monokle_perf.empty else float('nan')

                if not np.isnan(demaf_mean) and not np.isnan(monokle_mean):
                    diff = demaf_mean - monokle_mean
                    perf_diffs.append((diff, original_col_name))

            # Sort and select top 2 for each tool
            demaf_better = sorted([x for x in perf_diffs if x[0] > 0], key=lambda x: -x[0])[:2]
            monokle_better = sorted([x for x in perf_diffs if x[0] < 0], key=lambda x: x[0])[:2]

            f.write(f"\nTop 2 questions where DeMAF performed best for {task}:\n")
            for diff, colname in demaf_better:
                f.write(f' "{colname}": DeMAF better by {diff:.2f} percentage points\n')

            f.write(f"\nTop 2 questions where Monokle performed best for {task}:\n")
            for diff, colname in monokle_better:
                f.write(f' "{colname}": Monokle better by {abs(diff):.2f} percentage points\n')

            # Aggregated, Normalized performance
            agg_col = f"{task}_perf_agg"
            if agg_col in df.columns:
                original_agg_col_name = mapping_dict.get(agg_col, agg_col) if mapping_dict else agg_col

                condition_demaf = df[f"{task}_UQ_1"] == "demaf"
                condition_monokle = df[f"{task}_UQ_1"] == "monokle"
                demaf_agg = pd.to_numeric(df.loc[condition_demaf, agg_col], errors="coerce").dropna()
                monokle_agg = pd.to_numeric(df.loc[condition_monokle, agg_col], errors="coerce").dropna()
                demaf_mean = demaf_agg.mean() if not demaf_agg.empty else float('nan')
                monokle_mean = monokle_agg.mean() if not monokle_agg.empty else float('nan')

                if np.isnan(demaf_mean) and np.isnan(monokle_mean):
                    verdict = "No data for either tool."
                elif np.isnan(demaf_mean):
                    verdict = "Only Monokle has data."
                elif np.isnan(monokle_mean):
                    verdict = "Only DeMAF has data."
                elif demaf_mean > monokle_mean:
                    verdict = f"DeMAF performed better by {demaf_mean - monokle_mean:.2f} percentage points."
                elif monokle_mean > demaf_mean:
                    verdict = f"Monokle performed better by {monokle_mean - demaf_mean:.2f} percentage points."
                else:
                    verdict = "Both tools performed equally."

                f.write(f"\n{original_agg_col_name} (Aggregated, Normalized):\n")
                f.write(f"  DeMAF mean: {demaf_mean:.2f}\n")
                f.write(f"  Monokle mean: {monokle_mean:.2f}\n")
                f.write(f"  Verdict: {verdict}\n")

                # Plot aggregated boxplot
                max_length = max(len(demaf_agg), len(monokle_agg))
                demaf_list = demaf_agg.tolist() + [np.nan] * (max_length - len(demaf_agg))
                monokle_list = monokle_agg.tolist() + [np.nan] * (max_length - len(monokle_agg))
                plot_df = pd.DataFrame({
                    "DeMAF": demaf_list,
                    "Monokle": monokle_list
                })
                plot_boxplot(
                    plot_df,
                    ["DeMAF", "Monokle"],
                    output_file=f"{plot_output_folder}/{agg_col}_agg_comparison.png",
                    tile=f"Aggregated, Normalized Performance Comparison for {original_agg_col_name}",
                    x_label="Tool",
                    y_label="Performance (%)",
                    y_range=(-0.5, 100.5)
                )

def comparison_questions_eval(df, header_mapping=None, output_file="eval/comparison_questions_stats.txt", plot_output_folder="plots/comparison-questions"):
    """
    Evaluates and plots the data for columns CQ_1 to CQ_6.
    For CQ_1 to CQ_5 (Likert scale): computes mean, median, std, mode, writes to file, and creates boxplots.
    For CQ_6 (string answers): counts unique answers, writes to file, and creates a histogram.
    Optionally uses header_mapping to convert mapped column names to original/question text names.
    Additionally, creates a combined boxplot for all Likert columns using their column names.
    """
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Prepare mapping dictionary if provided
    mapping_dict = None
    if header_mapping is not None:
        mapping_dict = dict(zip(header_mapping['new_header'], header_mapping['original_header']))

    likert_cols = [f"CQ_{i}" for i in range(1, 6)]
    text_col = "CQ_6"

    with open(output_file, "w") as f:
        f.write("Comparison Questions Evaluation (CQ_1 to CQ_6):\n")

        # Likert scale questions
        likert_data = {}
        for col in likert_cols:
            display_name = f'"{mapping_dict[col]}"' if mapping_dict and col in mapping_dict else col
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                mean_val = round(series.mean(), 2) if not series.empty else np.nan
                median_val = round(series.median(), 2) if not series.empty else np.nan
                std_val = round(series.std(ddof=1), 2) if len(series) > 1 else np.nan
                try:
                    mode_val = int(mode(series)) if not series.empty else np.nan
                except StatisticsError:
                    mode_val = "No unique mode"

                f.write(f"\nStatistics for {display_name}:\n")
                f.write(f"  Values: {series.tolist()}\n")
                f.write(f"  Mean: {mean_val}, Median: {median_val}, Std Dev: {std_val}, Mode: {mode_val}\n")

                # Boxplot for individual question
                plot_boxplot(
                    pd.DataFrame({display_name: series}),
                    [display_name],
                    output_file=f"{plot_output_folder}/{col}_boxplot.png",
                    tile=f"Boxplot for {display_name}",
                    x_label="",
                    y_label="Likert Scale Value",
                    y_range=(0.5, 5.5)
                )
                # Collect for combined plot (use col name as key)
                likert_data[col] = series.reset_index(drop=True)
            else:
                f.write(f"\n{display_name}: Column not found in DataFrame.\n")

        # Combined boxplot for all Likert columns using their column names
        if likert_data:
            max_len = max(len(s) for s in likert_data.values())
            for k in likert_data:
                likert_data[k] = likert_data[k].tolist() + [np.nan] * (max_len - len(likert_data[k]))
            combined_df = pd.DataFrame(likert_data)
            plot_boxplot(
                combined_df,
                list(likert_data.keys()),
                output_file=f"{plot_output_folder}/all_likert_boxplot.png",
                tile="Comparison Questions Likert Scale Overview",
                x_label="Question",
                y_label="Preference (1-DeMAF, 5-Monokle)",
                y_range=(0.5, 5.5),
                size=(10, 5)
            )

        # Text question
        display_name = f'"{mapping_dict[text_col]}"' if mapping_dict and text_col in mapping_dict else text_col
        if text_col in df.columns:
            answers = df[text_col].dropna().astype(str)
            answers = answers.replace("other tool", "monokle")  # Replace before counting
            counts = answers.value_counts()
            f.write(f"\nAnswer counts for {display_name}:\n")
            for answer, count in counts.items():
                f.write(f"  \"{answer}\": {count}\n")

            # Histogram
            plot_histogram(
                answers,
                bins=len(counts),
                output_file=f"{plot_output_folder}/{text_col}_histogram.png",
                title=f"Histogram for {display_name}",
                x_label="Count",
                y_label="Answers",
                swap_axes=True,
                figsize=(21, 5),
                text_size=20
            )
        else:
            f.write(f"\n{display_name}: Column not found in DataFrame.\n")


def performance_stats_by_knowledge(
    df, mapping_df=None,
    output_file="eval/perf_by_knowledge.txt",
    plot_output_folder="plots/perf_by_knowledge"
):
    """
    For each task and each performance column, computes mean, median, std, and mode of performance
    split by all values of the knowledge column (KQ_4), for Both, DeMAF, and Monokle.
    Writes results to a file and creates plots for all of them.
    At the end of each task, also evaluates and plots the aggregated performance column.
    Additionally, creates a grouped boxplot for each task (like time_stats_by_knowledge).
    """
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header'])) if mapping_df is not None else {}

    tasks = ["T1", "T2"]
    kq_col = "KQ_4"
    mapped_kq_col = mapping_dict.get(kq_col, kq_col)
    kq_values = list(range(1, 6))

    with open(output_file, "w") as f:
        for task in tasks:
            f.write(f"\n=== {task} Performance by Knowledge ({mapped_kq_col}) ===\n")
            perf_cols = [col for col in df.columns if col.startswith(f"{task}_UQ_") and col.endswith("_perf") and "agg" not in col]
            condition_demaf = df[f"{task}_UQ_1"].str.lower() == "demaf"
            condition_monokle = df[f"{task}_UQ_1"].str.lower() == "monokle"

            # --- Grouped boxplot data collection ---
            group_labels = [str(kq) for kq in kq_values]
            subgroup_labels = ["Both", "DeMAF", "Monokle"]
            plot_data = {}

            for kq in kq_values:
                # Select participants at this knowledge level
                mask_kq = df[kq_col] == kq
                # For each participant, compute their mean performance across all perf_cols
                overall_vals = df.loc[mask_kq, perf_cols].mean(axis=1, skipna=True).tolist()
                demaf_vals = df.loc[mask_kq & condition_demaf, perf_cols].mean(axis=1, skipna=True).tolist()
                monokle_vals = df.loc[mask_kq & condition_monokle, perf_cols].mean(axis=1, skipna=True).tolist()
                # Pad to same length for DataFrame
                max_len = max(len(overall_vals), len(demaf_vals), len(monokle_vals), 1)
                overall_vals += [np.nan] * (max_len - len(overall_vals))
                demaf_vals += [np.nan] * (max_len - len(demaf_vals))
                monokle_vals += [np.nan] * (max_len - len(monokle_vals))
                plot_data[f"{kq}_Both"] = overall_vals
                plot_data[f"{kq}_DeMAF"] = demaf_vals
                plot_data[f"{kq}_Monokle"] = monokle_vals

            # After filling plot_data for all kq:
            max_len = max(len(lst) for lst in plot_data.values())
            for key in plot_data:
                plot_data[key] += [np.nan] * (max_len - len(plot_data[key]))
            plot_df = pd.DataFrame(plot_data)

            # The columns must be in the order [1_Both, 1_DeMAF, 1_Monokle, 2_Both, ...]
            ordered_columns = []
            for kq in group_labels:
                for subgroup in subgroup_labels:
                    ordered_columns.append(f"{kq}_{subgroup}")

            # Build custom x-tick labels with counts
            xtick_labels = []
            for kq in group_labels:
                counts = []
                for subgroup in subgroup_labels:
                    col = f"{kq}_{subgroup}"
                    count = plot_df[col].notna().sum()
                    counts.append(str(count))
                xtick_labels.append(f"{kq}\nn=[{','.join(counts)}]")

            plot_grouped_boxplot(
                plot_df[ordered_columns],
                group_labels=xtick_labels,
                subgroup_labels=subgroup_labels,
                output_file=f"{plot_output_folder}/{task}_grouped_perf_by_knowledge.png",
                title=f"{task} Performance by Knowledge Level and Tool",
                x_label=f"{mapped_kq_col} (Knowledge)",
                y_label="Performance (%)",
                y_range=(-0.5, 100.5),
                tight=True
            )

            # --- Per-question and aggregated boxplots ---
            for col in perf_cols:
                base_col = col[:-5] if col.endswith('_perf') else col
                mapped_col = mapping_dict.get(base_col, col)
                f.write(f"\nStatistics for {mapped_col} split by {mapped_kq_col}:\n")
                for group_name, group_mask in [
                    ("Both", pd.Series([True] * len(df))),
                    ("DeMAF", condition_demaf),
                    ("Monokle", condition_monokle)
                ]:
                    means, medians, stds, modes, counts = [], [], [], [], []
                    for kq in kq_values:
                        vals = pd.to_numeric(df.loc[(df[kq_col] == kq) & group_mask, col], errors="coerce").dropna()
                        count = len(vals)
                        if count == 0:
                            means.append(np.nan)
                            medians.append(np.nan)
                            stds.append(np.nan)
                            modes.append(np.nan)
                            counts.append(0)
                            continue
                        mean = np.mean(vals)
                        median = np.median(vals)
                        std = np.std(vals, ddof=1) if count > 1 else np.nan
                        try:
                            mode_val = mode(vals)
                        except StatisticsError:
                            mode_val = np.nan
                        means.append(mean)
                        medians.append(median)
                        stds.append(std)
                        modes.append(mode_val)
                        counts.append(count)
                        f.write(f"  {group_name} - Knowledge={kq}: N={count}, mean={mean:.2f}, median={median:.2f}, std={std:.2f}, mode={mode_val}\n")

                    # Prepare data for boxplot (always 1-5)
                    labels = [f"{kq} (n={counts[i]})" for i, kq in enumerate(kq_values)]
                    data = [
                        pd.to_numeric(df.loc[(df[kq_col] == kq) & group_mask, col], errors="coerce").dropna().tolist()
                        for kq in kq_values
                    ]
                    # Pad lists to the same length
                    max_len = max(len(lst) for lst in data) if data else 1
                    data = [list(lst) + [np.nan] * (max_len - len(lst)) for lst in data]
                    plot_df = pd.DataFrame({label: vals for label, vals in zip(labels, data)})
                    plot_boxplot(
                        plot_df,
                        plot_df.columns.tolist(),
                        output_file=f"{plot_output_folder}/{col}_{group_name}_by_knowledge.png",
                        tile=f"{mapped_col} by knowledge [{group_name}]",
                        x_label=f"{mapped_kq_col} (Knowledge)",
                        y_label="Performance (%)",
                        y_range=(-0.5, 100.5),
                        tight=True,
                    )

            # --- Aggregated, Normalized performance for the task ---
            agg_col = f"{task}_perf_agg"
            if agg_col in df.columns:
                base_agg_col = agg_col[:-5] if agg_col.endswith('_perf_agg"') else agg_col
                mapped_agg_col = mapping_dict.get(base_agg_col, agg_col)
                f.write(f"\nStatistics for {mapped_agg_col} (Aggregated, Normalized) split by {mapped_kq_col}:\n")
                for group_name, group_mask in [
                    ("Both", pd.Series([True] * len(df))),
                    ("DeMAF", condition_demaf),
                    ("Monokle", condition_monokle)
                ]:
                    stats_by_kq = []
                    for kq in kq_values:
                        vals = pd.to_numeric(df.loc[(df[kq_col] == kq) & group_mask, agg_col], errors="coerce").dropna()
                        count = len(vals)
                        if count == 0:
                            stats_by_kq.append((kq, [], 0, np.nan, np.nan, np.nan, np.nan))
                            continue
                        mean = np.mean(vals)
                        median = np.median(vals)
                        std = np.std(vals, ddof=1) if count > 1 else np.nan
                        try:
                            mode_val = mode(vals)
                        except StatisticsError:
                            mode_val = np.nan
                        stats_by_kq.append((kq, vals, count, mean, median, std, mode_val))
                        f.write(f"  {group_name} - Knowledge={kq}: N={count}, mean={mean:.2f}, median={median:.2f}, std={std:.2f}, mode={mode_val}\n")

                    # Boxplot for this group using plot_boxplot (always 1-5)
                    labels = [f"{kq} (n={stats_by_kq[i][2]})" for i, kq in enumerate(kq_values)]
                    data = [stats_by_kq[i][1] for i in range(len(kq_values))]
                    max_len = max(len(lst) for lst in data) if data else 1
                    data = [list(lst) + [np.nan] * (max_len - len(lst)) for lst in data]
                    plot_df = pd.DataFrame({label: vals for label, vals in zip(labels, data)})
                    plot_boxplot(
                        plot_df,
                        plot_df.columns.tolist(),
                        output_file=f"{plot_output_folder}/{agg_col}_{group_name}_by_knowledge.png",
                        tile=f"Task {mapped_agg_col[1:2]} performance by knowledge [{group_name}]",
                        x_label=f"{mapped_kq_col} (knowledge)",
                        y_label="Performance (%)",
                        y_range=(-0.5, 100.5),
                        tight=True,
                    )
                # --- Write summary stats for normalized aggregated performance per knowledge level ---
                f.write(f"\nSummary statistics for {mapped_agg_col} (Aggregated, Normalized) by {mapped_kq_col}:\n")
                for kq in kq_values:
                    vals = pd.to_numeric(df.loc[df[kq_col] == kq, agg_col], errors="coerce").dropna()
                    count = len(vals)
                    if count == 0:
                        mean = median = std = mode_val = np.nan
                    else:
                        mean = np.mean(vals)
                        median = np.median(vals)
                        std = np.std(vals, ddof=1) if count > 1 else np.nan
                        try:
                            mode_val = mode(vals)
                        except StatisticsError:
                            mode_val = np.nan
                    f.write(f"  Knowledge={kq}: N={count}, mean={mean:.2f}, median={median:.2f}, std={std:.2f}, mode={mode_val}\n")


def time_stats_by_knowledge(
    df, mapping_df=None,
    output_file="eval/time_by_knowledge.txt",
    plot_output_folder="plots/time_by_knowledge"
):
    """
    For each task time column, computes count, mean, median, std, and mode of time
    split by all values of the knowledge column (KQ_4), for Both, DeMAF, and Monokle.
    Writes results to a file and creates boxplots for all of them.
    """
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header'])) if mapping_df is not None else {}

    kq_col = "KQ_4"
    mapped_kq_col = mapping_dict.get(kq_col, kq_col)
    kq_values = sorted(df[kq_col].dropna().unique())
    time_cols = ["T1_time", "T2_time"]

    with open(output_file, "w") as f:
        for time_col in time_cols:
            mapped_col = mapping_dict.get(time_col, time_col)
            f.write(f"\nStatistics for {mapped_col} split by {mapped_kq_col}:\n")

            # Prepare data for grouped boxplot
            group_labels = [str(kq) for kq in range(1, 6)]
            subgroup_labels = ["Both", "DeMAF", "Monokle"]
            plot_data = {}

            for kq in range(1, 6):
                # DeMAF
                vals_demaf = pd.to_numeric(
                    df.loc[(df[kq_col] == kq) & (df["T1_UQ_1"].str.lower() == "demaf" if time_col == "T1_time" else df["T2_UQ_1"].str.lower() == "demaf"), time_col],
                    errors="coerce"
                ).dropna().tolist()
                # Monokle
                vals_monokle = pd.to_numeric(
                    df.loc[(df[kq_col] == kq) & (df["T1_UQ_1"].str.lower() == "monokle" if time_col == "T1_time" else df["T2_UQ_1"].str.lower() == "monokle"), time_col],
                    errors="coerce"
                ).dropna().tolist()
                # Both (all tools)
                vals_overall = pd.to_numeric(
                    df.loc[df[kq_col] == kq, time_col],
                    errors="coerce"
                ).dropna().tolist()

                # Write stats for this knowledge level
                for label, vals in zip(subgroup_labels, [vals_overall, vals_demaf, vals_monokle]):
                    count = len(vals)
                    mean = np.mean(vals) if count > 0 else np.nan
                    median = np.median(vals) if count > 0 else np.nan
                    std = np.std(vals, ddof=1) if count > 1 else np.nan
                    try:
                        mode_val = mode(vals)
                    except StatisticsError:
                        mode_val = np.nan
                    fitting = "  " if label == "DeMAF" else ""
                    f.write(f"  {label}{fitting} - Knowledge={kq}: N={count}, mean={mean:.2f}, median={median:.2f}, std={std:.2f}, mode={mode_val}\n")

                # Pad to same length for DataFrame
                max_len = max(len(vals_demaf), len(vals_monokle), len(vals_overall), 1)
                vals_demaf += [np.nan] * (max_len - len(vals_demaf))
                vals_monokle += [np.nan] * (max_len - len(vals_monokle))
                vals_overall += [np.nan] * (max_len - len(vals_overall))

                plot_data[f"{kq}_Both"] = vals_overall
                plot_data[f"{kq}_DeMAF"] = vals_demaf
                plot_data[f"{kq}_Monokle"] = vals_monokle

            # After filling plot_data for all kq:
            max_len = max(len(lst) for lst in plot_data.values())
            for key in plot_data:
                plot_data[key] += [np.nan] * (max_len - len(plot_data[key]))
            plot_df = pd.DataFrame(plot_data)

            # The columns must be in the order [1_Both, 1_DeMAF, 1_Monokle, 2_Both, ...]
            ordered_columns = []
            for kq in group_labels:
                for subgroup in subgroup_labels:
                    ordered_columns.append(f"{kq}_{subgroup}")

            # Build custom x-tick labels with counts
            xtick_labels = []
            for kq in group_labels:
                counts = []
                for subgroup in subgroup_labels:
                    col = f"{kq}_{subgroup}"
                    # Count non-NaN values for this group/subgroup
                    count = plot_df[col].notna().sum()
                    counts.append(str(count))
                xtick_labels.append(f"{kq}\nn=[{','.join(counts)}]")

            plot_grouped_boxplot(
                plot_df[ordered_columns],
                group_labels=xtick_labels,
                subgroup_labels=subgroup_labels,
                output_file=f"{plot_output_folder}/{time_col}_grouped_by_knowledge.png",
                title=f"{mapped_col} by Knowledge Level and Tool",
                x_label=f"{mapped_kq_col} (Knowledge)",
                y_label="Time (seconds)",
                y_range=(600, 2000),
                tight=True
            )

def prefered_tool_by_knowledge(
    df, mapping_df=None,
    output_file="eval/prefered_tool_by_knowledge.txt",
    plot_output_folder="plots/prefered_tool_by_knowledge"
):
    """
    For each value of the knowledge column (KQ_4), computes the count of each favorite tool (CQ_6).
    Writes results to a file and creates stacked bar plots.
    """
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header'])) if mapping_df is not None else {}

    kq_col = "KQ_4"
    tool_col = "CQ_6"
    mapped_kq_col = mapping_dict.get(kq_col, kq_col)

    # Map "other tool" to "monokle"
    df[tool_col] = df[tool_col].replace("other tool", "monokle")

    count_df = df.groupby([kq_col, tool_col]).size().unstack(fill_value=0)

    with open(output_file, "w") as f:
        f.write(f"Favorite Tool distribution by Knowledge ({mapped_kq_col}):\n")
        # Ensure all knowledge levels 1-5 are present, even if count is 0
        for k in range(1, 6):
            if k not in count_df.index:
                count_df.loc[k] = [0] * count_df.shape[1]
        count_df = count_df.sort_index()

         # Write the table to the file
        f.write(count_df.to_string())
        f.write("\n")

        # Plot: grouped bar chart (not stacked)
        ax = count_df.plot(
            kind='bar',
            stacked=False,
            figsize=(10, 6),
            colormap='tab20c'
        )
        plt.title(f"Favorite Tool by Knowledge Level ({mapped_kq_col})")
        plt.xlabel(f"{mapped_kq_col} (Knowledge)")
        plt.ylabel("number of participants")
        plt.legend(title="Favorite Tool")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{plot_output_folder}/favorite_tool_by_knowledge.png", dpi=300)
        plt.close()

def check_comprehension_data_requirements(df):
    """
    Checks data requirements for t-tests on comprehension metrics between DeMAF and Monokle.
    Prints warnings if requirements are not met.
    """
    tasks = ["T1", "T2"]
    metrics = {
        "Performance (Aggregated, Normalized)": "_perf_agg",
        "Don't know count": "_dont-know_agg",
        "Task time": "_time",
        "NASA Likert (Aggregated, Normalized)": "_LQ_agg"
    }

    for task in tasks:
        print(f"\n--- Checking requirements for {task} ---")
        for metric_name, suffix in metrics.items():
            if suffix == "_time":
                col = f"{task}_time"
            else:
                col = f"{task}{suffix}"
            if col not in df.columns:
                print(f"  [!] Column {col} not found, skipping.")
                continue

            demaf = pd.to_numeric(df[df[f"{task}_UQ_1"].str.lower() == "demaf"][col], errors="coerce").dropna()
            monokle = pd.to_numeric(df[df[f"{task}_UQ_1"].str.lower() == "monokle"][col], errors="coerce").dropna()

            print(f"\n  Metric: {metric_name} ({col})")
            print(f"    DeMAF: n={len(demaf)}")
            print(f"    Monokle: n={len(monokle)}")

            # Sample size check
            if len(demaf) < 8 or len(monokle) < 8:
                print("    [!] Warning: Small sample size for t-test (n<8). Consider non-parametric test.")

            # Normality check (Shapiro-Wilk)
            if len(demaf) >= 3:
                stat, p = stats.shapiro(demaf)
                print(f"    DeMAF normality p={p:.3f} ({'OK' if p>0.05 else 'NOT normal'})")
            if len(monokle) >= 3:
                stat, p = stats.shapiro(monokle)
                print(f"    Monokle normality p={p:.3f} ({'OK' if p>0.05 else 'NOT normal'})")

            # Variance check (Levene's test)
            if len(demaf) >= 2 and len(monokle) >= 2:
                stat, p = stats.levene(demaf, monokle)
                print(f"    Variance equality p={p:.3f} ({'Equal' if p>0.05 else 'NOT equal'})")

        # --- Check all Likert scale columns {task}_LQ_1 to {task}_LQ_15 ---
        for i in range(1, 16):
            col = f"{task}_LQ_{i}"
            if col not in df.columns:
                print(f"  [!] Likert column {col} not found, skipping.")
                continue

            demaf = pd.to_numeric(df[df[f"{task}_UQ_1"].str.lower() == "demaf"][col], errors="coerce").dropna()
            monokle = pd.to_numeric(df[df[f"{task}_UQ_1"].str.lower() == "monokle"][col], errors="coerce").dropna()

            print(f"\n  Likert: {col}")
            print(f"    DeMAF: n={len(demaf)}")
            print(f"    Monokle: n={len(monokle)}")

            # Sample size check
            if len(demaf) < 8 or len(monokle) < 8:
                print("    [!] Warning: Small sample size for t-test (n<8). Consider non-parametric test.")

            # Normality check (Shapiro-Wilk)
            if len(demaf) >= 3:
                stat, p = stats.shapiro(demaf)
                print(f"    DeMAF normality p={p:.3f} ({'OK' if p>0.05 else 'NOT normal'})")
            if len(monokle) >= 3:
                stat, p = stats.shapiro(monokle)
                print(f"    Monokle normality p={p:.3f} ({'OK' if p>0.05 else 'NOT normal'})")

            # Variance check (Levene's test)
            if len(demaf) >= 2 and len(monokle) >= 2:
                stat, p = stats.levene(demaf, monokle)
                print(f"    Variance equality p={p:.3f} ({'Equal' if p>0.05 else 'NOT equal'})")




def comprehension_ttest_pipeline(df, metrics=None, group_cols=None, output_file="eval/comprehension_ttest_results.txt"):
    """
    Runs the appropriate test (t-test, Welch's t-test, or Mann-Whitney U) for each metric and task.
    Writes all results to a file and returns a dict of all relevant results.
    Additionally, combines T1 and T2 p-values for each metric using Fisher's method.
    """
    tasks = ["T1", "T2"]
    if metrics is None:
        metrics = {
            "Performance (Aggregated, Normalized)": "_perf_agg",
            "Don't know count": "_dont-know_agg",
            "Task time": "_time",
            "NASA Likert (Aggregated, Normalized)": "_LQ_agg"
        }
    if group_cols is None:
        group_cols = {"T1": "T1_UQ_1", "T2": "T2_UQ_1"}

    all_results = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for task in tasks:
            group_col = group_cols[task]
            f.write(f"\n--- {task} ---\n")
            # Standard metrics
            for metric_name, suffix in metrics.items():
                if suffix == "_time":
                    metric_col = f"{task}_time"
                else:
                    metric_col = f"{task}{suffix}"
                if metric_col not in df.columns:
                    f.write(f"  [!] Column {metric_col} not found, skipping.\n")
                    continue

                demaf = pd.to_numeric(df[df[group_col].str.lower() == "demaf"][metric_col], errors="coerce").dropna()
                monokle = pd.to_numeric(df[df[group_col].str.lower() == "monokle"][metric_col], errors="coerce").dropna()

                # Normality
                demaf_normal = shapiro(demaf)[1] > 0.05 if len(demaf) >= 3 else False
                monokle_normal = shapiro(monokle)[1] > 0.05 if len(monokle) >= 3 else False

                # Variance
                equal_var = levene(demaf, monokle)[1] > 0.05 if len(demaf) >= 2 and len(monokle) >= 2 else False

                # Decide test
                if demaf_normal and monokle_normal:
                    if equal_var:
                        stat, p = ttest_ind(demaf, monokle, equal_var=True)
                        test_used = "t-test"
                    else:
                        stat, p = ttest_ind(demaf, monokle, equal_var=False)
                        test_used = "Welch's t-test"
                else:
                    stat, p = mannwhitneyu(demaf, monokle, alternative="two-sided")
                    test_used = "Mann-Whitney U"

                result = {
                    "test_used": test_used,
                    "p_value": p,
                    "demaf_mean": demaf.mean(),
                    "monokle_mean": monokle.mean(),
                    "demaf_median": np.median(demaf),
                    "monokle_median": np.median(monokle),
                    "demaf_n": len(demaf),
                    "monokle_n": len(monokle),
                    "demaf_normal": demaf_normal,
                    "monokle_normal": monokle_normal,
                    "equal_var": equal_var
                }
                all_results[(task, metric_name)] = result

                # Write results to file
                f.write(f"  {metric_name} ({metric_col}):\n")
                f.write(f"    DeMAF: n={result['demaf_n']}, mean={result['demaf_mean']:.3f}, median={result['demaf_median']:.3f}, normal={result['demaf_normal']}\n")
                f.write(f"    Monokle: n={result['monokle_n']}, mean={result['monokle_mean']:.3f}, median={result['monokle_median']:.3f}, normal={result['monokle_normal']}\n")
                f.write(f"    Variance equal: {result['equal_var']}\n")
                f.write(f"    Test used: {result['test_used']}\n")
                f.write(f"    p-value: {result['p_value']:.5f}\n\n")

            # --- Add Likert scale columns {task}_LQ_1 to {task}_LQ_15 ---
            for i in range(1, 16):
                col = f"{task}_LQ_{i}"
                if col not in df.columns:
                    f.write(f"  [!] Likert column {col} not found, skipping.\n")
                    continue

                demaf = pd.to_numeric(df[df[group_col].str.lower() == "demaf"][col], errors="coerce").dropna()
                monokle = pd.to_numeric(df[df[group_col].str.lower() == "monokle"][col], errors="coerce").dropna()

                # Normality
                demaf_normal = shapiro(demaf)[1] > 0.05 if len(demaf) >= 3 else False
                monokle_normal = shapiro(monokle)[1] > 0.05 if len(monokle) >= 3 else False

                # Variance
                equal_var = levene(demaf, monokle)[1] > 0.05 if len(demaf) >= 2 and len(monokle) >= 2 else False

                # Decide test
                if demaf_normal and monokle_normal:
                    if equal_var:
                        stat, p = ttest_ind(demaf, monokle, equal_var=True)
                        test_used = "t-test"
                    else:
                        stat, p = ttest_ind(demaf, monokle, equal_var=False)
                        test_used = "Welch's t-test"
                else:
                    stat, p = mannwhitneyu(demaf, monokle, alternative="two-sided")
                    test_used = "Mann-Whitney U"

                result = {
                    "test_used": test_used,
                    "p_value": p,
                    "demaf_mean": demaf.mean(),
                    "monokle_mean": monokle.mean(),
                    "demaf_median": np.median(demaf),
                    "monokle_median": np.median(monokle),
                    "demaf_n": len(demaf),
                    "monokle_n": len(monokle),
                    "demaf_normal": demaf_normal,
                    "monokle_normal": monokle_normal,
                    "equal_var": equal_var
                }
                all_results[(task, col)] = result

                # Write results to file
                f.write(f"  Likert: {col}:\n")
                f.write(f"    DeMAF: n={result['demaf_n']}, mean={result['demaf_mean']:.3f}, median={result['demaf_median']:.3f}, normal={result['demaf_normal']}\n")
                f.write(f"    Monokle: n={result['monokle_n']}, mean={result['monokle_mean']:.3f}, median={result['monokle_median']:.3f}, normal={result['monokle_normal']}\n")
                f.write(f"    Variance equal: {result['equal_var']}\n")
                f.write(f"    Test used: {result['test_used']}\n")
                f.write(f"    p-value: {result['p_value']:.5f}\n\n")

        # --- Fisher's method: combine T1 and T2 p-values for each metric ---
        f.write("\n=== Fisher's Method: Combined p-values for each metric ===\n")
        for metric_name in metrics.keys():
            pvals = []
            for task in tasks:
                key = (task, metric_name)
                if key in all_results:
                    pvals.append(all_results[key]["p_value"])
            if len(pvals) == 2:
                stat, combined_p = combine_pvalues(pvals, method='fisher')
                f.write(f"{metric_name}: Fisher's X^2 = {stat:.3f}, combined p-value = {combined_p:.5g}\n")
                all_results[("Fisher", metric_name)] = {
                    "fisher_stat": stat,
                    "combined_p_value": combined_p
                }

        # --- Fisher's method for Likert items ---
        f.write("\n=== Fisher's Method: Combined p-values for each Likert item ===\n")
        for i in range(1, 16):
            pvals = []
            for task in tasks:
                key = (task, f"{task}_LQ_{i}")
                if key in all_results:
                    pvals.append(all_results[key]["p_value"])
            if len(pvals) == 2:
                stat, combined_p = combine_pvalues(pvals, method='fisher')
                f.write(f"LQ_{i}: Fisher's X^2 = {stat:.3f}, combined p-value = {combined_p:.5g}\n")
                all_results[("Fisher", f"LQ_{i}")] = {
                    "fisher_stat": stat,
                    "combined_p_value": combined_p
                }

    return all_results

def comprehension_ttest_pipeline_combined(df, metrics=None, group_cols=None, output_file="eval/comprehension_ttest_results_combined.txt"):
    """
    Runs the appropriate test (t-test, Welch's t-test, or Mann-Whitney U) for each metric,
    but combines T1 and T2 data for each tool before testing.
    Writes all results to a file and returns a dict of all relevant results.
    """
    if metrics is None:
        metrics = {
            "Performance (Aggregated, Normalized)": "_perf_agg",
            "Don't know count": "_dont-know_agg",
            "Task time": "_time",
            "NASA Likert (Aggregated, Normalized)": "_LQ_agg"
        }
    if group_cols is None:
        group_cols = {"T1": "T1_UQ_1", "T2": "T2_UQ_1"}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_results = {}

    with open(output_file, "w") as f:
        f.write("--- Combined T1+T2 Analysis ---\n")
        # Standard metrics
        for metric_name, suffix in metrics.items():
            # Collect DeMAF and Monokle values from both tasks
            demaf_vals = []
            monokle_vals = []
            for task in ["T1", "T2"]:
                group_col = group_cols[task]
                if suffix == "_time":
                    metric_col = f"{task}_time"
                else:
                    metric_col = f"{task}{suffix}"
                if metric_col not in df.columns:
                    continue
                demaf_vals.append(pd.to_numeric(df[df[group_col].str.lower() == "demaf"][metric_col], errors="coerce").dropna())
                monokle_vals.append(pd.to_numeric(df[df[group_col].str.lower() == "monokle"][metric_col], errors="coerce").dropna())
            demaf = pd.concat(demaf_vals, ignore_index=True) if demaf_vals else pd.Series(dtype=float)
            monokle = pd.concat(monokle_vals, ignore_index=True) if monokle_vals else pd.Series(dtype=float)

            # Normality
            demaf_normal = shapiro(demaf)[1] > 0.05 if len(demaf) >= 3 else False
            monokle_normal = shapiro(monokle)[1] > 0.05 if len(monokle) >= 3 else False

            # Variance
            equal_var = levene(demaf, monokle)[1] > 0.05 if len(demaf) >= 2 and len(monokle) >= 2 else False

            # Decide test
            if demaf_normal and monokle_normal:
                if equal_var:
                    stat, p = ttest_ind(demaf, monokle, equal_var=True)
                    test_used = "t-test"
                else:
                    stat, p = ttest_ind(demaf, monokle, equal_var=False)
                    test_used = "Welch's t-test"
            else:
                stat, p = mannwhitneyu(demaf, monokle, alternative="two-sided")
                test_used = "Mann-Whitney U"

            result = {
                "test_used": test_used,
                "p_value": p,
                "demaf_mean": demaf.mean(),
                "monokle_mean": monokle.mean(),
                "demaf_median": np.median(demaf),
                "monokle_median": np.median(monokle),
                "demaf_n": len(demaf),
                "monokle_n": len(monokle),
                "demaf_normal": demaf_normal,
                "monokle_normal": monokle_normal,
                "equal_var": equal_var
            }
            all_results[metric_name] = result

            # Write results to file
            f.write(f"\n{metric_name} (T1+T2 combined):\n")
            f.write(f"    DeMAF: n={result['demaf_n']}, mean={result['demaf_mean']:.3f}, median={result['demaf_median']:.3f}, normal={result['demaf_normal']}\n")
            f.write(f"    Monokle: n={result['monokle_n']}, mean={result['monokle_mean']:.3f}, median={result['monokle_median']:.3f}, normal={result['monokle_normal']}\n")
            f.write(f"    Variance equal: {result['equal_var']}\n")
            f.write(f"    Test used: {result['test_used']}\n")
            f.write(f"    p-value: {result['p_value']:.5f}\n\n")

        # --- Add Likert scale columns LQ_1 to LQ_15 (combined T1+T2) ---
        for i in range(1, 16):
            demaf_vals = []
            monokle_vals = []
            for task in ["T1", "T2"]:
                group_col = group_cols[task]
                col = f"{task}_LQ_{i}"
                if col not in df.columns:
                    continue
                demaf_vals.append(pd.to_numeric(df[df[group_col].str.lower() == "demaf"][col], errors="coerce").dropna())
                monokle_vals.append(pd.to_numeric(df[df[group_col].str.lower() == "monokle"][col], errors="coerce").dropna())
            demaf = pd.concat(demaf_vals, ignore_index=True) if demaf_vals else pd.Series(dtype=float)
            monokle = pd.concat(monokle_vals, ignore_index=True) if monokle_vals else pd.Series(dtype=float)

            # Normality
            demaf_normal = shapiro(demaf)[1] > 0.05 if len(demaf) >= 3 else False
            monokle_normal = shapiro(monokle)[1] > 0.05 if len(monokle) >= 3 else False

            # Variance
            equal_var = levene(demaf, monokle)[1] > 0.05 if len(demaf) >= 2 and len(monokle) >= 2 else False

            # Decide test
            if demaf_normal and monokle_normal:
                if equal_var:
                    stat, p = ttest_ind(demaf, monokle, equal_var=True)
                    test_used = "t-test"
                else:
                    stat, p = ttest_ind(demaf, monokle, equal_var=False)
                    test_used = "Welch's t-test"
            else:
                stat, p = mannwhitneyu(demaf, monokle, alternative="two-sided")
                test_used = "Mann-Whitney U"

            result = {
                "test_used": test_used,
                "p_value": p,
                "demaf_mean": demaf.mean(),
                "monokle_mean": monokle.mean(),
                "demaf_median": np.median(demaf),
                "monokle_median": np.median(monokle),
                "demaf_n": len(demaf),
                "monokle_n": len(monokle),
                "demaf_normal": demaf_normal,
                "monokle_normal": monokle_normal,
                "equal_var": equal_var
            }
            all_results[f"LQ_{i}"] = result

            # Write results to file
            f.write(f"  Likert: LQ_{i} (T1+T2 combined):\n")
            f.write(f"    DeMAF: n={result['demaf_n']}, mean={result['demaf_mean']:.3f}, median={result['demaf_median']:.3f}, normal={result['demaf_normal']}\n")
            f.write(f"    Monokle: n={result['monokle_n']}, mean={result['monokle_mean']:.3f}, median={result['monokle_median']:.3f}, normal={result['monokle_normal']}\n")
            f.write(f"    Variance equal: {result['equal_var']}\n")
            f.write(f"    Test used: {result['test_used']}\n")
            f.write(f"    p-value: {result['p_value']:.5f}\n\n")

    return all_results

def write_text_answers_to_file(df, mapping_df, output_file="eval/text_answers.txt"):
    """
    Writes all text-based answers for specified columns to a file, using the original column names from the mapping_df.
    For columns containing T1 or T2, answers are separated by tool.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    text_cols = [
        "T1_LQ_17", "T1_LQ_18", "T2_LQ_17", "T2_LQ_18",
        "CQ_7", "FQ_1", "FQ_2", "FQ_3", "FQ_8"
    ]
    mapping_dict = dict(zip(mapping_df['new_header'], mapping_df['original_header']))

    with open(output_file, "w", encoding="utf-8") as f:
        for col in text_cols:
            if col in df.columns:
                orig_col_name = mapping_dict.get(col, col)
                # For T1/T2 columns, separate by tool
                if col.startswith("T1") or col.startswith("T2"):
                    tool_col = "T1_UQ_1" if col.startswith("T1") else "T2_UQ_1"
                    f.write(f"\n--- {orig_col_name} ({col}) ---\n")
                    for tool in ["demaf", "monokle"]:
                        f.write(f"\n[{tool.capitalize()}]\n")
                        tool_mask = (df[tool_col].str.lower() == tool)
                        for idx, val in df.loc[tool_mask, col].items():
                            uuid = df.iloc[idx]["UUID"] if "UUID" in df.columns else f"Row {idx+1}"
                            answer = str(val).strip()
                            if answer and answer.lower() not in ["nan", "none"]:
                                f.write(f"Participant {uuid}: {answer}\n")
                else:
                    f.write(f"\n--- {orig_col_name} ({col}) ---\n")
                    for idx, val in df[col].items():
                        uuid = df.iloc[idx]["UUID"] if "UUID" in df.columns else f"Row {idx+1}"
                        answer = str(val).strip()
                        if answer and answer.lower() not in ["nan", "none"]:
                            f.write(f"Participant {uuid}: {answer}\n")
            else:
                f.write(f"\n--- {col} not found in DataFrame ---\n")


def main():
    # Specify the file path and columns to plot
    main_data_path = 'prepared data/study_data_renamed.csv'
    header_mapping_path = 'prepared data/header_mapping.csv'
    task_solutions_path = 'prepared data/task_solutions.csv'

    # Read the CSV into a DataFrame
    df = read_csv_to_df(main_data_path)
    df = df.replace(r'\n', ' ', regex=True)
    if df is not None:
        df_demaf_first, df_monokle_first = split_data_by_task_tool(df)
        df_task_solutions = read_csv_to_df(task_solutions_path)
        mapping_df = read_csv_to_df(header_mapping_path)

        plot_task_times(df, [600, 2000])

        #compute_all_likert_stats_and_plots(df, mapping_df, 'eval/likert_stats.txt')
        #compute_all_likert_stats_and_plots_combined(df, mapping_df)

        # plot_demographics(df)

        #compute_and_plot_kq_stats(df, mapping_df)        
        plot_knowledge_terms_grouped_barchart(df, mapping_df)

        # comparison_questions_eval(df,mapping_df)

        # prefered_tool_by_knowledge(df, mapping_df)

        # time_stats_by_knowledge(df, mapping_df)

        # write_text_answers_to_file(df, mapping_df)

        # plot_demographics_pie_charts(df)

        #------ Performance df-----------------------

        #df_perf = evaluate_task_performance(df, df_task_solutions)

        # compute_and_plot_dontknow_counts(df_perf)

        # performance_stats_by_knowledge(df_perf, mapping_df)

        # evaluate_relative_performance(df_perf, mapping_df, plot_output_folder='plots/performance-comparison')

        #plot_and_save_performance(df_perf)

        # check_comprehension_data_requirements(df_perf)

        #comprehension_ttest_pipeline(df_perf)
        #comprehension_ttest_pipeline_combined(df_perf)

if __name__ == "__main__":
    main()
