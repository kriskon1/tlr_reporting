import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def read_format_data(xlsx):
    L2_SVS = pd.read_excel(xlsx, sheet_name="L2_SVS_2024-2025", header=None)

    L2_SVS.columns = L2_SVS.iloc[0] + " - " + L2_SVS.iloc[1]
    L2_SVS = L2_SVS[3:].reset_index(drop=True)

    L2_SVS = L2_SVS.fillna(0)
    L2_SVS.columns = [f"Unnamed_{i}" if pd.isna(col) else col for i, col in enumerate(L2_SVS.columns)]

    L2_SVS = L2_SVS.rename(columns={L2_SVS.columns[0]: "Date",
                                    L2_SVS.columns[71]: "wk",
                                    L2_SVS.columns[72]: "ok",
                                    L2_SVS.columns[73]: "nok",
                                    L2_SVS.columns[74]: "total",
                                    L2_SVS.columns[75]: "TLR"})

    L2_SVS['month'] = L2_SVS['Date'].dt.to_period('M')
    # isocalendar is needed so that week numbers aren't starting from 0, e.g. 2025-01-01 is not 2025_00, rather 2025_01
    L2_SVS['week'] = L2_SVS['Date'].dt.isocalendar().year.astype(str) + "-" + L2_SVS[
        'Date'].dt.isocalendar().week.astype(str).str.zfill(2)

    L2_SVS = L2_SVS.drop(columns=L2_SVS.columns[71])

    cols_to_int = [66, 71, 72, 73, 74]
    L2_SVS = L2_SVS.astype({L2_SVS.columns[i]: int for i in cols_to_int})

    return L2_SVS


def top5_month_sort(L2_SVS):
    L2_SVS_top5_by_month = L2_SVS.groupby('month').apply(
            lambda x: x.iloc[:, 1:71].astype(int).sum().nlargest(5)).reset_index()

    monthly_prod = L2_SVS.groupby('month')["total"].sum().astype(int)

    merged_L2_SVS_top5_by_month = pd.merge(L2_SVS_top5_by_month, monthly_prod, on='month')

    merged_L2_SVS_top5_by_month = merged_L2_SVS_top5_by_month.rename(columns={
                                                                        merged_L2_SVS_top5_by_month.columns[1]: 'error',
                                                                        merged_L2_SVS_top5_by_month.columns[2]: 'nok'})
    merged_L2_SVS_top5_by_month["TLR"] = np.where(merged_L2_SVS_top5_by_month["total"] != 0,
        (merged_L2_SVS_top5_by_month["nok"].astype(int) / merged_L2_SVS_top5_by_month["total"].astype(int)) * 1_000_000,
        0)
    merged_L2_SVS_top5_by_month["TLR"] = merged_L2_SVS_top5_by_month["TLR"].fillna(0).astype(int)

    # Filter out rows from the current month onwards
    merged_L2_SVS_top5_by_month = merged_L2_SVS_top5_by_month[
        merged_L2_SVS_top5_by_month['month'] < pd.to_datetime(datetime.now().strftime('%Y-%m')).to_period('M')]

    return merged_L2_SVS_top5_by_month


def top5_month_chart_bar_sum(merged_L2_SVS_top5_by_month):
    unique_errors_month = merged_L2_SVS_top5_by_month['error'].unique()
    color_map = dict(zip(unique_errors_month, plt.cm.tab20.colors[:len(unique_errors_month)]))

    fig, ax = plt.subplots(figsize=(14, 10))

    x_positions = []  # Store modified x-axis positions
    x_labels = []  # Store corresponding labels
    gap = 0  # Extra space for every 5th element

    for i, (index, row) in enumerate(merged_L2_SVS_top5_by_month.iterrows()):
        if i % 5 == 0 and i != 0:
            gap += 1

        x_pos = i + gap
        x_positions.append(x_pos)
        x_labels.append(f"{row['month']} - {row['error']}")

        color = color_map[row['error']]
        ax.bar(x_pos, row.iloc[4], color=color, label=row['error'])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, ha='right')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Failure Types')

    plt.title('Top 5 Failures by Month')
    plt.xlabel('Failures')
    plt.ylabel('TLR')
    plt.tight_layout()
    plt.show()


def top5_month_chart_line_sum(merged_L2_SVS_top5_by_month):
    unique_errors = merged_L2_SVS_top5_by_month['error'].unique()
    color_map = dict(zip(unique_errors, plt.cm.tab20.colors[:len(unique_errors)]))

    fig, ax = plt.subplots(figsize=(14, 10))

    merged_L2_SVS_top5_by_month['month'] = merged_L2_SVS_top5_by_month['month'].astype(str)

    for error in unique_errors:
        error_data = merged_L2_SVS_top5_by_month[merged_L2_SVS_top5_by_month['error'] == error]
        ax.plot(error_data["month"], error_data["TLR"], label=error, color=color_map[error], marker='o', linestyle='-')

    plt.xticks(merged_L2_SVS_top5_by_month["month"].unique(), rotation=45)

    plt.legend(title='Error Types')

    plt.title('Top 5 Failures by Month')
    plt.xlabel('Month')
    plt.ylabel('TLR')
    plt.tight_layout()
    plt.show()


def top5_month_chart_line_failure_mode(merged_L2_SVS_top5_by_month):
    unique_errors_month = merged_L2_SVS_top5_by_month['error'].unique().tolist()

    for error in unique_errors_month:
        error_data = merged_L2_SVS_top5_by_month[merged_L2_SVS_top5_by_month['error'] == error]

        plt.figure(figsize=(5, 3))
        plt.plot(error_data["month"], error_data['TLR'], marker='o')
        plt.title(f'TLR Trend for {error}')
        plt.xlabel('Month')
        plt.ylabel('TLR')
        plt.xticks(error_data["month"].unique(), rotation=45)
        plt.show()


def top5_week_sort(L2_SVS):
    L2_SVS_top5_by_week_num = L2_SVS.groupby('week').apply(
        lambda x: x.iloc[:, 3:71].astype(int).sum().nlargest(5)).reset_index()

    weekly_prod = L2_SVS.groupby('week')["total"].sum().astype(int)

    merged_L2_SVS_top5_by_week_num = pd.merge(L2_SVS_top5_by_week_num, weekly_prod, on='week')

    merged_L2_SVS_top5_by_week_num = merged_L2_SVS_top5_by_week_num.rename(
        columns={merged_L2_SVS_top5_by_week_num.columns[1]: 'error', merged_L2_SVS_top5_by_week_num.columns[2]: 'nok'})
    merged_L2_SVS_top5_by_week_num["TLR"] = np.where(merged_L2_SVS_top5_by_week_num["total"] != 0,
        (merged_L2_SVS_top5_by_week_num["nok"].astype(int) / merged_L2_SVS_top5_by_week_num["total"].astype(int)) * 1_000_000,
        0)
    merged_L2_SVS_top5_by_week_num["TLR"] = merged_L2_SVS_top5_by_week_num["TLR"].fillna(0).astype(int)

    # Filter out rows for current and future weeks
    current_week_tuple = (datetime.today().year, datetime.today().isocalendar()[1])

    # Convert 'week' column from string to (year, week) tuples
    merged_L2_SVS_top5_by_week_num['week_tuple'] = merged_L2_SVS_top5_by_week_num['week'].apply(
        lambda x: tuple(map(int, x.split('-'))))

    merged_L2_SVS_top5_by_week_num = merged_L2_SVS_top5_by_week_num[
        merged_L2_SVS_top5_by_week_num['week_tuple'] < current_week_tuple]
    merged_L2_SVS_top5_by_week_num = merged_L2_SVS_top5_by_week_num.drop(columns=['week_tuple'])

    # Filter out rows with 0 TLR values
    merged_L2_SVS_top5_by_week_num.set_index(merged_L2_SVS_top5_by_week_num.columns[0], inplace=True)

    return merged_L2_SVS_top5_by_week_num


def top5_week_chart_bar_sum(merged_L2_SVS_top5_by_week_num):
    unique_errors_week = merged_L2_SVS_top5_by_week_num['error'].unique()
    colormap = plt.get_cmap('tab20b', len(unique_errors_week))
    color_map = {error: colormap(i) for i, error in enumerate(unique_errors_week)}

    fig, ax = plt.subplots(figsize=(40, 18))

    x_positions = []
    x_labels = []
    gap = 0

    for i, (index, row) in enumerate(merged_L2_SVS_top5_by_week_num.iterrows()):
        if i % 5 == 0 and i != 0:
            gap += 1

        x_pos = i + gap
        x_positions.append(x_pos)
        x_labels.append(f"{index} - {row['error']}")

        color = color_map[row['error']]
        ax.bar(x_pos, row["TLR"], color=color, label=row['error'])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Error Types')

    plt.title('Top 5 Failures by Week with Gaps')
    plt.xlabel('Failures')
    plt.ylabel('TLR')
    plt.tight_layout()
    plt.show()


def top5_week_chart_line_sum(merged_L2_SVS_top5_by_week_num):
    if 'week' in merged_L2_SVS_top5_by_week_num.index.names:
        # Reset the index to bring 'week_num' back as a column
        merged_L2_SVS_top5_by_week_column = merged_L2_SVS_top5_by_week_num.reset_index()

    unique_errors_week = merged_L2_SVS_top5_by_week_column['error'].unique()
    colormap = plt.get_cmap('tab20b', len(unique_errors_week))
    color_map = {error: colormap(i) for i, error in enumerate(unique_errors_week)}

    # Convert week values to a numeric index for plotting
    week_mapping = {week: i for i, week in enumerate(merged_L2_SVS_top5_by_week_column["week"].unique())}
    merged_L2_SVS_top5_by_week_column["week_numeric"] = merged_L2_SVS_top5_by_week_column["week"].map(week_mapping)

    fig, ax = plt.subplots(figsize=(14, 10))

    for error in unique_errors_week:
        error_data = merged_L2_SVS_top5_by_week_column[merged_L2_SVS_top5_by_week_column['error'] == error]
        ax.plot(error_data["week_numeric"], error_data["TLR"], label=error, color=color_map.get(error), marker='o')

    ax.set_xticks(list(week_mapping.values()))
    ax.set_xticklabels(list(week_mapping.keys()), rotation=45)

    ax.legend(title='Error Types')

    plt.title('Top 5 Errors by Week')
    plt.xlabel('Week Number')
    plt.ylabel('TLR')
    plt.tight_layout()
    plt.show()


def top5_week_chart_line_failure_mode(merged_L2_SVS_top5_by_week_num):
    unique_errors_week = merged_L2_SVS_top5_by_week_num['error'].unique().tolist()

    for error in unique_errors_week:
        error_data = merged_L2_SVS_top5_by_week_num[merged_L2_SVS_top5_by_week_num['error'] == error]
        if len(error_data) > 15:
            plt.figure(figsize=(15, 3))
        else:
            plt.figure(figsize=(5, 3))
        plt.plot(error_data.index, error_data['TLR'], marker='o')
        plt.title(f'TLR Trend for {error}')
        plt.xlabel('Month')
        plt.ylabel('TLR')
        plt.xticks(error_data.index, rotation=90)
        plt.show()
