import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def plot_matched(
    data=None,
    x=None,
    y=None,
    x_order=None,
    match_col=None,
    x_match_sort=None,
    sort_idx = None,
    title=None,
    x_xoffset=0.2,
    ax=None,
    figsize=(9, 6),
    sort_marker="$",
    legendmarker=False, #If False, plot normal Markers, else plot legend number or string as marker
    error="amend",
    estimator='mean',
    cp=None,
):
    if bool(x_match_sort) and bool(sort_idx):
        raise ValueError("Either x_match_sort or sort_idx should be passed, not both!")

    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor="white", figsize=figsize)
    if x_order is not None:
        #data = data.loc[data[x].str.match("|".join([re.escape(xo) for xo in x_order]))]
        x_order = list(x_order)
        data = data.loc[data[x].isin(x_order)]
        ux = data[x].unique()
        clean_order = [x_order[i] for i in range(len(x_order)) for ux1 in ux if x_order[i] == ux1] #added extra for loop instead of in, because testing with tuples throws error
        if len(clean_order) < len(x_order):
            print("Warning: Truncating ordering, as some pipelines not existing in data frame.")
            x_order = clean_order
        if x_match_sort is not None and x_match_sort not in x_order:
            errorwarn_string = (
                f"Cannot sort by {x_match_sort} as it is not in {x_order} / data array."
            )
            if error == "raise":
                raise ValueError(errorwarn_string)
            else:
                print(f"WARNING: {errorwarn_string}")
                x_match_sort = None
    else:
        x_order = list(data[x].unique())
        if x_match_sort is not None:
            x_order.remove(x_match_sort)
            x_order.insert(0, x_match_sort)
    cp = sns.color_palette() if cp is None else cp
    marker_arr = ["^", "s", "p", "H", "o"]
    n_markers = len(marker_arr)
    num_x = len(data[x].unique())
    num_matched = len(data[match_col].unique())
    legend_labels = data[match_col].unique()
    if x_match_sort is not None:
        sort_idx = (
            data.loc[data[x] == x_match_sort]
            .sort_values(by=match_col, ascending=True)
            .reset_index()
            .sort_values(by=y, ascending=True)
            .index.copy()
        )
    if sort_idx is not None:
        legend_labels = legend_labels[sort_idx]
    c_offs_left = 2
    c_offs_right = 2
    gmap = LinearSegmentedColormap.from_list(
        "custom",
        [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)],
        N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right),
    )
    legend_handles = []
    m_scores = []
    for i, x_main in enumerate(x_order):
        base_col = cp[i]
        cmap = LinearSegmentedColormap.from_list(
            "custom",
            [(0, 0, 0), base_col, (1, 1, 1)],
            N=(num_matched // n_markers + 1 + c_offs_left + c_offs_right),
        )
        r = data.loc[data[x] == x_main]  # .sort_values(by=match_col, ascending=True).reset_index()
        # if sort_idx is None:
        #     sort_idx = r.sort_values(by='score', ascending=True).index.copy()
        if sort_idx is not None:
            r = r.iloc[sort_idx]
        x_center = i + 1
        x_width = 0.5 - x_xoffset / 2
        x_space = np.linspace(x_center - x_width, x_center + x_width, num_matched)
        if estimator == 'mean':
            m_score, m_std = r[y].aggregate((np.mean, np.std))
        elif estimator == 'median':
            m_score, m_std = r[y].aggregate((np.median, np.std))
        else:
            raise ValueError(f'{estimator} is an invalid estimator, only "mean" or "median" are allowed')
        m_scores.append(m_score)
        m_err = m_std / np.sqrt(num_matched)
        err_artists = ax.errorbar(
            x_center,
            m_score,
            2 * m_err,
            capsize=0,
            zorder=15,
            color="k",
            linewidth=1,
            alpha=1,
            dash_capstyle="round",
        )
        err_artists[2][0].set_capstyle("round")
        ax.scatter(
            x_center,
            m_score,
            marker="X",
            color=base_col,
            edgecolor="k",
            linewidth=0.4,
            zorder=20,
            s=20,
        )
        for j, x_j in enumerate(x_space):
            score = r[y].iloc[j]
            if legendmarker:
                m = f"${r[match_col].iloc[j]}$" # Use Strings as markers, usually subject number
            else:
                m = marker_arr[j % len(marker_arr)]
            sdef = 35
            s = (
                sdef * 0.66 if m in ["s", "D"] else sdef
            )  # these two markers are unexplicably larger in default pyplot
            ax.scatter(
                x_j,
                score,
                alpha=0.8,
                marker=m,
                linewidth=0.4,
                edgecolor=(0.8, 0.8, 0.8),
                s=s,
                color=cmap((j // n_markers) + c_offs_left),
            )
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=m,
                        color="w",
                        label=r[match_col].iloc[j],
                        markerfacecolor=gmap((j // n_markers) + c_offs_left),
                        markersize=1.2 * np.sqrt(s),
                    )
                )
                # need to translate markersize between scatter and plt function, 1.2*sqrt() seems to work kind of
    ax.axhline(np.max(m_scores), linestyle='--', c="k", linewidth=1)
    ax.set_xticks(np.arange(1, num_x + 1))
    xticklabels = list(x_order)
    if x_match_sort is not None:
        xticklabels[x_order.index(x_match_sort)] = (
            sort_marker + str(x_order[x_order.index(x_match_sort)])
        )
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, num_x + 1)
    for i, tick in enumerate(ax.get_xticklabels()):
        if len(tick.get_text()) > 0:
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
            tick.set_rotation(20)
            tick.set_color(cp[i])
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax, (legend_handles, legend_labels)



def plot_matched_diffs(data=None,
                       base_value=None,
                       compare_column=None,
                       score_metric='score',
                       x_order=None,
                       ylim=None,
                       ):
    '''plot subject-wise differences between pipelines
    base_pipeline: is the pipeline that will be subtracted from the others
    x_order: contains all other pipelines, can still include the base pipeline'''
    if base_value is None:
        base_value = data.pipeline[0]
        print('No base pipeline passed, selecting first pipeline in data')
    if compare_column is None:
        compare_column = "pipeline"

    base_df = (data.loc[data.pipeline == base_value]
                   .sort_values('subject')
                   .set_index([compare_column, 'dataset', 'subject', 'samples'])
                   .reset_index(drop=True)
                   .loc[:, score_metric]
                   )

    compare_df = (data.loc[data.pipeline != base_value]
                      .sort_values([compare_column,'subject'])
                      .set_index([compare_column, 'dataset', 'subject', 'samples'])
                      .loc[:, score_metric]
                      )
    compare_index = compare_df.index

    diff_df = compare_df.reset_index(drop=True) - base_df.iloc[
        np.tile(np.arange(len(base_df)),len(data[compare_column].unique()) - 1)].reset_index(drop=True)
    diff_df.index = compare_index
    diff_df = diff_df.reset_index()
    if ylim is not None:
        ymin, ymax = ylim
    else:
        ymin, ymax = diff_df[score_metric].min(), diff_df[score_metric].max()

    if x_order is None:
        x_order = diff_df[compare_column].sort_values().unique().tolist()
    else:
        cp = sns.color_palette()
        del cp[x_order.index(base_value)]
        x_order.remove(base_value)
    # diff_df = diff_df.append(
    #     plot_df.loc[plot_df.pipeline == base_pipeline, ['pipeline', 'dataset', 'subject', 'samples', 'score']])
    #x_order.append(base_pipeline)
    sort_idx = (
        data.loc[data['pipeline'] == base_value]
            .sort_values(by="subject", ascending=True)
            .reset_index()
            .sort_values(by=score_metric, ascending=True)
            .index.copy()
    )

    fig, ax = plt.subplots(1, 1, dpi=200)
    ax, legend_stuff = plot_matched(
        x='pipeline',
        y=score_metric,
        data=diff_df,
        match_col="subject",
        # x_match_sort=x_match_sort,
        x_order=x_order,
        sort_idx=sort_idx,
        # x_labels=get_clean_xlabels(subset_pipelines),
        ax=ax,
        sort_marker="",
        estimator='mean',
        cp=cp,
    )

    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_ylim([ymin - 0.05, ymax + 0.05])
    return ax, legend_stuff