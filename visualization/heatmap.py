import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
import os
import math
# import copy


operations = {
    'max': np.max,
    'min': np.min,
    'avg': np.mean,
    'med': np.median
}


def draw(index, aux_index=None, op='avg', title='', highlight=False):
    """
    Params
    """

    e_order = [1, 2, 4, 8, 16]  # hardcoded

    # TODO: highlight_scale_factor
    main_scaled_factor = 0.1 * 1  # (index) +/- scaling target  # +/- 10%
    aux_th = 100.0 * 1  # (aux_index) timeout th  # 10%
    denom = 1
    if title == 'latency' or title == 'latency_highlight' or title == 'latency_timeout' or title == 'latency_timeout_highlight':
        # denom = 86400 / 12.06 # 7164.1791044776
        denom = 7164

    """
    Open pkl
    """

    data = {}
    aux_data = {}
    pickle_files = [f for f in os.listdir(
        './save_heatmap/objects') if f.endswith('.pkl')]
    for file in pickle_files:
        # Extract QC, E, D from the filename
        parts = file.split('_')
        E = int(parts[4].replace('E', ''))
        D = int(parts[5].replace('D', ''))
        QC = int(parts[6].replace('QC', ''))

        # index
        with open(f'./save_heatmap/objects/{file}', 'rb') as f:
            loaded_data = pickle.load(f)
            target_data = loaded_data[index]

        avg_target_data = operations[op](target_data) / denom

        if E not in data:
            data[E] = []
        data[E].append((QC, D, avg_target_data))

        # aux_index
        if (aux_index != None):
            with open(f'./save_heatmap/objects/{file}', 'rb') as f:
                loaded_data = pickle.load(f)
                target_data = loaded_data[aux_index]

            avg_target_data = operations[op](target_data)

            if E not in aux_data:
                aux_data[E] = []
            aux_data[E].append((QC, D, avg_target_data))

    """
    Draw
    """
    markers_palette = [(0.00784313725490196, 0.24313725490196078, 1.0),
                       (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
                       (1.0, 0.48627450980392156, 0.0),
                       (0.9098039215686274, 0.0, 0.043137254901960784),
                       (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                       (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
                       (0.6235294117647059, 0.2823529411764706, 0.0),
                       (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
                       (1.0, 0.7686274509803922, 0.0),
                       (0.0, 0.8431372549019608, 1.0)]
    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

    # Setup the figure and axes for the heatmaps + 1 extra for the colorbar
    fig, axes = plt.subplots(1, len(e_order) + 1, figsize=(3 * len(e_order) + 1.5, 3),
                             gridspec_kw={'width_ratios': [1]*len(e_order) + [0.05]})
    axes[-1].axis('off')  # Hide the last subplot (used for colorbar)
    plt.subplots_adjust(wspace=0.1)

    # Global color scale for all heatmaps
    all_target_data = [val for sublist in data.values()
                       for _, _, val in sublist]
    if max(all_target_data) <= 100:
        vmin, vmax = math.floor(min(all_target_data)), math.ceil(
            max(all_target_data))
    else:
        vmin, vmax = 0, math.ceil(
            max(all_target_data) / 100) * 100
    if aux_index:
        all_target_data = [val for sublist in aux_data.values()
                           for _, _, val in sublist]
        if max(all_target_data) <= 100:
            aux_vmin, aux_vmax = math.floor(min(all_target_data)), math.ceil(
                max(all_target_data))
        else:
            aux_vmin, aux_vmax = 0, math.ceil(
                max(all_target_data) / 100) * 100

    # Define a custom colormap with a higher number of discrete levels
    num_levels = 256
    custom_cmap = plt.cm.get_cmap('Greys', num_levels)
    aux_cmap = plt.cm.get_cmap('Blues', num_levels)
    # colors = aux_cmap(np.linspace(0.3, 0.8, num_levels))
    # aux_cmap = LinearSegmentedColormap.from_list("half_blues", colors)

    for i, E in enumerate(e_order):
        ax = axes[i]

        values = data.get(E, [])
        if not values:
            continue  # Skip if no data for this E

        df = pd.DataFrame(values, columns=['QC', 'D', 'Value'])
        pivot_df = df.pivot(index="D", columns="QC", values="Value")

        # Plot each heatmap without a colorbar
        sns.heatmap(pivot_df, ax=ax, cbar=False,
                    vmin=vmin, vmax=vmax,
                    annot=True, fmt=".2f",
                    cmap=custom_cmap)

        # Iterate over the data and create a Triangle patch for each cell
        if (aux_index != None):
            values = aux_data.get(E, [])
            if values:
                aux_df = pd.DataFrame(values, columns=['QC', 'D', 'Value'])
                aux_pivot_df = aux_df.pivot(
                    index="D", columns="QC", values="Value")

                for ri in range(len(aux_pivot_df)):
                    for rj in range(len(aux_pivot_df.columns)):
                        side_length = 1.0

                        # Original vertices before flipping
                        bottom_left = (rj, ri)
                        bottom_right = (rj + side_length, ri)
                        top_left = (rj, ri + side_length)
                        # Flip horizontally by reflecting the x-coordinates around the midpoint of the square
                        midpoint_x = rj + side_length / 2
                        flipped_bottom_left = (
                            midpoint_x - (bottom_left[0] - midpoint_x), bottom_left[1])
                        flipped_bottom_right = (
                            midpoint_x - (bottom_right[0] - midpoint_x), bottom_right[1])
                        flipped_top_left = (
                            midpoint_x - (top_left[0] - midpoint_x), top_left[1])

                        # Determine color based on some value; here we use normed_values[ri, rj] directly
                        normed_values = (aux_pivot_df.to_numpy() -
                                         aux_vmin) / (aux_vmax - aux_vmin)
                        color = aux_cmap(normed_values[ri, rj])

                        # Create and add the flipped triangle with color
                        triangle = patches.Polygon(
                            [flipped_bottom_left, flipped_bottom_right,
                                flipped_top_left], closed=True,
                            color=color, fill=True, alpha=0.5)
                        ax.add_patch(triangle)

        # ax.set_title(f'(E = {E})')
        # ax.set_title(None)

        ax.invert_yaxis()  # Flip the y-axis

        # qcs = copy.deepcopy(ax.get_xticklabels())
        qcs = [int(t.get_text()) for t in ax.get_xticklabels()]
        # ds = copy.deepcopy(ax.get_yticklabels())
        ds = [int(t.get_text()) for t in ax.get_yticklabels()]

        ax.set_xlabel(None)
        # ax.set_xlabel(rf'$Q_C$' + ' ' + rf'$(E = {E})$')
        # ax.xaxis.label.set_color('black')
        # ax.xaxis.label.set_path_effects([
        #     patheffects.withStroke(linewidth=4, foreground=color_hex)
        # ])
        label_y = -0.18
        ax.text(0.355, label_y, r"$Q_C$", color='black',
                transform=ax.transAxes)
        ax.text(0.445, label_y, rf'$(E = {E})$', color=markers_palette[i],
                transform=ax.transAxes)

        ax.set_ylabel(None)  # Hide the y-axis title
        if i != 0:
            ax.set_yticklabels([])  # Hide the y-axis tick labels
        else:
            ax.set_yticklabels(
                ['6.25%', '12.5%', '25%', '50%'], rotation=45)
        # ax.tick_params(axis='y', length=0)  # Optionally, hide the y-axis tick marks

        # Highlight cells
        if highlight:
            if aux_index == None:
                for ytick, xtick in np.ndindex(pivot_df.shape):
                    value = pivot_df.iat[ytick, xtick]
                    # print(qcs[xtick], ds[ytick], value)
                    # target = E * 256/ds[ytick] * qcs[xtick] / 21
                    if 32 >= value:
                    # if (1.09 > value):
                        rect = patches.Rectangle(
                            (xtick+0.05, ytick+0.05), 1-0.05*2, 1-0.05*2,
                            linewidth=2, edgecolor=markers_palette[i],
                            alpha=0.25,
                            hatch=hatches[i],
                            facecolor='none')
                        ax.add_patch(rect)
            else:
                for ytick, xtick in np.ndindex(pivot_df.shape):
                    value = pivot_df.iat[ytick, xtick]
                    aux_value = aux_pivot_df.iat[ytick, xtick]
                    # print(qcs[xtick], ds[ytick], value)
                    # target = E * 256/ds[ytick] * qcs[xtick] / 21
                    if (32 >= value) and (aux_value <= aux_th):
                    # if (1.09 > value) and (aux_value <= aux_th):
                        rect = patches.Rectangle(
                            (xtick+0.05, ytick+0.05), 1-0.05*2, 1-0.05*2,
                            linewidth=2, edgecolor=markers_palette[i],
                            alpha=0.25,
                            hatch=hatches[i],
                            facecolor='none')
                        ax.add_patch(rect)

    # Create a colorbar in the last subplot
    sm = plt.cm.ScalarMappable(
        cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm,
                        # left, bottom, width, height
                        cax=fig.add_axes([0.904, 0.114, 0.012, 0.767]),
                        orientation='vertical')
    cbar.set_label('Staleness', labelpad=-52)
    # cbar.set_label('Latency (x7164)', labelpad=-48.5)
    if aux_index != None:
        sm = plt.cm.ScalarMappable(
            cmap=aux_cmap, norm=plt.Normalize(vmin=aux_vmin, vmax=aux_vmax))
        sm.set_array([])
        aux_cbar = fig.colorbar(sm,
                                # left, bottom, width, height
                                cax=fig.add_axes([0.954, 0.114, 0.012, 0.767]),
                                orientation='vertical')
        aux_cbar.set_label('# of Timed out', labelpad=-51.5)

    # plt.tight_layout()
    # plt.show()
    plt.savefig(f"./save_heatmap/heatmaps_{op}_{title}.png",
                bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    # # TPS
    # draw(-4, op="max", title="tps")
    # draw(-4, op="min", title="tps")
    draw(-4, op="avg", title="tps")
    # draw(-4, op="med", title="tps")

    # # Timeout
    # draw(3, op="max", title=imeout")
    # draw(3, op="min", title=imeout")
    draw(3, op="avg", title="timeout")
    # draw(3, op="med", title=imeout")

    # Latency
    # draw(-2, op="max", title="latency")
    # draw(-2, op="min", title="latency")
    draw(-2, op="avg", title="latency", highlight=False)
    draw(-2, op="avg", title="latency_highlight", highlight=True)
    draw(-2, aux_index=3, op="avg", title="latency_timeout", highlight=False)
    draw(-2, aux_index=3, op="avg",
         title="latency_timeout_highlight", highlight=True)
    # draw(-2, op="med", title="latency")

    # Stale
    # draw(-1, op="max", title="stale")
    # draw(-1, op="min", title="stale")
    draw(-1, op="avg", title="stale", highlight=False)
    draw(-1, op="avg", title="stale_highlight", highlight=True)
    draw(-1, aux_index=3, op="avg", title="stale_timeout", highlight=False)
    draw(-1, aux_index=3, op="avg",
         title="stale_timeout_highlight", highlight=True)
    # draw(-1, op="med", title="stale")
