import numpy as np
from webcolors._definitions import _CSS21_NAMES_TO_HEX
from webcolors import hex_to_rgb, rgb_to_name, hex_to_name

def closest_color(color_vals):
    color_dists = {}

    # get closest CSS21 color based on RGB values
    for name, key in _CSS21_NAMES_TO_HEX.items():
        exact_vals = [val for val in hex_to_rgb(key)]
        dist = np.linalg.norm(np.array(exact_vals) - np.array(color_vals))
        color_dists[dist] = name

    return color_dists[min(color_dists.keys())]


# return RGB float values or -1 if invalid
def get_color_vals(row):
    color_vals = []
    for idx in range(1, 4, 1):
        r = row['object_color_r' + str(idx)]
        g = row['object_color_g' + str(idx)]
        b = row['object_color_b' + str(idx)]
        if '_' in [r, g, b] or None in [r, g, b] or '' in [r, g, b]:
            color_vals.append([-1, -1, -1])
        else:
            color_vals.append([float(c) for c in [r, g, b]])

    return color_vals


# TODO: replace with updated color schema used during preprocessing
def get_color_labels_old(color_vals):
    colors = []
    for rgb in color_vals:
        if -1.0 in rgb:
            colors.append("N/A")
            continue

        # convert values to RGB range
        for i in range(len(rgb)):
            rgb[i] = int(rgb[i]*255)

            try:
                # match color based on exact RGB values
                color = rgb_to_name((rgb[0], rgb[1], rgb[2]), spec='css21')
            except ValueError:
                # otherwise, get closest color based on RGB values
                color = closest_color(rgb)

        colors.append(color)

    return colors

def get_color_labels(row):
    color_labels = []
    for i in range(1, 4):
        if row[f'object_color_scheme{i}'] in ['_', '', None]:
            color_labels.append('N/A')
        else:
            color_labels.append(row[f'object_color_scheme{i}'])
    return color_labels

def get_color_percentages(row):
    color_percentages = []
    for i in range(1, 4):
        if row[f'object_color_scheme_percentage{i}'] in ['_', '', None]:
            color_percentages.append('N/A')
        else:
            color_percentages.append(row[f'object_color_scheme_percentage{i}'])
    return color_percentages
