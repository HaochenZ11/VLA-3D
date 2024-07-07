import numpy as np
from collections import Counter
import matplotlib
import webcolors


color_scheme = {
    'lightsalmon': 'red', 'salmon': 'red', 'darksalmon': 'red', 'lightcoral': 'red', 'indianred': 'maroon', 'crimson': 'red', 'firebrick': 'maroon', 'red': 'red', 'darkred': 'maroon', \
    'coral': 'orange', 'tomato': 'orange', 'orangered': 'orange', 'gold': 'yellow', 'orange': 'orange', 'darkorange': 'orange', \
    'lightyellow': 'yellow', 'lemonchiffon': 'yellow', 'lightgoldenrodyellow': 'yellow', 'papayawhip': 'yellow', 'moccasin': 'yellow', 'peachpuff': 'yellow', \
    'palegoldenrod': 'yellow', 'khaki': 'yellow', 'darkkhaki': 'yellow', 'yellow': 'yellow', \
    'lawngreen': 'green', 'chartreuse': 'green', 'limegreen': 'green', 'lime': 'green', 'forestgreen': 'green', 'green': 'green', 'darkgreen': 'green', 'greenyellow': 'green', 'yellowgreen': 'green', \
    'springgreen': 'green', 'mediumspringgreen': 'green', 'lightgreen': 'green', 'palegreen': 'green', 'darkseagreen': 'green', 'mediumseagreen': 'green', 'seagreen': 'green', 'olive': 'olive', 'darkolivegreen': 'olive', 'olivedrab': 'olive', \
    'lightcyan': 'aqua', 'cyan': 'aqua', 'aqua': 'aqua', 'aquamarine': 'aqua', 'mediumaquamarine': 'aqua', 'paleturquoise': 'aqua', 'turquoise': 'aqua', \
    'mediumturquoise': 'aqua', 'darkturquoise': 'aqua', 'lightseagreen': 'aqua', 'cadetblue': 'aqua', 'darkcyan': 'aqua', 'teal': 'aqua', \
    'powderblue': 'blue', 'lightblue': 'blue', 'lightskyblue': 'blue', 'skyblue': 'blue', 'deepskyblue': 'blue', 'lightsteelblue': 'blue', 'dodgerblue': 'blue', 'cornflowerblue': 'blue', 'steelblue': 'blue', \
    'royalblue': 'blue', 'blue': 'blue', 'mediumblue': 'blue', 'darkblue': 'navy', 'navy': 'navy', 'midnightblue': 'navy', \
    'mediumslateblue': 'purple', 'slateblue': 'purple', 'darkslateblue': 'purple', 'lavender': 'purple', 'thistle': 'purple', 'plum': 'purple', 'violet': 'purple', \
    'orchid': 'purple', 'fuchsia': 'pink', 'magenta': 'pink', 'mediumorchid': 'purple', 'mediumpurple': 'purple', 'blueviolet': 'purple', 'darkviolet': 'purple', 'darkorchid': 'purple', 'darkmagenta': 'purple', 'purple': 'purple', 'indigo': 'purple', \
    'pink': 'pink', 'lightpink': 'pink', 'hotpink': 'pink', 'deeppink': 'pink', 'palevioletred': 'pink', 'mediumvioletred': 'pink', \
    'white': 'white', 'snow': 'white', 'honeydew': 'white', 'mintcream': 'white', 'azure': 'white', 'aliceblue': 'white', 'ghostwhite': 'white', 'whitesmoke': 'white', 'seashell': 'white', \
    'beige': 'white', 'oldlace': 'white', 'floralwhite': 'white', 'ivory': 'white', 'antiquewhite': 'white', 'linen': 'white', 'lavenderblush': 'white', 'mistyrose': 'white', \
    'gainsboro': 'gray', 'lightgray': 'gray', 'silver': 'gray', 'darkgray': 'gray', 'gray': 'gray', 'dimgray': 'gray', 'lightslategray': 'gray', 'slategray': 'gray', 'darkslategray': 'aqua', \
    'black': 'black', \
    'cornsilk': 'yellow', 'blanchedalmond': 'brown', 'bisque': 'brown', 'navajowhite': 'brown', 'wheat': 'brown', 'burlywood': 'brown', 'tan': 'brown', 'rosybrown': 'purple', 'sandybrown': 'brown', \
    'goldenrod': 'yellow', 'darkgoldenrod': 'yellow', 'peru': 'brown', 'chocolate': 'orange', 'saddlebrown': 'brown', 'sienna': 'brown', 'brown': 'maroon', 'maroon': 'maroon'
    }
                

def rgb2lab(color_rgb):
    def func(t):
        if (t > 0.008856):
            return np.power(t, 1/3.0)
        else:
            return 7.787 * t + 16 / 116.0

    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]]

    cie_colors = np.dot(matrix, np.transpose(color_rgb))
    lab_colors = []

    for i in range(cie_colors.shape[1]):
        cie = cie_colors[:, i]
        cie[0] = cie[0] /0.950456
        cie[2] = cie[2] /1.088754 

        # Calculate the L
        L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]

        # Calculate the a 
        a = 500*(func(cie[0]) - func(cie[1]))

        # Calculate the b
        b = 200*(func(cie[1]) - func(cie[2]))

        #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
        Lab = [b , a, L]

        lab_colors.append(Lab)
    
    return np.array(lab_colors)

def generate_color_anchors(standard):
    if standard == 'html4':
        mapping = webcolors._definitions._HTML4_HEX_TO_NAMES.items()
    elif standard == 'css2':
        mapping = webcolors._definitions._CSS2_HEX_TO_NAMES.items()
    elif standard == 'css21':
        mapping = webcolors._definitions._CSS21_HEX_TO_NAMES.items()
    elif standard == 'css3':
        mapping = webcolors._definitions._CSS3_HEX_TO_NAMES.items()
    else:
        raise ValueError('Not a supported color standard.')
    anchor_colors_array = []
    anchor_colors_array_hsv = []
    anchor_colors_name = []
    for key, name in mapping:
        r, g, b = webcolors.hex_to_rgb(key)
        anchor_colors_array.append([r, g, b])
        anchor_colors_name.append(name)

    anchor_colors_array = np.array(anchor_colors_array)
    anchor_colors_array_hsv = rgb2lab(anchor_colors_array/255)
    return anchor_colors_array, anchor_colors_array_hsv, anchor_colors_name


def judge_color(rgb_colors, tree, anchor_colors_array, anchor_colors_name):
    lab_colors = rgb2lab(rgb_colors)
    distances, indexes = tree.query(lab_colors)
    colors = np.array(anchor_colors_name)[indexes]
    color_count = Counter(indexes)
    color_sche = list(map(lambda x: color_scheme[x], colors))
    most_3 = Counter(color_sche).most_common(3)
    color_3 = []
    for index_pair in most_3:
        if index_pair[1] >= 0.1*len(indexes):
            for c in sorted(color_count.items(), key=lambda s: (-s[1])):
                if color_scheme[anchor_colors_name[c[0]]] == index_pair[0]:
                    color_most = anchor_colors_array[c[0], :]
                    break
            color_3 += list(color_most)
            percentage = index_pair[1]/len(indexes)
            i = np.where(np.array(color_sche) == index_pair[0])
            average_dist = np.average(distances[i])
            color_3 += [index_pair[0], percentage, average_dist]
        else:
            color_3 += ['_', '_', '_', '_', '_', '_']
    return color_3
