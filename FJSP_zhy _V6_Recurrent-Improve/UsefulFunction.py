import colorsys
import random


def check_circle(G):
    node_set = set()
    r = len(G)
    have_in_zero = True
    while have_in_zero:
        have_in_zero = False
        for i in range(r):
            if i not in node_set and not any([row[i] for row in G]):
                node_set.add(i)
                G[i] = [0] * r
                have_in_zero = True
                break
    return 0 if len(node_set)==r else -1


def encoding_color_list2str(color_list):
    color_str = '#'
    for each_color in color_list:
        color_str += ('0'*(4-len(hex(each_color)))+hex(each_color)[2:])
    return color_str


def get_color(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors

    def ncolors(num):
        rgb_colors = []
        if num < 1:
            return rgb_colors
        hls_colors = get_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
            rgb_colors.append([r, g, b])

        return rgb_colors

    return ncolors(num)

