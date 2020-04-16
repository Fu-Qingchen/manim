'''
这个文件中主要处理了颜色
'''

import random

from colour import Color
import numpy as np

from manimlib.constants import PALETTE
from manimlib.constants import WHITE
from manimlib.utils.bezier import interpolate
from manimlib.utils.simple_functions import clip_in_place
from manimlib.utils.space_ops import normalize


def color_to_rgb(color):
    '''
    将颜色转换为RGB值
    color可以为字符串(例"#66CCFF")，也可以为Color类
    '''
    if isinstance(color, str):
        return hex_to_rgb(color)
    elif isinstance(color, Color):
        return np.array(color.get_rgb())
    else:
        raise Exception("Invalid color type")


def color_to_rgba(color, alpha=1):
    '''
    将颜色转换为RGB加上alpha透明度
    color可以为字符串(例"#66CCFF")，也可以为Color类
    '''
    return np.array([*color_to_rgb(color), alpha])


def rgb_to_color(rgb):
    '''
    将RGB颜色转换为Color类
    '''
    try:
        return Color(rgb=rgb)
    except:
        return Color(WHITE)


def rgba_to_color(rgba):
    '''
    将RGBA前三个数RGB转换为Color类
    '''
    return rgb_to_color(rgba[:3])


def rgb_to_hex(rgb):
    '''
    将RGB转换为十六进制字符串表示
    '''
    return "#" + "".join('%02x' % int(255 * x) for x in rgb)


def hex_to_rgb(hex_code):
    '''
    将十六进制字符串转换为RGB
    '''
    hex_part = hex_code[1:]
    if len(hex_part) == 3:
        "".join([2 * c for c in hex_part])
    return np.array([
        int(hex_part[i:i + 2], 16) / 255
        for i in range(0, 6, 2)
    ])


def invert_color(color):
    '''
    返回color的反色
    '''
    return rgb_to_color(1.0 - color_to_rgb(color))


def color_to_int_rgb(color):
    '''
    将颜色转化为整数RGB
    '''
    return (255 * color_to_rgb(color)).astype('uint8')


def color_to_int_rgba(color, opacity=1.0):
    '''
    将颜色转化为整数RGBA
    '''
    alpha = int(255 * opacity)
    return np.append(color_to_int_rgb(color), alpha)


def color_gradient(reference_colors, length_of_output):
    '''
    返回长度为length_of_output的颜色梯度数组
    '''
    if length_of_output == 0:
        return reference_colors[0]
    rgbs = list(map(color_to_rgb, reference_colors))
    alphas = np.linspace(0, (len(rgbs) - 1), length_of_output)
    floors = alphas.astype('int')
    alphas_mod1 = alphas % 1
    # End edge case
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [
        rgb_to_color(interpolate(rgbs[i], rgbs[i + 1], alpha))
        for i, alpha in zip(floors, alphas_mod1)
    ]


def interpolate_color(color1, color2, alpha):
    '''
    在color1和color2之间插值，返回Color类表示的颜色
    '''
    rgb = interpolate(color_to_rgb(color1), color_to_rgb(color2), alpha)
    return rgb_to_color(rgb)


def average_color(*colors):
    '''
    返回colors的平均颜色
    '''
    rgbs = np.array(list(map(color_to_rgb, colors)))
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color():
    '''
    随机亮色
    '''
    color = random_color()
    curr_rgb = color_to_rgb(color)
    new_rgb = interpolate(
        curr_rgb, np.ones(len(curr_rgb)), 0.5
    )
    return Color(rgb=new_rgb)


def random_color():
    '''
    随机颜色
    '''
    return random.choice(PALETTE)


def get_shaded_rgb(rgb, point, unit_normal_vect, light_source):
    '''
    获取从光源light_source到point着色的RGB
    '''
    to_sun = normalize(light_source - point)
    factor = 0.5 * np.dot(unit_normal_vect, to_sun)**3
    if factor < 0:
        factor *= 0.5
    result = rgb + factor
    clip_in_place(rgb + factor, 0, 1)
    return result
