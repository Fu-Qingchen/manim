'''
这个文件中主要处理了和贝塞尔曲线、插值有关的函数
'''

from scipy import linalg
import numpy as np

from manimlib.utils.simple_functions import choose

CLOSED_THRESHOLD = 0.001


def bezier(points):
    '''
    返回由点集(锚点，控制点)确定的参数方程
    贝塞尔曲线的次数由points中点的个数确定
    '''
    n = len(points) - 1
    return lambda t: sum([
        ((1 - t)**(n - k)) * (t**k) * choose(n, k) * point
        for k, point in enumerate(points)
    ])


def partial_bezier_points(points, a, b):
    """
    Given an array of points which define
    a bezier curve, and two numbers 0<=a<b<=1,
    return an array of the same size, which
    describes the portion of the original bezier
    curve on the interval [a, b].
    输入贝塞尔曲线的点数组和两个0,1之间的数a,b
    返回一个大小相同的数组
    该数组描述原始贝塞尔曲线在间隔[a,b]上的部分

    This algorithm is pretty nifty, and pretty dense.
    这个算法很漂亮，也很复杂
    """
    if a == 1:
        return [points[-1]] * len(points)

    a_to_1 = np.array([
        bezier(points[i:])(a)
        for i in range(len(points))
    ])
    end_prop = (b - a) / (1. - a)
    return np.array([
        bezier(a_to_1[:i + 1])(end_prop)
        for i in range(len(points))
    ])


# Linear interpolation variants

def interpolate(start, end, alpha):
    '''
    线性插值
    '''
    return (1 - alpha) * start + alpha * end


def integer_interpolate(start, end, alpha):
    """
    alpha is a float between 0 and 1.  This returns
    an integer between start and end (inclusive) representing
    appropriate interpolation between them, along with a
    "residue" representing a new proportion between the
    returned integer and the next one of the
    list.
    整数插值
    返回两个数，第一个为插值结果(整数)，第二个为和线性插值相差的小数部分

    For example, if start=0, end=10, alpha=0.46, This
    would return (4, 0.6).
    """
    if alpha >= 1:
        return (end - 1, 1.0)
    if alpha <= 0:
        return (start, 0)
    value = int(interpolate(start, end, alpha))
    residue = ((end - start) * alpha) % 1
    return (value, residue)


def mid(start, end):
    '''
    返回两个数的平均值
    '''
    return (start + end) / 2.0


def inverse_interpolate(start, end, value):
    '''
    由插值的结果value，返回alpha
    '''
    return np.true_divide(value - start, end - start)


def match_interpolate(new_start, new_end, old_start, old_end, old_value):
    '''
    匹配插值
    给出原插值范围old_start,old_end和结果old_value
    返回以相同比例，插值范围为new_start,new_end的插值结果
    '''
    return interpolate(
        new_start, new_end,
        inverse_interpolate(old_start, old_end, old_value)
    )


# Figuring out which bezier curves most smoothly connect a sequence of points


def get_smooth_handle_points(points):
    '''
    给出一系列锚点points
    返回经过points的平滑贝塞尔曲线的一系列控制点
    '''
    points = np.array(points)
    num_handles = len(points) - 1
    dim = points.shape[1]
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))
    # Must solve 2*num_handles equations to get the handles.
    # l and u are the number of lower an upper diagonal rows
    # in the matrix to solve.
    l, u = 2, 1
    # diag is a representation of the matrix in diagonal form
    # See https://www.particleincell.com/2012/bezier-splines/
    # for how to arive at these equations
    diag = np.zeros((l + u + 1, 2 * num_handles))
    diag[0, 1::2] = -1
    diag[0, 2::2] = 1
    diag[1, 0::2] = 2
    diag[1, 1::2] = 1
    diag[2, 1:-2:2] = -2
    diag[3, 0:-3:2] = 1
    # last
    diag[2, -2] = -1
    diag[1, -1] = 2
    # This is the b as in Ax = b, where we are solving for x,
    # and A is represented using diag.  However, think of entries
    # to x and b as being points in space, not numbers
    b = np.zeros((2 * num_handles, dim))
    b[1::2] = 2 * points[1:]
    b[0] = points[0]
    b[-1] = points[-1]

    def solve_func(b):
        return linalg.solve_banded((l, u), diag, b)
    use_closed_solve_function = is_closed(points)
    if use_closed_solve_function:
        # Get equations to relate first and last points
        matrix = diag_to_matrix((l, u), diag)
        # last row handles second derivative
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        # first row handles first derivative
        matrix[0, :] = np.zeros(matrix.shape[1])
        matrix[0, [0, -1]] = [1, 1]
        b[0] = 2 * points[0]
        b[-1] = np.zeros(dim)

        def closed_curve_solve_func(b):
            return linalg.solve(matrix, b)

    handle_pairs = np.zeros((2 * num_handles, dim))
    for i in range(dim):
        if use_closed_solve_function:
            handle_pairs[:, i] = closed_curve_solve_func(b[:, i])
        else:
            handle_pairs[:, i] = solve_func(b[:, i])
    return handle_pairs[0::2], handle_pairs[1::2]


def diag_to_matrix(l_and_u, diag):
    """
    Converts array whose rows represent diagonal
    entries of a matrix into the matrix itself.
    See scipy.linalg.solve_banded
    LDU分解还原
    """
    l, u = l_and_u
    dim = diag.shape[1]
    matrix = np.zeros((dim, dim))
    for i in range(l + u + 1):
        np.fill_diagonal(
            matrix[max(0, i - u):, max(0, u - i):],
            diag[i, max(0, u - i):]
        )
    return matrix


def is_closed(points):
    '''
    检查曲线是否闭合(首尾锚点重合)
    '''
    return np.allclose(points[0], points[-1])
