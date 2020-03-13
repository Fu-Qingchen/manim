'''
  > File Name        : daily_try
  > Author           : Fu_Qingchen
  > Creating Time    : 2020-02-04
'''

from manimlib.imports import *
from from_kindergarten.imports import *

class Polygon0204(Scene):
    '''
    由正三角形到正n边形变化的动画
    '''
    def construct(self):
        tracker = ValueTracker(3)
        vg = Integer(3).scale(2).add_updater(lambda n:n.set_value(tracker.get_value()).center())
        polygon = RegularPolygon(int(tracker.get_value())).scale(2).add_updater(
            lambda n: n.become(RegularPolygon(int(tracker.get_value())).scale(2))
            )
        self.add(tracker, vg, polygon)
        self.play(tracker.set_value, 50, rate_func=smooth, run_time = 5)


class MinGroupGraph0205(Scene):
    '''
    最小集标准形式的解释动画
    '''
    CONFIG = {
        "circle_color": [RED, GREEN, BLUE],
        "circle_label": ['A', 'B', 'C']
    }
    def construct(self):
        rect = Rectangle(height=9, width=16).shift(DOWN*0.5).set_height(1.5*FRAME_Y_RADIUS)
        circles = VGroup(*[
            Circle().scale(2).set_color(self.circle_color[i]).shift(RIGHT).rotate_about_origin(TAU/3*i)
            for i in range(3)
            ]).rotate(PI/6).scale(0.8).shift(rect.get_center() + UP*0.6)
        circle_labels = VGroup(*[
            TextMobject(self.circle_label[i]).set_color(self.circle_color[i]) 
            for i in range(3)
            ]).arrange(RIGHT, buff = 3).next_to(rect, UP)

        labels_ABC = TexMobject('A', 'B', 'C')
        labels_ABC_bin = TexMobject("1", "1", "1", buff = 0.5)
        label_M111 = TexMobject("{}_".format("M"), "{111}")

        labels_BC = TexMobject('\\bar A', 'B', 'C').shift(LEFT, DOWN*0.8)
        labels_BC_bin = TexMobject("0", "1", "1", buff = 0.5).shift(LEFT, DOWN*0.8)
        label_M011 = TexMobject("{}_".format("M"), "{011}").shift(LEFT, DOWN*0.8)

        self.play(ShowCreation(rect))
        self.play(ShowCreation(circles), Write(*[circle_labels]))

        self.play(TransformFromCopy(circle_labels, labels_ABC))
        self.play(TransformFromCopy(circle_labels, labels_BC))

        self.play(ReplacementTransform(labels_ABC, labels_ABC_bin))
        self.play(ReplacementTransform(labels_BC, labels_BC_bin))

        self.play(FadeInFrom(label_M111[0], UP), ReplacementTransform(labels_ABC_bin, label_M111[1]))
        self.play(FadeInFrom(label_M011[0], UP), ReplacementTransform(labels_BC_bin, label_M011[1]))

        self.wait()


class VectorField0206(Scene):
    '''
    真空静止点电荷电场图
    点电荷的场强公式:    E = KQ/r²
    '''
    CONFIG = {
        "q1_position": np.array([0.001, 1.43, 0]),
        "q2_position": np.array([0.001, -1.43, 0]),
        "q1": 1,
        "q2": 1,
        "stream_line_animation_config": {
            "line_anim_class": ShowPassingFlash,
        },
        "stream_lines_config": {
            "virtual_time": 3,
            "cutoff_norm": 2
        }
    }

    def construct(self):
        grid = NumberPlane()
        grid.add(grid.get_axis_labels())
        # vectors = VGroup(*[
        #     Vector(self.get_direction(x + 0.01, y +0.01)).shift(x*RIGHT + y*UP).set_color(YELLOW)
        #     for x, y in it.product(np.arange(-7, 7, 0.5), np.arange(-4, 4, 0.5))
        # ])
        # vectors_change_len = VGroup(*[
        #     Vector(self.get_length(x + 0.01, y + 0.01)).shift(x*RIGHT + y*UP).set_color(YELLOW)
        #     for x, y in it.product(np.arange(-7, 7, 0.5), np.arange(-4, 4, 0.5))
        # ])
        vector_field = VectorField(lambda n: self.get_length(n[0], n[1]))
        stream_lines = StreamLines(lambda n: self.get_length(n[0], n[1]), **self.stream_lines_config)
        self.add(grid)
        # self.play(ShowCreation(vectors))
        self.play(
            ShowCreation(Circle().scale(0.25).shift(self.q1_position)), 
            Write(TextMobject("+").shift(self.q1_position).set_color(YELLOW)),
            ShowCreation(Circle().scale(0.25).shift(self.q2_position)), 
            Write(TextMobject("-").shift(self.q1_position).set_color(YELLOW)))
        # self.play(ApplyMethod(vectors.become, vectors_change_len))
        # self.play(ReplacementTransform(vectors, vector_field))
        self.play(ShowCreation(vector_field))
        # self.play(LaggedStartMap(ShowPassingFlashWithThinningStrokeWidth, stream_lines))
        self.begin_flow(stream_lines)
        self.wait(10)

    def begin_flow(self, stream_lines):
        stream_line_animation = AnimatedStreamLines(stream_lines, **self.stream_line_animation_config)
        self.add(stream_line_animation)

    def get_direction(self, column, row):
        '''
        INPUT:  向量位置
        OUTPUT: 向量的方向
        '''
        length1 = ((column-self.q1_position[0])**2 + (row-self.q1_position[1])**2)**0.5 * 2
        length2 = ((column-self.q2_position[0])**2 + (row-self.q2_position[1])**2)**0.5 * 2
        e1 = np.array([(column-self.q1_position[0])/length1, (row-self.q1_position[1])/length1, 0])*self.q1
        e2 = np.array([(column-self.q2_position[0])/length2, (row-self.q2_position[1])/length2, 0])*self.q2
        return e1+e2

    def get_length(self, column, row):
        '''
        INPUT:              向量位置
        OUTPUT:             向量的长度
        '''
        r1 = ((column-self.q1_position[0])**2 + (row-self.q1_position[1])**2)**0.5
        r2 = ((column-self.q2_position[0])**2 + (row-self.q2_position[1])**2)**0.5
        e1 = np.array([(column-self.q1_position[0])/r1*(1/(r1**2)*10), (row-self.q1_position[1])/r1*(1/(r1**2)*10), 0])*self.q1
        e2 = np.array([(column-self.q2_position[0])/r2*(1/(r2**2)*10), (row-self.q2_position[1])/r2*(1/(r2**2)*10), 0])*self.q2
        return e1 +e2


class VectorField3D0208(ThreeDScene):
    CONFIG = {
        "q1_position": np.array([2.51, 0.001, 0]),
        "q2_position": np.array([-2.51, 0.001, 0]),
        "q1": 1,
        "q2": -1,
    }
    def construct(self):
        three_d_axes = ThreeDAxes()
        three_d_vector_field = ThreeDVectorField(
            lambda n:self.get_length(n[0], n[1], n[2])
        )
        
        self.set_camera_orientation(phi = 80 * DEGREES, theta = -45 * DEGREES)
        self.play(ShowCreation(three_d_axes), ShowCreation(three_d_vector_field))
        self.move_camera(phi = 0 * DEGREES, theta = 0 * DEGREES, run_time = 5)

    def get_length(self, x, y, z):
        '''
        INPUT:              向量位置
        OUTPUT:             向量的长度
        '''
        r1 = ((x-self.q1_position[0])**2 + (y-self.q1_position[1])**2 + (z-self.q1_position[2])**2)**0.5
        r2 = ((x-self.q2_position[0])**2 + (y-self.q2_position[1])**2 + (z-self.q2_position[2])**2)**0.5
        e1 = np.array([
            (x-self.q1_position[0])/r1*(1/(r1**2)*10), (y-self.q1_position[1])/r1*(1/(r1**2)*10), (z-self.q1_position[2])/r1*(1/(r1**2)*10)
            ])*self.q1
        e2 = np.array([
            (x-self.q2_position[0])/r2*(1/(r2**2)*10), (y-self.q2_position[1])/r2*(1/(r2**2)*10), (z-self.q2_position[2])/r2*(1/(r2**2)*10)
            ])*self.q2
        return e1 +e2


class LorenzAttractor0208(ThreeDScene):
    '''
    绘制洛伦兹吸引子曲线
    函数：
    dx/dt = \sigma*(y-x)
    dy/dt = x*(R - z) - y
    dz/dt = xy - \beta*z
    '''
    CONFIG = {
        "sigma": 10, 
        "R": 28, 
        "beta": 8/3, 
        "position": [1, 1, 1],
        "delta_t": 0.001,
        "length": 50000,
        "three_d_axes_config": {
            "x_min": -5.5,
            "x_max": 5.5,
            "y_min": -5.5,
            "y_max": 5.5,
            "z_min": -3.5,
            "z_max": 3.5,
        }
    }
    def construct(self):
        self.set_camera_orientation(distance = 1000)
        three_d_axes = ThreeDAxes(**self.three_d_axes_config).scale(10)

        formula = TexMobject(
            "&{{\\partial ", "x", "}\\over{\\partial t}} = \\sigma(", "y", "-", "x", ")\\\\",
            "&{{\\partial ", "y", "}\\over{\\partial t}} = ", "x", "(R - ", "z", ") - ", "y", "\\\\"
            "&{{\\partial ", "z", "}\\over{\\partial t}} = ", "x", "", "y", " - \\beta ", "z", "\\\\"
            ).set_color_by_tex_to_color_map({"x": RED, "y": GREEN, "z":BLUE}).scale(10)

        self.x, self.y, self.z = self.position
        dots = VGroup(*[
            Dot(self.get_value()).set_color(YELLOW).scale(1.2)
            for i in range(self.length)
        ])

        self.play(Write(formula))
        self.play(formula.scale, 0.8, formula.shift, UL*30, ShowCreation(three_d_axes))

        self.add(three_d_axes.get_axis_labels())
        self.move_camera(phi=45 * DEGREES, theta=-45*DEGREES, frame_center = np.array([0, 0, 30]), run_time = 3)
        self.begin_ambient_camera_rotation()
        self.play(ShowCreation(dots), rate_func=linear, run_time = 15)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi= 85 * DEGREES,theta= -45 * DEGREES, run_time = 5)
        self.wait()

    def get_value(self):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.R - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        self.x += dx * self.delta_t
        self.y += dy * self.delta_t
        self.z += dz * self.delta_t
        return np.array([self.x, self.y, self.z])


class ShowFunction0211(ThreeDScene):
    '''
    显示函数曲线方程
    f(x,t) = sin(1.5*x)*e^{-0.15t}
    '''
    CONFIG = {
        "u_min": 0,
        "u_max": 4*PI,
        "v_min": 0,
        "v_max": 4*PI,
    }
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi = 80 * DEGREES, theta = -45 * DEGREES, distance = 1000)
        self.get_objects()
        self.update_objects()
        self.add(axes)
        self.begin_ambient_camera_rotation()
        self.play(Write(self.surface), run_time = 3)
        self.play(Write(self.plane))
        self.play(Write(self.line))
        self.play(ApplyMethod(self.timer.increment_value, PI, run_time = 3))
        self.stop_ambient_camera_rotation()
        self.move_camera(phi = 90 * DEGREES, theta = 90 * DEGREES, run_time = 3)
        self.wait()

    def get_objects(self):
        self.timer = ValueTracker(-PI)
        self.plane = Rectangle(width = 2*PI, height = 6, shade_in_3d=True, fill_opacity=0.2).rotate(PI/2, X_AXIS).shift(Y_AXIS*-PI)
        self.surface = ParametricSurface(lambda x, y: np.array([
            x, 
            y, 
            np.sin(1.5*x)*np.e**(-0.15*y)
        ]), u_min = -PI, u_max = PI, v_min = -PI, v_max = PI, fill_opacity = 0.1)
        self.line = ParametricFunction(lambda x:np.array([
            x, 
            -PI, 
            np.sin(1.5*x)*np.e**(-0.15*-PI)
        ]), t_min = -PI, t_max = PI, shade_in_3d = True).set_color(YELLOW)

    def update_objects(self):
        self.plane.add_updater(lambda n:n.move_to(Y_AXIS*self.timer.get_value()))
        self.line.add_updater(lambda n:n.become(ParametricFunction(lambda x:np.array([
            x, 
            self.timer.get_value(), 
            np.sin(1.5*x)*np.e**(-0.15*self.timer.get_value())
        ]), t_min = -PI, t_max = PI, shade_in_3d = True).set_color(YELLOW)))


class Programing_process(Scene):

    def construct(self):

        s = 0.25
        to_corner_loc = LEFT * 2 + UP * 0.1
        MAGENTA = '#6A6CCB'

        bg_rect = Rectangle(fill_color=DARK_GRAY, fill_opacity=0.275).scale(20)
        self.add(bg_rect)
        text_import = Text('import numpy as np\nimport matplotlib.pyplot as plt',
                           font='Consolas').scale(s).to_corner(to_corner_loc)
        text_import.set_color_by_t2c({'import':ORANGE, 'as': ORANGE})

        t0 = Text('import', font='Consolas').scale(s)
        h = text_import.get_height() - t0.get_height()
        w = t0.get_width()/len(t0)
        text_func = Text('iter_func = lambda z, c: (z ** 2 + c) # iteration function', font='Consolas').scale(s).to_corner(to_corner_loc).shift(DOWN * 3 * h)
        text_func.set_color_by_t2c({'lambda': ORANGE, '2': BLUE, '# iteration function': GRAY, ',': ORANGE})

        text_setvalue_01 = Text('def set_value(c, max_iter_num=128):', font='Consolas').scale(s).to_corner(to_corner_loc).shift(DOWN * 5 * h)
        text_setvalue_01.set_color_by_t2c({'def': ORANGE, 'set_value': GOLD_C, '128': BLUE, ',': ORANGE})
        text_setvalue_02 = Text('    z = complex(0, 0) # initial value of z\n'
                                '    num = 0\n'
                                '    while abs(z) < 2 and num < max_iter_num:\n'
                                '        z = iter_func(z, c)\n'
                                '        num += 1\n'
                                '    return num', font='Consolas')\
            .scale(s).to_corner(to_corner_loc).shift(DOWN * 6 * h + 4 * w * RIGHT)

        text_setvalue_02.set_color_by_t2c({'complex': MAGENTA, '0': BLUE, '# initial value of z': GRAY, 'abs':MAGENTA,
                                           'while': ORANGE, 'and': ORANGE, '1': BLUE, '2': BLUE, 'return': ORANGE})
        text_setvalue = VGroup(text_setvalue_01, text_setvalue_02)

        text_showfunc_01 = Text('def display_mandelbrot(x_num=1000, y_num=1000):', font='Consolas')\
            .scale(s).to_corner(to_corner_loc).shift(DOWN * 13 * h)
        text_showfunc_01.set_color_by_t2c({'def': ORANGE, 'display_mandelbrot': GOLD_C, '1': BLUE, '0':BLUE, ',': ORANGE})

        text_showfunc_02 = Text('    X, Y = np.meshgrid(np.linspace(-2, 2, x_num+1), np.linspace(-2, 2, y_num+1))\n'
                                '    C = X + Y * 1j\n'
                                '    result = np.zeros((y_num+1, x_num+1))', font='Consolas')\
            .scale(s).to_corner(to_corner_loc).shift(DOWN * 15 * h + 4 * w * RIGHT)
        text_showfunc_02.set_color_by_t2c({'1': BLUE, '2':BLUE, 'j': BLUE, ',': ORANGE})
        text_showfunc_03 = Text('    for i in range(y_num+1):\n'
                                '        for j in range(x_num+1):\n'
                                '            result[i, j] = set_value(C[i, j])', font='Consolas')\
            .scale(s).to_corner(to_corner_loc).shift(DOWN * 19 * h + 4 * w * RIGHT)

        text_showfunc_03.set_color_by_t2c({'for': ORANGE, 'in': ORANGE, 'range': MAGENTA, '1': BLUE, ',': ORANGE})

        text_showfunc_04 = Text('plt.imshow(result, interpolation="bilinear", cmap=plt.cm.hot,\n'
                                '           vmax=abs(result).max(), vmin=abs(result).min(),\n'
                                '           extent=[-2, 2, -2, 2])\n'
                                'plt.show()', font='Consolas')\
            .scale(s).to_corner(to_corner_loc).shift(DOWN * 23 * h + 4 * w * RIGHT)
        text_showfunc_04.set_color_by_t2c({'interpolation':RED, '"bilinear"': GREEN_D, 'cmap': RED,
                                           'vmax':RED, 'vmin':RED, 'extent':RED, 'abs': MAGENTA, ',': ORANGE})

        text_showfunc = VGroup(text_showfunc_01, text_showfunc_02, text_showfunc_03, text_showfunc_04)

        text_main = Text('if __name__ == "__main__":\n'
                         '    \n'
                         # '    iter_num = 200 # maximum iteration num\n'
                         '    display_mandelbrot(2000, 2000)', font='Consolas')\
            .scale(s).to_corner(to_corner_loc).shift(DOWN * 28 * h)
        text_main.set_color_by_t2c({'if': ORANGE, '"__main__"': GREEN_D, '2': BLUE, '0': BLUE,
                                    '# maximum iteration num': GRAY, ',': ORANGE})
        separate_line = Line(UP * 10, DOWN * 10, color=GRAY, stroke_width=1).to_corner(to_corner_loc * RIGHT).shift(w * LEFT)
        line_num = VGroup()
        for i in range(50):
            tex_i = Text(str(i), color=GRAY, font='Consolas').scale(s).shift(DOWN * h * i).scale(0.8)
            if i > 9:
                tex_i.shift(LEFT * w/2)
            line_num.add(tex_i)
        line_num.next_to(separate_line, LEFT * 1.5).to_corner(to_corner_loc * UP).shift(DOWN * h * 0.1).to_corner(LEFT * 0.2)

        rect_gray = Rectangle(stroke_width=0, fill_color=GRAY, fill_opacity=0.15, height=10, width=3).align_to(separate_line, RIGHT)

        self.add(separate_line, rect_gray)
        self.play(Write(line_num))
        self.wait()

        dt = 0.01
        self.play(Write(text_import), run_time=len(text_import)*dt)
        self.wait()
        self.play(Write(text_func), run_time=len(text_func)*dt)
        self.wait()
        self.play(Write(text_setvalue), run_time=(len(text_setvalue_01)+len(text_setvalue_02))*dt)
        self.wait()
        self.play(Write(text_showfunc), run_time=(len(text_showfunc_01)+len(text_showfunc_02)+len(text_showfunc_03)+len(text_showfunc_04))*dt)
        self.wait()
        self.play(Write(text_main), run_time=len(text_main)*dt)
        self.wait(4)


class Wave3D(ThreeDScene):
    '''
    画一个3D的波浪
    '''
    def construct(self):
        self.set_camera_orientation(phi = 66 * DEGREES, theta = -60 * DEGREES)
        surface = ParametricSurface(lambda u, v: 
        X_AXIS * u +
        Y_AXIS * v +
        Z_AXIS * np.sin(u**2 + v**2)*np.e**(-((u**2 + v**2)**0.5)*0.5),
        u_min = -TAU, u_max = TAU, v_min = -TAU, v_max = TAU, checkerboard_colors = None).set_opacity(0.75)
        time = ValueTracker(0)
        surface.add_updater(lambda s:s.become(ParametricSurface(lambda u, v: 
        X_AXIS * u +
        Y_AXIS * v +
        Z_AXIS * np.sin((u**2 + v**2) + time.get_value())*np.e**(-((u**2 + v**2)**0.5)*0.5),
        u_min = -TAU, u_max = TAU, v_min = -TAU, v_max = TAU, checkerboard_colors = None).set_opacity(0.75)))
        self.add(time)
        plane = ParametricSurface(lambda u, v: 
        X_AXIS * (2*u) +
        Y_AXIS * (2*v) + 
        Z_AXIS * (0), 
        u_min = -TAU, u_max = TAU, v_min = -TAU, v_max = TAU, checkerboard_colors = None).set_opacity(0.75)
        self.begin_ambient_camera_rotation()
        self.play(ShowCreation(plane), run_time = 2)
        self.play(ReplacementTransform(plane, surface), run_time = 3)
        self.play(time.increment_value, 10, run_time = 5)
        self.stop_ambient_camera_rotation()
        self.wait()


class FadeInRandom0222(SpecialThreeDScene):
    '''
    让3D物体随机的出现
    '''
    def construct(self):
        axes = ThreeDAxes()
        self.add(axes)
        self.set_camera_to_default_position()
        surface = ParametricSurface(lambda u, v: 
        X_AXIS * u + Y_AXIS * v + Z_AXIS * np.sin(u**2 + v**2)*np.e**(-((u**2 + v**2)**0.5))*5,
        u_min = -TAU, u_max = TAU, v_min = -TAU, v_max = TAU, checkerboard_colors = None, resolution = (50, 50)).set_opacity(0.75)
        self.play(*[FadeInFrom(surface[i], 
        surface[i].get_center()[0]*3*X_AXIS + surface[i].get_center()[1]*3*Y_AXIS + surface[i].get_center()[2]*3*Z_AXIS
        , run_time = np.random.random()*2, rate_func = running_start)
            for i in range(len(surface))
        ])


class SteamFlowThreeD(StreamLines):
    CONFIG = {
        "x_min": -4,
        "x_max": 4,
        "y_min": -4,
        "y_max": 4,
        "z_min": -4,
        "z_max": 4,
        "delta_x": 0.5,
        "delta_y": 0.5,
        "delta_z": 0.5,
    }

    def get_start_points(self):
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max
        z_min = self.z_min
        z_max = self.z_max
        delta_x = self.delta_x
        delta_y = self.delta_y
        delta_z = self.delta_z
        n_repeats = self.n_repeats
        noise_factor = self.noise_factor

        if noise_factor is None:
            noise_factor = delta_y / 2
        return np.array(self.get_points_area(x_min, x_max, delta_x, y_min, y_max, delta_y, z_min, z_max, delta_z, noise_factor, n_repeats))

    def get_points_area(self, x_min, x_max, delta_x, y_min, y_max, delta_y, z_min, z_max, delta_z, noise_factor, n_repeats):
        area = []
        for x in np.arange(x_min, x_max + delta_x, delta_x):
            for y in np.arange(y_min, y_max + delta_y, delta_y):
                for z in np.arange(z_min, z_max + delta_z, delta_z):
                    if (x**2 + y**2 + z**2)> (1**2):
                        for n in range(n_repeats):
                            area.append(x * X_AXIS + y * Y_AXIS + z * Z_AXIS + noise_factor * np.random.random(3))
        return area


class ShowFlowThreeD(ThreeDScene):
    # TODO: 把这玩意搞成3D的

    CONFIG = {
        "flow_time": 20,
        "U": 1,
        "R": 1,
    }

    def construct(self):
        self.set_camera_orientation(phi = 60 * DEGREES, theta = -45 * DEGREES)
        stream_lines = SteamFlowThreeD(
            lambda n:self.func(n[0], n[1], n[2])
        )
        animated_stream_lines = AnimatedStreamLines(
            stream_lines,
            line_anim_class=ShowPassingFlashWithThinningStrokeWidth,
        )

        circle = Sphere(checkerboard_colors = None).scale(self.R)
        self.begin_ambient_camera_rotation()
        self.add(animated_stream_lines, circle)
        self.wait(self.flow_time)
        self.stop_ambient_camera_rotation()

    def func(self, x, y, z):
        r = (x**2 + y**2 + z**2)**0.5
        cos_theta = x/r
        sin_theta = (y**2 + z**2)**0.5/r
        v_r = self.U*(1 - (self.R**2/r**2))*cos_theta
        v_theta = -1*self.U*(1 + (self.R**2/r**2))*sin_theta
        v_x = v_r*cos_theta - v_theta*sin_theta
        v_y = v_r*sin_theta + v_theta*cos_theta
        v_z = v_y
        return (v_x)*X_AXIS + (v_y)*Y_AXIS + (v_z)*Z_AXIS


class DoublePendulum0308(Scene):
    # TODO: 学微分方程数值解，然后把这个解出来
    CONFIG = {
        "m_1": 1,    #第一根杆的质量
        "m_2": 1,    #第二根杆的质量
        "l_1": 2,    #第一根杆的长度
        "l_2": 2,    #第二根杆的长度
        "phi1_init": 30* DEGREES,
        "phi2_init": 45* DEGREES,
        "g": 9.8
    }

    def construct(self):
        # Create points and lines
        point_o = Dot()
        point_a = Dot().shift(self.get_point_A_position(self.phi1_init))
        point_b = Dot().shift(self.get_point_B_position(self.phi1_init, self.phi2_init))
        line_oa = Line(point_o, point_a)
        line_ab = Line(point_a, point_b)

        # Create texts
        time_text = VGroup(TextMobject("时间："), DecimalNumber(0)).arrange(RIGHT).to_corner(UL)
        value_tracker = ValueTracker(0)

        # Create Animation
        self.add(point_o, point_a, point_b, line_oa, time_text)
        self.wait()

    def get_point_A_position(self, phi1):
        x = self.l_1*np.sin(phi1)
        y = self.l_1*np.cos(phi1)*-1
        return x* RIGHT + y * UP

    def get_point_B_position(self, phi1, phi2):
        x_A, y_A, z_A = self.get_point_A_position(phi1)
        x = x_A + self.l_2*np.sin(phi2)
        y = y_A - self.l_2*np.cos(phi2)
        return x* RIGHT + y * UP

    def get_phi(self):
        dt = 0.001
        ddphi1 = 0.001
        ddphi2 = 0.001
        T = 0.5*(0.5*self.m_1*self.l_1**2)*dphi1**2 + 0.5*self.m_2*(dphi1*self.l_1)**2 + 0.5*(0.5*self.m_2*self.l_2**2)*dphi2**2
        V = self.m_1*self.g*y_A*0.5 + self.m_1*self.g*(y_A + y_B)*0.5
        L = T - V
        dL_dphi1 = L/ddphi1
        dL_dphi2 = L/ddphi2
        dL_phi1 = L/dphi1
        dL_phi2 = L/dphi2


class PiPicture0309(ZoomedScene):
    '''
    PI的可视化
    PI的小数点位数为极坐标下r, PI的那位数字为theta
    '''
    CONFIG = {
        "x_color": GREEN,
        "y_color": RED,
        "r_color": YELLOW,
        "theta_color": LIGHT_PINK,
        "file": r"E:\GitHub\manim\manim-master\assets\pi.txt"
    }

    def construct(self):
        self.camera.frame.scale(0.8)
        self.plane = PolarPlane()
        self.dot = Dot(self.plane.coords_to_point(1, PI))
        self.polar_grid = self.plane
        self.play(
            Write(self.polar_grid, run_time=2),
        )
        self.show_polar_coordinates()
        #self.read_Pi()
        #self.show_all_pi()

    def read_Pi(self):
        with open(self.file) as file:
            strs = file.read()
            self.theta_list = [int(str_) for str_ in strs]
             
    def show_all_pi(self):
        all_pi_group = VGroup(*[
            Dot(self.plane.coords_to_point(r, r*self.theta_list[r])).scale(10) for r in range(len(self.theta_list))
        ])

    def show_polar_coordinates(self):
        dot = self.dot
        plane = self.plane
        origin = plane.c2p(0, 0)

        r_color = self.r_color
        theta_color = self.theta_color

        r_line = Line(origin, dot.get_center())
        r_line.set_color(r_color)
        r_value = r_line.get_length()
        theta_value = r_line.get_angle()

        coord_label = self.get_coord_label(r_value, theta_value, r_color, theta_color)
        r_coord = coord_label.x_coord
        theta_coord = coord_label.y_coord

        coord_label.add_updater(lambda m: m.next_to(dot, UL, buff=SMALL_BUFF))
        r_coord.add_updater(lambda d: d.set_value(
            get_norm(dot.get_center())
        ))
        theta_coord.add_background_rectangle()
        theta_coord.add_updater(lambda d: d.set_value(
            (angle_of_vector(dot.get_center()) % TAU)
        ))
        coord_label[-1].add_updater(
            lambda m: m.next_to(theta_coord, RIGHT, SMALL_BUFF)
        )

        non_coord_parts = VGroup(*[
            part
            for part in coord_label
            if part not in [r_coord, theta_coord]
        ])

        r_label = TexMobject("r")
        r_label.set_color(r_color)
        r_label.add_updater(lambda m: m.next_to(r_coord, UP))
        theta_label = TexMobject("\\theta")
        theta_label.set_color(theta_color)
        theta_label.add_updater(lambda m: m.next_to(theta_coord, UP))

        r_coord_copy = r_coord.copy()
        r_coord_copy.add_updater(
            lambda m: m.next_to(r_line.get_center()*2, UL, buff=SMALL_BUFF)
        )

        degree_label = DecimalNumber(0, num_decimal_places=0)
        arc = Arc(radius=1, angle=theta_value)
        arc.set_color(theta_color)
        degree_label.set_color(theta_color)

        # Show r
        self.play(
            ShowCreation(r_line, run_time=2),
            ChangeDecimalToValue(r_coord_copy, r_value, run_time=2),
            VFadeIn(r_coord_copy, run_time=0.5),
            Write(dot)
        )
        r_coord.set_value(r_value)
        self.add(non_coord_parts, r_coord_copy)
        self.play(
            FadeIn(non_coord_parts),
            ReplacementTransform(r_coord_copy, r_coord),
            FadeInFromDown(r_label),
        )

        # Show theta
        degree_label.next_to(arc.get_start(), UR, SMALL_BUFF)
        line = r_line.copy()
        line.rotate(-theta_value, about_point=ORIGIN)
        line.set_color(theta_color)
        self.play(
            ShowCreation(arc),
            Rotate(line, theta_value, about_point=ORIGIN),
            VFadeInThenOut(line),
            ChangeDecimalToValue(degree_label, theta_value),
        )
        self.play(
            degree_label.scale, 0.9,
            degree_label.move_to, theta_coord,
            FadeInFromDown(theta_label),
        )
        self.theta_label = TexMobject("\\theta").set_color(self.theta_color).to_corner(UL).shift(UL*-2 + UP)
        self.r_label = TexMobject("r").set_color(self.r_color).next_to(self.theta_label, DOWN, buff =MED_LARGE_BUFF)
        self.play(
            FadeOut(r_line), 
            FadeOut(arc),
            ReplacementTransform(theta_label, self.theta_label), 
            ReplacementTransform(r_label, self.r_label)
        )
        self.wait()

    def get_coord_label(self,
                        x=0,
                        y=0,
                        x_color=WHITE,
                        y_color=WHITE,
                        include_background_rectangle=True,
                        **decimal_kwargs):
        coords = VGroup()
        for n in x, y:
            if isinstance(n, numbers.Number):
                coord = DecimalNumber(n, num_decimal_places=0, **decimal_kwargs)
            elif isinstance(n, str):
                coord = TexMobject(n)
            else:
                raise Exception("Invalid type")
            coords.add(coord)

        x_coord, y_coord = coords
        x_coord.set_color(x_color)
        y_coord.set_color(y_color)

        coord_label = VGroup(
            TexMobject("("), x_coord,
            TexMobject(","), y_coord,
            TexMobject(")")
        )
        coord_label.arrange(RIGHT, buff=SMALL_BUFF)
        coord_label[2].align_to(coord_label[0], DOWN)

        coord_label.x_coord = x_coord
        coord_label.y_coord = y_coord
        if include_background_rectangle:
            coord_label.add_background_rectangle()
        return coord_label

    def get_arc(self, theta, r=1, color=None):
        if color is None:
            color = self.theta_color
        return ParametricFunction(
            lambda t: self.plane.coords_to_point(1 + 0.025 * t, t),
            t_min=0,
            t_max=theta,
            dt=0.25,
            color=color,
            stroke_width=3,
        )


class TestCreature(GraphScene):
    CONFIG = {
        "point_x": [0, 1, 2, 3, 4, 5],
        "point_y": [1, 2, 4, 3, 5, 2],
        "x_min": 0,
        "x_max": 5,
        "x_axis_width": 5,
        "x_tick_frequency": 1,
        "y_min": 0,
        "y_max": 5,
        "y_axis_height": 5,
        "graph_origin": -2.5 * UP + -1 * RIGHT,
    }

    def construct(self):
        # object
        self.setup_axes(animate=True)
        points = VGroup(*[
            Dot(self.coords_to_point(self.point_x[i], self.point_y[i]))
            for i in range(len(self.point_y))
        ])
        self.graph = self.get_graph(self.interpolation_function, x_min=0,x_max=5)
        formular = TexMobject(r"L_n(x)=\sum\limits_{i=0}^ny_i\prod\limits_{j=0,j\neq i}^n {(x-x_j)\over(x_i-x_j)}")\
            .add_background_rectangle().move_to(self.graph.get_center())
        self.f_creature = FCreature("happy").shift(LEFT*4)

        #   animation
        self.play(FadeInFrom(self.f_creature, LEFT))
        for i in range(len(points)):
            self.play(ShowCreation(points[i]), self.f_creature.look_at, points[i], run_time=0.5)
        self.play(ShowCreation(self.graph), self.f_creature.look_at, self.graph, run_time = 1.5)
        self.wait()
        self.play(Blink(self.f_creature))
        self.play(Write(formular), self.f_creature.look_at, formular, self.f_creature.change_mode, "plain", run_time = 2)
        self.play(self.f_creature.look, LEFT)
        self.play(self.f_creature.shift, LEFT*4, rate_func = linear, run_time = 8)
        self.wait()


    def interpolation_function(self, x):
        L = 0
        for i in range(len(self.point_y)):
            L += self.point_y[i]*self.get_li(x, i)
        return L
    
    def get_li(self, x, i):
        li = 1
        for xi in self.point_x:
            if xi!=self.point_x[i]:
                li*=(x-xi)/(self.point_x[i]-xi)
        return li


class debug(Scene):
    def construct(self):
        a = FCreature().scale(2)
        self.add(a)
        debugTeX(self, a)
        self.wait()


# TODO: 绘制负载转速图


# TODO: 绘制分形


# TODO: 加入物理引擎
