'''
  > File Name        : have_fun
  > Author           : Fu_Qingchen
  > Creating Time    : 2020-01-28
'''

from manimlib.imports import *

class Try3DScence(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle = Circle()
        text = TextMobject("Test 3D Scence").scale(2).rotate(PI/4)
        self.set_camera_orientation(phi = 80 * DEGREES, theta = -45 * DEGREES)
        self.play(ShowCreation(axes))
        self.play(ShowCreation(circle))
        self.play(Write(text))


class GraphTest(GraphScene):
    CONFIG = {
        "y_max": 8,
        "y_axis_height": 5,
    }
    def construct(self):
        self.setup_axes()
        def func(x):
            return 0.1 * (x + 3-5) * (x - 3-5) * (x-5) + 5

        self.graph = self.get_graph(func,x_min=0.3,x_max=9.2)
        graph2 = self.get_derivative_graph(self.graph, x_min=0.3,x_max=9.2)
        
        self.add(self.graph, graph2)
        self.wait()


class ParametricCurve2(ThreeDScene):
    def construct(self):
        curve1=ParametricFunction(
                lambda u : np.array([
                1.2*np.cos(u),
                1.2*np.sin(u),
                u/2
            ]),color=RED,t_min=-TAU,t_max=TAU,
            )
        curve2=ParametricFunction(
                lambda u : np.array([
                1.2*np.cos(u),
                1.2*np.sin(u),
                u
            ]),color=RED,t_min=-TAU,t_max=TAU,
            )

        curve1.set_shade_in_3d(True)
        curve2.set_shade_in_3d(True)

        axes = ThreeDAxes()

        self.add(axes)

        self.set_camera_orientation(phi=80 * DEGREES,theta=-60*DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1) 
        self.play(ShowCreation(curve1))
        self.wait()
        self.play(Transform(curve1,curve2),rate_func=there_and_back,run_time=3)
        self.wait()


class TemperatureGraphScene(SpecialThreeDScene):
    CONFIG = {
        "axes_config": {
            "x_min": 0,
            "x_max": TAU,
            "y_min": 0,
            "y_max": 10,
            "z_min": -3,
            "z_max": 3,
            "x_axis_config": {
                "tick_frequency": TAU / 8,
                "include_tip": False,
            },
            "num_axis_pieces": 1,
        },
        "default_graph_style": {
            "stroke_width": 2,
            "stroke_color": YELLOW,
            #"background_image_file": "VerticalTempGradient",
        },
        "default_surface_config": {
            "fill_opacity": 0.1,
            "checkerboard_colors": [LIGHT_GREY],
            "stroke_width": 0.5,
            "stroke_color": WHITE,
            "stroke_opacity": 0.5,
        },
        "temp_text": "f(x,t) = 2\\sin(1.5x)",
    }

    def get_three_d_axes(self, include_labels=True, include_numbers=False, **kwargs):
        config = dict(self.axes_config)
        config.update(kwargs)
        axes = ThreeDAxes(**config)
        axes.set_stroke(width=2)

        if include_numbers:
            self.add_axes_numbers(axes)

        if include_labels:
            self.add_axes_labels(axes)

        # Adjust axis orientation
        axes.x_axis.rotate(
            90 * DEGREES, RIGHT,
            about_point=axes.c2p(0, 0, 0),
        )
        axes.y_axis.rotate(
            90 * DEGREES, UP,
            about_point=axes.c2p(0, 0, 0),
        )

        # Add xy-plane
        input_plane = self.get_surface(
            axes, lambda x, t: 0
        )
        input_plane.set_style(
            fill_opacity=0.5,
            fill_color=BLUE_B,
            stroke_width=0.5,
            stroke_color=WHITE,
        )

        axes.input_plane = input_plane

        return axes

    def add_axes_numbers(self, axes):
        x_axis = axes.x_axis
        y_axis = axes.y_axis
        tex_vals = [
            ("\\pi \\over 2", TAU / 4),
            ("\\pi", TAU / 2),
            ("3\\pi \\over 2", 3 * TAU / 4),
            ("2\\pi", TAU)
        ]
        x_labels = VGroup()
        for tex, val in tex_vals:
            label = TexMobject(tex)
            label.scale(0.5)
            label.next_to(x_axis.n2p(val), DOWN)
            x_labels.add(label)
        x_axis.add(x_labels)
        x_axis.numbers = x_labels

        y_axis.add_numbers()
        for number in y_axis.numbers:
            number.rotate(90 * DEGREES)
        return axes

    def add_axes_labels(self, axes):
        x_label = TexMobject("x")
        x_label.next_to(axes.x_axis.get_end(), RIGHT)
        axes.x_axis.label = x_label

        t_label = TextMobject("t")
        t_label.rotate(90 * DEGREES, OUT)
        t_label.next_to(axes.y_axis.get_end(), UP)
        axes.y_axis.label = t_label

        temp_label = TexMobject(self.temp_text)
        temp_label.rotate(90 * DEGREES, RIGHT)
        temp_label.next_to(axes.z_axis.get_zenith(), RIGHT)
        axes.z_axis.label = temp_label
        for axis in axes:
            axis.add(axis.label)
        return axes

    def get_time_slice_graph(self, axes, func, t, **kwargs):
        config = dict()
        config.update(self.default_graph_style)
        config.update({
            "t_min": axes.x_min,
            "t_max": axes.x_max,
        })
        config.update(kwargs)
        return ParametricFunction(
            lambda x: axes.c2p(
                x, t, func(x)
            ),
            **config,
        )

    def get_initial_state_graph(self, axes, func, **kwargs):
        return self.get_time_slice_graph(
            axes,
            lambda x: func(x),
            t = 0,
            **kwargs
        )

    def get_surface(self, axes, func, **kwargs):
        config = {
            "u_min": axes.y_min,
            "u_max": axes.y_max,
            "v_min": axes.x_min,
            "v_max": axes.x_max,
            "resolution": (
                (axes.y_max - axes.y_min) // axes.y_axis.tick_frequency,
                (axes.x_max - axes.x_min) // axes.x_axis.tick_frequency,
            ),
        }
        config.update(self.default_surface_config)
        config.update(kwargs)
        return ParametricSurface(
            lambda t, x: axes.c2p(
                x, t, func(x, t)
            ),
            **config
        )

    def orient_three_d_mobject(self, mobject,
                               phi=85 * DEGREES,
                               theta=-80 * DEGREES):
        mobject.rotate(-90 * DEGREES - theta, OUT)
        mobject.rotate(phi, LEFT)
        return mobject

    def get_rod_length(self):
        return self.axes_config["x_max"]

    def get_const_time_plane(self, axes):
        t_tracker = ValueTracker(0)
        plane = Polygon(
            *[
                axes.c2p(x, 0, z)
                for x, z in [
                    (axes.x_min, axes.z_min),
                    (axes.x_max, axes.z_min),
                    (axes.x_max, axes.z_max),
                    (axes.x_min, axes.z_max),
                ]
            ],
            stroke_width=0,
            fill_color=WHITE,
            fill_opacity=0.2
        )
        plane.add_updater(lambda m: m.shift(
            axes.c2p(
                axes.x_min,
                t_tracker.get_value(),
                axes.z_min,
            ) - plane.points[0]
        ))
        plane.t_tracker = t_tracker
        return plane


class SimpleCosExpGraph(TemperatureGraphScene):
    def construct(self):
        axes = self.get_three_d_axes()
        cos_graph = self.get_cos_graph(axes)
        cos_exp_surface = self.get_cos_exp_surface(axes)

        self.set_camera_orientation(
            phi=80 * DEGREES,
            theta=-80 * DEGREES,
        )
        self.camera.frame_center.shift(3 * RIGHT)
        self.begin_ambient_camera_rotation(rate=0.01)

        self.add(axes)
        self.play(ShowCreation(cos_graph))
        self.play(UpdateFromAlphaFunc(
            cos_exp_surface,
            lambda m, a: m.become(
                self.get_cos_exp_surface(axes, v_max=a * PI * 2)
            ),
            run_time=3
        ))

        self.add(cos_graph.copy().set_color(WHITE))

        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        cos_graph.add_updater(
            lambda m: m.become(self.get_time_slice_graph(
                axes,
                lambda x: self.cos_exp(x, get_t()),
                t=get_t()
            ))
        )

        plane = Rectangle(
            stroke_width=0,
            fill_color=WHITE,
            fill_opacity=0.1,
        )
        plane.rotate(90 * DEGREES, RIGHT)
        plane.match_width(axes.x_axis)
        plane.match_depth(axes.z_axis, stretch=True)
        plane.move_to(axes.c2p(0, 0, 0), LEFT)

        self.add(plane, cos_graph)
        self.play(
            ApplyMethod(
                t_tracker.set_value, 10,
                run_time=10,
                rate_func=linear,
            ),
            ApplyMethod(
                plane.shift, 10 * UP,
                run_time=10,
                rate_func=linear,
            ),
            VFadeIn(plane),
        )
        self.wait(1)

    #
    def cos_exp(self, x, t, A=2, omega=1.5, k=0.1):
        return A * np.sin(omega * x) * np.exp(-k * (omega) * t)

    def get_cos_graph(self, axes, **config):
        return self.get_initial_state_graph(
            axes,
            lambda x: self.cos_exp(x, 0),
            **config
        )

    def get_cos_exp_surface(self, axes, **config):
        return self.get_surface(
            axes,
            lambda x, t: self.cos_exp(x, t),
            **config
        )


class TestPaFunction(Scene):
    def construct(self):
        grid = NumberPlane()
        a = 1
        path = ParametricFunction(lambda t:np.array([1, 0, 0])*a*np.cos(t)+np.array([0, 1, 0])*a*np.sin(t)**2, t_min = 0, t_max = 4*PI)
        self.play(ShowCreation(grid))
        self.play(ShowCreation(path))
        self.wait()


class Tutur15(GraphScene):
    CONFIG = {
        "y_max": 30,
        "x_max": 10,
        "x_axis_label" : "$t$",
        "y_axis_label" : "$f(t)$",
        "y_labeled_nums": range(0,60,10),
        "x_labeled_nums": list(np.arange(2, 7.0+0.5, 0.5))
    }

    def construct(self):
        self.setup_axes(animate = True)
        xs = range(0,5001)
        ys = [x**2 for x in xs]
        dots_group = VGroup()
        for x in xs:
            dots_group.add(Dot(self.coords_to_point(x, ys[x])))
        self.play(ShowCreation(dots_group))
        self.wait()


class TestCBB(Scene):
    def construct(self):
        self.add(*[Dot(point) for point in [UP, RIGHT, DOWN, LEFT]])\
            .play(ShowCreation(CubicBezier([UP, RIGHT, DOWN, LEFT])))


class RadomCircle(Scene):
    def construct(self):
        self.play(
            ShowCreation(
                Circle().add_updater(
                    lambda m: m.move_to(
                        [np.random.randint(-600,600)/100, np.random.randint(-300,300)/100, 0]
                    )
                )
            )
        )


class TestGraph(GraphScene):
    CONFIG = {
        "x_min": 0,
        "x_max": 4,
        "x_axis_width": 9,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None,  # Change if different from x_min
        "x_labeled_nums": None,
        "x_axis_label": "$x$",
        "y_min": -1,
        "y_max": 3,
        "y_axis_height": 6,
        "y_tick_frequency": 1,
        "y_bottom_tick": None,  # Change if different from y_min
        "y_labeled_nums": None,
        "y_axis_label": "$y$",
        "axes_color": GREY,
        "graph_origin": 2.5 * DOWN + 4 * LEFT,
        "exclude_zero_label": True,
        "default_graph_colors": [BLUE, GREEN, YELLOW],
        "default_derivative_color": GREEN,
        "default_input_color": YELLOW,
        "default_riemann_start_color": BLUE,
        "default_riemann_end_color": GREEN,
        "area_opacity": 0.8,
        "num_rects": 50,
    }
    def construct(self):
        self.setup_axes()
        graph1 = self.get_graph(lambda x:x**0.5)
        graph2 = self.get_graph(lambda x:x)
        graph3 = self.get_graph(lambda x:x**2)
        self.play(ShowCreation(graph1))
        self.play(ShowCreation(graph2))
        self.play(ShowCreation(graph3))


class TestCircle(Scene):
    def construct(self):
        grid = NumberPlane()
        circle = Circle().scale(2).set_color(YELLOW)
        point = Dot().shift(RIGHT*2 + UP*0.001).set_color(WHITE)
        sin_line = Line(point.get_center()[0]*RIGHT, point.get_center(), color = RED)
        cos_line = Line(ORIGIN, point.get_center()[0]*RIGHT, color = ORANGE)
        x1 = DecimalNumber(point.get_center()[0]/2, num_decimal_places = 2).move_to(cos_line.get_center())
        x2 = DecimalNumber(point.get_center()[1]/2, num_decimal_places = 2).move_to(sin_line.get_center())

        sin_line.add_updater(lambda m: m.put_start_and_end_on(np.array([point.get_center()[0], 0, 0]), point.get_center()))
        cos_line.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, np.array([point.get_center()[0], 0, 0])))
        x1.add_updater(lambda m: m.set_value((point.get_center()[0]/2)).next_to(cos_line.get_center()))
        x2.add_updater(lambda m: m.set_value((point.get_center()[1]/2)).next_to(sin_line.get_center()))

        # self.play(ShowCreation(grid))
        self.play(ShowCreation(circle))
        self.add(point, sin_line, cos_line, x1, x2)
        self.play(Rotating(point, about_point = ORIGIN))
        self.wait()


class Test3DGraph(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi = 90*DEGREES, gamma = -90 * DEGREES, distance = 1000)
        self.play(ShowCreation(ThreeDAxes(z_min = -8, z_max = 8)))
        self.play(ShowCreation(ParametricFunction(
            lambda t:np.array([np.cos(2*t), np.sin(2*t), t]), color=RED, t_min = -TAU, t_max = TAU
            )))
        self.move_camera(phi = 0*DEGREES, gamma = 0 * DEGREES, run_time = 3)
        self.wait()

