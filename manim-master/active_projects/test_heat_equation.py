'''
  > File Name        : test_heat_equation
  > Creating Time    : 2020-01-28
'''

from manimlib.imports import *
from from_3b1b.active.diffyq.part2.shared_constructs import *

class TalkThrough1DHeatGraph(SpecialThreeDScene):
    CONFIG = {
        "step_size": 0.05,
        "axes_config": {
            "x_min": -1,
            "x_max": 11,
            "y_min": -10,
            "y_max": 100,
            "x_axis_config": {
                "tick_frequency": TAU / 8,
                "include_tip": False,
            },
            "y_axis_config": {
                "unit_size": 0.06,
                "tick_frequency": 10,
            },
        },
        "y_labels": range(20, 100, 20),
        "graph_x_min": 0,
        "graph_x_max": 10,
        # "freq_amplitude_pairs": [
        #     (1, 0.5),
        #     (2, 1),
        #     (3, 0.5),
        #     (4, 0.3),
        #     (5, 0.3),
        #     (7, 0.2),
        # ],
        "surface_resolution": 20,
        "graph_slice_step": 10 / 20,
        # "alpha": 0.1,
        "y_label_text": "f(x,t) = 2\\sin(1.5x)e^{-0.15t} + 60"
    }

    # 要画出来的函数方程
    def temp_func(self, x, t, A=15, omega=1.5, k=0.1):
        return A * np.sin(omega * x) * np.exp(-k * (omega) * t) + 60
        # new_x = TAU * x / 10
        # return 50 + 20 * np.sum([
        #     amp * np.sin(freq * new_x) *
        #     np.exp(-(self.alpha * freq**2) * t)
        #     for freq, amp in self.freq_amplitude_pairs
        # ])

    def construct(self):
        self.add_axes()
        self.add_graph()
        self.show_surface()
        

    def setup_axes(self):
        axes = Axes(**self.axes_config)
        axes.center().to_edge(UP)

        y_label = axes.get_y_axis_label(self.y_label_text)
        y_label.to_edge(UP)
        axes.y_axis.label = y_label
        axes.y_axis.add(y_label)
        axes.y_axis.add_numbers(*self.y_labels)

        self.axes = axes

    def add_axes(self):
        self.setup_axes()
        self.play(ShowCreation(self.axes))
        #self.add(self.axes)

    def add_graph(self):
        self.graph = self.get_graph()
        self.play(ShowCreation(self.graph))
        # self.add(self.graph)

        self.wait()

    def show_surface(self):
        axes = self.axes
        graph = self.graph
        t_min = 0
        t_max = 10

        axes_copy = axes.deepcopy()
        self.original_axes = self.axes

        # Set rod final state
        final_graph = self.get_graph(t_max)
        curr_graph = self.graph
        self.graph = final_graph
        self.graph = curr_graph

        # Time axis
        t_axis = NumberLine(
            x_min=t_min,
            x_max=t_max,
        )
        origin = axes.c2p(0, 0)
        t_axis.shift(origin - t_axis.n2p(0))
        t_axis.add_numbers(
            *range(1, t_max + 1),
            direction=UP,
        )
        time_label = TextMobject("Time")
        time_label.scale(1.5)
        time_label.next_to(t_axis, UP)
        t_axis.time_label = time_label
        t_axis.add(time_label)
        t_axis.rotate(90 * DEGREES, UP, about_point=origin)

        # New parts of graph
        step = self.graph_slice_step
        graph_slices = VGroup(*[
            self.get_graph(time=t).shift(
                t * IN
            )
            for t in np.arange(0, t_max + step, step)
        ])
        graph_slices.set_stroke(width=1) #.set_color_by_gradient([BLUE, TEAL, GREEN, YELLOW, "#ff0000"])
        graph_slices.set_shade_in_3d(True)

        # Input plane
        x_axis = self.axes.x_axis
        y = axes.c2p(0, 0)[1]
        surface_config = {
            "u_min": self.graph_x_min,
            "u_max": self.graph_x_max,
            "v_min": t_min,
            "v_max": t_max,
            "resolution": self.surface_resolution,
        }
        input_plane = ParametricSurface(
            lambda x, t: np.array([
                x_axis.n2p(x)[0],
                y,
                t_axis.n2p(t)[2],
            ]),
            **surface_config,
        )
        input_plane.set_style(
            fill_opacity=0.5,
            fill_color=BLUE_B,
            stroke_width=0.5,
            stroke_color=WHITE,
        )

        # Surface
        y_axis = axes.y_axis
        # 参数曲面用法
        surface = ParametricSurface(
            lambda x, t: np.array([
                x_axis.n2p(x)[0],
                y_axis.n2p(self.temp_func(x, t))[1],
                t_axis.n2p(t)[2],
            ]),
            **surface_config,
        )
        surface.set_style(
            fill_opacity=0.1,
            fill_color=LIGHT_GREY,
            stroke_width=0.5,
            stroke_color=WHITE,
            stroke_opacity=0.5,
        )

        # Rotate everything on screen and move camera
        # in such a way that it looks the same
        curr_group = Group(*self.get_mobjects())
        curr_group.clear_updaters()
        self.set_camera_orientation(
            phi=90 * DEGREES,
        )
        mobs = [
            curr_group,
            graph_slices,
            t_axis,
            input_plane,
            surface,
        ]
        for mob in mobs:
            self.orient_mobject_for_3d(mob)

        self.move_camera(
            phi=80 * DEGREES,
            theta=-85 * DEGREES,
            added_anims=[
                Write(input_plane),
                Write(t_axis)
            ]
        )
        self.begin_ambient_camera_rotation()
        self.add(*graph_slices, *self.get_mobjects())
        self.play(
            FadeIn(surface),
            LaggedStart(*[
                TransformFromCopy(graph, graph_slice)
                for graph_slice in graph_slices
            ], lag_ratio=0.02)
        )
        self.wait()

        # Show slices
        self.axes = axes_copy  # So get_graph works...
        slicing_plane = Rectangle(
            stroke_width=0,
            fill_color=WHITE,
            fill_opacity=0.2,
        )
        slicing_plane.set_shade_in_3d(True)
        slicing_plane.replace(
            Line(axes_copy.c2p(0, 0), axes_copy.c2p(10, 100)),
            stretch=True
        )
        self.orient_mobject_for_3d(slicing_plane)

        def get_time_slice(t):
            new_slice = self.get_graph(t)
            new_slice.set_shade_in_3d(True)
            self.orient_mobject_for_3d(new_slice)
            new_slice.shift(t * UP)
            return new_slice

        graph.set_shade_in_3d(True)
        t_tracker = ValueTracker(0)
        graph.add_updater(lambda g: g.become(
            get_time_slice(t_tracker.get_value())
        ))
        
        kw = {"run_time": 10, "rate_func": linear}
        self.play(
            ApplyMethod(t_tracker.set_value, 10, **kw),
            ApplyMethod(slicing_plane.shift, 10 * UP, **kw),
        )
        self.wait()

        self.set_variables_as_attrs(
            t_axis,
            input_plane,
            surface,
            graph_slices,
            slicing_plane,
            t_tracker,
        )

    #
    def get_graph(self, time=0):
        graph = self.axes.get_graph(
            lambda x: self.temp_func(x, time),
            x_min=self.graph_x_min,
            x_max=self.graph_x_max,
            step_size=self.step_size,
        )
        graph.time = time
        # graph.color_using_background_image("VerticalTempGradient")
        graph.set_color(YELLOW_A)
        return graph

    def get_graph_time_change_animation(self, graph, new_time, **kwargs):
        old_time = graph.time
        graph.time = new_time
        config = {
            "run_time": abs(new_time - old_time),
            "rate_func": linear,
        }
        config.update(kwargs)

        return UpdateFromAlphaFunc(
            graph,
            lambda g, a: g.become(
                self.get_graph(interpolate(
                    old_time, new_time, a
                ))
            ),
            **config
        )

    def orient_mobject_for_3d(self, mob):
        mob.rotate(
            90 * DEGREES,
            axis=RIGHT,
            about_point=ORIGIN
        )
        return mob
