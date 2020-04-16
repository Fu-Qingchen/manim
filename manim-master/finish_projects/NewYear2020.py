'''
  > File Name        : NewYear2020
'''

from manimlib.imports import *
from from_3b1b.active.diffyq.part4.long_fourier_scenes import FourierSeriesExampleWithRectForZoom
from from_3b1b.active.diffyq.part4.long_fourier_scenes import ZoomedInFourierSeriesExample
from from_3b1b.active.diffyq.part2.fourier_series import FourierOfTrebleClef

class Introduce(Scene):
    CONFIG = {
        "camera_config": {
            "background_color": "#b2032e"
        }
    }

    def construct(self):
        intro1 = TextMobject("Manim在去年更新了一个用傅里叶级数画图的库", color = YELLOW).shift(UP*2)
        intro2 = TextMobject("寒假终于有时间可以来玩一玩了！", color = YELLOW).next_to(intro1, DOWN)
        text1 = TextMobject("原理是用到了傅里叶级数的复数形式", color = YELLOW).next_to(intro1, DOWN).shift(DOWN)
        formula = TexMobject(r"x(t) = ", r"\sum_{n = -\infty}^{\infty}C_ne^{jn2\pi ft}", color = YELLOW).next_to(text1, DOWN)
        text2 = TextMobject("它在复平面上可以看作是一个个旋转的向量首尾相接", color = YELLOW).next_to(formula, DOWN)
        self.play(FadeInFromDown(intro1))
        self.play(FadeInFromDown(intro2))
        self.play(FadeInFromDown(text1))
        self.play(FadeInFromDown(formula))
        self.play(FadeInFromDown(text2))


class Textscence1(Scene):
    CONFIG = {
        "camera_config": {
            "background_color": "#b2032e"
        }
    }
    def construct(self):
        text = TextMobject("向量越多，越接近原函数", color = YELLOW)
        self.play(FadeInFromDown(text))


class Textscence2(Scene):
    CONFIG = {
        "camera_config": {
            "background_color": "#b2032e"
        }
    }
    def construct(self):
        text = TextMobject("下面是1000个向量画成的老鼠", color = YELLOW)
        self.play(FadeInFromDown(text))


class ComplexFourierSeriesExample(FourierOfTrebleClef):
    CONFIG = {
        "file_name": "good",
        "run_time": 10,
        "n_vectors": 200,
        "n_cycles": 2,
        "max_circle_stroke_width": 0.75,
        "drawing_height": 5,
        "center_point": DOWN,
        "top_row_center": 3 * UP,
        "top_row_label_y": 2,
        "top_row_x_spacing": 1.75,
        "top_row_copy_scale_factor": 0.9,
        "plane_config": {
            "axis_config": {"unit_size": 2},
            "y_min": -1.25,
            "y_max": 1.25,
            "x_min": -2.5,
            "x_max": 2.5,
            "background_line_style": {
                "stroke_width": 1,
                "stroke_color": LIGHT_GREY,
            },
        },
        "top_rect_height": 2.5,
        "camera_config": {
            "background_color": "#b2032e"
        }
    }

    def get_top_row_labels(self, vector_copies):
        labels = VGroup()
        for vector_copy in vector_copies:
            freq = vector_copy.freq
            label = Integer(freq)
            label = TexMobject(r"e^{j", str(freq), r"·2\pi t}").scale(0.8)
            label[1].set_color(YELLOW)
            label.move_to(np.array([
                freq * self.top_row_x_spacing,
                self.top_row_label_y,
                0
            ]))
            labels.add(label)
        return labels

    def construct(self):
        self.add_vectors_circles_path()
        self.add_top_row(self.vectors, self.circles)
        # self.write_title()
        # self.highlight_vectors_one_by_one()
        self.change_shape()

    def write_title(self):
        title = TextMobject("Complex\\\\Fourier series")
        title.scale(1.5)
        title.to_edge(LEFT)
        title.match_y(self.path)

        self.wait(11)
        self.play(FadeInFromDown(title))
        self.wait(2)
        self.title = title

    def highlight_vectors_one_by_one(self):
        # Don't know why these vectors can't get copied.
        # That seems like a problem that will come up again.
        labels = self.top_row[-1]
        next_anims = []
        for vector, circle, label in zip(self.vectors, self.circles, labels):
            # v_color = vector.get_color()
            c_color = circle.get_color()
            c_stroke_width = circle.get_stroke_width()

            rect = SurroundingRectangle(label, color=PINK)
            self.play(
                # vector.set_color, PINK,
                circle.set_stroke, RED, 3,
                FadeIn(rect),
                *next_anims
            )
            self.wait()
            next_anims = [
                # vector.set_color, v_color,
                circle.set_stroke, c_color, c_stroke_width,
                FadeOut(rect),
            ]
        self.play(*next_anims)

    def change_shape(self):
        # path_mob = TexMobject("\\pi")
        path_mob = SVGMobject("good")
        new_path = path_mob.family_members_with_points()[0]
        new_path.set_height(4)
        new_path.move_to(self.path, DOWN)
        new_path.shift(0.5 * UP)

        self.transition_to_alt_path(new_path)
        for n in range(self.n_cycles):
            self.run_one_cycle()

    def transition_to_alt_path(self, new_path, morph_path=False):
        new_coefs = self.get_coefficients_of_path(new_path)
        new_vectors = self.get_rotating_vectors(
            coefficients=new_coefs
        )
        new_drawn_path = self.get_drawn_path(new_vectors)

        self.vector_clock.suspend_updating()

        vectors = self.vectors
        anims = []

        for vect, new_vect in zip(vectors, new_vectors):
            new_vect.update()
            new_vect.clear_updaters()

            line = Line(stroke_width=0)
            line.put_start_and_end_on(*vect.get_start_and_end())
            anims.append(ApplyMethod(
                line.put_start_and_end_on,
                *new_vect.get_start_and_end()
            ))
            vect.freq = new_vect.freq
            vect.coefficient = new_vect.coefficient

            vect.line = line
            vect.add_updater(
                lambda v: v.put_start_and_end_on(
                    *v.line.get_start_and_end()
                )
            )
        if morph_path:
            anims.append(
                ReplacementTransform(
                    self.drawn_path,
                    new_drawn_path
                )
            )
        else:
            anims.append(
                FadeOut(self.drawn_path)
            )

        self.play(*anims, run_time=3)
        for vect in self.vectors:
            vect.remove_updater(vect.updaters[-1])

        if not morph_path:
            self.add(new_drawn_path)
            self.vector_clock.set_value(0)

        self.vector_clock.resume_updating()
        self.drawn_path = new_drawn_path

    #
    def get_path(self):
        path = super().get_path()
        path.set_height(self.drawing_height)
        path.to_edge(DOWN)
        return path

    def add_top_row(self, vectors, circles, max_freq=3):
        self.top_row = self.get_top_row(
            vectors, circles, max_freq
        )
        self.add(self.top_row)

    def get_top_row(self, vectors, circles, max_freq=3):
        vector_copies = VGroup()
        circle_copies = VGroup()
        for vector, circle in zip(vectors, circles):
            if vector.freq > max_freq:
                break
            vcopy = vector.copy()
            vcopy.clear_updaters()
            ccopy = circle.copy()
            ccopy.clear_updaters()
            ccopy.original = circle
            vcopy.original = vector

            vcopy.center_point = op.add(
                self.top_row_center,
                vector.freq * self.top_row_x_spacing * RIGHT,
            )
            ccopy.center_point = vcopy.center_point
            vcopy.add_updater(self.update_top_row_vector_copy)
            ccopy.add_updater(self.update_top_row_circle_copy)
            vector_copies.add(vcopy)
            circle_copies.add(ccopy)

        dots = VGroup(*[
            TexMobject("\\dots").next_to(
                circle_copies, direction,
                MED_LARGE_BUFF,
            )
            for direction in [LEFT, RIGHT]
        ])
        labels = self.get_top_row_labels(vector_copies)
        return VGroup(
            vector_copies,
            circle_copies,
            dots,
            labels,
        )

    def update_top_row_vector_copy(self, vcopy):
        vcopy.become(vcopy.original)
        vcopy.scale(self.top_row_copy_scale_factor)
        vcopy.shift(vcopy.center_point - vcopy.get_start())
        return vcopy

    def update_top_row_circle_copy(self, ccopy):
        ccopy.become(ccopy.original)
        ccopy.scale(self.top_row_copy_scale_factor)
        ccopy.move_to(ccopy.center_point)
        return ccopy

    def setup_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.shift(self.center_point)
        plane.add_coordinates()

        top_rect = Rectangle(
            width=FRAME_WIDTH,
            fill_color="#b2032e",
            fill_opacity=1,
            stroke_width=0,
            height=self.top_rect_height,
        )
        top_rect.to_edge(UP, buff=0)

        self.plane = plane
        self.add(plane)
        self.add(top_rect)

    def get_path_end(self, vectors, stroke_width=None, **kwargs):
        if stroke_width is None:
            stroke_width = self.drawn_path_st
        full_path = self.get_vector_sum_path(vectors, **kwargs)
        path = VMobject()
        path.set_stroke(
            self.drawn_path_color,
            stroke_width
        )

        def update_path(p):
            alpha = self.get_vector_time() % 1
            p.pointwise_become_partial(
                full_path,
                np.clip(alpha - 0.01, 0, 1),
                np.clip(alpha, 0, 1),
            )
            p.points[-1] = vectors[-1].get_end()

        path.add_updater(update_path)
        return path

    def get_drawn_path_alpha(self):
        return super().get_drawn_path_alpha() - 0.002

    def get_drawn_path(self, vectors, stroke_width=2, **kwargs):
        odp = super().get_drawn_path(vectors, stroke_width, **kwargs)
        return VGroup(
            odp,
            self.get_path_end(vectors, stroke_width, **kwargs),
        )

    def get_vertically_falling_tracing(self, vector, color, stroke_width=3, rate=0.25):
        path = VMobject()
        path.set_stroke(color, stroke_width)
        path.start_new_path(vector.get_end())
        path.vector = vector

        def update_path(p, dt):
            p.shift(rate * dt * DOWN)
            p.add_smooth_curve_to(p.vector.get_end())
        path.add_updater(update_path)
        return path


class FourierOfMouse(FourierSeriesExampleWithRectForZoom):
    CONFIG = {
        # "file_name": "mouse",
        "file_name": r"E:\GitHub\manim\manim-master\assets\svg_images\mouse.svg",
        "drawing_height": 7,
        "n_vectors": 1000,
        "drawn_path_color": YELLOW,
        "drawn_path_stroke_width": 10,
        "camera_config": {
            "background_color": "#b2032e"
        }
    }

    def get_rect(self):
        return ScreenRectangle(
            color=WHITE,
            stroke_width=self.rect_stroke_width,
        )


class FourierOfMouseWithNoRect(FourierSeriesExampleWithRectForZoom):
    CONFIG = {
        "file_name": "mouse",
        "drawing_height": 7,
        "n_vectors": 1000,
        "drawn_path_color": YELLOW,
        "drawn_path_stroke_width": 0,
        "camera_config": {
            "background_color": "#b2032e"
        }
    }

    def get_rect(self):
        return ScreenRectangle(
            color=BLUE,
            stroke_width=0,
        )


class FourierOfMouseZoomedIn(ZoomedInFourierSeriesExample):
    CONFIG = {
        "file_name": "mouse",
        "drawing_height": 7,
        "n_vectors": 1000,
        "drawn_path_color": YELLOW,
        "drawn_path_stroke_width": 5,
        "max_circle_stroke_width": 0.3,
        "camera_config": {
            "background_color": "#b2032e"
        }
    }


class IncreaseOrderOfApproximation(ComplexFourierSeriesExample):
    CONFIG = {
        "file_name": "coin",
        "drawing_height": 6,
        "n_vectors": 250,
        "parametric_function_step_size": 0.001,
        "run_time": 10,
        # "n_vectors": 25,
        # "parametric_function_step_size": 0.01,
        # "run_time": 5,
        "slow_factor": 0.05,
        "camera_config": {
            "background_color": "#b2032e"
        }
    }

    def construct(self):
        path = self.get_path()
        path.to_edge(DOWN)
        path.set_stroke(YELLOW, 2)
        freqs = self.get_freqs()
        coefs = self.get_coefficients_of_path(
            path, freqs=freqs,
        )
        vectors = self.get_rotating_vectors(freqs, coefs)
        circles = self.get_circles(vectors)

        n_tracker = ValueTracker(2)
        n_label = VGroup(
            TextMobject("用"),
            Integer(100).set_color(YELLOW),
            TextMobject("个向量拟合")
        )
        n_label.arrange(RIGHT)
        n_label.to_corner(UL)
        n_label.add_updater(
            lambda n: n[1].set_value(
                n_tracker.get_value()
            ).align_to(n[2], DOWN)
        )

        changing_path = VMobject()
        vector_copies = VGroup()
        circle_copies = VGroup()

        def update_changing_path(cp):
            n = n_label[1].get_value()
            cp.become(self.get_vector_sum_path(vectors[:n]))
            cp.set_stroke(YELLOW, 2)
            # While we're at it...
            vector_copies.submobjects = list(vectors[:n])
            circle_copies.submobjects = list(circles[:n])

        changing_path.add_updater(update_changing_path)

        self.add(n_label, n_tracker, changing_path)
        self.add(vector_copies, circle_copies)
        self.play(
            n_tracker.set_value, self.n_vectors,
            rate_func=smooth,
            run_time=self.run_time,
        )
        self.wait(5)


class Safe(FourierSeriesExampleWithRectForZoom):
    CONFIG = {
        "file_name": "mouse",
        "drawing_height": 7,
        "n_vectors": 1000,
        "drawn_path_color": YELLOW,
        "drawn_path_stroke_width": 10,
        "max_circle_stroke_width": 0,
        "camera_config": {
            "background_color": "#b2032e"
        },
        "background_line_style": {
            "stroke_width": 0,
            "stroke_color": LIGHT_GREY,
        },
        "vector_config": {
            "buff": 0,
            "max_tip_length_to_length_ratio": 0,
            "tip_length": 0,
            "max_stroke_width_to_length_ratio": 0,
            "stroke_width": 0,
        },
    }
    def get_rect(self):
        return ScreenRectangle(
            color=BLUE,
            stroke_width=0,
        )


class LOGO(Scene):
    CONFIG = {
        "Author": "@Fu\\_Qingchen",
        "author_colors": [BLUE, YELLOW, ORANGE],
        "camera_config": {
            "background_color": "#b2032e"
        }
    }

    def construct(self):
        logo_re = ImageMobject("E:\GitHub\Project\Science3Min\Resources\V4.3.png").scale(1.5).shift(UP*0.5)
        author = TextMobject(
            self.Author,
            tex_to_color_map={self.Author: self.author_colors}
        ).scale(1.5).next_to(logo_re, DOWN).shift(DOWN*0.5)
        self.play(FadeInFrom(logo_re))
        self.play(Write(author))
        self.play(FadeOut(logo_re), FadeOut(author))


class Cover(Scene):
    CONFIG = {
        "camera_config": {
            "background_color":"#b2032e"
        }
    }

    def construct(self):
        mouse_pre_svg = SVGMobject("mouse_pre").set_color("#cd6466").scale(2.5)
        Text = TextMobject("用", "傅里叶级数", "画只", "老鼠？").scale(1.5)
        Text[1].set_color(YELLOW)
        Text[3].set_color(YELLOW)
        self.play(ShowCreation(mouse_pre_svg))
        self.play(ShowCreation(Text))

