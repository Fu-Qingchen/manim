'''
  > File Name        : Interpolation
  > Author           : Fu_Qingchen
  > Creating Time    : 2020-03-11
'''

from manimlib.imports import *
from from_kindergarten.imports import *

def interpolation_function(x, points_x, points_y):
    L = 0
    for i in range(len(points_y)):
        L += points_y[i] * get_li(x, i, points_x, points_y)
    return L
    

def get_li(x, i, points_x, points_y):
    li = 1
    for xi in points_x:
        if xi != points_x[i]:
            li *= (x-xi) / (points_x[i] - xi)
    return li


class Introduction(GraphScene):
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
        #   objects
        self.setup_axes(animate=True)
        points = VGroup(*[
            Dot(self.coords_to_point(self.point_x[i], self.point_y[i]))
            for i in range(len(self.point_y))
        ])
        self.graph = self.get_graph(
            lambda x:interpolation_function(x, points_x = self.point_x, points_y = self.point_y), x_min=0,x_max=5)
        formular = TexMobject(r"L_n(x)=\sum\limits_{i=0}^ny_i\prod\limits_{j=0,j\neq i}^n {(x-x_j)\over(x_i-x_j)}")\
            .add_background_rectangle().move_to(self.graph.get_center())
        self.f_creature = FCreature("happy").shift(LEFT*4)

        #   animations
        self.play(FadeInFrom(self.f_creature, LEFT))
        for i in range(len(points)):
            self.play(ShowCreation(points[i]), self.f_creature.look_at, points[i], run_time=0.25)
        self.wait()
        self.play(ShowCreation(self.graph), self.f_creature.look_at, self.graph, run_time = 1.5)
        self.wait()
        self.play(Blink(self.f_creature))
        self.wait()
        self.play(Write(formular), self.f_creature.look_at, formular, self.f_creature.change_mode, "plain", run_time = 2)
        self.play(self.f_creature.look, LEFT)
        self.play(self.f_creature.shift, LEFT*1, rate_func = linear, run_time = 2)
        self.play(FadeOut(formular), self.f_creature.look_at, self.graph)
        self.play(self.f_creature.shift, RIGHT*1, self.f_creature.change_mode, "happy")
        self.wait()


class ShowLOGO(LOGO):
    pass


class ShowDate(GraphScene):
    CONFIG = {
        "point_x": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "point_y": [4, 3, 2, 1, 2, 3, 3, 3, 4],
        "x_labeled_nums": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "y_labeled_nums": [0, 1, 2, 3, 4],
        "x_min": 0,
        "x_max": 8,
        "x_axis_width": 8,
        "x_tick_frequency": 1,
        "y_min": 0,
        "y_max": 4,
        "y_axis_height": 4,
        "graph_origin": 3 * DOWN + 4 * LEFT,
        "x_color": LIGHT_PINK,
        "y_color": YELLOW,
    }

    def construct(self):
        # objects
        number_points = len(self.point_x)
        self.x_text = TexMobject("x").scale(1.2).set_color(self.x_color).to_corner(UL, buff=LARGE_BUFF)
        self.y_text = TexMobject("y").scale(1.2).set_color(self.y_color).next_to(self.x_text, DOWN, buff = MED_LARGE_BUFF)
        self.x_numbers = VGroup(*[
            TextMobject(str(self.point_x[i])).scale(1.2)
            for i in range(number_points)
        ]).arrange(RIGHT, buff = LARGE_BUFF).next_to(self.x_text, buff = LARGE_BUFF)
        self.y_numbers = VGroup(*[
            TextMobject(str(self.point_y[i])).scale(1.2)
            for i in range(number_points)
        ]).arrange(RIGHT, buff = LARGE_BUFF).next_to(self.y_text, buff = LARGE_BUFF)
        
        # animations
        self.play(FadeInFromDown(self.x_text), FadeInFromDown(self.y_text), FadeInFromDown(self.x_numbers), FadeInFromDown(self.y_numbers))
        self.setup_axes(animate=True)

        # objects
        self.dots = VGroup(*[
            Dot(self.coords_to_point(self.point_x[i], self.point_y[i]))
            for i in range(number_points)
        ])
        self.lines = VGroup(*[
            Line(self.coords_to_point(i, 0), self.dots[i].get_center())
            for i in range(number_points)
        ])
        self.graph_main = self.get_graph(
            lambda x:interpolation_function(x, points_x = self.point_x, points_y = self.point_y), x_min=0,x_max=8)
        func_label = TexMobject("f(x)=", "?").set_color(BLUE).shift(UP*0.5)

        # animations
        self.play(ShowCreation(self.lines), ShowCreation(self.dots))
        self.wait()
        self.play(ShowCreation(self.graph_main), run_time = 2)
        self.play(FadeInFromDown(func_label))

        # objects
        data_group = VGroup(self.x_text, self.y_text, self.x_numbers, self.y_numbers)
        self.func_demo = VGroup(TexMobject("f(x)").set_color(BLUE), TexMobject("="), TexMobject("a_0x^0+a_1x^1+...+a_nx^n"))\
            .arrange(RIGHT).to_edge(UP, buff = LARGE_BUFF)
        func_demo_a = VGroup(*self.func_demo[2][0][0:2], *self.func_demo[2][0][5:7], *self.func_demo[2][0][14:16]).set_color(YELLOW)
        func_demo_x = VGroup(*self.func_demo[2][0][2:4], *self.func_demo[2][0][7:9], *self.func_demo[2][0][16:]).set_color(GREEN)
        func_demo_2 = TexMobject("a_0+\sum\limits_{i=1}^n(a_i\cos ix+b_i\sin ix)").next_to(self.func_demo[1])
        func_demo_2_a = VGroup(*func_demo_2[0][0:2], *func_demo_2[0][9:11], *func_demo_2[0][17:19]).set_color(YELLOW)
        func_demo_2_x = VGroup(func_demo_2[0][15], func_demo_2[0][23]).set_color(GREEN)
        
        # animations
        self.play(FadeOut(data_group))
        self.play(func_label[0].move_to, self.func_demo[0:1], FadeOut(func_label[1]))
        self.play(Write(self.func_demo[2]))
        self.wait()
        self.play(FadeOutAndShiftDown(self.func_demo[2]), FadeInFrom(func_demo_2, UP))
        self.wait()
        self.play(FadeOutAndShiftDown(func_demo_2), FadeInFrom(self.func_demo[2], UP))
        self.wait()


class SolveFunctionsRegular(ShowDate):
    def construct(self):
        # objects in ShowDate
        self.show_objects_before()
        self.get_function()
        self.update_first_li()
        self.wait()

    def show_objects_before(self):
        self.setup_axes()
        self.number_points = len(self.point_x)
        self.dots = VGroup(*[
            Dot(self.coords_to_point(self.point_x[i], self.point_y[i]))
            for i in range(self.number_points)
        ])
        self.lines = VGroup(*[
            Line(self.coords_to_point(i, 0), self.dots[i].get_center())
            for i in range(self.number_points)
        ])
        self.graph_main = self.get_graph(
            lambda x:interpolation_function(x, points_x = self.point_x, points_y = self.point_y), x_min=0,x_max=8)
        self.graph_group = VGroup(self.dots, self.lines, self.graph_main)
        self.func_demo = VGroup(TexMobject("f(x)").set_color(BLUE), TexMobject("="), TexMobject("a_0x^0+a_1x^1+...+a_nx^n"))\
            .arrange(RIGHT).to_edge(UP, buff = LARGE_BUFF)
        func_demo_a = VGroup(*self.func_demo[2][0][0:2], *self.func_demo[2][0][5:7], *self.func_demo[2][0][14:16]).set_color(YELLOW)
        func_demo_x = VGroup(*self.func_demo[2][0][2:4], *self.func_demo[2][0][7:9], *self.func_demo[2][0][16:]).set_color(GREEN)
        self.add(self.dots, self.lines, self.graph_main, self.func_demo)

    def get_function(self):
        for i in range(self.number_points):
            y_tex = TexMobject(str(self.point_y[i])).set_color(BLUE)
            x_tex = TexMobject(str(self.point_x[i])).set_color(GREEN)
            self.play(Transform(self.func_demo[0], y_tex.move_to(self.func_demo[0].get_center())), 
                Transform(self.func_demo[2][0][2], x_tex.move_to(self.func_demo[2][0][2].get_center())), 
                Transform(self.func_demo[2][0][7], x_tex.copy().move_to(self.func_demo[2][0][7].get_center())), 
                Transform(self.func_demo[2][0][16], x_tex.copy().move_to(self.func_demo[2][0][16].get_center())),
                Flash(self.dots[i]), run_time = 0.2)
            self.wait()
        y_tex = TexMobject("f(x)").set_color(BLUE)
        x_tex = TexMobject("x").set_color(GREEN)
        self.play(Transform(self.func_demo[0], y_tex.move_to(self.func_demo[0].get_center())), 
                Transform(self.func_demo[2][0][2], x_tex.move_to(self.func_demo[2][0][2].get_center())), 
                Transform(self.func_demo[2][0][7], x_tex.copy().move_to(self.func_demo[2][0][7].get_center())), 
                Transform(self.func_demo[2][0][16], x_tex.copy().move_to(self.func_demo[2][0][16].get_center())))
        self.wait()

    def update_first_li(self):
        li_tex = TexMobject(r"f_0(x) + f_1(x) +...+ f_n(x)").next_to(self.func_demo[1])
        li_tex_li = VGroup(*li_tex[0][0:2], *li_tex[0][6:8], *li_tex[0][16:18]).set_color(YELLOW)
        li_tex_x = VGroup(*li_tex[0][3:4], *li_tex[0][9:10], *li_tex[0][19:20]).set_color(GREEN)
        li_tex_dark = TexMobject("f_0(x)").next_to(self.func_demo[1])
        li_tex_dark[0][0:2].set_color(YELLOW)
        li_tex_dark[0][3:4].set_color(GREEN)
        self.play(Transform(self.func_demo[2], li_tex))
        self.wait()
        f_0_formular = TexMobject(r"f_0\left( x_i \right) =\begin{cases}	y_0&		i=0\\	0&		i\ne 0\\\end{cases}")\
            .next_to(li_tex_dark, DOWN).shift(RIGHT)
        f_0_formular[0][0:2].set_color(YELLOW)
        f_0_formular[0][3:4].set_color(GREEN)
        self.play(Write(f_0_formular))
        self.wait()
        graph_group = self.graph_group.copy()
        for i in range(self.number_points):
            #   TODO: 做一个鼠标类
            update_graphs = self.get_first_li_graph(i)
            self.play(self.graph_group.set_color, DARK_GREY, 
                        Transform(graph_group, update_graphs), 
                        VGroup(*li_tex[0][5:]).set_color, DARK_GREY)
        self.play(TransformFromCopy(graph_group[2], li_tex_dark))
        self.wait()

        li_formular = VGroup(
            li_tex_dark.copy().next_to(li_tex_dark, DOWN),
            TexMobject("="),
            TexMobject(r"y_0"),
            TexMobject(r"\prod\limits_{i=1}^n {(x-x_i)\over(x_0-x_i)}")
        ).arrange(RIGHT).add_background_rectangle().shift(UP)
        li_formular[4][0][6].set_color(GREEN)
        li_text = TextMobject(r"拉格朗日插值基函数$l_0(x)$").set_color(YELLOW).next_to(li_formular[4], DOWN).add_background_rectangle()
        self.play(FadeOut(f_0_formular))
        self.play(Write(li_formular))
        rect1 = SurroundingRectangle(li_formular[4][0][5:11])
        rect2 = SurroundingRectangle(VGroup(*update_graphs[0][1:]))
        self.wait()
        self.play(ShowCreation(rect1))
        self.play(Transform(rect1, rect2))
        self.play(FadeOut(rect1))
        self.wait()
        rect_3 = SurroundingRectangle(li_formular[4])
        self.play(ShowCreation(rect_3))
        self.play(Write(li_text))
        self.wait()
        self.play(FadeOut(li_formular), FadeOut(li_text), FadeOut(rect_3))

    def get_first_li_graph(self, i):
        point_x = self.point_x
        point_y = self.point_y
        for ii in range(1,i + 1):
            point_y[ii] = 0
        return VGroup(
            VGroup(*[Dot(self.coords_to_point(point_x[i], point_y[i])) for i in range(self.number_points)]), 
            VGroup(*[Line(self.coords_to_point(i, 0), self.coords_to_point(point_x[i], point_y[i])) for i in range(self.number_points)]), 
            self.get_graph(lambda x:interpolation_function(x, points_x = point_x, points_y = point_y), x_min=0,x_max=8).set_color(YELLOW)
        )


class ShowLi(SolveFunctionsRegular):
    def construct(self):
        # objects
        self.setup_axes()
        self.number_points = len(self.point_x)
        self.get_origin_graph()
        self.func_demo = VGroup(TexMobject("f(x)").set_color(BLUE), TexMobject("="), TexMobject("a_0x^0+a_1x^1+...+a_nx^n"))\
            .arrange(RIGHT).to_edge(UP, buff = LARGE_BUFF)
        li_tex = TexMobject(r"f_0(x) + f_1(x) +...+ f_n(x)").next_to(self.func_demo[1])
        li_tex_li = VGroup(*li_tex[0][0:2], *li_tex[0][6:8], *li_tex[0][16:18]).set_color(YELLOW)
        li_tex_x = VGroup(*li_tex[0][3:4], *li_tex[0][9:10], *li_tex[0][19:20]).set_color(GREEN)
        li_tex_copy = li_tex.copy()

        # animations
        graph_group = self.graph_group.copy()
        graph_grou_back = self.graph_group.copy().set_color(DARK_GREY)
        self.add(self.func_demo[0:2], li_tex, graph_grou_back, self.graph_group)
        for i in range(self.number_points):
            self.play(
                Transform(li_tex, self.set_color_li_tex(i, li_tex)),
                Transform(self.graph_group, self.get_li_graph(i)))
            self.wait(0.5)
        self.wait()

        sum_formular = TexMobject(r"\sum\limits_{i=0}^ny_il_i(x)").next_to(self.func_demo[1])
        sum_formular[0][5:7].set_color(BLUE)
        sum_formular[0][7:9].set_color(YELLOW)
        sum_formular[0][10:11].set_color(GREEN)
        full_formular = TexMobject(r"=\sum\limits_{i=0}^ny_i\prod\limits_{i=1}^n {(x-x_i)\over(x_0-x_i)}")\
            .next_to(self.func_demo[0]).shift(DOWN*1.6).add_background_rectangle()
        self.play(Transform(self.graph_group, graph_group), Transform(li_tex, sum_formular))
        self.wait()
        self.play(Transform(self.func_demo[0], TexMobject(r"L(x)").move_to(self.func_demo[0]).set_color(BLUE)))
        self.play(Write(full_formular))
        self.wait()

    def set_color_li_tex(self, i, li_tex):
        if i == 0:
            li_tex.set_color(DARK_GREY)
            li_tex[0][0:2].set_color(YELLOW)
            li_tex[0][3:4].set_color(GREEN)
        elif i == 1:
            li_tex.set_color(DARK_GREY)
            li_tex[0][6:8].set_color(YELLOW)
            li_tex[0][9:10].set_color(GREEN)
        elif i == 8:
            li_tex.set_color(DARK_GREY)
            li_tex[0][16:18].set_color(YELLOW)
            li_tex[0][19:20].set_color(GREEN)
        else: 
            li_tex.set_color(DARK_GREY)
            li_tex[0][12:15].set_color(YELLOW)
        return li_tex

    
    def get_origin_graph(self):
        self.dots = VGroup(*[
            Dot(self.coords_to_point(self.point_x[i], self.point_y[i]))
            for i in range(self.number_points)
        ])
        self.lines = VGroup(*[
            Line(self.coords_to_point(i, 0), self.dots[i].get_center())
            for i in range(self.number_points)
        ])
        self.graph_main = self.get_graph(
            lambda x:interpolation_function(x, points_x = self.point_x, points_y = self.point_y), x_min=0, x_max=8)
        self.graph_group = VGroup(self.dots, self.lines, self.graph_main)


    def get_li_graph(self, i):
        point_x = self.point_x
        point_y = [0 for i in range(self.number_points)]
        point_y[i] = self.point_y[i]
        return VGroup(
            VGroup(*[Dot(self.coords_to_point(point_x[i], point_y[i])) for i in range(self.number_points)]), 
            VGroup(*[Line(self.coords_to_point(i, 0), self.coords_to_point(point_x[i], point_y[i])) for i in range(self.number_points)]), 
            self.get_graph(lambda x:interpolation_function(x, points_x = point_x, points_y = point_y), x_min=0,x_max=8).set_color(YELLOW)
        )

