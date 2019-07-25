#!/usr/bin/env python

from big_ol_pile_of_manim_imports import *

# To watch one of these scenes, run the following:
# python extract_scene.py file_name <SceneName> -p
#
# Use the flat -l for a faster rendering at a lower
# quality, use -s to skip to the end and just show
# the final frame, and use -n <number> to skip ahead
# to the n'th animation of a scene.

class Color_Map_Output(Scene):
    def construct(self):
        text1 = TextMobject("Red")
        text1.set_color(RED)
        text1.to_edge(LEFT)
        text2 = TextMobject("Orange")
        text2.set_color(ORANGE)
        text2.add_updater(lambda d: d.next_to(text1, RIGHT))
        text3 = TextMobject("Yellow")
        text3.set_color(YELLOW)
        text3.add_updater(lambda d: d.next_to(text2, RIGHT))
        text4 = TextMobject("Gold")
        text4.set_color(GOLD)
        text4.add_updater(lambda d: d.next_to(text3, RIGHT))
        text5 = TextMobject("Green")
        text5.set_color(GREEN)
        text5.add_updater(lambda d: d.next_to(text4, RIGHT))
        text6 = TextMobject("Teal")
        text6.set_color(TEAL)
        text6.add_updater(lambda d: d.next_to(text5, RIGHT))
        text7 = TextMobject("Blue")
        text7.set_color(BLUE)
        text7.add_updater(lambda d: d.next_to(text6, RIGHT))
        text8 = TextMobject("Purple")
        text8.set_color(PURPLE)
        text8.add_updater(lambda d: d.next_to(text7, RIGHT))
        text9 = TextMobject("Pink")
        text9.set_color(PINK)
        text9.add_updater(lambda d: d.next_to(text8, RIGHT))
        # self.add(text1,text2,text3)
        self.play(Write(text1))#,Write(text2),Write(text3),Write(text4),Write(text5),Write(text6),Write(text7),Write(text8),Write(text9))
        self.play(Write(text2))
        self.play(Write(text3))
        self.play(Write(text4))
        self.play(Write(text5))
        self.play(Write(text6))
        self.play(Write(text7))
        self.play(Write(text8))
        self.play(Write(text9))
        self.wait()


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        self.play(ShowCreation(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))


class WarpSquare(Scene):
    def construct(self):
        square = Square()
        self.play(ApplyPointwiseFunction(
            lambda point: complex_to_R3(np.exp(R3_to_complex(point))),
            square
        ))
        self.wait()


class WriteStuff(Scene):
    CONFIG = { "plane_kwargs" : {
        "x_line_frequency" : 2,
        "y_line_frequency" :2
        }
    }
    def construct(self):
        example_text = TextMobject(
            "Hydraulic Elements \& Transition"
        )
        example_text.scale(1.5)
        my_plane = NumberPlane(**self.plane_kwargs)
        self.add(my_plane)
        self.play(Write(example_text))
        self.wait()


class Shapes(Scene):
    #A few simple shapes
    def construct(self):
        circle = Circle()
        square = Square()
        line=Line(np.array([4,0,0]),np.array([5,0,1]))
        triangle=Polygon(np.array([0,0,0]),np.array([1,1,0]),np.array([1,-1,0]))

        self.add(line)
        self.play(ShowCreation(circle))
        self.play(FadeOut(circle))
        self.play(GrowFromCenter(square))
        self.play(Transform(square,triangle))

class MoreShapes(Scene):
    def construct(self):
        circle = Circle(color=PURPLE_A)
        square = Square(fill_color=GOLD_B, fill_opacity=1, color=GOLD_A)
        square.move_to(UP+LEFT)
        circle.surround(square)
        rectangle = Rectangle(height=2, width=3)
        ellipse=Ellipse(width=3, height=1, color=RED)
        ellipse.shift(2*DOWN+2*RIGHT)
        pointer = CurvedArrow(2*RIGHT,5*RIGHT,color=MAROON_C)
        arrow = Arrow(LEFT,UP)
        arrow.next_to(circle,DOWN+LEFT)
        rectangle.next_to(arrow,DOWN+LEFT)
        ring=Annulus(inner_radius=.5, outer_radius=1, color=BLUE)
        ring.next_to(ellipse, RIGHT)

        self.add(pointer)
        self.play(FadeIn(square))
        self.play(Rotating(square),FadeIn(circle))
        self.play(GrowArrow(arrow))
        self.play(GrowFromCenter(rectangle), GrowFromCenter(ellipse), GrowFromCenter(ring))

class AddingText(Scene):
    #Adding text on the screen
    def construct(self):
        my_first_text=TextMobject("Writing with manim is fun")
        second_line=TextMobject("and easy to do!")
        second_line.next_to(my_first_text,DOWN)
        third_line=TextMobject("for me and you!")
        third_line.next_to(my_first_text,DOWN)

        self.add(my_first_text, second_line)
        self.wait(2)
        self.play(Transform(second_line,third_line))
        self.wait(2)
        second_line.shift(3*DOWN)
        self.play(ApplyMethod(my_first_text.shift,3*UP))

class AddingMoreText(Scene):
    #Playing around with text properties
    def construct(self):
        quote = TextMobject("Imagination is more important than knowledge")
        quote.set_color(RED)
        quote.to_edge(UP)
        quote2 = TextMobject("A person who never made a mistake never tried anything new")
        quote2.set_color(YELLOW)
        author=TextMobject("-Albert Einstein")
        author.scale(0.75)
        author.next_to(quote.get_corner(DOWN+RIGHT),DOWN)

        self.add(quote)
        self.add(author)
        self.wait(2)
        self.play(Transform(quote,quote2),
          ApplyMethod(author.move_to,quote2.get_corner(DOWN+RIGHT)+DOWN+2*LEFT))

        self.play(ApplyMethod(author.scale,1.5))
        author.match_color(quote2)
        self.play(FadeOut(quote))

class BasicEquations(Scene):
    #A short script showing how to use Latex commands
    def construct(self):
        eq1=TextMobject("$\\vec{X}_0 \\cdot \\vec{Y}_1 = 3$")
        eq1.shift(2*UP)
        eq2=TexMobject(r"\vec{F}_{net} = \sum_i \vec{F}_i")
        eq2.shift(2*DOWN)

        self.play(Write(eq1))
        self.play(Write(eq2))

class DCMotor(Scene):
    """docstring for DCMotor."""
    def construct(self):
        eq1_text=["E","=","K_e","\\Phi","n"]
        n_eq1=TexMobject(*eq1_text)
        n_eq1.shift(2*UP)
        n_eq1.set_color(RED)
        eq2_text=["U","-","E","=","I_a","R_a"]
        n0_eq2=TexMobject(*eq2_text)
        n0_eq2.set_color(YELLOW)
        eq3_text=["T","=","K_t","\\Phi","I_a"]
        T_eq3=TexMobject(*eq3_text)
        T_eq3.set_color(GREEN)
        T_eq3.shift(2*DOWN)
        eq4_text=["U","-","K_e","\\Phi","n","=","I_a","R_a"]
        T_eq4=TexMobject(*eq4_text)
        T_eq4.set_color(YELLOW)
        T_eq4.set_color_by_tex_to_color_map({
            "K_e":RED,
            "\\Phi":RED,
            "n":RED
        })
        eq5_text=["U","-","K_e","\\Phi","n","=","\\frac T{K_t\\Phi}","R_a"]
        T_eq5=TexMobject(*eq5_text)
        T_eq5.set_color(YELLOW)
        T_eq5.set_color_by_tex_to_color_map({
            "K_e":RED,
            "\\Phi":RED,
            "n":RED,
            "\\frac T{K_t\\Phi}":GREEN
        })
        eq6=TexMobject(r"n=\frac{U}{K_e\Phi}-\frac{R_a}{K_eK_t{\Phi}^2}T")
        n_eq1.scale(2)
        n0_eq2.scale(2)
        T_eq3.scale(2)
        T_eq4.scale(2)
        T_eq5.scale(2)
        eq6.scale(2)

        self.play(Write(n_eq1),Write(n0_eq2),Write(T_eq3))
        self.play(Transform(n0_eq2,T_eq4))
        self.play(Transform(n0_eq2,T_eq5))
        self.wait(0.5)
        self.play(Transform(n0_eq2,eq6),FadeOut(n_eq1),FadeOut(T_eq3))
        self.wait(1)

class UsingBracesConcise(Scene):
    #A more concise block of code with all columns aligned
    def construct(self):
        eq1_text=["4","x","+","3","y","=","0"]
        eq2_text=["5","x","-","2","y","=","3"]
        eq1_mob=TexMobject(*eq1_text)
        eq2_mob=TexMobject(*eq2_text)
        eq1_mob.set_color_by_tex_to_color_map({
            "x":RED_B,
            "y":GREEN_C
            })
        eq2_mob.set_color_by_tex_to_color_map({
            "x":RED_B,
            "y":GREEN_C
            })
        for i,item in enumerate(eq2_mob):
            item.align_to(eq1_mob[i],LEFT)
        eq1=VGroup(*eq1_mob)
        eq2=VGroup(*eq2_mob)
        eq2.shift(DOWN)
        eq_group=VGroup(eq1,eq2)
        braces=Brace(eq_group,LEFT)
        eq_text = braces.get_text("A pair of equations")

        self.play(Write(eq1),Write(eq2))
        self.play(GrowFromCenter(braces),Write(eq_text))

class UdatersExample(Scene):
    def construct(self):
        decimal = DecimalNumber(
            0,
            show_ellipsis=True,
            num_decimal_places=3,
            include_sign=True,
        )
        decima2 = DecimalNumber(
            0,
            show_ellipsis=True,
            num_decimal_places=3,
        )
        decima2.set_color(RED)
        square = Square().to_edge(UP)

        decimal.add_updater(lambda d: d.next_to(square, RIGHT))
        decima2.add_updater(lambda d: d.next_to(square, LEFT))
        decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))
        decima2.add_updater(lambda d: d.set_value(square.get_center()[1]))
        self.add(square, decimal, decima2)
        self.play(
            square.to_edge, DOWN,
            rate_func = there_and_back,
            run_time = 5,
        )
        self.wait()

# See old_projects folder for many, many more
