from big_ol_pile_of_manim_imports import *

# python -m manim helloworld.py HelloWorld -w

class LoveDeathRobots(Scene):

    def construct(self):

        ## Make Love
        circle_loveLeft = Circle(color=RED, fill_color=RED, fill_opacity=0.4)
        circle_loveRight = Circle(color=RED, fill_color=RED, fill_opacity=0.4)
        circle_loveLeft.shift(np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0]))
        circle_loveRight.shift(np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0]))
        square_love = Square(color=RED, fill_color=RED, fill_opacity=0.25)
        square_love.rotate(np.pi/4)
        love_Group = VGroup(circle_loveLeft, circle_loveRight, square_love)
        self.play(Write(love_Group))
        self.play(ApplyMethod(love_Group.shift, np.array([-4, 0, 0])))

        ## Make Death
        rect1 = Rectangle(height=4, width=1, color=RED, fill_color=RED, fill_opacity=0.4)
        rect1.rotate(np.pi/4)
        rect2 = Rectangle(height=4, width=1, color=RED, fill_color=RED, fill_opacity=0.4)
        rect2.rotate(-np.pi/4)
        death_group = VGroup(rect1, rect2)
        death_group.scale((4/np.sqrt(2))/(3*np.sqrt(2)/2+1))
        self.play(Write(death_group))

        ## Make Robots
        rect = Square(color=RED, fill_color=RED, fill_opacity=0.4)
        circle_robotLeft = Circle(color=BLACK, fill_color=BLACK, fill_opacity=0.2)
        circle_robotRight = Circle(color=BLACK, fill_color=BLACK, fill_opacity=0.2)
        circle_robotLeft.scale(.25)
        circle_robotRight.scale(.25)
        circle_robotLeft.shift(np.array([-0.4, 0.4, 0]))
        circle_robotRight.shift(np.array([0.4, 0.4, 0]))
        robot_Group = VGroup(rect, circle_robotLeft, circle_robotRight)
        robot_Group.scale((3*np.sqrt(2)/2+1)/2)
        robot_Group.shift(np.array([4, 0, 0]))
        self.play(Write(robot_Group))

        ## Make Line
        picture_Group = VGroup(love_Group, death_group, robot_Group)
        self.play(ApplyMethod(love_Group.set_fill, RED,1), ApplyMethod(death_group.set_fill, RED,1),
                  ApplyMethod(robot_Group.set_fill, RED,1), ApplyMethod(circle_robotLeft.set_fill, BLACK, 1),
                  ApplyMethod(circle_robotRight.set_fill, BLACK, 1))
        line = Line(np.array([-6, 0, 0]),np.array([6, 0, 0]), color=RED)
        line.next_to(picture_Group, DOWN)
        # line.shift(DOWN)
        self.play(Write(line))
        text = TextMobject("LOVE, DEATH \& ROBOTS", color=RED)
        text.scale(1.75)
        text.next_to(picture_Group, DOWN)
        # text.shift(DOWN)
        self.play(Transform(line, text))
        all_group = VGroup(picture_Group, line)
        self.play(ApplyMethod(all_group.shift, UP*0.5))

class Shoot(Scene):

    def construct(self):

        ## Making aim-scope
        circle_outer = Circle(color=BLUE)
        circle_inner = Circle(color=RED, fill_color=RED, fill_opacity=1)
        circle_inner.scale(.1)
        line_h = Line(np.array([-1, 0, 0]), np.array([1, 0, 0]), color=RED)
        line_v = Line(np.array([0, -1, 0]), np.array([0, 1, 0]), color=RED)
        group_aimScope = VGroup(circle_outer, circle_inner, line_h, line_v)

        ## Making target
        target_list = []
        for i in range(3):
            for j in range(5):
                circle_target = Circle(color=YELLOW, fill_color=YELLOW, fill_opacity=.4)
                circle_target.scale(.4)
                circle_target.shift(np.array([-4 + 2*j, 2 - 2*i, 0]))
                self.play(FadeIn(circle_target))
                target_list.append(circle_target)

        ## Move and Shoot
        def shoot(i, j):
            self.play(ApplyMethod(group_aimScope.move_to, np.array([-4 + 2*i, 2 - 2*j, 0])))
            self.play(ApplyMethod(target_list[i + 5*j].set_fill, GREY),ApplyMethod(target_list[i + 5*j].set_color, GREY))
            text =  TextMobject("(%d,%d)" % (i, j), color=GREY)
            text.next_to(target_list[i + 5*j], DOWN)
            self.play(FadeIn(text))


        self.add(group_aimScope)
        self.wait()
        shoot(0, 0)
        shoot(1, 1)
        shoot(3, 2)
        shoot(2, 1)


class BasicShapes(Scene):

    def construct(self):

        ## Object
        ring = Annulus(inner_radius=.4, outer_radius=1, color=BLUE)
        square = Square(color=ORANGE, fill_color=ORANGE, fill_opacity=0.5)
        rect = Rectangle(height=3.2, width=1.2, color=PINK, fill_color=PINK, fill_opacity=0.5)

        line1 = Line(np.array([0, 3.6, 0]), np.array([0, 2, 0]))
        line2 = Line(np.array([-1, 2, 0]), np.array([-1, -1, 0]))
        line3 = Line(np.array([1, 2, 0]), np.array([1, 0.5, 0]))

        ## Position
        ring.shift(UP * 2)
        square.shift(LEFT + DOWN * 2)
        rect.shift(RIGHT + DOWN * (3.2/2 - 0.5))

        ## Animation
        self.add(line1)
        self.play(GrowFromCenter(ring))
        self.wait(0.5)
        self.play(FadeIn(line2),FadeIn(line3))
        self.wait(0.5)
        self.play(FadeInFromDown(square))
        self.play(FadeInFromDown(rect))
        self.wait(0.5)


class HelloWorld(Scene):
    def construct(self):
        helloworld_Text = TextMobject("Hello World", color=RED)

        rectangle = Rectangle(color=BLUE)
        rectangle.surround(helloworld_Text)

        group1 = VGroup(helloworld_Text,rectangle)

        helloManim_Text = TextMobject("Hello Manim", color=BLUE)
        helloManim_Text.scale(2.5)

        self.play(Write(helloworld_Text))
        self.wait(1)
        self.play(FadeIn(rectangle))
        self.wait(1)
        self.play(ApplyMethod(group1.scale, 2.5))
        self.wait(1)
        self.play(Transform(helloworld_Text, helloManim_Text))
        self.wait(1)

class Latex(Scene):
    def construct(self):
        title = TextMobject("\\This is some \\LaTeX")
        basel = TexMobject(
            "\\sum_{n=1}^{\\infty}\\frac{1}{n^2}=\\frac{\\pi^2}{6}"
        )
        basel.next_to(title, DOWN)
        self.play(Write(title), FadeInFrom(basel, DOWN))
