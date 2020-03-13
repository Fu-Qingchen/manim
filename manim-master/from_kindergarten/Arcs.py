# from @cigar666

from manimlib.imports import *

class Arcs(VGroup):
    '''
    饼状图
    '''
    CONFIG = {
        'colors': [RED, YELLOW, BLUE, PINK],
        'radius': 1,
        'start_angle':0,
        'angle_list': [30 * DEGREES, 60 * DEGREES, 90 * DEGREES],   #饼图的各个部分
        'stroke_width': 40,     #好像是50的时候全部填充满

    }

    def __init__(self, **kwargs):

        VMobject.__init__(self, **kwargs)
        self.create_arcs()

    def create_arcs(self, **kwargs):
        angle = self.start_angle
        colors = color_gradient(self.colors, len(self.angle_list))
        for i in range(len(self.angle_list)):
            self.add(Arc(radius=self.radius, start_angle=angle, angle=self.angle_list[i], color=colors[i], stroke_width=self.stroke_width, **kwargs))
            angle += self.angle_list[i]

class Arcs_Test(Scene):

    def construct(self):

        arcs_01 = Arcs(stroke_width=80).shift(LEFT * 4.5)
        arcs_02 = Arcs(angle_list=np.array([10, 20, 30, 40, 50, 60, 70, 80]) * DEGREES, stroke_width=90)
        arcs_03 = Arcs(angle_list=np.array([10, 15, 20, 30]) * DEGREES, stroke_width=200).set_stroke(opacity=0.25).shift(RIGHT * 4)
        arcs_04 = Arcs(angle_list=np.array([10, 15, 20, 30]) * DEGREES, radius=2, stroke_width=10).shift(RIGHT * 4)
        group = VGroup(arcs_01, arcs_02, arcs_03, arcs_04)

        self.play(ShowCreation(arcs_01))
        self.wait()
        self.play(ShowCreation(arcs_02))
        self.wait()
        self.play(ShowCreation(VGroup(arcs_03, arcs_04)))
        self.add(group)

        self.wait(4)

