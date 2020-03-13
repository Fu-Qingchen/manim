from manimlib.imports import *

class LOGO(Scene):
    '''
    显示作者相关信息
    '''
    CONFIG = {
        "author": "@Fu\\_Qingchen",
        "colors": [BLUE, YELLOW, ORANGE, YELLOW]
    }

    def construct(self):
        logo = ImageMobject("E:\GitHub\Project\Science3Min\Resources\V4.3.png").scale(1.5).shift(UP*0.5)
        author = TextMobject(
            self.author, tex_to_color_map = {self.author: self.colors}
        ).scale(1.25).next_to(logo, DOWN).shift(DOWN*0.5)
        self.play(FadeInFrom(logo))
        self.play(Write(author))
        self.play(FadeOut(logo), FadeOut(author))
