# from @鹤翔万里

from manimlib.imports import *
import itertools

def debugTeX(self, texm):
    '''
    用于分析TexMobject的下标。
    在使用时先 self.add(tex) 然后再 debugTeX(self, tex)，导出最后一帧。观察每段字符上的标号，即为下标
    '''
    for i, j in enumerate(texm):
        tex_id = Text(str(i), font="Consolas").scale(0.1).set_color(PURPLE)
        tex_id.move_to(j)
        self.add(tex_id)