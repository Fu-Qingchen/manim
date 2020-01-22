'''
  > File Name        : Science3Min
  > Author           : Fu_Qingchen
'''

from manimlib.imports import *

# python -m manim Science3Mins.py S1 -pl
# python -m manim Science3Mins.py -al

class S1(Scene):
    def construct(self):
        grid = NumberPlane()

        self.add(grid)  # Make sure title is on top of grid
        self.play(
            ShowCreation(grid, run_time=3, lag_ratio=0.1),
        )

        ## S1.1-S1.2
        Gear_Han_image = ImageMobject(
            "E:\GitHub\Project\Science3Min\Resources\Gear_Han.png"
        ).scale(4)

        self.play(FadeInFromDown(Gear_Han_image))
        self.play(ApplyMethod(Gear_Han_image.scale,0.75))
        self.play(ApplyMethod(Gear_Han_image.shift,LEFT*3))
        #self.add(Gear_Han_image)

        Gear_Han_text = TextMobject(
            "\\kaishu 汉铁齿轮"
        ).shift(RIGHT*2.5+UP*1.5).scale(2.25)

        Gear_Han_line = Line(
            np.array([0,2,0]),
            np.array([0,0,0])
        ).shift(RIGHT*0.25)

        Zhengzhou_text = TextMobject(
            "郑州博物馆"
        ).scale(0.75).next_to(Gear_Han_text.get_corner(DL),RIGHT,0).shift(DOWN*0.5)

        Gear_Han_name_group = VGroup(
            Gear_Han_text,
            Zhengzhou_text
        ).next_to(Gear_Han_line).shift(RIGHT*0.25)

        Gear_Han_Detail_text = VGroup(
            TextMobject("汉代时期的机械，推断这些仪器均采用有齿轮。"),
            TextMobject("这件铁齿轮的出土，对于研究我国古代机械工程"),
            TextMobject("的发展，提供了重要的实物资料。")
        ).arrange_submobjects(DOWN,aligned_edge=LEFT,buff=MED_SMALL_BUFF).scale(0.5)
        Gear_Han_Detail_text.next_to(Gear_Han_line.get_corner(DL),RIGHT,0).shift(DOWN*1)

        Gear_Han_group = VGroup(
            Gear_Han_line,Gear_Han_name_group,Gear_Han_Detail_text
        ).shift(DOWN*0.25)

        self.play(
            Write(Gear_Han_line),
            FadeInFrom(Gear_Han_text,LEFT),
            FadeInFrom(Zhengzhou_text,LEFT)
        )
        self.play(Write(Gear_Han_Detail_text))
        self.wait()

        ## S1.3

        ## Cycloidal Gear
        CycloidalGear_image = ImageMobject(
            "E:\GitHub\Project\Science3Min\Resources\CycloidalGear.png"
        ).scale(2.75).shift(LEFT * 3)

        CycloidalGear_text = TextMobject(
            "\\kaishu 摆线齿轮"
        ).shift(RIGHT * 2.5 + UP * 1.5).scale(2.25)

        CycloidalGear_line = Line(
            np.array([0, 2, 0]),
            np.array([0, 0, 0])
        ).shift(RIGHT * 0.25)

        CycloidalGear_time_text = TextMobject(
            "1674"
        ).scale(0.75).next_to(CycloidalGear_text.get_corner(DL), RIGHT, 0).shift(DOWN * 0.5)

        CycloidalGear_name_group = VGroup(
            CycloidalGear_text,
            CycloidalGear_time_text
        ).next_to(CycloidalGear_line).shift(RIGHT * 0.25)

        CycloidalGear_Detail_text = VGroup(
            TextMobject("1674年丹麦天文学家罗默(Ole Rømer)提出用外"),
            TextMobject("摆线齿形能使齿轮等速运动。")
        ).arrange_submobjects(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF).scale(0.5)
        CycloidalGear_Detail_text.next_to(CycloidalGear_line.get_corner(DL), RIGHT, 0).shift(DOWN * 1)

        CycloidalGear_group = VGroup(
            CycloidalGear_line,
            CycloidalGear_name_group,
            CycloidalGear_Detail_text
        ).shift(DOWN * 0.25)

        self.play(
            FadeOutAndShift(Gear_Han_image,LEFT),
            FadeOutAndShift(Gear_Han_group,LEFT)
        )
        self.play(
            FadeInFrom(CycloidalGear_image, RIGHT),
            FadeInFrom(CycloidalGear_group, RIGHT)
        )
        self.wait()

        ## Involute Gear
        Involute_Gear_image = ImageMobject(
            "E:\GitHub\Project\Science3Min\Resources\Involute_Gear.png"
        ).scale(3).shift(LEFT * 3)

        Involute_Gear_text = TextMobject(
            "\\kaishu 渐开线齿轮"
        ).shift(RIGHT * 2.5 + UP * 1.5).scale(2.25)

        Involute_Gear_line = Line(
            np.array([0, 2, 0]),
            np.array([0, 0, 0])
        ).shift(RIGHT * 0.25)

        Involute_Gear_time_text = TextMobject(
            "1765"
        ).scale(0.75).next_to(Involute_Gear_text.get_corner(DL), RIGHT, 0).shift(DOWN * 0.5)

        Involute_Gear_name_group = VGroup(
            Involute_Gear_text,
            Involute_Gear_time_text
        ).next_to(Involute_Gear_line).shift(RIGHT * 0.25)

        Involute_Gear_Detail_text = VGroup(
            TextMobject("1765年，欧拉(Euler)提出渐开线齿形解析研究"),
            TextMobject("的数学基础。")
        ).arrange_submobjects(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF).scale(0.5)
        Involute_Gear_Detail_text.next_to(Involute_Gear_line.get_corner(DL), RIGHT, 0).shift(DOWN * 1)

        Involute_Gear_group = VGroup(
            Involute_Gear_line,
            Involute_Gear_name_group,
            Involute_Gear_Detail_text
        ).shift(DOWN * 0.25)

        self.play(
            FadeOutAndShift(CycloidalGear_image, LEFT),
            FadeOutAndShift(CycloidalGear_group, LEFT)
        )
        self.play(
            FadeInFrom(Involute_Gear_image, RIGHT),
            FadeInFrom(Involute_Gear_group, RIGHT)
        )
        self.wait()
        self.play(FadeOut(Involute_Gear_group),buff = 0.5)
        self.play(
            ApplyMethod(Involute_Gear_image.shift,np.array([3,0,0]))
        )

class S2(Scene):
    def construct(self):
        grid = NumberPlane()

        G30_svg = SVGMobject("E:\GitHub\-\Science3Min\Resources\GB_gear_2M_30T.svg",color = WHITE)\
            .scale(2.5).shift(LEFT*1.96+DOWN*0.05)
        G2030_image_2 = ImageMobject("E:\GitHub\-\Science3Min\Resources\G2030_600_2.png").scale(4)
        self.add(grid)
        self.add(G2030_image_2)
        self.play(
            ShowCreation(G30_svg,run_time = 3),
            FadeOut(G2030_image_2,run_time = 3)
        )
        Invoke_text = TextMobject("渐开线").scale(2).next_to(G30_svg).shift(RIGHT)
        self.play(FadeInFromDown(Invoke_text))
        group = VGroup(
            G30_svg,Invoke_text
        )
        title = TextMobject("如何制造这样的齿轮？").scale(2)
        self.play(
            FadeOutAndShift(Invoke_text,RIGHT),
            FadeOutAndShift(G30_svg,RIGHT)
        )
        self.play(ShowCreation(title,run_time = 2))

class S3_P1(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)

        ## S3_12
        self.S3_12()
        self.S3_13()

        ## S3_13 Part1
    def S3_13(self):
        gear1 = ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_30.png").scale(5)
        d_chinese_text = TextMobject("分度圆").scale(1.75)
        d_english_text = TextMobject("reference circle").next_to(d_chinese_text,DOWN)
        d_text_group = VGroup(
            d_chinese_text,d_english_text
        ).move_to(np.array([0,0,0]))
        self.play(
            FadeInFrom(gear1, DOWN),
            FadeIn(d_text_group)
        )
        reference_circle_circle = Circle(color=WHITE).scale(3)
        self.play(
            ShowCreation(reference_circle_circle)
        )
        self.wait()


    def S3_12(self):
        d_chinese_text = TextMobject("分度圆").scale(1.75)
        d_english_text = TextMobject("reference circle").next_to(d_chinese_text,DOWN)
        d_text_group = VGroup(
            d_chinese_text,d_english_text
        ).next_to(np.array([0,0,0]),LEFT,buff = 1)

        m_chinese_text = TextMobject("模数").scale(1.75)
        m_english_text = TextMobject("module").next_to(m_chinese_text, DOWN)
        m_text_group = VGroup(
            m_chinese_text, m_english_text
        ).next_to(np.array([0, 0, 0]), RIGHT,buff = 1.5)

        self.play(FadeInFrom(d_text_group,DOWN))
        self.play(FadeInFrom(m_text_group,DOWN))
        self.play(
            FadeOutAndShift(m_text_group,RIGHT),
            ApplyMethod(d_text_group.move_to,np.array([0,0,0]))
        )

class S3_P2(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)

        ## S3_13 Part2
        gear1_img = ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_30.png").scale(5)
        d_chinese_text = TextMobject("分度圆").scale(1.75)
        d_english_text = TextMobject("reference circle").next_to(d_chinese_text, DOWN)
        d_text_group = VGroup(
            d_chinese_text, d_english_text
        ).move_to(np.array([0, 0, 0]))
        gear1_reference_circle_circle = Circle(color = WHITE).scale(3)
        gear1_group = Group(
            gear1_img,
            gear1_reference_circle_circle
        )
        self.add(gear1_group)
        self.play(FadeOut(d_text_group))

        #S3_14
        gear12_img = ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_30.png").scale(5).shift(LEFT*2).rotate(PI)
        gear2_img = ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_20.png").scale(3.95)
        gear2_reference_circle_circle = Circle(color = WHITE).scale(2)
        gear2_group = Group(
            gear2_img,gear2_reference_circle_circle
        ).shift((RIGHT*3))
        self.play(
            ApplyMethod(gear1_group.shift,LEFT*2),
            FadeInFrom(gear2_group,RIGHT)
        )

        o1_point = Point(np.array([-2,0,0]))
        o2_point = Point(np.array([3,0,0]))
        o1_text = TexMobject("O_1",color = YELLOW).next_to(o1_point,LEFT)
        o2_text = TexMobject("O_2",color = YELLOW).next_to(o2_point)
        o1o2_line = Line(np.array([-2,0,0]),np.array([3,0,0]),color = YELLOW)
        self.play(
            Rotate(gear1_group,PI,np.array([0,0,1]),run_time=4),
            Rotate(gear2_group,PI/2*3,np.array([0,0,-1]),run_time=4),
            ShowCreation(o1_text),
            ShowCreation(o2_text),
            ShowCreation(o1o2_line)
        )

        ## S3 15
        self.play(
            FadeIn(gear12_img),
            FadeOut(gear1_reference_circle_circle),
            FadeOut(o1_text),
            FadeOut(o2_text),
            FadeOut(o1o2_line),
            ApplyMethod(gear2_img.set_opacity,0.15),
            ApplyMethod(gear2_reference_circle_circle.fade,0.85)
        )

        tooth_thickless_arc = Arc(2*PI/120*11,2*PI/60,color = RED,radius = 3).shift(LEFT*2)
        tooth_thickless_text1 = TextMobject("齿厚(tooth thickness)",color = RED).next_to(tooth_thickless_arc)
        tooth_thickless_text2 = TextMobject("齿厚(tooth thickness)",color = RED).next_to(tooth_thickless_arc)

        spacewidth_arc = Arc(2*PI/120*1,2*PI/60,color = YELLOW,radius = 3).shift(LEFT*2)
        spacewidth_text1 = TextMobject("齿槽宽(spacewidth)",color = YELLOW).next_to(spacewidth_arc)
        spacewidth_text2 = TextMobject("齿槽宽(spacewidth)",color = YELLOW).next_to(spacewidth_arc)

        pitch_arc = Arc(2 * PI / 120 * 111, 2 * PI / 60*2, color=GREEN, radius=3).shift(LEFT * 2)
        pitch_text1 = TextMobject("齿距","(pitch)", color=GREEN).next_to(pitch_arc)
        pitch_text2 = TextMobject("齿距(pitch)", color=GREEN).next_to(pitch_arc)

        self.play(
            ShowCreation(tooth_thickless_text1),ShowCreation(tooth_thickless_text2),
            ShowCreation(spacewidth_text1), ShowCreation(spacewidth_text2),
            ShowCreation(pitch_text1), ShowCreation(pitch_text2)
        )
        self.play(Transform(tooth_thickless_text1,tooth_thickless_arc))
        self.play(Transform(spacewidth_text1, spacewidth_arc))
        self.wait()
        self.play(Transform(pitch_text1, pitch_arc))

        module_formular = TexMobject("{\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\over\\pi}=")\
            .next_to(pitch_text2,DOWN).shift(np.array([0.3,0.1,0]))
        module_text = TextMobject("模数").next_to(module_formular).shift(UP*0.25)
        self.play(ShowCreation(module_formular))
        self.play(ShowCreation(module_text))

class S3_P3(Scene):
    def construct(self):
        ## S3_16
        grid = NumberPlane()
        self.add(grid)

        GBT_1_img = ImageMobject("E:\GitHub\-\Science3Min\Resources\GBT_1357_2008_1.png").scale(5).shift(LEFT*3+DOWN*2)
        GBT_4_img = ImageMobject("E:\GitHub\-\Science3Min\Resources\GBT_1357_2008_4.png").scale(5)\
            .move_to(GBT_1_img).shift(np.array([-0.2,0.2,0]))

        GBT_name_text_group = VGroup(
            TextMobject("\\kaishu 通用机械和重型机械用"),
            TextMobject("\\kaishu 圆柱齿轮\\ 模数")
        ).arrange_submobjects(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF).shift(RIGHT * 2.5 + UP * 1.5)

        GBT_line = Line(
            np.array([0, 3, 0]),
            np.array([0, 1, 0])
        ).shift(RIGHT)

        GBT_lable_text = TextMobject(
            "GB/T 1357-2008 "
        ).scale(0.65).next_to(GBT_name_text_group.get_corner(DL), RIGHT, 0).shift(DOWN * 0.5)

        GBT_name_group = VGroup(
            GBT_name_text_group,
            GBT_lable_text
        ).next_to(GBT_line).shift(RIGHT * 0.25)

        self.play(
            GrowFromEdge(GBT_4_img, DOWN),
            GrowFromEdge(GBT_1_img,DOWN)
        )
        self.play(
            Write(GBT_line),
            FadeInFrom(GBT_name_group, LEFT)
        )
        self.play(
            FadeOutAndShift(GBT_1_img,RIGHT),
            FadeOutAndShift(GBT_name_group,RIGHT),
            FadeOutAndShift(GBT_line,RIGHT),
            ApplyMethod(GBT_4_img.shift,4)
        )
        self.play(
            ApplyMethod(GBT_4_img.scale,1.75,run_time = 0.5)
        )

        rect = Rectangle(color = "#1a73e8",height = 5.5).shift(np.array([-1.5,-1,0]))
        self.play(ShowCreation(rect))

        ## S3_17
        module_type_group = VGroup(
            TextMobject("1"),
            TextMobject("1.25"),
            TextMobject("1.5"),
            TextMobject("2"),
            TextMobject("2.5"),
            TextMobject("3"),
            TextMobject("4"),
            TextMobject("5"),
            TextMobject("6")
        ).arrange_submobjects(DOWN,buff=MED_LARGE_BUFF*2.1).shift(np.array([-1.56,-0.88,0])).scale(0.4)
        self.play(ShowCreation(module_type_group))
        self.play(
            FadeOut(GBT_4_img),
            FadeOut(rect)
        )
        module_example_text = TextMobject("模数").to_corner(UL)
        module_serial_text = TextMobject("模数","系列").scale(2).to_edge(UP).shift(DOWN*0.5)
        module_group = VGroup(
            TextMobject("1"),
            TextMobject("1.25"),
            TextMobject("1.5"),
            TextMobject("2"),
            TextMobject("2.5"),
            TextMobject("3"),
            TextMobject("4"),
            TextMobject("5"),
            TextMobject("6")
        ).arrange_submobjects(RIGHT,buff=MED_LARGE_BUFF*6).scale(0.75)\
            .align_to(module_example_text,LEFT).shift(RIGHT*1.1+UP*0.5)
        self.play(
            ShowCreation(module_serial_text),
            Transform(module_type_group,module_group)
        )

        module_frac_group = VGroup(
            TexMobject("\\frac{60}{60}"),
            TexMobject("\\frac{60}{48}"),
            TexMobject("\\frac{60}{40}"),
            TexMobject("\\frac{60}{30}"),
            TexMobject("\\frac{60}{24}"),
            TexMobject("\\frac{60}{20}"),
            TexMobject("\\frac{60}{60}"),
            TexMobject("\\frac{60}{60}"),
            TexMobject("\\frac{60}{60}")
        ).arrange_submobjects(RIGHT, buff=MED_LARGE_BUFF * 6).scale(0.75) \
            .align_to(module_example_text, LEFT).shift(RIGHT * 1.1 + UP * 0.5)

        module_d_group = VGroup(
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm"),
            TexMobject("60mm")
        ).arrange_submobjects(RIGHT, buff=MED_SMALL_BUFF * 8).scale(0.75) \
            .align_to(module_example_text, LEFT).shift(RIGHT * 1.1 + DOWN * 3)

        gear_group = Group(
            ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_60.png"),
            ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_48.png"),
            ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_40.png"),
            ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_30.png"),
            ImageMobject("E:\GitHub\-\Science3Min\Resources\Gear_24.png")
        ).scale(1.5).arrange_submobjects(RIGHT,buff = MED_LARGE_BUFF*-5.65).shift(DOWN*1.5)


        self.play(
            FadeInFromDown(gear_group)
        )

        ## S3_18
        module_formular_text = TextMobject("模数","=","分度圆直径","/","齿数").scale(1.5).to_edge(UP).shift(DOWN*0.5)
        self.play(Transform(module_serial_text,module_formular_text))
        self.wait()
        self.play(Write(module_d_group,run_time=3))
        self.play(
            Transform(module_type_group,module_frac_group,run_time=3)
        )

class Gird(Scene):
    def construct(self):
        grid = NumberPlane()

        self.add(grid)  # Make sure title is on top of grid
        self.play(
            ShowCreation(grid, run_time=3, lag_ratio=0.1),
        )

class S4_P1(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)  # Make sure title is on top of grid

        ramming_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\S4_Ramming.png").scale(1.2)
        casting_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\S4_Casting.png")\
            .next_to(ramming_image,LEFT).scale(1.2)
        cutting_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\S4_Cutting.png")\
            .next_to(ramming_image,RIGHT).scale(1.2)

        casting_text = TextMobject("铸造").next_to(casting_image,DOWN).scale(0.75)
        ramming_text = TextMobject("冲压").next_to(ramming_image,DOWN).scale(0.75)
        cutting_text = TextMobject("切削").next_to(cutting_image,DOWN).scale(0.75)

        self.play(
            FadeInFromDown(casting_image),
            Write(casting_text)
        )
        self.play(
            FadeInFromDown(ramming_image),
            Write(ramming_text)
        )
        self.play(
            FadeInFromDown(cutting_image),
            Write(cutting_text)
        )
        self.play(
            FadeOutAndShift(casting_image,LEFT),
            FadeOutAndShift(casting_text,LEFT),
            FadeOutAndShift(ramming_image,LEFT),
            FadeOutAndShift(ramming_text,LEFT),
            ApplyMethod(cutting_image.shift,np.array([-4.5,2.5,0])),
            ApplyMethod(cutting_text.shift,np.array([-4.5,2.6,0]))
        )

        fangxing_sceen = ScreenRectangle(height = 3.125,fill_color = BLACK).next_to(cutting_text.get_corner(DL),DL)
        zhancheng_sceen = ScreenRectangle(height = 3.125,fill_color = BLACK).next_to(cutting_text.get_corner(DR),DR)

        fangxing_start_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\Fangxing360P.1.png")\
            .scale(1.5).move_to(fangxing_sceen.get_center())
        zhancheng_start_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\zhancheng360P.1.png")\
            .scale(1.5).move_to(zhancheng_sceen.get_center())
        fangxing_after_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\Fangxing.1.png") \
            .scale(1.5).move_to(fangxing_sceen.get_center())
        zhancheng_after_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\zhancheng.1.png") \
            .scale(1.5).move_to(zhancheng_sceen.get_center())

        fangxing_text = TextMobject("仿形法").next_to(fangxing_start_image,DOWN).scale(0.75)
        zhancheng_text = TextMobject("展成法").next_to(zhancheng_start_image,DOWN).scale(0.75)
        self.add(fangxing_after_image,zhancheng_after_image)
        self.play(
            ShowCreation(fangxing_sceen),
            FadeIn(fangxing_start_image),
            Write(fangxing_text)
        )
        self.play(
            ShowCreation(zhancheng_sceen),
            FadeIn(zhancheng_start_image),
            Write(zhancheng_text)
        )
        self.play(
            FadeOut(zhancheng_start_image),
            FadeOut(fangxing_start_image)
        )
        self.wait()

class S4_P2(Scene):
    def construct(self):
        grid = NumberPlane()
        fangxing_image = ImageMobject("E:\GitHub\-\Science3Min\Resources\Fangxing.180.png").scale(4)
        G30_svg = SVGMobject("E:\GitHub\-\Science3Min\Resources\GB_gear_2M_30T.svg", color=YELLOW) \
            .scale(3).shift(DOWN * 4.4)
        #self.add(grid)
        #self.add(fangxing_image)
        self.play(ShowCreation(G30_svg,run_time = 6))

class S5(Scene):
    def construct(self):
        image = ImageMobject("E:\GitHub\-\Science3Min\Resources\zhancheng.600.png").scale(4)
        text1 = TextMobject("齿坯").scale(1.5)
        text2 = TextMobject("刀具").scale(1.5).next_to(text1,LEFT,buff=MED_LARGE_BUFF*8)
        self.add(image)
        self.play(Write(text2),Write(text1))

class LOGO(Scene):
    CONFIG = {
        "Author": "@Fu\\_Qingchen",
        "author_colors": [BLUE, YELLOW, ORANGE, RED],
    }
    def construct(self):
        logo = ImageMobject("E:\GitHub\-\Science3Min\Resources\V4.3.png").scale(1.5).shift(UP*0.5)
        author = TextMobject(
            self.Author,
            tex_to_color_map={self.Author: self.author_colors}
        ).scale(1.5).next_to(logo,DOWN).shift(DOWN*0.5)
        self.play(FadeInFromDown(logo))
        self.play(Write(author))
        self.play(FadeOut(logo),FadeOut(name))


