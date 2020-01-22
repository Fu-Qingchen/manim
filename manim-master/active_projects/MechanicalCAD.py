'''
  > File Name        : MechanicalCAD
  > Author           : Fu_Qingchen
  > Creating Time    : 2019-12-25
'''

from manimlib.imports import *

# python -m manim MechanicalCAD.py P -pl

# 机械原理课程设计机构

class P(GraphScene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)
        pointA_dot = Dot()
        pointA_text = TextMobject("A").next_to(pointA_dot).scale(0.8)
        pointB_dot = Dot(np.array([0, -3, 0]))
        pointB_text = TextMobject("B").next_to(pointB_dot).scale(0.8)
        pointC_dot = Dot(np.array([-1, 0, 0]))

        pointC_text = TextMobject("C").next_to(pointC_dot).scale(0.8)

        def get_pointD(array_c):
            for x_d in range(-5000,5000):
                x_d = x_d * 1.0/1000
                y_d = math.sqrt(25 - x_d**2) - 3
                if abs(x_d * (array_c[1] + 3)) - abs(array_c[0] * (y_d + 3)) < 0.001:
                    return x_d*(-array_c[0]/abs(array_c[0])), y_d

        def get_pointE(array_d):
            x_e = -math.sqrt(4 - (array_d[1] - 1)**2) + array_d[0]
            return np.array([x_e, 1, 0])
        x_d, y_d = get_pointD(pointC_dot.get_center())
        pointD_dot = Dot(np.array([float(x_d), float(y_d), 0]))
        pointD_text = TextMobject("D").scale(0.8).next_to(pointD_dot)
        xe_number = abs(get_pointE(pointD_dot.get_center())[0])
        xe_coor = DecimalNumber(xe_number, color = YELLOW, num_decimal_places = 2).scale(0.8)
        pointE_dot = Dot(get_pointE(pointD_dot.get_center()))

        pointE_text = TextMobject("E").scale(0.8).next_to(pointE_dot)
        lineAC_line = Line(pointA_dot.get_center(), pointC_dot.get_center())
        lineBD_line = Line(pointB_dot.get_center(), pointD_dot.get_center())
        lineDE_line = Line(pointD_dot.get_center(), pointE_dot.get_center())
        lineS_line = Line(UP*2.5, np.array([pointE_dot.get_center()[0], 2.5, 0]), color = YELLOW)
        lineS_text = TextMobject("s = ", "0.00", color = YELLOW).scale(0.8)

        S_group = VGroup(lineS_text[0], xe_coor.move_to(lineS_text[1]))
        pointC_text.add_updater(lambda x: x.next_to(pointC_dot))
        pointD_dot.add_updater(lambda d: d.move_to(np.array([get_pointD(pointC_dot.get_center())[0],
                                                             get_pointD(pointC_dot.get_center())[1],
                                                             0])))

        pointD_text.add_updater(lambda d:d.next_to(pointD_dot))
        pointE_dot.add_updater(lambda e:e.move_to(get_pointE(pointD_dot.get_center())))
        pointE_text.add_updater(lambda e:e.next_to(pointE_dot))
        lineAC_line.add_updater(lambda ac:ac.put_start_and_end_on(pointA_dot.get_center(), pointC_dot.get_center()))
        lineBD_line.add_updater(lambda bd:bd.put_start_and_end_on(pointB_dot.get_center(), pointD_dot.get_center()))
        lineDE_line.add_updater(lambda de:de.put_start_and_end_on(pointD_dot.get_center(), pointE_dot.get_center()))
        lineS_line.add_updater(lambda s:s.put_start_and_end_on(UP*2.5, np.array([pointE_dot.get_center()[0], 2.5, 0])))
        xe_coor.add_updater(lambda xe:xe.set_value(pointE_dot.get_center()[0]*-1))
        S_group.add_updater(lambda sg:sg.next_to(lineS_line, UP))


        Mechanical_group = VGroup(lineAC_line, lineBD_line, lineDE_line,
                 lineS_line, S_group,
                 pointA_dot, pointA_text,
                 pointB_dot, pointB_text,
                 pointC_dot, pointC_text,
                 pointD_dot, pointD_text,
                 pointE_dot, pointE_text)

        self.add(Mechanical_group)
        self.play(Rotating(pointC_dot, radians=TAU, about_point=pointA_dot.get_center()), run_time = 10)



# 机械CAD

## 线条图处理技术

### 基本图形扫描转换算法

class C2_1_2(MovingCameraScene):
    def construct(self):
        grid = NumberPlane()
        dots = VGroup()

        point1_array = np.array([-50, -30, 0])
        point2_array = np.array([50, 30, 0])

        point1_point = Dot(point1_array)
        point2_point = Dot(point2_array)

        line_example = Line(point1_array, point2_array)

        self.add(grid)
        dots.add(point1_point, point2_point)

        if abs(point1_array[0]-point2_array[0]) < abs(point1_array[1]-point2_array[1]):
            steps = abs(point1_array[1] - point2_array[1])
        else:
            steps = abs(point1_array[0]-point2_array[0])

        deltaX = (point1_array[0]-point2_array[0])/steps
        deltaY = (point1_array[1]-point2_array[1])/steps

        x = point2_array[0]
        y = point2_array[1]

        print(range(0, steps))
        for i in range(0, steps):
            x = x + deltaX
            y = y + deltaY
            dots.add(Dot(np.array([int(x + 0.5), int(y + 0.5), 0])))

        self.play(ShowCreation(dots))
        self.play(ApplyMethod(self.camera.frame.scale, 10),run_time = 2)
        self.play(ShowCreation(line_example))
