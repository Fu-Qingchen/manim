# from @魔与方(ThreeDVector) & @Fu_Qingchen(ThreeDVectorField)

from manimlib.imports import *


DEFAULT_SCALAR_FIELD_COLORS = [BLUE_E, GREEN, YELLOW, RED]


class ThreeDVector(VMobject):
    """
    这个类的原理：
    向量主体是圆柱体，向量顶端是圆锥，
    圆柱体和圆锥的底面是用多边形近似的圆，
    圆柱体侧面也是用多边形拼接的，
    所以d_theta控制多边形近似圆的精度的
    """
    CONFIG = {
    "start":None,           #起点
    "end":None,             #终点
    "fill_opacity":0.5,     #透明度
    "bottom_radius":0.025,  #圆柱体的底面半径
    "bottom_circle_d_theta":0.7,
    "tip_bottom_radius":0.05,#顶端圆锥的底面半径,建议是bottom_radius的两倍
    "tip_bottom_circle_d_theta":0.7,
    "tip_length":0.1,       #顶端圆锥的长(圆锥的高)，建议是tip_bottom_radius的两倍
    }

    def __init__(self, direction,**kwargs):
        VMobject.__init__(self, **kwargs)
        self.start = ORIGIN
        self.end = direction
        self.n = self.end - self.start

        self.points1 = self.get_three_d_cicle_points(
            self.start,self.n,self.bottom_radius,self.bottom_circle_d_theta)
        self.points2 = self.get_three_d_cicle_points(
            self.end,self.n,self.bottom_radius,self.bottom_circle_d_theta)

        self.get_bottom_circles()   #圆柱体底圆
        self.get_side()             #圆柱体侧面
        self.get_tip()              #箭头

        self.add(self.cir1,self.cir2,self.side,self.tip)

    def dot_product(self,a,b):
        return np.sum(a*b)

    def vector_product(self,a,b):
        return np.array([\
            a[1]*b[2]-a[2]*b[1],\
            a[2]*b[0]-a[0]*b[2],\
            a[0]*b[1]-a[1]*b[0]\
            ])

    def get_length(self,a):
        return np.sqrt(np.sum(np.square(a)))

    def get_unit_vector(self,a):
        return a/self.get_length(a)

    def get_three_d_cicle_points(self,c,n,r,d_theta):
        if self.get_length(n) == 0.:
            raise AssertionError("Error: The length of 'n' equals to '0'")

        n0 = self.get_unit_vector(n)
        if (self.get_length(n0)==1.) and (n0[0]==1.):
            a = self.vector_product(n0,np.array([0,1,0]))
        else:
            a = self.vector_product(n0,np.array([1,0,0]))

        b = self.vector_product(n0,a)

        def func(theta,a=a,b=b):
            s1 = np.array([[r*np.cos(theta)],
                           [r*np.sin(theta)]])
            s2 = np.vstack((a,b))
            s2 = s1*s2
            s3 = s2[0]+s2[1]+c
            return s3

        points = []
        for theta in np.arange(0,2*np.pi+d_theta,d_theta):
            points.append(func(theta=theta))
        return points

    def get_bottom_circles(self):
        self.cir1 = Polygon(
            *self.points1,
            color=self.color,
            fill_opacity=self.fill_opacity,
            stroke_width=1,
            stroke_color=self.color)
        self.cir2 = Polygon(
            *self.points2,
            color=self.color,
            fill_opacity=self.fill_opacity,
            stroke_width=1,
            stroke_color=self.color)

    def get_side(self):
        if not (len(self.points1)==len(self.points2)):
            raise AssertionError("Error:points1 doesn't equal to points2 ")
        list_length = len(self.points1)
        self.side = VGroup()
        list1 = list(range(0,list_length-1))
        list1.append(-1)
        for i in list1:
            polygon = Polygon(
                *[self.points1[i],self.points2[i],self.points2[i+1],self.points1[i+1]],
                color=self.color,
                fill_opacity=self.fill_opacity,
                stroke_width=0)
            self.side.add(polygon)
        
    def get_tip(self):
        self.tip = VGroup()
        points = self.get_three_d_cicle_points(
            self.end,self.n,self.tip_bottom_radius,self.bottom_circle_d_theta)
        bottom = Polygon(
            *points,
            color=self.color,
            fill_opacity=self.fill_opacity,
            stroke_width=1,
            stroke_color=self.color)
        self.tip.add(bottom)

        vertex = self.end + self.get_unit_vector(self.n)*self.tip_length
        list1 = list(range(0,len(points)-1))
        list1.append(-1)
        for i in list1:
            polygon = Polygon(
                *[points[i],vertex,points[i+1]],
                color=self.color,
                fill_opacity=self.fill_opacity,
                stroke_width=0)
            self.tip.add(polygon)


class ThreeDVectorField(VGroup):
    CONFIG = {
        "delta_x": 1,
        "delta_y": 1,
        "delta_z": 1,
        "x_min": -5,
        "x_max": 5,
        "y_min": -5,
        "y_max": 5,
        "z_min": -3,
        "z_max": 3,
        "min_magnitude": 0,
        "max_magnitude": 2,
        "colors": DEFAULT_SCALAR_FIELD_COLORS,
        # Takes in actual norm, spits out displayed norm
        "length_func": lambda norm: 0.45 * sigmoid(norm),
        "opacity": 1.0,
        "vector_config": {},
    }

    def __init__(self, func, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.func = func
        self.rgb_gradient_function = get_rgb_gradient_function(
            self.min_magnitude,
            self.max_magnitude,
            self.colors,
            flip_alphas=False
        )
        x_range = np.arange(
            self.x_min,
            self.x_max + self.delta_x,
            self.delta_x
        )
        y_range = np.arange(
            self.y_min,
            self.y_max + self.delta_y,
            self.delta_y
        )
        z_range = np.arange(
            self.z_min,
            self.z_max + self.delta_z,
            self.delta_z
        )
        for x, y in it.product(x_range, y_range):
            for z in z_range:
                point = x * RIGHT + y * UP + z*Z_AXIS
                self.add(self.get_vector(point))
        self.set_opacity(self.opacity)

    def get_vector(self, point, **kwargs):
        output = np.array(self.func(point))
        norm = get_norm(output)
        if norm == 0:
            output *= 0
        else:
            output *= self.length_func(norm) / norm
        vector_config = dict(self.vector_config)
        vector_config.update(kwargs)
        vect = ThreeDVector(output, **vector_config)
        vect.shift(point)
        fill_color = rgb_to_color(
            self.rgb_gradient_function(np.array([norm]))[0]
        )
        vect.set_color(fill_color)
        return vect