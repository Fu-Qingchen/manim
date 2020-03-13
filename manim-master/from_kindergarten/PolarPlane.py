# from @Fu_Qingchen

from manimlib.imports import *

class PolarPlane(Axes):
    '''
    极坐标平面
    '''
    CONFIG = {
        "grid_style": {
            "color": BLUE,
            "stroke_width": 1,
        },
        "max_radius": 25,
        "rays_number": 16,      # 射线根数
        "circle_frequency": 1,
    }
    def __init__(self, **kwargs):
        VGroup.__init__(self, **kwargs)
        digest_config(self, kwargs)
        self.init_background_lines()

    def init_background_lines(self):
        self.grid = self.get_polar_grid()
        self.add_to_back(self.grid)
        
    def get_polar_grid(self):
        circles = VGroup(*[
            Circle(radius=r, **self.grid_style)
            for r in np.arange(self.circle_frequency, int(self.max_radius), self.circle_frequency)
        ])
        rays = VGroup(*[
            Line(
                ORIGIN, self.max_radius * RIGHT,
                **self.grid_style,
            ).rotate(angle, about_point=ORIGIN)
            for angle in np.arange(0, TAU, TAU / self.rays_number)
        ])
        labels = VGroup(*[
            Integer(n).scale(0.5).next_to(
                RIGHT*n, DR, SMALL_BUFF
            )
            for n in range(1, int(self.max_radius))
        ])
        return VGroup(
            circles, rays, labels
        )

    def coords_to_point(self, r, theta):
        return np.array([r * np.cos(theta), r * np.sin(theta), 0])


class TestPolarPlane(Scene):
    def construct(self):
        polor = PolarPlane()
        grid = NumberPlane().set_opacity(0.2)
        point_in_polor = Dot(polor.coords_to_point(2, 1)).set_color(RED)
        point_in_grid = Dot(grid.coords_to_point(2, -1)).set_color(YELLOW)
        self.play(ShowCreation(polor))
        self.play(ShowCreation(grid))
        self.play(ShowCreation(point_in_polor))
        self.play(ShowCreation(point_in_grid))

