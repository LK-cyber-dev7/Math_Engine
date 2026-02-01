from __future__ import annotations

import turtle
import math

MAX_JUMP = 20

def _max_min(func, x1, x2, step):
    x = x1
    max_val = func(x)
    min_val = func(x)
    while x <= x2:
        val = func(x)
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        x += step

    return max_val,min_val

def scale(r1_start, r1_end, r1_point, r2_start, r2_end):
    if r1_start == r1_end:
        return (r2_start + r2_end) / 2
    return r2_start + ((r1_point - r1_start) * (r2_end - r2_start)) / (r1_end - r1_start)

def get_step(r1_start,r1_end,r2_start,r2_end, step=1):
    return (step*(r2_end-r2_start))/(r1_end-r1_start)

def log_10(n):
    result = 1
    while result < n:
        result *= 10

    return result//10

def nice_step(span, target_ticks=8):
    raw = span / target_ticks
    power = 10 ** math.floor(math.log10(raw))
    fraction = raw / power

    if fraction < 1.5:
        nice = 1
    elif fraction < 3:
        nice = 2
    elif fraction < 7:
        nice = 5
    else:
        nice = 10

    return nice * power

def first_tick(start, step):
    return math.ceil(start / step) * step

class Graph:
    def __init__(self, func, width=800, height=600, name="Function Graph", padding:int=25):
        self.func = func

        self.__screen = turtle.Screen()
        self.width = width
        self.height = height
        self.__screen.setup(width=self.width+(padding*2), height=self.height+(padding*2), startx=100, starty=20)
        self.__screen.title(name)

        self.__t = turtle.Turtle()
        self.__t.hideturtle()
        self.__t.speed(0)

    def _draw_axis(self, x=None, y=None) -> None:
        """
        Draw X and Y axes.
        x, y: pixel coordinates of the origin (intersection point)
        """
        t = self.__t  # shorthand
        t.penup()

        # Draw vertical Y-axis at x
        if x is not None:
            t.goto(x, -self.height / 2)  # bottom
            t.pendown()
            t.goto(x, self.height / 2)  # top
            t.penup()

        # Draw horizontal X-axis at y
        if y is not None:
            t.goto(-self.width / 2, y)  # left
            t.pendown()
            t.goto(self.width / 2, y)  # right
            t.penup()

    def origin(self, x1, x2):
        step = get_step(-self.width/2, self.width/2, x1, x2)
        y2, y1 = _max_min(func=self.func, x1=x1, x2=x2, step=step)
        print(y1,y2)
        if y2 < 0:
            y = scale(y1, 0, 0, -self.height/2, self.height/2)
        elif y1 > 0:
            y = scale(0, y2, 0, -self.height/2, self.height/2)
        else:
            y = scale(y1, y2, 0, -self.height/2, self.height/2)

        x = scale(x1, x2, 0, -self.width/2, self.width/2)
        print(x,y)
        if abs(x) > self.width/2:
            self._draw_axis(y=y)
        else:
            self._draw_axis(x=x, y=y)

        self.label(x1,x2,y1,y2,y,x)
        return y1,y2

    def label(self, x1, x2, y1, y2, x_axis, y_axis):
        step_x = nice_step(x2 - x1)
        step_y = nice_step(y2 - y1)

        # X-axis labels
        x_val = first_tick(x1, step_x)
        while x_val <= x2:
            x = scale(x1, x2, x_val, -self.width / 2, self.width / 2)
            self.__t.penup()
            self.__t.goto(x, x_axis - 15)
            self.__t.write(f"{x_val:g}", align="center")
            x_val += step_x

        # Y-axis labels
        y_val = first_tick(y1, step_y)
        while y_val <= y2:
            y = scale(y1, y2, y_val, -self.height / 2, self.height / 2)
            self.__t.penup()
            self.__t.goto(y_axis - 10, y)
            self.__t.write(f"{y_val:g}", align="right")
            y_val += step_y

    def draw_graph(self, x1, x2):
        t = self.__t
        self.__screen.tracer(0)
        y1,y2 = self.origin(x1, x2)
        step = get_step(-self.width//2, self.width//2, x1, x2)
        x_val = x1
        x = -self.width//2
        last_y = None
        while x_val <= x2 and x <= self.width/2:
            y_val = self.func(x_val)
            y = int(scale(y1, y2, y_val, -self.height//2, self.height//2))
            if last_y is None or abs(y - last_y) > MAX_JUMP:
                t.penup()
                t.goto(x, y)
                t.pendown()
            else:
                t.goto(x, y)

            last_y = y
            x += 1
            x_val += step
        self.__screen.update()

    def show(self):
        self.__screen.mainloop()

# sim = turtle.Screen()
# tim = turtle.Turtle()
# tim.hideturtle()
# tim.speed(0)
# plot(tim, 0, 40)
# sim.mainloop()