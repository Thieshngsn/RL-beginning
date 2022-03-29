from pickle import TUPLE3
from events import WatchedKey
import turtle
import random
import numpy

STEPS = 5000


class Pong_env(object):
    """
    Observations:
    size: 5
    
    0: tile x  [0;1]
    1: ball x  [0;1]
    2: ball y  [0;1]
    3: ball vx [-1;1]
    4: ball vy -1 or 1
    """
    def __init__(self):
        self.action_space_sample = numpy.array([0,0])
        self.observation_space_sample = numpy.array([0,0,0,0])
        
        self.x_bound = 270
        self.width=600
        self.height=600
        self.win = turtle.Screen()    # Create a screen
        self.win.title('Pong')      # Set the title to paddle
        #self.win.bgcolor('black')     # Set the color to black
        self.win.tracer(0)
        self.win.setup(width=600, height=600)   # Set the width and height to 600
        # Keyboard Control
        self.left = WatchedKey('Left')
        self.right = WatchedKey('Right')
        self.win.listen()


        self.tile = turtle.Turtle()    # Create a turtle object
        self.tile.shape('square')      # Select a square shape
        self.tile.speed(0)           
        self.tile.shapesize(stretch_wid=1, stretch_len=5)   # Streach the length of square by 5 
        self.tile.penup()
        self.tile.color('black')       # Set the color to white
        self.tile.goto(0, -265)        # Place the shape on bottom of the screen

        self.ball = turtle.Turtle()      # Create a turtle object
        self.ball.speed(0)
        self.ball.shape('circle')        # Select a circle shape
        self.ball.color('red')           # Set the color to red
        self.ball.penup()
        self.ball.goto(0, -200)           # Place the shape in middle
        #print(len(self.win.turtles()))
        if random.random() > 0.5:
            self.ball.dx = random.random() *8 + 2   # ball's x-axis velocity 
            
        else:
            self.ball.dx = -random.random() *8 -2
        self.ball.dy = 5  # ball's y-axis velocity 
        
        self.is_done = False
        self.steps_done = 0
        self.hit = False
        

    # Pong Movement
    def move(self, act) -> None:
        x = self.tile.xcor()
        m = (act-0.5) * 2
        if (m ==-1 and x > -self.x_bound) or (m == 1 and x < self.x_bound):
            self.tile.setx(x + (m * 10))
            
    def tile_within_bounds(self) -> bool:
        if -self.x_bound-5 < self.tile.xcor() < self.x_bound+5:
            return True
        return False
    
    #updates the balls location
    def update(self) -> None:
        self.ball.setx(self.ball.xcor() + self.ball.dx)  # update the ball's x-location using velocity
        self.ball.sety(self.ball.ycor() + self.ball.dy)  # update the ball's y-location using velocity  
        
        # Ball-Walls collision  
        if self.ball.xcor() > (self.width/2)-10:    # If ball touch the right wall
            self.ball.setx((self.width/2)-10)
            self.ball.dx *= -1        # Reverse the x-axis velocity

        if self.ball.xcor() < -(self.width/2)+10:   # If ball touch the left wall
            self.ball.setx(-(self.width/2)+10)
            self.ball.dx *= -1        # Reverse the x-axis velocity

        if self.ball.ycor() > (self.height/2)-10:    # If ball touch the upper wall
            self.ball.sety((self.height/2)-10)
            self.ball.dy *= -1        # Reverse the y-axis velocity
            #ball.dx = random.random() * 10-5

        # Ball-Paddle collision
        if abs(self.ball.ycor() + (self.height/2)-50) < 2 and abs(self.tile.xcor() - self.ball.xcor()) < 55:
            self.hit = True
            self.ball.dy *= -1
            if random.random() > 0.5:
                self.ball.dx = random.random() * 8 + 2  # ball's x-axis velocity
            else:
                self.ball.dx = -random.random() * 8 -2   # ball's x-axis velocity

        # Ball-Ground collison            
        if self.ball.ycor() < -(self.height/2)-10:   # If ball touch the ground 
            self.is_done = True
    
    def step(self,act) -> tuple:
        """
        Observations:
        size: 4
    
        0: tile x - ball x  [0;1]
        1: ball y  [0;1]
        2: ball vx [-1;1]
        3: ball vy -1 or 1
        """
        rew = 1
        self.steps_done += 1
        if self.steps_done > STEPS:
            self.is_done = True
        self.move(act)
        self.update()
        obs = numpy.zeros((4))
        obs[0] = (self.tile.xcor() + (self.width/2)) / self.width - (self.ball.xcor() + (self.width/2)) / self.width
        obs[1] = (self.ball.ycor() + (self.height/2)) / self.height
        obs[2] = self.ball.dx / 4
        obs[3] = self.ball.dy / abs(self.ball.dy)
        if self.hit:
            self.hit = False
            rew = 100
        #else:
        #    rew = ((self.width - abs(self.tile.xcor() - self.ball.xcor()))/ self.width * 2) ** 2
        #elif obs[3] > 0:
        #    rew = abs(self.width/2 - abs(self.tile.xcor())) / 100
        
        return obs, rew, self.is_done
    
    def reset(self):
        self.tile.reset()
        self.ball.reset()
        self.win.clear()
        self.__init__()
        obs = numpy.zeros((4))
        obs[0] = (self.tile.xcor() + (self.width/2)) / self.width - (self.ball.xcor() + (self.width/2)) / self.width
        obs[1] = (self.ball.ycor() + (self.height/2)) / self.height
        obs[2] = self.ball.dx / 4
        obs[3] = self.ball.dy / abs(self.ball.dy)
        return obs
        
    def show(self):
        self.win.update()
        
            

    

    