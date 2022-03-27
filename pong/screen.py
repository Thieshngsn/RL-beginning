
import turtle
import time
import random

class WatchedKey:
    def __init__(self, key):
        self.key = key
        self.down = False
        turtle.onkeypress(self.press, key)
        turtle.onkeyrelease(self.release, key)

    def press(self):
        self.down = True

    def release(self):
        self.down = False

win = turtle.Screen()    # Create a screen
win.title('Paddle')      # Set the title to paddle
win.bgcolor('black')     # Set the color to black
win.tracer(0)
win.setup(width=500, height=800)   # Set the width and height to 600



# Paddle
paddle = turtle.Turtle()    # Create a turtle object
paddle.shape('square')      # Select a square shape
paddle.speed(0)             
paddle.shapesize(stretch_wid=1, stretch_len=5)   # Streach the length of square by 5 
paddle.penup()
paddle.color('white')       # Set the color to white
paddle.goto(0, -365)        # Place the shape on bottom of the screen

# Ball
ball = turtle.Turtle()      # Create a turtle object
ball.speed(0)
ball.shape('circle')        # Select a circle shape
ball.color('red')           # Set the color to red
ball.penup()
ball.goto(0, -256)           # Place the shape in middle

# Paddle Movement
def paddle_right():
    x = paddle.xcor()        # Get the x position of paddle
    if x < 230:
        paddle.setx(x+2)    # increment the x position by 20

def paddle_left():
    x = paddle.xcor()        # Get the x position of paddle
    if x > -240:
        paddle.setx(x-2)    # decrement the x position by 20

# Keyboard Control
left = WatchedKey('Left')
right = WatchedKey('Right')
win.listen()

ball.dx = random.random() + 3   # ball's x-axis velocity 
ball.dy = 5  # ball's y-axis velocity 









while True:   # same loop4
    if right.down:
        paddle_right()
    if left.down:
        paddle_left()
    win.update()
    start = time.time()
    ball.setx(ball.xcor() + ball.dx)  # update the ball's x-location using velocity
    ball.sety(ball.ycor() + ball.dy)  # update the ball's y-location using velocity  
    time.sleep(0.01-(time.time()-start))
    
    
    # Ball-Walls collision  
    if ball.xcor() > 240:    # If ball touch the right wall
        ball.setx(240)
        ball.dx *= -1        # Reverse the x-axis velocity

    if ball.xcor() < -240:   # If ball touch the left wall
        ball.setx(-240)
        ball.dx *= -1        # Reverse the x-axis velocity

    if ball.ycor() > 390:    # If ball touch the upper wall
        ball.sety(390)
        ball.dy *= -1        # Reverse the y-axis velocity
        #ball.dx = random.random() * 10-5

    # Ball-Paddle collision
    if abs(ball.ycor() + 350) < 2 and abs(paddle.xcor() - ball.xcor()) < 55:
        ball.dy *= -1
        if random.random() > 0.5:
            ball.dx = random.random() * 2 + 2  # ball's x-axis velocity
        else:
            ball.dx = -random.random() * 2 -2   # ball's x-axis velocity

    # Ball-Ground collison            
    if ball.ycor() < -390:   # If ball touch the ground 
        ball.goto(0, -265)    # Reset the ball position  
        ball.dx = random.random() + 3   # ball's x-axis velocity 
        ball.dy = 5  # ball's y-axis velocity 

     #ball.setx(ball.xcor() + dx)  # update the ball's x-location using velocity
     #ball.sety(ball.ycor() + dy)  # update the ball's y-location using velocity44444444