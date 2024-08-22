import random
from enum import Enum
from collections import namedtuple
import numpy as np

class Direction(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3

Point = namedtuple('Point', 'x, y')

ZONE_SIZE = 8
BEGIN_LENGTH = 3

class QSnake:
	def __init__(self, w=ZONE_SIZE, h=ZONE_SIZE):
		self.w = w
		self.h = h
		self.setup()
	
	def setup(self):
		self.direction = Direction.RIGHT

		self.head = Point(self.w//2, self.h//2)
		self.snake = [
			self.head,
			Point(self.head.x-1, self.head.y),
			Point(self.head.x-2, self.head.y)
		]
		self.food = None
		self.frame = 0
		self.gen_food()
	
	def gen_food(self):
		x = random.randint(0, (self.w-1))
		y = random.randint(0, (self.h-1))
		self.food = Point(x, y)
		if self.food in self.snake:
			self.gen_food()
	
	def collision(self, pt=None):
		if pt is None:
			pt = self.head
		if pt.x > self.w-1 or pt.x < 0 or pt.y > self.h-1 or pt.y < 0:
			return True
		if pt in self.snake[1:]:
			return True
		return False
	
	def move(self, action):
		clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		idx = clockwise.index(self.direction)

		if np.array_equal(action, [1, 0, 0]):
			new_dir = clockwise[idx] # go straight
		elif np.array_equal(action, [0, 1, 0]):
			new_dir = clockwise[(idx+1) % 4] # turn right
		else:
			new_dir = clockwise[(idx-1) % 4] # turn left

		self.direction = new_dir

		x = self.head.x
		y = self.head.y

		if self.direction == Direction.UP:
			y -= 1
		elif self.direction == Direction.DOWN:
			y += 1
		elif self.direction == Direction.LEFT:
			x -= 1
		elif self.direction == Direction.RIGHT:
			x += 1
		
		self.head = Point(x, y)
	
	def step(self, choice: Direction):
		self.move(choice)
		self.snake.insert(0, self.head)
		self.frame += 1
		reward = -0.1

		if self.head == self.food:
			self.gen_food()
			reward = 10
		else:
			self.snake.pop()
		
		if self.collision() or self.frame > 100*len(self.snake):
			return -10, True, len(self.snake) - BEGIN_LENGTH

		return reward, False, len(self.snake) - BEGIN_LENGTH
	
	def get_vision(self, view_dim=3):
		left_bound_val = 0 - view_dim // 2
		right_bound_val = view_dim // 2 + 1

		# get the head position
		head = self.snake[0]

		# vision array
		vision = [
			# food position
			self.food.x < self.head.x,  # food left
			self.food.x > self.head.x,  # food right
			self.food.y < self.head.y,  # food up
			self.food.y > self.head.y,  # food down
			self.food.x == self.head.x, # food in line with head horizontal
			self.food.y == self.head.y,  # food in line with head vertical
			# direction
			self.direction == Direction.LEFT,
			self.direction == Direction.RIGHT,
			self.direction == Direction.UP,
			self.direction == Direction.DOWN
		]

		# get the view of the snake
		for i in range(left_bound_val, right_bound_val):
			for j in range(left_bound_val, right_bound_val):
				# get the point
				point = Point(head.x + i, head.y + j)
				# check if point is in the snake or out of bounds
				if point in self.snake or point.x < 0 or point.x >= self.w or point.y < 0 or point.y >= self.h:
					vision.append(1)
				else:
					vision.append(0)
		
		return np.array(vision, dtype=int)
	
	# def display_board(self):
	# 	for row in self.board:
	# 		print("".join("🟩 " if x == 1 else "⬜️ " if x == 0 else "🍎 " for x in row))