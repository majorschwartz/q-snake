import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
from typing import Tuple

class Direction(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3

Point = namedtuple('Point', 'x, y')

ZONE_SIZE = 20
BEGIN_LENGTH = 3

class QSnake:
	def __init__(self, w: int = ZONE_SIZE, h: int = ZONE_SIZE):
		self.w = w
		self.h = h
		self.setup()
	
	def setup(self) -> None:
		# Set the direction to right
		self.direction = Direction.RIGHT

		# Set the head to the center of the zone
		self.head = Point(self.w//2, self.h//2)

		# Initialize the snake
		self.snake = [
			self.head,
			Point(self.head.x-1, self.head.y),
			Point(self.head.x-2, self.head.y)
		]
		self.food = None
		self.frame = 0
		self.gen_food()
	
	def gen_food(self) -> None:
		x = random.randint(0, (self.w-1))
		y = random.randint(0, (self.h-1))
		self.food = Point(x, y)
		if self.food in self.snake:
			self.gen_food()
	
	def collision(self, pt: Point = None) -> bool:
		if pt is None:
			pt = self.head
		if not (0 <= pt.x < self.w and 0 <= pt.y < self.h):
			return True
		return pt in set(self.snake[1:])
	
	def move(self, action: torch.Tensor) -> None:
		# action: torch.Tensor([go straight, turn right, turn left])
		clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		idx = clockwise.index(self.direction)

		if torch.equal(action, torch.tensor([1, 0, 0], dtype=torch.int)):
			new_dir = clockwise[idx] # go straight
		elif torch.equal(action, torch.tensor([0, 1, 0], dtype=torch.int)):
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
	
	def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		self.move(action)
		self.snake.insert(0, self.head)
		self.frame += 1
		reward = torch.tensor(-0.1, dtype=torch.float)

		if self.head == self.food:
			self.gen_food()
			reward = torch.tensor(10, dtype=torch.float)
		else:
			self.snake.pop()
		
		if self.collision() or self.frame > 100*len(self.snake):
			return torch.tensor(-10, dtype=torch.float), torch.tensor(True, dtype=torch.bool), torch.tensor(len(self.snake) - BEGIN_LENGTH, dtype=torch.int)

		return reward, torch.tensor(False, dtype=torch.bool), torch.tensor(len(self.snake) - BEGIN_LENGTH, dtype=torch.int)
	
	def get_vision(self, view_dim=3) -> torch.Tensor:
		left_bound_val = 0 - view_dim // 2
		right_bound_val = view_dim // 2 + 1

		# get the head position
		head = self.snake[0]

		# vision tensor
		vision = torch.tensor([
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
		], dtype=torch.int)

		# create a grid of points around the head
		x_range = torch.arange(left_bound_val, right_bound_val)
		y_range = torch.arange(left_bound_val, right_bound_val)
		grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
		grid_x = grid_x.flatten()
		grid_y = grid_y.flatten()

		# calculate the points
		points_x = head.x + grid_x
		points_y = head.y + grid_y

		# check if points are in the snake or out of bounds
		in_snake = torch.tensor([Point(x, y) in self.snake for x, y in zip(points_x, points_y)], dtype=torch.int)
		out_of_bounds = (points_x < 0) | (points_x >= self.w) | (points_y < 0) | (points_y >= self.h)

		# combine the results
		vision = torch.cat((vision, in_snake | out_of_bounds.int()))

		# print(vision, vision.shape)
		return vision.float()

	# def display_board(self):
	# 	for row in self.board:
	# 		print("".join("🟩 " if x == 1 else "⬜️ " if x == 0 else "🍎 " for x in row))