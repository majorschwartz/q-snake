import numpy as np

class Direction():
	UP = 1
	DOWN = 2
	LEFT = 3
	RIGHT = 4

ZONE_SIZE = 10

class QSnake:
	def __init__(self, w=ZONE_SIZE, h=ZONE_SIZE):
		self.w = w
		self.h = h
		self.wait_time = 0.1
		self.setup()
	
	def setup(self):
		self.direction = Direction.RIGHT
		self.board = np.zeros((self.w, self.h), dtype=np.int8)
		self.snake = [
			(ZONE_SIZE//2, ZONE_SIZE//2 - 2),
			(ZONE_SIZE//2, ZONE_SIZE//2 - 1),
			(ZONE_SIZE//2, ZONE_SIZE//2)
		]
		for x, y in self.snake:
			self.board[x, y] = 1
		self.food = None
		self.gen_food()
	
	def gen_food(self):
		self.food = (np.random.randint(0, self.board.shape[0]), np.random.randint(0, self.board.shape[1]))
		while self.board[self.food] == 1:
			self.food = (np.random.randint(0, self.board.shape[0]), np.random.randint(0, self.board.shape[1]))
		self.board[self.food] = 2
	
	def move(self, dir):
		head_y, head_x = self.snake[-1]

		if dir == Direction.UP:
			head_y -= 1
		elif dir == Direction.DOWN:
			head_y += 1
		elif dir == Direction.LEFT:
			head_x -= 1
		elif dir == Direction.RIGHT:
			head_x += 1
		self.snake.append((head_y, head_x))
	
	def step(self, choice: Direction):
		self.move(choice)
		
		if self.snake[-1] == self.food:
			self.gen_food()
		else:
			self.snake.pop(0)
		
		if self.snake[-1][0] >= self.board.shape[0] or self.snake[-1][0] < 0 or self.snake[-1][1] >= self.board.shape[1] or self.snake[-1][1] < 0:
			return False, len(self.snake)
		
		if self.snake[-1] in self.snake[:-1]:
			return False, len(self.snake)
		
		self.board = np.zeros((self.w, self.h), dtype=np.int8)
		for x, y in self.snake:
			self.board[x, y] = 1
		self.board[self.food] = 2
		return True, len(self.snake)
	
	def get_vision(self):
		return self.snake, self.food, self.board
	
	def display_board(self):
		for row in self.board:
			print("".join("🟩 " if x == 1 else "⬜️ " if x == 0 else "🍎 " for x in row))