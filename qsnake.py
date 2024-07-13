import numpy as np

class Direction():
	UP = [1, 0, 0, 0]
	DOWN = [0, 1, 0, 0]
	LEFT = [0, 0, 1, 0]
	RIGHT = [0, 0, 0, 1]

ZONE_SIZE = 10
BEGIN_LENGTH = 3

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
		reward = 0

		if self.snake[-1] == self.food:
			self.gen_food()
			reward = 1
		else:
			self.snake.pop(0)
		
		if self.snake[-1][0] >= self.board.shape[0] or self.snake[-1][0] < 0 or self.snake[-1][1] >= self.board.shape[1] or self.snake[-1][1] < 0:
			return -1, True, len(self.snake) - BEGIN_LENGTH
		
		if self.snake[-1] in self.snake[:-1]:
			return -1, True, len(self.snake) - BEGIN_LENGTH
		
		self.board = np.zeros((self.w, self.h), dtype=np.int8)
		for x, y in self.snake:
			self.board[x, y] = 1
		self.board[self.food] = 2
		return reward, False, len(self.snake) - BEGIN_LENGTH
	
	def get_state(self, view_dim=3):
		left_bound_val = 0 - view_dim // 2
		right_bound_val = view_dim // 2 + 1
		head = self.snake[-1]
		
		# vision array (one-hots)
		vision = []
		# food dir ([up: 0, down: 1], [left: 0, right: 1)]
		food_dir = [0, 0]

		# for i in range(left_bound_val, right_bound_val):
		# 	for j in range(left_bound_val, right_bound_val):
		# 		if head[0] + i < 0 or head[0] + i >= self.board.shape[0] or head[1] + j < 0 or head[1] + j >= self.board.shape[1]:
		# 			vision.append([1, 0, 0, 0])
		# 		elif self.board[head[0] + i][head[1] + j] == 0:
		# 			vision.append([0, 1, 0, 0])
		# 		elif self.board[head[0] + i][head[1] + j] == 1:
		# 			vision.append([0, 0, 1, 0])
		# 		elif self.board[head[0] + i][head[1] + j] == 2:
		# 			vision.append([0, 0, 0, 1])

		for i in range(left_bound_val, right_bound_val):
			for j in range(left_bound_val, right_bound_val):
				if head[0] + i < 0 or head[0] + i >= self.board.shape[0] or head[1] + j < 0 or head[1] + j >= self.board.shape[1]:
					vision.append(1)
				elif self.board[head[0] + i][head[1] + j] == 0:
					vision.append(0)
				elif self.board[head[0] + i][head[1] + j] == 1:
					vision.append(1)
				elif self.board[head[0] + i][head[1] + j] == 2:
					vision.append(0)

		if head[0] < self.food[0]:
			food_dir[0] = 1
		if head[1] < self.food[1]:
			food_dir[1] = 1

		return list(np.array(vision).flatten()) + food_dir #+ self.direction
	
	def display_board(self):
		for row in self.board:
			print("".join("🟩 " if x == 1 else "⬜️ " if x == 0 else "🍎 " for x in row))