import numpy as np
import torch
from qsnake import QSnake, Direction
from qmodel import QModel, Train

class Runner():
	def __init__(self):
		self.n_game = 0
		self.memory = []
		self.model = QModel()
		self.trainer = Train(self.model, 0.001, 0.9)
		

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
	
	def train_memory(self, long=False):
		if not long:
			self.trainer.train_step(self.memory[-1][0], self.memory[-1][1], self.memory[-1][2], self.memory[-1][3], self.memory[-1][4])
		else:
			for state, action, reward, next_state, done in self.memory:
				self.trainer.train_step(state, action, reward, next_state, done)
			self.memory = []
	
	def act(self, state):
		pred = self.model(torch.tensor(state, dtype=torch.float))
		action = torch.argmax(pred).item()
		return action

def training_loop():
	score = 0
	record = 0
	runner = Runner()
	game = QSnake()

	while True:
		state_old = game.get_state(game)
		final_move = runner.act(state_old)
		reward, done, score = game.step(final_move)
		state_new = game.get_state(game)

		runner.remember(state_old, final_move, reward, state_new, done)

		runner.train_memory()

		if done:
			game.setup()
			runner.n_game += 1
			for _ in range(5):
				runner.train_memory(long=True)
			if score > record:
				record = score
			
			print(f'Game {runner.n_game + 1} | Score: {score} | Record: {record}')

if __name__ == '__main__':
	training_loop()