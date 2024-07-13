import numpy as np
import random
import torch
from qsnake import QSnake
from qmodel import QModel, Train
import time
import os

class Runner():
	def __init__(self, view_dim=3):
		self.n_game = 0
		self.memory = []

		self.view_dim = view_dim
		self.state_dim = (self.view_dim**2 * 4) + 2 + 4
		self.hidden_dim = 128
		self.action_dim = 4

		self.model = QModel(self.state_dim, self.hidden_dim, self.action_dim)
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
		print(pred, action)
		move = [0, 0, 0, 0]
		move[random.randint(0, 3)] = 1
		return move

def training_loop():
	score = 0
	steps = 0
	record = 0
	view_dim = 3
	runner = Runner()
	game = QSnake()

	while True:
		time.sleep(0.1)
		os.system('clear')
		game.display_board()
		print(f'Score: {score} | Record: {record} | Steps: {steps}')
		print(f'Direction: {game.direction}')
		state_old = game.get_state(view_dim)
		final_move = runner.act(state_old)
		print(f'Final Move: {final_move}')
		reward, dead, score = game.step(final_move)
		state_new = game.get_state(view_dim)

		runner.remember(state_old, final_move, reward, state_new, dead)

		runner.train_memory()

		if dead:
			game.setup()
			runner.n_game += 1
			for _ in range(5):
				runner.train_memory(long=True)
			if score > record:
				record = score
			score = 0
			steps = 0

			print(f'Game {runner.n_game + 1} | Score: {score} | Record: {record}')
		else:
			steps += 1

if __name__ == '__main__':
	training_loop()