import numpy as np
import random
import torch
from collections import deque
from qsnake import QSnake
from qmodel import QModel, Train
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005

VIEW_DIM = 5
assert VIEW_DIM % 2 == 1, "View dimension must be odd"

class Runner():
	def __init__(self):
		self.n_games = 0
		self.epsilon = 0
		self.gamma = 0.9

		self.food_dim = 6
		self.dir_dim = 4
		
		self.memory = deque(maxlen=MAX_MEMORY)
		
		self.state_dim = (VIEW_DIM ** 2) + self.food_dim + self.dir_dim
		self.hidden_dim = 256
		self.action_dim = 3

		self.model = QModel(self.state_dim, self.hidden_dim, self.action_dim)
		self.trainer = Train(self.model, lr=LR, gamma=self.gamma)
		

	def remember(self, state, action, reward, next_state, dead):
		self.memory.append((state, action, reward, next_state, dead))
	
	def train_long(self):
		if len(self.memory) > BATCH_SIZE:
			mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
		else:
			mini_sample = self.memory

		states, actions, rewards, next_states, dones = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, dones)

	def train_short(self, state, action, reward, next_state, done):
		self.trainer.train_step(state, action, reward, next_state, done)
	
	def act(self, state):
		move = [0, 0, 0]
		self.epsilon = 500 - self.n_games
		if random.randint(0, 700) < self.epsilon:
			move[random.randint(0, 2)] = 1
		else:
			pred = self.model(torch.tensor(np.array(state), dtype=torch.float))
			best = torch.argmax(pred).item()
			# print(pred, best)
			move[best] = 1
			# print(f"Move: {move}")
		return move

def calculate_minutes_and_seconds(sec):
	minutes = int(sec // 60)
	seconds = int(sec % 60)
	return minutes, seconds

def training_loop():
	score = 0
	record = 0
	steps = 0
	total_steps = 0
	runner: Runner = Runner()
	game: QSnake = QSnake()
	start_time = time.time()

	while True:
		state_old = game.get_vision(VIEW_DIM)
		final_move = runner.act(state_old)

		reward, dead, score = game.step(final_move)
		state_new = game.get_vision(VIEW_DIM)

		runner.train_short(state_old, final_move, reward, state_new, dead)

		runner.remember(state_old, final_move, reward, state_new, dead)

		if dead:
			elapsed_time = time.time() - start_time
			steps_per_minute = total_steps / elapsed_time * 60
			games_per_minute = runner.n_games / elapsed_time * 60
			minutes, seconds = calculate_minutes_and_seconds(elapsed_time)
			print(f'Game: {runner.n_games + 1:4d} | Score: {score:3d} | Record: {record:3d} | Steps: {steps:4d} | Steps/min: {steps_per_minute:6.0f} | Games/min: {games_per_minute:4.0f} | Elapsed time: {minutes:2d}m {seconds:2d}s')
			game.setup()
			runner.n_games += 1
			runner.train_long()
			if score > record:
				record = score
			score = 0
			steps = 0
		else:
			steps += 1
			total_steps += 1

if __name__ == '__main__':
	training_loop()