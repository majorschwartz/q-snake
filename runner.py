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

		# Initialization creates game counter, epsilon, gamma, the dimensions, and the model + trainer

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
		# Remember the state, action, reward, next state, and dead
		self.memory.append((state, action, reward, next_state, dead))
	
	def train_long(self):
		# If the memory is greater than the batch size, sample a random sample of the memory
		if len(self.memory) > BATCH_SIZE:
			mini_sample = random.sample(self.memory, BATCH_SIZE)
		else:
			# Otherwise, use the entire memory
			mini_sample = self.memory

		# Unzip the mini sample into states, actions, rewards, next states, and dones
		states, actions, rewards, next_states, dones = zip(*mini_sample)
		
		# Convert lists to tensors with a batch dimension
		states = torch.stack(states)
		actions = torch.stack(actions)
		rewards = torch.stack(rewards)
		next_states = torch.stack(next_states)
		dones = torch.stack(dones)
		
		# print(f"{states[:2].shape=}\n{actions[:2].shape=}\n{rewards[:2].shape=}\n{next_states[:2].shape=}\n{dones[:2].shape=}")
		# print(f"{states[:2]=}\n{actions[:2]=}\n{rewards[:2]=}\n{next_states[:2]=}\n{dones[:2]=}")
		# Train the model
		self.trainer.train_step(states, actions, rewards, next_states, dones)

	def train_short(self, state, action, reward, next_state, done):
		# Train the model
		self.trainer.train_step(state, action, reward, next_state, done)
	
	def act(self, state: torch.Tensor):
		# Create a move tensor
		move = torch.zeros(3, dtype=torch.int)
		# Update epsilon
		self.epsilon = 500 - self.n_games
		# If the random number is less than epsilon, make a random move
		if random.randint(0, 700) < self.epsilon:
			# Make a random move
			move[random.randint(0, 2)] = 1
		else:
			# Otherwise, make a move based on the model
			state = state.float()
			# Get the prediction
			pred = self.model(state)
			# Get the best action
			best = torch.argmax(pred).item()
			# Set the best action to 1
			move[best] = 1
		return move

def calculate_minutes_and_seconds(sec):
	# Calculate the minutes and seconds
	minutes = int(sec // 60)
	seconds = int(sec % 60)
	return minutes, seconds

def training_loop():
	# Initialize the record, steps, and total steps
	record = 0
	steps = 0
	total_steps = 0
	# Initialize the runner and game
	runner: Runner = Runner()
	game: QSnake = QSnake()
	# Initialize the start time
	start_time = time.time()

	# Loop forever
	while True:
		# Get the old state
		state_old: torch.Tensor = game.get_vision(VIEW_DIM)
		# Get the next "final" move
		final_move: torch.Tensor = runner.act(state_old)

		# Get the reward, dead, and score from the step we take
		reward, dead, score = game.step(final_move)
		# Get the new state
		state_new: torch.Tensor = game.get_vision(VIEW_DIM)

		# Train the model
		# print(f"{state_old.shape=}\n{final_move.shape=}\n{reward.shape=}\n{state_new.shape=}\n{dead.shape=}")
		# print(f"{state_old=}\n{final_move=}\n{reward=}\n{state_new=}\n{dead=}")
		runner.train_short(state_old, final_move, reward, state_new, dead)

		# Remember the state, action, reward, next state, and dead
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
			steps = 0
		else:
			steps += 1
			total_steps += 1

if __name__ == '__main__':
	training_loop()