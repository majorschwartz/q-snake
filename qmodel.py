import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QModel(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super().__init__()
		self.h1 = nn.Linear(state_dim,	hidden_dim)
		self.h2 = nn.Linear(hidden_dim, hidden_dim)
		self.h3 = nn.Linear(hidden_dim, hidden_dim//2)
		self.output = nn.Linear(hidden_dim//2, action_dim)

	def forward(self, x):
		x = F.relu(self.h1(x))
		x = F.relu(self.h2(x))
		x = F.relu(self.h3(x))
		return self.output(x)

class Train():
	def __init__(self, model, lr, gamma):
		self.lr = lr
		self.gamma = gamma
		self.model = model
		self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()

	def train_step(self, state, action, reward, next_state, done):
		# Compute Q values for current state
		pred = self.model(state)

		target = pred.clone()
		for idx in range(done.shape[0]):
			Q_new = reward[idx]
			if not done[idx]:
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

			target[idx][torch.argmax(action[idx]).item()] = Q_new

		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward()

		self.optimizer.step()