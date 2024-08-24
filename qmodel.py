import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from settings import DEBUG

device = torch.device("cuda" if cuda.is_available() else "cpu")

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
		self.model = model.to(device)
		self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()

	def train_step(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor):
		if state.dim() == 1:
			state = state.unsqueeze(0)
			action = action.unsqueeze(0)
			reward = reward.unsqueeze(0)
			next_state = next_state.unsqueeze(0)
			done = done.unsqueeze(0)
		
		if DEBUG:
			print(f"{state.shape=}\n{action.shape=}\n{reward.shape=}\n{next_state.shape=}\n{done.shape=}")

		# Compute Q values for current state
		pred = self.model(state)

		target = pred.clone()
		for idx in range(done.size(0)):
			if DEBUG:
				print(f"{pred[idx]=}\n{reward[idx]=}\n{next_state[idx]=}\n{done[idx]=}")
			Q_new = reward[idx]
			if not done[idx].item():
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

			target[idx][torch.argmax(action[idx]).item()] = Q_new

		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward()

		self.optimizer.step()