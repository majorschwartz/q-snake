import torch
import torch.nn as nn

class QModel(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QModel, self).__init__()
		self.fc1 = nn.Linear(state_dim, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, action_dim)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Train():
	def __init__(self, model, lr, gamma):
		self.lr = lr
		self.gamma = gamma
		self.model = model
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss()
	
	def train_step(self, state, action, reward, next_state, done):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)
		done = torch.tensor(done, dtype=torch.bool)
		
		pred = self.model(state)
		print(pred)
