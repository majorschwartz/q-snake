import torch
import torch.nn as nn

class QModel(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(QModel, self).__init__()
		self.fc1 = nn.Linear(state_dim,	hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, action_dim)

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
    
    def train_step(self, state, action, reward, next_state, dead):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        dead = torch.tensor(dead, dtype=torch.bool)
        
        # Get current Q values
        pred = self.model(state)
        
        # print(f"\n\n{state=}\n{next_state=}\n{action=}\n{reward=}\n{dead=}\n{pred=}\n\n")
        
        # Get next Q values
        next_pred = self.model(next_state)
        
        # Calculate target Q values
        target = pred.clone()
        Q_new = reward
        if not dead.item():
            Q_new = reward + self.gamma * torch.max(next_pred)
        target[action] = Q_new
        
        # Calculate loss
        loss = self.criterion(pred, target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
