import torch
import torch.nn as nn
import torch.optim as optim
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layers defined here
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation for the first layer
        x = torch.relu(self.fc2(x))  # Activation for the second layer
        x = self.fc3(x)  # Output layer
        return x

    def save(self, file_name='model.pth'):
        # Save the model to the specified file path
        model_folder_path = './model'  # Define the directory to save the model
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # Create directory if not exists

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)  # Save model's state_dict (weights)

    def load(self, file_name='model.pth'):
        # Load the model weights from a file
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()  # Set the model to evaluation mode (disables dropout, etc.)
            print(f"Model loaded from {file_name}")
        else:
            print(f"Model file {file_name} not found!")


class QTrainer:
    def __init__(self, model, lr, gamma, weight_decay=0.0):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for Q-value estimation

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Ensure inputs are in the correct shape (batch size, input_size)
        if len(state.shape) == 1:  # If a single state is passed (not a batch)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Ensure 'done' is a tuple to avoid errors

        # Step 1: Get predicted Q values for the current state
        pred = self.model(state)

        # Create a target tensor with the same values as the predicted Q-values initially
        target = pred.clone()

        # Step 2: Calculate the target Q-values
        for idx in range(len(done)):
            Q_new = reward[idx]  # Start with the current reward
            if not done[idx]:
                # If the game is not done, calculate future Q-value (discounted)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the Q-value of the chosen action
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Step 3: Compute the loss and optimize the model
        self.optimizer.zero_grad()  # Clear previous gradients
        loss = self.criterion(target, pred)  # Calculate the loss (MSE)
        loss.backward()  # Backpropagation to compute gradients
        self.optimizer.step()  # Update model parameters based on gradients

        return loss.item()  # Return the loss for monitoring
