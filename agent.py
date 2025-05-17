import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point  # Assuming these are defined in your 'game.py'
from model import Linear_QNet, QTrainer  # Assuming model architecture and trainer are defined in 'model.py'
from helper import plot  # Assuming the plot function is in 'helper.py'

MAX_MEMORY = 100_000  # Maximum memory size for the agent
BATCH_SIZE = 1000     # Mini-batch size for training
LR = 0.001            # Learning rate for training the model


class Agent:
    def __init__(self):
        self.n_games = 0  # Track the number of games played
        self.epsilon = 0  # Exploration rate (random actions)
        self.gamma = 0.9   # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Memory to store experiences (state, action, reward, next_state, done)
        self.model = Linear_QNet(11, 256, 3)  # Model to predict actions (input size: 11, hidden layer size: 256, output size: 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Trainer to optimize the model


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)  # Left of the snake's head
        point_r = Point(head.x + 20, head.y)  # Right of the snake's head
        point_u = Point(head.x, head.y - 20)  # Up of the snake's head
        point_d = Point(head.x, head.y + 20)  # Down of the snake's head

        # Check if the snake will collide in the respective directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction (1 if true, 0 if false)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location (relative to snake's head)
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Append experience to memory

    def train_long_memory(self):
        # Train on a batch of experiences (mini-batch)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Randomly sample experiences
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # Unzip experience tuples
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Train on mini-batch

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)  # Train on single experience

    def get_action(self, state):
        # Exploration vs Exploitation: choose random action with probability epsilon, else predict from the model
        self.epsilon = 80 - self.n_games  # Decay epsilon (increasing exploitation over time)
        final_move = [0, 0, 0]  # Default: no action

        if random.randint(0, 200) < self.epsilon:
            # Exploration: Random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: Use the model to predict the best action
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Model prediction (Q-values)
            move = torch.argmax(prediction).item()  # Select the action with highest Q-value
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []  # List to store individual game scores
    plot_mean_scores = []  # List to store mean scores over time
    total_score = 0  # Total score across all games
    record = 0  # Highest score achieved so far
    agent = Agent()  # Instantiate the agent
    game = SnakeGameAI()  # Instantiate the game

    while True:
        state_old = agent.get_state(game)  # Get the current state of the game

        final_move = agent.get_action(state_old)  # Get the action from the agent

        reward, done, score = game.play_step(final_move)  # Perform action and get result (reward, done, score)
        state_new = agent.get_state(game)  # Get new state after performing the action

        agent.train_short_memory(state_old, final_move, reward, state_new, done)  # Train on short-term memory
        agent.remember(state_old, final_move, reward, state_new, done)  # Store the experience in memory

        if done:
            game.reset()  # Reset the game when it's over
            agent.n_games += 1  # Increment the number of games played
            agent.train_long_memory()  # Train on long-term memory after a game

            if score > record:
                record = score  # Update the record score if necessary
                agent.model.save()  # Save the model if a new record is achieved

            print(f'Game {agent.n_games} Score {score} Record: {record}')

            plot_scores.append(score)  # Store the score for plotting
            total_score += score  # Add score to total
            mean_score = total_score / agent.n_games  # Calculate mean score
            plot_mean_scores.append(mean_score)  # Store mean score for plotting
            plot(plot_scores, plot_mean_scores)  # Plot scores over time


if __name__ == '__main__':
    train()  # Run the training process
