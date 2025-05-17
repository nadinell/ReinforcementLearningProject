import torch
from game import SnakeGameAI
from model import Linear_QNet
from agent import Agent
import numpy as np
import time  # Import time module for delay

def test():
    # Initialize the game and the agent
    game = SnakeGameAI()
    model = Linear_QNet(input_size=6, hidden_size=256, output_size=3)
    
    # Load the trained model
    model.load("model.pth")  # Load the model saved during training

    # Now initialize the agent with the loaded model
    agent = Agent(model)

    total_score = 0
    total_games = 0
    game_limit = 1000  # Set a limit for total games, or use a large number to keep testing
    continue_testing = True  # Flag to control whether to continue or stop testing

    while continue_testing:
        game.reset()
        state_old = game.get_state()  # Get the initial state
        done = False
        score = 0

        while not done:
            # Get the agent's action based on the state
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)

            # Get the new state after taking the action
            state_new = game.get_state()

            # Update the old state
            state_old = state_new

        total_score += score
        total_games += 1
        print(f"Game {total_games} - Score: {score}")

        # Calculate and print the average score so far
        avg_score = total_score / total_games
        print(f"Average score after {total_games} games: {avg_score}")

        # Optional: Save the model every few games (e.g., every 100 games)
        if total_games % 100 == 0:
            agent.save_model("model.pth")
            print(f"Game {total_games} - Model saved!")

        # If the game ended, wait for the game over sound to finish before continuing
        if done:
            time.sleep(1)  # Pause for 1 second to let the game over sound play
            # You can adjust the sleep time if needed to allow the full sound to play

        # You can set a condition here to stop the loop after a specific number of games
        if total_games >= game_limit:
            continue_testing = False  # Stop testing when the game limit is reached

    print("Testing finished. Final model saved!")
    print(f"Average score after {total_games} games: {total_score / total_games}")

if __name__ == "__main__":
    test()
