import torch
from model import Linear_QNet  # Ensure this is the correct import for your model

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Create the model and pass it to the agent
    model = Linear_QNet()  # Make sure to initialize your model here
    agent = Agent(model)  # Now pass the model to the Agent constructor

    game = SnakeGameAI()  # Initialize the Snake game

    while True:
        state_old = agent.get_state(game)  # Get the current state of the game

        # Get the action the agent wants to take (based on its current state)
        final_move = agent.get_action(state_old)

        # Perform the action, get the reward, and the new game state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train the agent with the new state and reward
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the state transition for future training
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()  # Reset the game after it is over
            agent.n_games += 1
            agent.train_long_memory()  # Train with long-term memory after each game

            if score > record:
                record = score
                agent.model.save()  # Save the model if it achieves a new high score

            print(f'Game {agent.n_games} Score {score} Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)  # Plot the score graph

if __name__ == '__main__':
    train()
