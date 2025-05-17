import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Interactive mode for real-time plotting

def plot(scores, mean_scores, energy_levels=None):
    # Clear the output and display the current plot
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Set the plot title and labels
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plotting Scores and Mean Scores
    plt.plot(scores, label="Scores", color='blue')
    plt.plot(mean_scores, label="Mean Scores", color='green')

    # Plot Energy Levels if provided
    if energy_levels is not None:
        plt.plot(energy_levels, label="Energy Levels", color="orange", linestyle="--")

    # Set Y-axis limits to start from 0 (no negative values for scores)
    plt.ylim(ymin=0)

    # Adding values to the last points in the graph
    plt.text(len(scores)-1, scores[-1], str(scores[-1]), fontsize=10, ha='center')
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), fontsize=10, ha='center')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show(block=False)
    plt.pause(0.1)  # Pause for a moment to update the plot
