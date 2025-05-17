import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

pygame.mixer.init()  # Initialize the mixer for playing sounds and music
font = pygame.font.Font('arial.ttf', 25)

# Load and play background music
pygame.mixer.music.load('snake_music.wav')  # Replace with the actual path if needed
pygame.mixer.music.play(-1)  # Loop indefinitely

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors and game settings
WHITE = (255, 255, 255)
RED = (200, 0, 0)
DARK_GREEN = (34, 139, 34)  # Realistic snake color
BROWN = (139, 69, 19)  # Obstacle color
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Load assets
        self.wood_texture = pygame.image.load('wood_texture.jpg')
        self.wood_texture = pygame.transform.scale(self.wood_texture, (self.w, self.h))
        self.death_sound = pygame.mixer.Sound('game_over.wav')  # Dramatic death sound
        self.tension_sound = pygame.mixer.Sound('tension.wav')  # Tension sound

        self.obstacles = []  # To store dynamic obstacles
        self.energy = 100  # Energy for the snake
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.obstacles = self._generate_obstacles()
        self.energy = 100
        self.frame_iteration = 0
        self._place_food()
        pygame.mixer.stop()  # Stop any ongoing sounds

    def _generate_obstacles(self, count=5):
        obstacles = []
        for _ in range(count):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            obstacles.append(Point(x, y))
        return obstacles

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

    def play_step(self, action):
        self.frame_iteration += 1
        self.energy -= 0.1  # Reduce energy slightly per step
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self._move(action)
        self.snake.insert(0, self.head)

        # Check for collisions
        reward = 0
        game_over = False
        if self._collision():
            reward = -10
            game_over = True
            self.death_sound.play()

        if self.head == self.food:
            self.food = None
            self._place_food()
            reward = 10  # Increase reward when food is eaten
            self.score += 1
        else:
            self.snake.pop()

        self.display.fill(BLACK)  # Clear the screen
        self._draw_snake()
        self._draw_food()
        self._draw_obstacles()
        self._draw_score()
        self._draw_energy()

        pygame.display.flip()

        if game_over or self.energy <= 0:
            return reward, True, self.score

        return reward, False, self.score

    def _collision(self):
        # Check if the snake hits the walls or itself
        if self.head.x < 0 or self.head.x >= self.w or self.head.y < 0 or self.head.y >= self.h:
            return True

        if self.head in self.snake[1:]:
            return True

        # Check if the snake hits any obstacles
        for obstacle in self.obstacles:
            if self.head == obstacle:
                return True

        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]  # Move straight
        elif action == [0, 1, 0]:
            new_dir = clock_wise[(idx + 1) % 4]  # Turn right
        elif action == [0, 0, 1]:
            new_dir = clock_wise[(idx - 1) % 4]  # Turn left
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def _draw_snake(self):
        for block in self.snake:
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))

    def _draw_food(self):
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

    def _draw_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, BROWN, pygame.Rect(obstacle.x, obstacle.y, BLOCK_SIZE, BLOCK_SIZE))

    def _draw_score(self):
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])

    def _draw_energy(self):
        energy_text = font.render(f"Energy: {int(self.energy)}", True, WHITE)
        self.display.blit(energy_text, [0, 20])

    def get_state(self):
        # Return the state representation: head position, food position, direction, energy level
        return np.array([
            self.head.x, self.head.y,  # Snake head position
            self.food.x, self.food.y,  # Food position
            self.direction.value,  # Current direction
            self.energy,  # Current energy level
            len(self.snake)  # Snake length (for a more informative state)
        ], dtype=int)
