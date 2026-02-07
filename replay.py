import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import neat
import pickle

from settings import *
from env import SnakeEnv

def play(genome,s = None):

    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

    winner_net = neat.nn.FeedForwardNetwork.create(genome, config)

    screen = pygame.display.set_mode((width,height))

    env = SnakeEnv()
    if s is None:
        saved_seed = env.reset()
    else:
        saved_seed = env.reset(int(s))

    # Draw
    draw_state(screen, env)

    run = True

    pygame.time.set_timer(pygame.USEREVENT, tick_rate)
    clock = pygame.time.Clock()

    while run:

        event = pygame.event.wait()
        if event.type == pygame.USEREVENT:

            inputs = env.get_inputs()
            output = winner_net.activate(inputs)
            direction = output.index(max(output))
            _, run_over = env.step(direction)
            if run_over:
                run = False

            draw_state(screen, env)


def draw_state(screen, env):
    screen.fill(background_color)
    pygame.draw.rect(
        screen,
        food_color,
        (env.food[0], env.food[1], block_size - 1, block_size - 1),
    )
    for obstacle in env.obstacles:
        pygame.draw.rect(
            screen,
            (100, 100, 100),
            (obstacle[0], obstacle[1], block_size - 1, block_size - 1),
        )
    pygame.draw.rect(
        screen,
        snake_head_color,
        (env.snake_body[0][0], env.snake_body[0][1], block_size - 1, block_size - 1),
    )
    for (x, y) in env.snake_body:
        if (x, y) != env.snake_body[0]:
            pygame.draw.rect(screen, snake_color, (x, y, block_size - 1, block_size - 1))
    pygame.display.update()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: replay.py genome_file [seed_file]")
        exit()

    winner = None
    load_seed = None
    with open(sys.argv[1], 'rb') as f:
        winner = pickle.load(f)
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'rb') as f:
            load_seed = pickle.load(f)
    play(winner,load_seed)
