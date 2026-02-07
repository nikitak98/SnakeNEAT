import os
import sys
import datetime
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import neat
import pickle
import visualize
import json

from settings import *
from env import SnakeEnv
import replay

counter = 0
all_time_max_fitness = 0
seed_to_play = None
genome_to_play = None
play = False

dir_save = 'run' + str(datetime.datetime.now())

def eval_genomes(genomes,config):

    global counter
    global all_time_max_fitness
    global seed_to_play
    global genome_to_play
    global play
    global dir_save

    for genome_id, genome in genomes:
        env = SnakeEnv()
        saved_seed = env.reset()

        net = neat.nn.FeedForwardNetwork.create(genome,config)
        run = True

        fitness = 0.0

        while run:
            inputs = env.get_inputs()
            output = net.activate(inputs)
            direction = output.index(max(output))
            reward, run_over = env.step(direction)
            fitness += reward
            if run_over:
                run = False

        genome.fitness = fitness

        if genome.fitness > all_time_max_fitness:
            play = True
            all_time_max_fitness = genome.fitness
            seed_to_play = saved_seed
            genome_to_play = genome

    if play:
        folder_name = dir_save + '/generation' + str(counter)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        with open(folder_name + '/genome', 'wb') as f:
            pickle.dump(genome_to_play, f)
        with open(folder_name + '/seed', 'wb') as f:
            pickle.dump(seed_to_play,f)
        print(all_time_max_fitness)
        replay.play(genome_to_play,seed_to_play)
    play = False
    counter += 1


def evaluate_genome(genome, config, seeds):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    results = []
    for seed in seeds:
        env = SnakeEnv()
        env.reset(seed)
        fitness = 0.0
        run = True
        while run:
            inputs = env.get_inputs()
            output = net.activate(inputs)
            direction = output.index(max(output))
            reward, run_over = env.step(direction)
            fitness += reward
            if run_over:
                run = False
        results.append(
            {"fitness": fitness, "steps": env.steps, "eaten": env.eaten}
        )
    return results

if __name__ == "__main__":

    pygame.init()
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

    winner = None
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    if len(sys.argv) > 1 and sys.argv[1] == '--load-checkpoint':
        p = neat.Checkpointer.restore_checkpoint(sys.argv[2])
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100,filename_prefix = dir_save + '/neat-checkpoint-'))

    winner = p.run(eval_genomes,1000)

    with open(dir_save + '/winner-snake','wb') as f:
        pickle.dump(winner,f)

    evaluation_seeds = list(range(10))
    eval_results = evaluate_genome(winner, config, evaluation_seeds)
    average_fitness = sum(r["fitness"] for r in eval_results) / len(eval_results)
    average_steps = sum(r["steps"] for r in eval_results) / len(eval_results)
    average_eaten = sum(r["eaten"] for r in eval_results) / len(eval_results)
    evaluation_summary = {
        "seeds": evaluation_seeds,
        "average_fitness": average_fitness,
        "average_steps": average_steps,
        "average_eaten": average_eaten,
        "runs": eval_results,
    }
    with open(dir_save + '/evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)

    # For these to work please install matplotlib
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
