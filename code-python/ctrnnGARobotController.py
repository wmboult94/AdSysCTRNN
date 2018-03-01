from __future__ import print_function

import os
import pickle
import numpy as np

import assignment
import neat
from neat.activations import sigmoid_activation
from neat.math_util import mean
import visualize

runs_per_net = 5
simulation_seconds = 60.0
time_const = 0.1

# steps for ctrnn:
# 1. build population of random nn parameters
# 2. run w.simulate for each of these genotypes
# -- ctrnn takes in sensor input, outputs an action,
# so controller needs to read in sensor, the nn, the time step
# 3. evaluate fitnesses
# 4. repeat for num generations
# 5. obtain best genotype, build nn with it and run simulation
def ctrnnController(sensors, state, dt):
	net = state
	action = net.advance(sensors,time_const,time_const)
	return (action,net)

# Use the CTRNN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
	net = neat.ctrnn.CTRNN.create(genome, config, time_const)

	fitnesses = []
	for runs in range(runs_per_net):
		w = assignment.World()
		net.reset()

		# Run the given simulation for up to num_steps time steps.
		fitness = 0.0
		poses, sensations, actions, states = w.simulateNN(ctrnnController,net)
		fitness1 = -500 if w.task1fitness(poses) == -np.inf else w.task1fitness(poses)
		fitness2 = -500 if w.task2fitness(poses) == -np.inf else w.task2fitness(poses)
		fitness3 = np.sum(sensations)
		fitness = float((fitness1 + fitness2 + fitness3) / 3)

		fitnesses.append(fitness)

		#print("{0} fitness {1}".format(net, fitness))


	# The genome's fitness is its worst performance across all runs.
	return min(fitnesses)


def eval_genomes(genomes, config):
	for genome_id, genome in genomes:
		genome.fitness = eval_genome(genome, config)


def run():
	# Load the config file, which is assumed to live in
	# the same directory as this script.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-ctrnn')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						neat.DefaultSpeciesSet, neat.DefaultStagnation,
						config_path)

	pop = neat.Population(config)
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)
	pop.add_reporter(neat.StdOutReporter(True))

	# if 0:
	# 	winner = pop.run(eval_genomes,n=50)
	# else:
	pe = neat.ParallelEvaluator(4, eval_genome)
	winner = pop.run(pe.evaluate,n=50)

	# Save the winner.
	with open('winner-ctrnn', 'wb') as f:
		pickle.dump(winner, f)

	print(winner)

	visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
	visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

	# node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
	visualize.draw_net(config, winner, True)

	visualize.draw_net(config, winner, view=True,
						filename="winner-ctrnn.gv")
	visualize.draw_net(config, winner, view=True,
						filename="winner-ctrnn-enabled.gv", show_disabled=False)
	visualize.draw_net(config, winner, view=True,
						filename="winner-ctrnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)

	return winner

def exampleCTRNN():
	# Create a fully-connected network of two neurons with no external inputs.
	node1_inputs = [(1, 3), (2, 0.2)]
	node2_inputs = [(1, -0.2), (2, -2)]

	node_evals = {
					1: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -2.75 / 5.0, 2.0, node1_inputs),
					2: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.75 / 5.0, 2.0, node2_inputs)
				}

	net = neat.ctrnn.CTRNN([1], [10, 10], node_evals)

	init1 = 0.0
	init2 = 0.0

	net.set_node_value(1, init1)
	net.set_node_value(2, init2)


if __name__ == '__main__':
	# exampleCTRNN()

	# Obtain best evolved neural net, and run simulation
	ctrnn = run()
	w = assignment.World()
	finalNet = neat.ctrnn.CTRNN.create(genome, config, time_const)
	poses, sensations, actions, states = w.simulateNN(ctrnnController,finalNet)
	# print(actions)
	print("Fitness on task 1: %f" % w.task1fitness(poses))
	print("Fitness on task 2: %f" % w.task2fitness(poses))
	ani = w.animate(poses, sensations)
