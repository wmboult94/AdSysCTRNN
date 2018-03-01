import assignment
import numpy as np

def constantController(sensors, state, dt):
	return ([5, 1], None)

# state is a tuple with values (motor_outputs,sensor_reading,sub_iter)
def basicController(sensors, state, dt):
	# if in initial state, just move forward
	if state == None:
		return ([5, 5], ([5,5],sensors,1))
	# while sensors readings are increasing, just move forward
	if sensors >= state[1]:
		return ([5, 5], ([5,5],sensors,1))
	# otherwise move in same direction for 4 steps, then choose a random new direction
	if state[2] > 3:
		# if sensors reading is quite a lot less than previous, back up one way
		if sensors < 0.5 * state[1]:
			return ([-3, 3], ([-3,3],sensors,1))
		rand = np.floor(np.random.rand()+0.5)
		if rand == 0:
			return ([1, 10], ([1,10],sensors,1))
		else:
			return ([10, 1], ([10,1],sensors,1))

	return (state[0],(state[0],state[1],state[2]+1))


w = assignment.World()
poses, sensations, actions, states = w.simulate(basicController)
print("Fitness on task 1: %f" % w.task1fitness(poses))
print("Fitness on task 2: %f" % w.task2fitness(poses))
ani = w.animate(poses, sensations)
