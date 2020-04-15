import pygame
from time import time
import math
import random
import numpy as np
import tensorflow as tf
pygame.init()

def dot(v1, v2):
	return (v1[0] * v2[0]) + (v1[1] * v2[1])

def castCollision(p1, p2, p3, p4):#Finds collision of 2 lines (if it exists)
	tuDenom = ((p1.x - p2.x) * (p3.y - p4.y)) - ((p1.y - p2.y) * (p3.x - p4.x))
	tNum = ((p1.x - p3.x) * (p3.y - p4.y)) - ((p1.y - p3.y) * (p3.x - p4.x))
	uNum = ((p1.x - p3.x) * (p1.y - p2.y)) - ((p1.y - p3.y) * (p1.x - p2.x))
	if (tuDenom == 0) or (not (0 <= tNum/tuDenom and 1 >= tNum/tuDenom and 0 <= uNum/tuDenom and 1 >= uNum/tuDenom)):
		return None
	xNum = (((p1.x * p2.y) - (p1.y * p2.x)) * (p3.x - p4.x)) - (((p3.x * p4.y) - (p3.y * p4.x)) * (p1.x - p2.x))
	yNum = (((p1.x * p2.y) - (p1.y * p2.x)) * (p3.y - p4.y)) - (((p3.x * p4.y) - (p3.y * p4.x)) * (p1.y - p2.y))
	
	return Point(xNum / tuDenom, yNum / tuDenom)

def calcAngle(obj): #Angle of an object relative to a point at the center. Used for a continuous reward function.
	return -math.atan2((450 - obj.getY()), (obj.getX() - 550))

def isContainingAxis(axis, o1, o2): #I made this up. It means one projection along the axis is contained within the other
	min1, max1, min2, max2 = float('+inf'), float('-inf'), float('+inf'), float('-inf')
	for p in o1.points:
		projection = dot(p, axis)
		min1 = min(min1, projection)
		max1 = max(max1, projection)
	for p in o2.points:
		projection = dot(p, axis)
		min2 = min(min2, projection)
		max2 = max(max2, projection)

	if (min2 >= min1 and max1 >= max2) or (min1 >= min2 and max2 >= max1):
		return True
	else:
		return False

def isSeperatingAxis(axis, o1, o2):#Determines if "axis" is a seperating axis of 2 objects.
	min1, max1, min2, max2 = float('+inf'), float('-inf'), float('+inf'), float('-inf')
	for p in o1.points:
		projection = dot(p, axis)
		min1 = min(min1, projection)
		max1 = max(max1, projection)
	for p in o2.points:
		projection = dot(p, axis)
		min2 = min(min2, projection)
		max2 = max(max2, projection)

	if max1 >= min2 and max2 >= min1:
		return False
	else:
		return True

class Object:
	def __init__(self, points): #Define in the counterclockwise direction starting at any point
		self.points = points

	def draw(self, screen):
		pygame.draw.polygon(screen, (0, 0, 0), self.points, 3)

	def getEdges(self):
		edges = []
		for i in range(len(self.points)):
			edges.append((self.points[i], self.points[(i + 1) % len(self.points)])) #Create sets of points representing each edge
		return tuple(edges)

	def getEdgeNormals(self):
		edges = self.getEdges() #Get and convert edges to list
		edges = list(edges)
		for i in range(len(edges)):
			edges[i] = (edges[i][1][0] - edges[i][0][0], edges[i][1][1] - edges[i][0][1]) #Convert edges to vectors
			edges[i] = (-edges[i][1], edges[i][0]) #Rotate by 90 degrees counterclockwise
			mag = math.sqrt((edges[i][0] * edges[i][0]) + (edges[i][1] * edges[i][1]))
			edges[i] = (edges[i][0] / mag, edges[i][1] / mag) #Normalize
		return tuple(edges)

	def isCollidingWith(self, obj):
		edgeNormals = list(self.getEdgeNormals()) + list(obj.getEdgeNormals())
		for e in edgeNormals:
			if isSeperatingAxis(e, self, obj):
				return False
		return True

	def isWithin(self, obj):
		edgeNormals = list(self.getEdgeNormals()) + list(obj.getEdgeNormals())
		for e in edgeNormals:
			if not isContainingAxis(e, self, obj):
				return False
		return True

class Gate:
	def __init__(self, pos1, pos2):
		self.pos1 = pos1
		self.pos2 = pos2
		self.passed = False
		self.within = False

	def check(self, car):
		if not self.passed and not self.within:
			if not castCollision(Point(self.pos1[0], self.pos1[1]), Point(self.pos2[0], self.pos2[1]), Point(car.getX(), car.getY()), Point(car.getX() + 10, car.getY() + 10)) is None:
				self.within = True

		if not self.passed and self.within:
			if castCollision(Point(self.pos1[0], self.pos1[1]), Point(self.pos2[0], self.pos2[1]), Point(car.getX(), car.getY()), Point(car.getX() + 10, car.getY() + 10)) is None:
				self.passed = True
				return True
		return False

	def draw(self, screen):
		pygame.draw.line(screen, (0, 0, 0), self.pos1, self.pos2)
	
	def reset(self):
		self.passed = False
		self.within = False


inner = Object(((200, 300), (200, 600), (500, 700), (1000, 600), (1100, 300), (500, 200)))#Inner Track
outer = Object(((50, 100), (100, 800), (400, 890), (1290, 750), (1290, 150), (500, 10)))#Outer Track
gates = (Gate((50, 300), (200, 300)), Gate((150, 70), (200, 300)), Gate((245, 55), (350, 250)), Gate((500, 10), (500, 200)), Gate((700, 55), (650, 230)), Gate((900, 75), (900, 300)), Gate((1290, 150), (1100, 300)), Gate((1290, 450), (1050, 450)), Gate((1290, 700), (1000, 600)), Gate((1000, 800), (990, 600)), Gate((700, 845), (750, 650)), Gate((400, 890), (500, 700)), Gate((250, 860), (260, 620)), Gate((80, 700), (200, 600)), Gate((50, 450), (200, 450)))
BATCH_SIZE = 50
GAMMA = 0.998


class RocketCar:
	length = 30
	width = 20
	mass = 1
	drag = 1
	def __init__(self, x, y, angle):
		self.x = x
		self.y = y
		self.xVel = 0
		self.yVel = 0
		self.xAcc = 0
		self.yAcc = 0
		self.angle = angle
		self.time = time()
		self.car = Object((()))

	def setPos(self, x, y, angle):
		self.x = x
		self.y = y
		self.angle = angle
		self.xVel = 0
		self.yVel = 0
		self.xAcc = 0
		self.yAcc = 0

	def raycast(self):
		edges = list(inner.getEdges()) + list(outer.getEdges())

		angles = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)

		points = []

		pCar = Point(self.x, self.y)

		for a in angles:
			minD = float('+inf')
			for e in edges:
				p = castCollision(Point(self.getX(), self.getY()), Point(self.getX() + (1600 * math.sin(math.radians(a + self.getTheta()))), self.getY() - (1600 * math.cos(math.radians(a + self.getTheta())))), Point(e[0][0], e[0][1]), Point(e[1][0], e[1][1]))
				if not (p is None):
					if pCar.distTo(p) < minD:
						minD = pCar.distTo(p)
			points.append(minD)
		return points

	def update(self, key):
		if key == 1:
			self.turn(-.4)
		if key == 2:
			self.turn(.4)
		if key == 3:
			self.setForce(250)
		self.time = time() - self.time
		self.x += self.xVel * self.time
		self.y += self.yVel * self.time
		self.xVel += (self.xAcc - (self.xVel * RocketCar.drag)) * self.time
		self.yVel += (self.yAcc - (self.yVel * RocketCar.drag)) * self.time
		self.xAcc = 0
		self.yAcc = 0
		self.time = time()
		cosine = math.cos(math.radians(self.angle))
		sine = math.sin(math.radians(self.angle))
		point1 = (self.x + (sine * 2 * RocketCar.length / 3), self.y - (cosine * 2 * RocketCar.length / 3))
		point2 = (self.x - (cosine * RocketCar.width / 2) - (sine * RocketCar.length / 3), self.y + (cosine * RocketCar.length / 3) - (sine * RocketCar.width / 2))
		point3 = (self.x + (cosine * RocketCar.width / 2) - (sine * RocketCar.length / 3), self.y + (cosine * RocketCar.length / 3) + (sine * RocketCar.width / 2))
		self.car = Object((point1, point2, point3))
		if inner.isCollidingWith(self.car) or not outer.isWithin(self.car): #Collision detection
			return True
		return False

	def draw(self, screen):
		self.car.draw(screen)

	def reset(self):
		self.setPos(140, 370, 0)

	def turn(self, theta):
		self.angle += theta

	def setForce(self, force):
		self.xAcc = math.sin(math.radians(self.angle)) * force / RocketCar.mass
		self.yAcc = -math.cos(math.radians(self.angle)) * force / RocketCar.mass
	
	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def getTheta(self):
		return self.angle

	def getVel(self):
		vel = []
		vel.append(self.xVel)
		vel.append(self.yVel)
		vel.append(self.angle)
		return vel

class Score:
	def __init__(self, obj):
		self.s = 0
		self.targetAngle = calcAngle(obj)

	def update(self, obj):
		temp = calcAngle(obj)
		diff = temp - self.targetAngle
		if not abs(diff) > 6:
			self.s += 100 * (diff)
		self.targetAngle = temp
		return diff

	def getScore(self):
		return self.s

	def reset(self, obj):
		self.s = 0
		self.targetAngle = calcAngle(obj)

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def distTo(self, p):
		xDist = self.x - p.x
		yDist = self.y - p.y
		return math.sqrt((xDist * xDist) + (yDist * yDist))

class Memory:#Simple memory buffer with random sampling
	def __init__(self, max):
		self.maxMem = max
		self.samples = []

	def add(self, sample):
		self.samples.append(sample)
		if len(self.samples) > self.maxMem:
			self.samples.pop(0)

	def getSample(self, number):
		if number > len(self.samples):
			return random.sample(self.samples, len(self.samples))
		else:
			return random.sample(self.samples, number)

class Model:
	def __init__(self, numStates, numActions):
		self.numStates = numStates
		self.numActions = numActions
		
		self.states = tf.placeholder(shape=[None, self.numStates], dtype=tf.float32)#Input layer
		self.qsa = tf.placeholder(shape=[None, self.numActions], dtype=tf.float32)#Ideal outputs/rewards/q value
		fc1 = tf.layers.dense(self.states, 50, activation=tf.nn.relu)#Hidden layer 1
		fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)#Hidden layer 2
		self.logits = tf.layers.dense(fc2, self.numActions)#Output layer
		loss = tf.losses.mean_squared_error(self.qsa, self.logits)#Loss function
		self.optimizer = tf.train.AdamOptimizer().minimize(loss)#Training function
		self.varInit = tf.global_variables_initializer()

	def predict(self, state, sess):
		return sess.run(self.logits, feed_dict={self.states:state.reshape(1,self.numStates)})#Sets self.states to be states (feed_dict part) then finds self.logits

	def predictBatch(self, states, sess):
		return sess.run(self.logits, feed_dict={self.states:states})#Sets self.states to be states (feed_dict part) then finds self.logits

	def trainBatch(self, sess, x_batch, y_batch):
		sess.run(self.optimizer, feed_dict={self.states: x_batch, self.qsa: y_batch})#Sets states and target based on x and y, then runs with optimizer given in init

class Runner:
	def __init__(self, sess, model, reward, car, screen, memory, maxE, minE, decay):#maxE, minE, and decay are all for the epsilon greedy function.
		self.sess = sess
		self.model = model
		self.memory = memory
		self.car = car
		self.reward = reward
		self.screen = screen
		self.maxE = maxE
		self.minE = minE
		self.decay = decay
		self.epsilon = self.maxE
		self.steps = 0 #To decay epsilon greedy
		self.maxRewards = [] #To graph progress

	def run(self):
		self.car.reset()
		self.car.update(0)
		state = np.array(self.car.raycast() + self.car.getVel())
		maxReward = 0
		for g in gates:
			g.reset()
		while True:
			#Drawing stuff
			self.screen.fill((255, 255, 255))
			self.car.draw(self.screen)
			inner.draw(self.screen)
			outer.draw(self.screen)
			pygame.display.flip()
			#Drawing stuff
			action = self.chooseAction(state)#Chose an action (epsilon greedy)
			done = self.car.update(action)#Update car, see if it's done(dead)
			nextState = np.array(self.car.raycast() + self.car.getVel())#Find the next state
			reward = -0.02 #Small negative reward applied every frame, to speed things up.

			if gates[len(gates) - 1].check(self.car):#If it passes the last gate, reset all gates.
				reward = 30
				maxReward += 30
				for g in gates:
					g.reset()

			for g in gates:
				if g.check(self.car):
					reward = 30
					maxReward += 30

			if done:
				reward -= 60#Because it hit a wall
				nextState = None

			self.memory.add((state, action, reward, nextState))
			self.replay()
			
			self.epsilon = self.epsilon * self.decay#Epsilon decay function except I'm bad at functions
			self.steps += 1

			state = nextState
			
			if done:
				self.maxRewards.append(self.reward.getScore())
				break
		print("Reward: {}, Eps: {}".format(maxReward, self.epsilon))

	def chooseAction(self, state):
		if random.random() < self.epsilon:
			return random.randint(0, self.model.numActions - 1)
		else:
			return np.argmax(self.model.predict(state, self.sess))

	def replay(self):
		batch = self.memory.getSample(BATCH_SIZE)#Get batch
		states = np.array([val[0] for val in batch])#Extract states
		nextStates = np.array([(np.zeros(self.model.numStates) if val[3] is None else val[3]) for val in batch]) #Extra stuff is to make a padded 0 state if it's an end state
		_qsa = self.model.predictBatch(states, self.sess)#Predict Q(s, a)
		_qsad = self.model.predictBatch(nextStates, self.sess)#Predict Q(s', a')
		x = np.zeros((len(batch), self.model.numStates))#A matrix batch filled with zeros
		y = np.zeros((len(batch), self.model.numActions))
		for i, b in enumerate(batch):
			state, action, reward, nextState = b[0], b[1], b[2], b[3]
			q = _qsa[i]
			if nextState is None:
				q[action] = reward
			else:
				q[action] = reward + (GAMMA * np.amax(_qsad[i]))
			x[i] = state
			y[i] = q
		self.model.trainBatch(self.sess, x, y)

#At this point, everthing is set up to begin training.

_screen = pygame.display.set_mode([1300, 900])#The screen
_car = RocketCar(140, 370, 0)#Car
_reward = Score(_car)#Reward Function
_model = Model(15, 4)
_mem = Memory(50000)

with tf.Session() as sess:
	sess.run(_model.varInit) #Necessary without eager execution for global variable initialization
	gameRunner = Runner(sess, _model, _reward, _car, _screen, _mem, .9, .1, .999997)
	episodes = 1000
	count = 0
	while count < episodes:
		count += 1
		print("Iteration {} out of 1000".format(count))
		gameRunner.run()

pygame.quit()
