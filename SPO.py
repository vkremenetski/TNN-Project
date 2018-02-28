#Swarm Particle Optimization

import numpy as np
#Useful Variable, Gates--------------------------------------------------------------------
j = 1j
sq = np.sqrt
r2 = np.sqrt(2)
nil = 10**(-6)
zero = np.array([1,0])
one = np.array([0,1])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
H = np.array([[1,1],[1,-1]])/r2
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
Y = np.array([[0,-j],[j,0]])
I = np.array([[1,0],[0,1]])
def normalize(state):
	norm = np.sqrt(sum([x*np.conj(x) for x in state]))
	return np.array([s/norm for s in state])
def TensorProd2q(a,b): #Takes the tensor product of two single-qubit gates, returns as a 4-by-4 matrix
	result = np.tensordot(a,b,0)
	result = np.transpose(result,[0,2,1,3])
	result = np.reshape(result,(4,4))
	return result
def rand_sphere(G,R): #Generates a uniformly random point on an n-sphere, given a center-point G and radius R (scalar). 
	coords = [np.random.normal() for i in range(len(G))]
	norm = np.sqrt(sum([c**2 for c in coords]))
	coords = np.array([((c/norm)*R) for c in coords])+G
	return coords



#Useful Classes/Methods--------------------------------------------------------------------
def fitness_maker(Hamiltonian):
	def fitness(position,Ham=Hamiltonian): 
		p = np.array([position[i]+j*position[i+1] for i in 2*np.array(range(len(position)//2))])
		return np.dot(np.conj(p),np.dot(Ham,p))
	return fitness

class Neighborhood:
	def __init__(self,particles):
		self.neighbors = particles
		self.best = min([p.prev_best for p in particles],key=fitness)
	def __repr__(self):
		return "Neighborhood("+str(self.neighbors)+")"
	def update(self):
		self.best = min([self.best,min([n.prev_best for n in self.neighbors],key=fitness)],key=fitness)
class Particle:
	def __init__(self,dimension,neighborhood=None):
		self.position = np.array([np.random.uniform() for i in range(2*dimension)])
		self.velocity = np.array([np.random.uniform(low = -self.position[i],high=1-self.position[i]) for i in range(2*dimension)])
		self.prev_best = self.position
		if(neighborhood):
			self.neighborhood = Neighborhood(neighborhood.neighbors.append(self))
		else:
			self.neighborhood = Neighborhood([self])
	def __repr__(self):
		return "Particle("+str(self.position)+")"
	def confine1(self):
		for i in range(len(self.position)):
			if self.position[i]>1:
				self.position[i] = 1
				self.velocity[i] = 0
			elif self.position[i]<-1:
				self.position[i] = -1
				self.velocity[i] = 0
	def confine2(self):
		for i in range(len(self.position)):
			if self.position[i]>1:
				self.position[i] = 1
				self.velocity[i] *= -0.5
			elif self.position[i]<-1:
				self.position[i] = -1
				self.velocity[i] *= -0.5
	def confine(self,key=1):
		if key==1:
			self.confine1()
		else:
			self.confine2()
	def update1(self):
		w,c = 1/(2*np.log(2)),0.5+np.log(2)
		U = np.random.uniform(0,c)
		self.velocity = (w*self.velocity)+(U*(self.prev_best-self.position))+(U*(self.neighborhood.best-self.position))
		self.position += self.velocity
		self.confine()
		self.prev_best = min([self.prev_best,self.position],key=fitness)
	def update2(self):
		w,c = 1/(2*np.log(2)),0.5+np.log(2)
		if(self.prev_best.all()==self.neighborhood.best.all()):
			G = self.position+c*(self.prev_best-self.position)/2
		else:
			G = self.position+c*(self.prev_best+self.neighborhood.best - 2*self.position)/3
		x_prime = rand_sphere(G,np.sqrt(np.sum([d**2 for d in G-self.position])))
		x = self.position
		self.position = self.velocity*w+x_prime
		self.velocity = w*self.velocity+x_prime-x
		self.confine()
		self.prev_best = min([self.prev_best,self.position],key=fitness)
	def update(self,key=1):
		if key==1:
			self.update1()
		else:
			self.update2()

#Actual Simulation Steps-------------------------------------------------------------------
fitness = fitness_maker(Z)
def PSO(Hamiltonian):
	fitness = fitness_maker(Hamiltonian)
	particles,neighborhoods = [Particle(len(Hamiltonian)) for i in range(35)],[]
	for i in range(35):
		neighborhoods.append(Neighborhood([particles[i-1],particles[i],particles[(i+1)%35]]))
		particles[i].neighborhood = neighborhoods[-1]
	for i in range(100):
		for p in particles:
			p.update()
			p.neighborhood.update()
	candidates = [n.best for n in neighborhoods]
	return min(candidates,key=fitness)