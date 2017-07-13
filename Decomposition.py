import numpy as np
j = 1j
r2 = np.sqrt(2)
zero = np.array([1,0])
one = np.array([0,1])
MaB = np.array([[1,j,0,0],[0,0,j,1],[0,0,j,-1],[1,-j,0,0]])/r2 #defined using unnormalized vectors because rt(2)'s cancel out when transforming matrix to magic basis
#Also neater to work with
MaBinv = np.linalg.inv(MaB)
MagicStates = [np.array([1,0,0,1])/r2,np.array([j,0,0,-j])/r2,np.array([0,j,j,0])/r2,np.array([0,1,-1,0])/r2]
def sign(x):
	if x>=0:
		return 1
	else:
		return -1
def get_zetas(states,operator):
	result = []
	states2 = [np.dot(operator,s) for s in states]
	for s in states2:
		if s[0]:
			if s[0]/s[3]>0:
				result.append(np.angle(MagicStates[0][0]/s[0]))
			else:
				result.append(np.angle(MagicStates[1][0]/s[0]))
		else:
			if s[1]/s[2]>0:
				result.append(np.angle(MagicStates[2][1]/s[1]))
			else:
				result.append(np.angle(MagicStates[3][1]/s[1]))
	return result
def xnot(qbit):
	return np.array([(x+1)%2 for x in qbit])
def TensorProd2q(a,b):
	result = np.tensordot(a,b,0)
	result = np.transpose(result,[0,2,1,3])
	result = np.reshape(result,(4,4))
	return result
def MB(U): #converts U into the magic basis
	return np.dot(MaBinv,np.dot(U,MaB))
def eigs(U): #finds the eigenvalues (converts to epsilons) and eigenvectors (in magic basis)
	UtU = np.dot(np.transpose(U),U)
	eigs = np.linalg.eig(UtU)
	evecs,evals = eigs[1],eigs[0]
	epsilons = [np.angle(evl)/2 for evl in evals]
	return [epsilons,evecs]

def ops1(ev): #takes in a set of eigenvectors in magic basis, and returns operators Va,Vb as well as zetas
	good_ev = [e for e in ev if e[2]==e[3]==0]
	good_ev = [np.dot(MaB,e) for e in good_ev]
	ef = good_ev[0]-j*good_ev[1]
	if np.absolute(ef[0])<10**(-6):
		e,f = np.array([0,1]),np.array([0,1])
	else:
		e,f = np.array([1,0]),np.array([1,0])
	Va,Vb = (np.tensordot(zero,e,0)+np.tensordot(one,xnot(e),0)),(np.tensordot(zero,f,0)+np.tensordot(one,xnot(f),0))
	OP = TensorProd2q(Va,Vb)
	standardEV = [np.dot(MaB,e) for e in ev]
	zetas = get_zetas(standardEV,OP)
	return [zetas,[Va,Vb]]




#--------------------------------------
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
H = np.array([[1,1],[1,-1]])/r2
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
Y = np.array([[0,-j],[j,0]])
I = np.array([[1,0],[0,1]])
