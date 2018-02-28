import numpy as np
#Useful variables/Quantum Gates------------------------------------------------------------
j = 1j
sq = np.sqrt
r2 = np.sqrt(2)
nil = 10**(-6)
zero = np.array([1,0])
one = np.array([0,1])
MaB = np.array([[1,0,j,0],[0,j,0,1],[0,j,0,-1],[1,0,-j,0]])/r2 #matrix of normalized magic states (each whole list is a row)

MaBinv = np.linalg.inv(MaB)
MagicStates = [np.array([1,0,0,1])/r2,np.array([0,j,j,0])/r2,np.array([j,0,0,-j])/r2,np.array([0,1,-1,0])]/r2 #list of magic states (each array is a magic state)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
H = np.array([[1,1],[1,-1]])/r2
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
Y = np.array([[0,-j],[j,0]])
I = np.array([[1,0],[0,1]])
def get_ev(fake_ev):#deals with the wretched convention of numpy's eigenvector output
	return np.array([fake_ev[:,i] for i in range(len(fake_ev))])
def complicate(n): #turns a purely real number into complex number so that numpy.log can be taken of negative numbers
	return np.real(n)+j*(np.imag(n))
def expm(m):#Computes e^M where M is a square matrix
	eigenvalues,eigenvectors = np.linalg.eig(m)[0],get_ev(np.linalg.eig(m)[1])
	eMatrix = np.zeros((len(m),len(m[0])))
	for i in range(len(eigenvalues)):
		eMatrix = eMatrix + ((np.e**eigenvalues[i])*np.tensordot(eigenvectors[i],np.conjugate(eigenvectors[i]),0))
	return eMatrix
def logm(m):#Computes log(M) where M is a square matrix
	eigenvalues,eigenvectors = np.linalg.eig(m)[0],get_ev(np.linalg.eig(m)[1])
	eMatrix = np.zeros((len(m),len(m[0])))
	for i in range(len(eigenvalues)):
		eMatrix = eMatrix + ((np.log(complicate(eigenvalues[i])))*np.tensordot(eigenvectors[i],np.conjugate(eigenvectors[i]),0))
	return eMatrix
def megadot(listy): #takes consecutive dot products of a list of matrices
	assert len(listy)>0,"No matrices to multiply together"
	if len(listy)==1:
		return listy[0]
	else:
		return np.dot(listy[0],megadot(listy[1:]))




#Ref[10] Decomposition Steps/components (PART ONE) ----------------------------------------

def zerofy(a): #Removes leftover bits from rounding errors in a list of numbers
	result = [np.real(r) if np.absolute(np.imag(r))<nil else r for r in result]
	result = [np.imag(r)*j if np.absolute(np.real(r))<nil else r for r in result]
	return result
def zm(m):
	for i in range(len(m)):
		m[i] = zerofy(m[i])
def oTimesVector(a,b): #takes tensor product of two vectors as a vector itself
	return np.array([i*j for i in a for j in b])
def get_zetas(states,operator): #returns the angle of the global phase imposed when an operator maps one maximally entangled state to a magic state
	result1 = []
	result2 = []
	states2 = [np.dot(operator,s) for s in states]
	for i in range(len(states2)):
		exponential = ([x for x in states2[i] if np.absolute(x)>nil][0])
		result1.append(np.angle(exponential))
		result2.append(np.array(zerofy(states2[i])))
	return [result1,result2]
def normalize(state):
	norm = np.sqrt(sum([x*np.conj(x) for x in state]))
	return np.array([s/norm for s in state])
def factorPS(product_state): #Factors a product state (product state must already be normalized)
	ps = product_state
	if (np.absolute(ps[0]*ps[3])>nil):
		atob,ctod = ps[0]/ps[2],ps[0]/ps[1]
		v1,v2 = normalize([atob,1]),normalize([ctod,1])
		v1 = v1*(ps[0]/(v1[0]*v2[0]))
	elif (np.absolute(ps[0])<nil):
		if (np.absolute(ps[1])>nil):
			if(np.absolute(ps[3])>nil):
				atob = ps[1]/ps[3]
				v1,v2 = normalize([atob,1]),np.array([0,1])
				v2 = v2*(ps[1]/v1[0])
			else:
				v1,v2 = np.array([ps[1],0]),np.array([0,1])
		elif(np.absolute(ps[2])>nil):
			if(np.absolute(ps[3])>nil):
				ctod = ps[2]/ps[3]
				v1,v2 = np.array([0,1]),normalize([ctod,1])
				v1 = v1*ps[2]/v2[0]
			else:
				v1,v2 = np.array([0,ps[2]]),np.array([1,0])
		else:
			v1,v2 = np.array([0,ps[3]]),np.array([0,1])
	else:
		if(np.absolute(ps[1])>nil):
			ctod = ps[0]/ps[1]
			v1,v2 = np.array([1,0]), normalize([ctod,1])
			v1 = v1*ps[0]/v2[0]
		elif(np.absolute(ps[2])>nil):
			atob = ps[0]/ps[2]
			v1,v2 = normalize([atob,1]),np.array([1,0])
			v2 = v2*ps[0]/v1[0]
		else:
			v1,v2 = np.array([ps[0],0]),np.array([1,0])
	return [v1,v2]			
def TensorProd2q(a,b): #Takes the tensor product of two single-qubit gates, returns as a 4-by-4 matrix
	result = np.tensordot(a,b,0)
	result = np.transpose(result,[0,2,1,3])
	result = np.reshape(result,(4,4))
	return result
def MB(U): #converts U into the magic basis
	return np.dot(MaBinv,np.dot(U,MaB))
def eigs(U): #finds the eigenvalues (converts to epsilons) and eigenvectors (in magic basis)
	UtU = np.dot(np.transpose(U),U)
	zm(UtU)
	eigs = np.linalg.eig(UtU)
	evecs,evals = eigs[1],eigs[0]
	evecs = get_ev(evecs)
	epsilons = [np.angle(evl)/2 for evl in evals]
	return [zerofy(epsilons),evecs]
def ops1(ev):
	ef,etft,psi3 = np.dot(MaB,(ev[0]-j*ev[1])/r2),np.dot(MaB,(ev[0]+j*ev[1])/r2),np.dot(MaB,ev[2])
	vecs1,vecs2 = factorPS(ef),factorPS(etft)
	e,f,et,ft = vecs1[0],vecs1[1],vecs2[0],vecs2[1]
	eft,etf = oTimesVector(e,ft),oTimesVector(et,f)
	delta = np.angle(np.dot(np.conj(eft),psi3)*j*r2)
	Va = (np.tensordot(zero,np.conj(e),0)+(np.e**(j*delta)*np.tensordot(one,np.conj(et),0)))
	Vb = (np.tensordot(zero,np.conj(f),0)+(np.e**(-j*delta)*np.tensordot(one,np.conj(ft),0)))
	OP = MB(TensorProd2q(Va,Vb))
	zetas,mStates = get_zetas(ev,OP)[0],get_zetas(ev,OP)[1]
	return [zerofy(zetas),[Va,Vb],mStates]
def ops2(epsilons,ev,U): #U and ev must both be in the magic basis; returns maximally entangled basis to which U maps UtU's eigenvectors
	result = []
	for i in range(len(epsilons)):
		result.append((np.e**(-j*epsilons[i])) * np.dot(U,ev[i]))
	return result
def ops3(zetas,epsilons,states): #states must still be in the magic basis
	exponentials, ops, mStates = ops1(states)[0],ops1(states)[1],ops1(states)[2]
	lambdas = []
	for i in range(len(exponentials)):
		lambdas.append(exponentials[i]-zetas[i]-epsilons[i])
	ops = [np.matrix.getH(o) for o in ops]
	lambdas = [(x+2*np.pi) if (x<-np.pi) else x for x in lambdas]
	return [zerofy(lambdas),ops,mStates]
def makeUd(lambdas,MagicStates): #returns Ud in *computational* Basis, assuming Ud 4 by 4
	Ud = np.zeros((4,4))
	for i in range(4):
		m = MagicStates[i]
		Ud = Ud + ((np.e**(-j*lambdas[i]))*np.tensordot(m,np.conjugate(m),0))
	return np.dot(MaB,np.dot(Ud,MaBinv))
def decomposition1(U): #returns lambdas, as well as Ua/b,Va/b,Ud operators from ref[10] paper
	U = MB(U)
	eigenstuffs = eigs(U)
	epsilons,eigenstates = eigenstuffs[0],eigenstuffs[1]
	zetasVs = ops1(eigenstates)
	zetas,Va,Vb = zetasVs[0],zetasVs[1][0],zetasVs[1][1]
	otherEntangledBasis = ops2(epsilons,eigenstates,U)
	lambdasUsMS = ops3(zetas,epsilons,otherEntangledBasis)
	lambdas,Ua,Ub,MS = lambdasUsMS[0],lambdasUsMS[1][0],lambdasUsMS[1][1],lambdasUsMS[2]
	Ud = makeUd(lambdas,MS)
	return [zerofy(lambdas),Ua,Ub,Va,Vb,Ud]

#------------------------------------------------------------------------------------------


#Vidal and Dawson Decomposition Steps/Components (PART TWO) -------------------------------
def getHam(Ud):#calculates the Hamiltonian Matrix of Ud
	Ham = j*logm(Ud)
	for i in range(4):
		Ham[i] = zerofy(Ham[i])
	return Ham
def get_raw_alphas(Ud): #Returns alphas that are possibly outside [0,pi/2] range (assumes Ud is 4 by 4 matrix)
	Ham = getHam(Ud)
	return [0.5*(Ham[3][0]+Ham[2][1]),0.5*(Ham[2][1]-Ham[3][0]),Ham[0][0]]
def get_and_set_alphas(raw_alphas,Ua,Ub,Va,Vb,Ud): #returns alphas all inside [0,pi/2] range and modifies U/V,Ud operators accordingly.
	ra = zerofy(raw_alphas)
	paulis = [X,Y,Z]
	for i in range(3):
		while ra[i]<0:
			ra[i] += np.pi/2
			Ua,Ub,Ud = np.dot(sq(j)*Ua,paulis[i]),np.dot(sq(j)*Ub,paulis[i]),np.dot(-j*TensorProd2q(paulis[i],paulis[i]),Ud)
		while ra[i]>=np.pi/2-nil:
			ra[i] -= np.pi/2
			Ua,Ub,Ud = np.dot(sq(-j)*Ua,paulis[i]),np.dot(sq(-j)*Ub,paulis[i]),np.dot(j*TensorProd2q(paulis[i],paulis[i]),Ud)
	return [zerofy(ra),Ua,Ub,Va,Vb,Ud]
def decomposition2a(alphas,Va,Vb,Ua,Ub): # First V&D method - decomposes using THREE CNOT gates (assumes that alpha_z is not equal to zero)
	u1,v1 = Va,Vb
	u2,v2 = j*np.dot(H,expm(-j*(alphas[0]-np.pi/4)*X)),expm(-j*alphas[2]*Z)
	u3,v3 = -j*H,expm(j*alphas[1]*Z)
	w,wh = (I-j*X)/r2,(I+j*X)/r2
	u4,v4 = np.dot(Ua,w)*np.e**(-j*np.pi/8),np.dot(Ub,wh)*np.e**(-j*np.pi/8)
	return [[u1,v1],CNOT,[u2,v2],CNOT,[u3,v3],CNOT,[u4,v4]]
def decomposition2b(alphas,Va,Vb,Ua,Ub): # Second V&D method - decomposes using TWO CNOT gates (assumes alpha_z IS equal to zero)
	u2,v2 = expm(-j*alphas[0]*X),expm(-j*alphas[1]*Z)
	w,wh= (I-j*X)/r2,(I+j*X)/r2
	u1,v1,u3,v3 = np.dot(wh,Va),np.dot(w,Vb),np.dot(Ua,w),np.dot(Ub,wh)
	return [[u1,v1],CNOT,[u2,v2],CNOT,[u3,v3]]

#------------------------------------------------------------------------------------------


#Grand Decomposition,Altogether! (PART THREE)----------------------------------------------

def decompose(U): #The grand decomposition function - returns single-qubit gates and 2 or 3 CNOT gates in the order they appear on the circuit.
	partone = decomposition1(U)
	lambdas,Ua,Ub,Va,Vb,Ud = partone[0],partone[1],partone[2], partone[3], partone[4],partone[5]
	ra = get_raw_alphas(Ud)
	q = get_and_set_alphas(ra,Ua,Ub,Va,Vb,Ud)
	alphas,Ua,Ub,Va,Vb = q[0],q[1],q[2],q[3],q[4]
	if (np.absolute(alphas[2])>nil):
		return decomposition2a(alphas,Va,Vb,Ua,Ub)
	else:
		return decomposition2b(alphas,Va,Vb,Ua,Ub)

#------------------------------------------------------------------------------------------





