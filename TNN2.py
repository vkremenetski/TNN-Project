import numpy as np
class Tensor:
	def __init__(self, index_dimensions):
		#Recursively creates an zero tensor given a list of index dimensions (**No vector/dual distinction**)
		#Early on, I dabbled with a few ways of representing them and I settled on Tensors as Trees with Scalars as Leaves
		"""For convenience, I've made it so that vectors are the lowest-ranking Tensor Objects possible.
		For scalars to occupy that role would have required a bunch of redundant variables added to each object and generally annoying modifications.
		Things work pretty well as it stands this way, I think."""
		self.rank = len(index_dimensions)
		self.IndexDim = index_dimensions
		if self.rank == 0:
			self.subTensors = 0
		else:
			self.subTensors = [Tensor(self.IndexDim[1:]) for x in range(self.IndexDim[0])]
	def __repr__(self):
		if(self.rank):
			return "Tensor(" + str(self.IndexDim) + ")"
		else:
			return str(self.subTensors)
	def get_rank(self):
		return self.rank
	def get_element(self, indices):
		#Since the tensor object is defined recursively, possible to get non-scalar elements as well
		if len(indices)<=self.rank:
			if(self.IndexDim==[] or indices ==[] or indices[0]<=self.IndexDim[0]):
				if(len(indices)==0):
					if(self.IndexDim):
						return self
					else:
						return self.subTensors
				else:
					return self.subTensors[indices[0]].get_element(indices[1:])
			else:
				raise IndexError("get_element: Value of requested index cannot exceed index dimension") #Identifying which method is causing trouble
		else:
			raise IndexError("get_element: Number of indices cannot exceed rank of Tensor")
	def set_element(self, indices, new):
		#For reasons stated above, possible to set subTensor elements as well
		if len(indices)<=self.rank:
			if(self.IndexDim==[] or indices ==[] or indices[0]<=self.IndexDim[0]):
				if(len(indices)==0):
					if(isinstance(new,Tensor)):
						self = new
					else:
						self.subTensors = new
				else:
					self.subTensors[indices[0]].set_element(indices[1:],new)
			else:
				raise IndexError("set_element: Value of requested index cannot exceed index dimension")
		else:
			raise IndexError("set_element: Number of indices cannot exceed rank of Tensor")
	def __add__(self, other):
		assert isinstance(other,Tensor), "Cannot add Tensor to non-Tensor"
		assert self.IndexDim==other.IndexDim,"Cannot be added; non-matching dimensions"
		result = Tensor(self.IndexDim)
		if(self.IndexDim):
			result.subTensors = [self.subTensors[i]+other.subTensors[i] for i in range(self.IndexDim[0])]
		else:
			result.subTensors = self.subTensors+other.subTensors
		return result
	def __sub__(self,other):
		assert isinstance(other,Tensor), "Cannot add Tensor to non-Tensor"
		assert self.IndexDim==other.IndexDim,"Cannot be subtracted; non-matching dimensions"
		result = Tensor(self.IndexDim)
		if(self.IndexDim):
			result.subTensors = [self.subTensors[i]-other.subTensors[i] for i in range(self.IndexDim[0])]
		else:
			result.subTensors = self.subTensors-other.subTensors
		return result
	def times_scalar(self, scalar):
		#Obviously "scalar" must be a scalar, not sure how to check directly though, since it might be complex, which doesn't fit into the primitive types
		result = Tensor(self.IndexDim)
		if self.rank==0:
			result.subTensors = self.subTensors*scalar
		else:
			result.subTensors = [x.times_scalar(scalar) for x in self.subTensors]
		return result
	def tensor_product(self,other):
		assert isinstance(other,Tensor), "Cannot have Tensor Product with non-Tensor"
		result = Tensor(self.IndexDim+other.IndexDim)
		if(self.rank==0):
			result = other.times_scalar(self.subTensors)
		else:
			result.subTensors = [x.tensor_product(other) for x in self.subTensors]
		return result
	def trace2(self, pair):#Here it is assumed that each index is assigned its own number, i.e. 0th, 1st, 2nd index, etc.
		assert max(pair)<self.rank, "Requested Trace of indices that don't exist"
		assert len(pair)==2,"Trace over more than two indices"
		if pair[0]>=0:
			result = Tensor(self.IndexDim[0:pair[0]]+self.IndexDim[pair[0]+1:pair[1]]+self.IndexDim[pair[1]+1:])
			if pair[0]>0:
				result.subTensors = [x.trace2([i-1 for i in pair]) for x in self.subTensors]
			else:
				pair[1] -=1
				for i in range(self.IndexDim[0]):
					pair[0] = -i-1
					result = result + self.subTensors[i].trace2(pair)
		else:
			result = Tensor(self.IndexDim[0:pair[1]]+self.IndexDim[pair[1]+1:])
			if pair[1] > 0:
				result.subTensors = [x.trace2([pair[0],pair[1]-1]) for x in self.subTensors]
			else:
				result = self.subTensors[-pair[0]-1]
		return result
	def contraction(self, other, index_pairs):
		result = self.tensor_product(other)
		for ip in index_pairs:
			result = result.trace2([ip[0],ip[1]+self.rank])
			for i in index_pairs[1:]:
				if i[0]>ip[0]: i[0]-=1
				if i[1]>ip[1]: i[1]-=2
				else: i[1]-=1
		return result
	def norm_sqrd(self):
		if(self.rank==0):
			return self.subTensors**2
		else:
			return sum([x.norm_sqrd() for x in self.subTensors])
	def norm(self):
		return np.sqrt(self.norm_sqrd())