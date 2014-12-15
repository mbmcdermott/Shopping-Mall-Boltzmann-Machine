import numpy

numpy.random.seed(2014)
class Boltzmann_Machine_Neural_Network:

	def __init__(self,nhid,nvis,lr):
		#l is the learning rate
		#nhid is the number of hidden units, a parameter we set (denoted by j)
		#nvis is an array with rows representing number of instances of data 
		#and columns the number of visible units (denoted by i)
		
		#initialize a small random matrix for the weights w_{ij}:
		self.w= numpy.random.normal(0,.5,(nvis,nhid))
		
		#initialize learning rate
		self.lr=lr
		
		#intialize hidden and visible sector dimensions
		self.nvis=nvis
		self.nhid=nhid
		

		
		
	def reconstruct1(self,vis_ar):
		"""
		Input
		---
		see reconstruct() 
		"""
		#for each instance of a visible unit vector calculate a hidden unit vector:
		#rows are instances (people) and columns are hidden units
		hprob = self.thermal_prob_vtoh(vis_ar,self.w)
		
		#turn on the hidden units with probability hprob
		h_ar = hprob > numpy.random.rand(numpy.shape(hprob)[0],numpy.shape(hprob)[1])
		
		#now that we have the hidden array, recontruct the visible array
		vprob = self.thermal_prob_htov(h_ar,self.w)
		
		#turn on the visible units with probability vprob
		v_ar = vprob > numpy.random.rand(numpy.shape(vprob)[0],numpy.shape(vprob)[1])
		
		#now calculate the w gradient (i.e. how much we'll be changing w by)
		#first calculate the correlations between hidden and visible sector given the intial visible vectors in the data.  NOTE: the dot product adds up all the instances for each {ij} so we divide by the number to get an average
		corgv = numpy.dot(vis_ar.T,hprob)/vis_ar.shape[0]   #gives an array of same dimensions as w_{ij} (weights)  
		#second calculate correlations in thermal state (which is approximated here by the reconstruction vector, the more times we reconstruct the more accurate this will be generally)
		cor   = numpy.dot(vprob.T,hprob)/vis_ar.shape[0]
		
		#and now we can calculate our new weights
		neww = self.w + self.lr*(corgv-cor)
		return {'corgv':corgv,'cor':cor,'neww':neww,'v_ar':v_ar, 'vprob':vprob}
	
	
	
	
	#function that we use to teach the machine
	def reconstruct(self,vis_ar,n=1):
		"""
		Creates a recontruction of the visible units with n Markov Chains.
		
		Input
		-------
		vis_ar: matrix of visible units supplied in the data, with columns representing
		the units and rows the different instances of the units.
		n: number of steps to run in Markov Chain before outputting reconstruction.
		
		Output
		-------
		matrix of reconstructed visible units.
		"""
		if n==1:
		  self.w = self.reconstruct1(vis_ar)['neww']
		
		
		
		
		elif n>1:
			#take corgv from first step
			corgv = self.reconstruct1(vis_ar)['corgv']
			#keep updating the visible array n-1 times (since reconstruction already does the first)
			for i in range(n-1):
				#pick out the last step cor value
				cor    = self.reconstruct1(vis_ar)['cor']
				#update visible array
				vis_ar = self.reconstruct1(vis_ar)['v_ar']
			
			#now that we have corgv and cor we can update weights w
			self.w = self.w + self.lr*(corgv - cor)
	
	
	def relax(self, vis_ar , n=1 , update_weight = 10 ):
		"""
		Relax to the system with optimal weights, updating the weights "update_weight" times
		Input
		---
		vis_ar: see reconstruct()
		n     : see reconstruct()
		"""
		for i in range(update_weight):
			self.reconstruct(vis_ar,n)
			
			
	
	
	def predict(self,which_unit,other_data,n=10,trials=100):
		"""
		A function which will, after being given a data set to learn from, predict the
		value of unit "which_unit" (counting the first unit as 1) given all the other data.
		In our example this means we can predict whether a customer went to a given store given all
		the other stores they went to.
		
		Input
		------
		which_unit: the index of the unit we are trying to predict
		
		Output
		------
		The probability of a unit "which_unit" being turned on
		"""
		#initialize our two test vectors, with [:] to make a copy rather than a new pointer
		v_on=other_data[:] 
		v_off=other_data[:]
		
		v_on.insert(which_unit-1,1)
		v_off.insert(which_unit-1,0)
		
		#make many rows of these same vectors to get probabilistic sample
		v_on_ar = numpy.array([v_on,]*trials).astype(float)
		v_off_ar = numpy.array([v_off,]*trials).astype(float)
		
		#print(self.reconstruct1(v_on_ar)['vprob'])
		#print(self.reconstruct1(numpy.array([[1,1,1,0,0,0,0],[1,1,1,0,0,0,1]]))['vprob'])
		#print(self.reconstruct1(numpy.array([[1,1,1,0,0,0,0], [ 9.55487987e-01,   9.99204473e-01,   8.93817644e-01,4.82083621e-03,   2.67039148e-01,   1.53815797e-02,2.20950729e-04]])))['vprob']
		#run n markov steps and update the unit we are trying to predict
		for i in range(n):
			v_on_ar[:,which_unit-1]  = self.reconstruct1(v_on_ar)['v_ar'][:,which_unit-1]
			v_off_ar[:,which_unit-1] = self.reconstruct1(v_off_ar)['v_ar'][:,which_unit-1]
			
			
		
		#average the trials (rows) to get a probability of unit "which_unit" is turned on
		avg_vis_unit = (sum(v_on_ar.astype(float)) + sum(v_off_ar.astype(float)) )/(2*trials)  #convert the boolean structure to float
		
		print(avg_vis_unit)
		#get prob of the unit we are trying to predict ("which_unit")
		return avg_vis_unit[which_unit-1]
	
	
	
	
	
	
	
	
	
	
	def thermal_prob_vtoh(self,vis,weight):
		#takes visible units and finds hidden unit probabilities
		#outputs a matrix with rows for instances and columns for hidden units
		return 1/(1+numpy.exp(-numpy.dot(vis,weight)))
	
	
	
	def thermal_prob_htov(self,hid,weight):
		#takes hidden units (rows being instances and columns being units) and finds visible unit probabilities
		#outputs a matrix with rows for instances and columns for visible units
		x =  1/(1+numpy.exp(-numpy.dot(weight,hid.T)))	
		#take the transpose to have the columns representing the vis units 
		#(rather than the rows)
		return x.T




if __name__ == '__main__':
	my_boltz = Boltzmann_Machine_Neural_Network(nhid=4,nvis=7,lr=0.1)
	vis_ar_data = numpy.array([[   1   ,  1  ,    1    ,     0    ,  0  ,       0      ,     0    ],
[   0   ,  0  ,    0    ,     1    ,  1  ,       0      ,     0     ],
[   1   ,  1  ,    1    ,     0    ,  0  ,       0      ,     0     ],
[   1   ,  1  ,    0    ,     0    ,  0  ,       0      ,     0     ],
[   0   ,  1  ,    1    ,     0    ,  0  ,       1      ,     0     ],
[   1   ,  1  ,    1    ,     0    ,  1  ,       0      ,     0     ],
[   0   ,  0  ,    0    ,     1    ,  1  ,       0      ,     0     ],
[   0   ,  0  ,    0    ,     0    ,  0  ,       1      ,     1     ],
[   0   ,  0  ,    0    ,     0    ,  0  ,       1      ,     1     ],
[   1   ,  0  ,    0    ,     0    ,  0  ,       1      ,     1     ],
[   1   ,  0  ,    0    ,     1    ,  0  ,       1      ,     0     ],
[   1   ,  0  ,    0    ,     1    ,  0  ,       1      ,     0     ],
[   0   ,  0  ,    0    ,     1    ,  1  ,       0      ,     1     ],
[   1   ,  0  ,    0    ,     1    ,  0  ,       1      ,     0     ]])
	my_boltz.reconstruct(vis_ar=vis_ar_data,n=1)
	print(my_boltz.w)
	my_boltz.relax(vis_ar=vis_ar_data,n=2,update_weight=4000)
	print repr(my_boltz.w)
	s=my_boltz.predict(which_unit=7,other_data=[1,1,1,1,1,1],n=1,trials=200000)
	print(s)




