import numpy as np
from numpy.random import rand
from tqdm import tqdm

class IsingSim():
  
  """This class performs Ising model simulations on a 2D grid. Interaction parameters are given by a matrix at each lattice site. 
  Field dependence is not supported at this time but will be in due course. The simulator outputs configurations after equlibrium
  as well as the trajectories, if specifically requested.
  Inputs:
    - N : (integer) - Size of lattice will be N^2. Only 2D square lattice is supported at this stage.
    - J_mat: (numpy matrix of shape(5,5)) - entries being floats for interaction parameters. Self-interaction (middle element of matrix)=0. 
              or: (list) of size(5,5) with each element belonging to scipy distribution from which to draw J value (for bond disorder)
    - T: (float) - Reduced temperature for simulation
    - save_trajectories: (Boolean) - whether to save trajectories, or only final state. Default False.
    - eqSteps: (integer) number of Monte-Carlo steps for equlibration before simulation starts. Default 750. AKA 'burn-in'.
    - mcSteps: (integer) number of Monte-Carlo for simulation. Default 750.
  Outputs: Several outputs are available, including trajectories (if called), configurations (i.e., the 2D states) and configurations histograms.
  These can be obtained by calling methods self.configurations(), self.histograms() and self.trajectories()"""

  def __init__(self, N = 40, J_mat = None, T = 2.7, save_trajectories = False,
               eqSteps = 750, mcSteps = 750):
    self.N = N
    
    #If no J matrix is provided we default to isotropic J interaction with NN with value 0.5
    if np.array(J_mat).all() == None:
      J_mat = np.zeros((5,5))
      J_mat[1,2] = J_mat[2,1] = J_mat[2,3] = J_mat[3,2] = 0.5 #Defaulting to 0.5 for NN, all others zeroed out.
    
    self.J_mat = J_mat

    try:
      rv = J_mat[1,2]
    except TypeError:
      rv = J_mat[1][2] #in case we have a list
      
    if 'dist' in str(type(rv)): #in this case we have random bond disorder situation
      self.bond_disorder = True
      self.J_lattice = self.make_J_lattice()
      self.J_mat = None
    else:
      self.bond_disorder = False
      self.J_lattice = None

    self.save_trajectories = save_trajectories
    self.eqSteps = eqSteps
    self.mcSteps = mcSteps
    self.config = self.initialState(random = True)
    self.T = T
    self.configs_list = self.make_configs_list()

  def initialState(self, random = False):   
    ''' Generates a lattice with spin configurations drawn randomly [-1 or 1] if random=True
    Else, the lattice is generated with all sites = 1 '''

    if random==True: state = 2*np.random.randint(2, size=(self.N,self.N))-1
    else: state = np.ones([self.N,self.N])

    return state

  def make_J_lattice(self):   

    ''' Return a matrix size (N,N,5,5) signifying interaction parameters at each lattice site, drawn from 
    distributions provided '''

    J_lattice = np.zeros(shape=(self.N, self.N, 5,5))
    for i in range(self.N):
      for j in range(self.N):
        for k in range(5):
          for l in range(5):
            if 'dist' in str(type(self.J_mat[k][l])):
              J_lattice[i,j,k,l] = self.J_mat[k][l].rvs(1)
            else: J_lattice[i,j,k,l] = self.J_mat[k][l]
    
    return J_lattice

  def mcmove(self, config):
    '''Monte Carlo move using Metropolis algorithm '''
    
    M = 5 #neighborhood size (M^2). Do not change!
    beta = 1.0/self.T

    for i in range(self.N):
      for j in range(self.N):
        if self.bond_disorder: J_mat = self.J_lattice[i,j,:,:]
        else: J_mat = self.J_mat

        cost = 0.0
        s =  np.copy(config[i, j])

        for p in range(-int(M/2), int(M/2)+1, 1):
          for q in range(-int(M/2), int(M/2)+1, 1):
            cost += 2 * J_mat[int(M/2)+p, int(M/2)+q] * config[(i+p)%self.N,(j+q)%self.N] * config[i,j]
        
        if cost < 0:
          s *= -1
        elif rand() < np.exp(-cost*beta):
          s *= -1
        config[i, j] = s
    return config


  def calcEnergy(self, config):
    '''Returns the energy of the current configuration'''
    M = 5
    energy = 0.0
    for i in range(len(config)):
      for j in range(len(config)):
        #In case we have bond disorder
        if self.bond_disorder: J_mat = self.J_lattice[i,j,:,:]
        else: J_mat = self.J_mat #otherwise, no

        s = config[i,j]
        for p in range(-int(M/2), int(M/2)+1, 1):
          for q in range(-int(M/2), int(M/2)+1, 1):
            energy += -J_mat[int(M/2)+p, int(M/2)+q] * config[(i+p)%self.N,(j+q)%self.N] * config[i,j]

    return energy/4.0


  def calcMag(self, config):
    '''Magnetization of a given configuration'''
    return np.sum(config)

  def calcAbsMag(self, config):
    ''' Absolute Magnetization of a given configuration'''
    return (np.abs(np.sum(config)))*1.0
  
  def performIsingSim(self):
    
    E1, M1, E2, M2 = 0.0,0.0,0.0,0.0    #These are all the average properties of all MC steps used
        
    config = np.copy(self.config)
    T = self.T
    if self.save_trajectories: config_mat = np.zeros([self.mcSteps,self.N,self.N])   #Saving all the configurations
    

    print('\n---Performing Equlibration---\n')
    for i in tqdm(range(self.eqSteps)):
        config = self.mcmove(config)

    print('\n---Finished...\n----Performing MC Moves----\n')
    for j in tqdm(range(self.mcSteps)):
        config = self.mcmove(config)
        Ene = self.calcEnergy(config)
        Mag = self.calcAbsMag(config)

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag * Mag
        E2 = E2 + Ene * Ene
        
        if self.save_trajectories: config_mat[j] = config
    
    print('Completed. Saving')    
    Energy = E1 / (self.mcSteps * self.N * self.N)
    Magnetization = M1 / (self.mcSteps * self.N * self.N)
    n1, n2  = 1.0/(self.mcSteps*self.N*self.N), 1.0/(self.mcSteps*self.mcSteps*self.N*self.N) 
    iT = 1.0/self.T
    iT2 = iT*iT
    SpecificHeat = (n1*E2 - n2*E1*E1)*iT2
    Susceptibility = (n1*M2 - n2*M1*M1)*iT
    # SpecificHeat = (E2 / self.mcSteps - E1 * E1 / (self.mcSteps * self.mcSteps)) / (self.N * self.T * self.T)
    # Susceptibility = (M2 / self.mcSteps - M1 * M1 / (self.mcSteps * self.mcSteps)) / (self.N * self.T)

    self.config = config
    config_hist, config_hist_norm = self.get_config_histogram()

    if self.save_trajectories:
      results_dict = {'config': config_mat, 'Energy': Energy, 'Magnetization': Magnetization,
      'SpecificHeat': SpecificHeat, 'Susceptibility': Susceptibility, 
      'Histogram': config_hist_norm}
    else:
      results_dict = {'Energy': Energy, 'Magnetization': Magnetization,
      'SpecificHeat': SpecificHeat, 'Susceptibility': Susceptibility,
      'Histogram': config_hist_norm}
    
    self.results = results_dict

    return 'Completed simulation'
    
  def make_configs_list(self):

    #Let's write down possible configurations for nearest neighbors
    all_configs = [
               #Configurations
               
               #all -1 -> join with last
               [-1,-1,-1,-1,-1],               #1

               #one 1 -> all together now
               [-1,-1,-1,-1,1],               #2
               [-1,-1,-1,1,-1],               #3
               [-1,-1,1,-1,-1],               #4
               [-1,1,-1,-1,-1],               #5
               [1,-1,-1,-1,-1],               #6
    
               #two 1s -> all together now
               [1,1,-1,-1,-1],               #7
               [1,-1,1,-1,-1],               #8
               [1,-1,-1,1,-1],               #9
               [1,-1,-1,-1,1],               #10
               [-1,1,1,-1,-1],               #11
               [-1,1,-1,1,-1],               #12
               [-1,1,-1,-1,1],               #13
               [-1,-1,1,1,-1],               #14
               [-1,-1,1,-1,1],               #15
               [-1,-1,-1,1,1],               #16
               
               #three ones -> go into two types
               [1,1,1,-1,-1],               #17
               [1,1,-1,1,-1],               #18
               [1,1,-1,-1,1],               #19
               [1,-1,1,1,-1],               #20
               [1,-1,1,-1,1],               #21
               [1,-1,-1,1,1],               #22
               [-1,1,1,1,-1],               #23
               [-1,1,-1,1,1],               #24
               [-1,-1,1,1,1],               #25
               [-1,1,1,-1,1],               #26

               #four 1s -> all together now
               [1,-1,1,1,1],               #27
               [1,1,-1,1,1],               #28
               [1,1,1,-1,1],               #29
               [1,1,1,1,-1],               #30
               [-1,1,1,1,1],               #31

               #five 1s -> join with first
               [1,1,1,1,1],               #32
      ]

    return all_configs

  def get_config_histogram(self):

    config_hist = np.zeros(shape=(len(self.configs_list)))

    for i in range(self.N):
        for j in range(self.N):
            cen = self.config[i,j]
            a = self.config[(i+1)%self.N,j]
            b = self.config[i,(j+1)%self.N]
            c = self.config[(i-1)%self.N,j]
            d = self.config[i,(j-1)%self.N]
            
            config_vec = [cen, a, b, c, d]
            #Now let's get the configuration number
            config_number = self.configs_list.index(config_vec)
            config_hist[int(config_number)]=config_hist[int(config_number)]+1

    #normalize it to get probabilities
    config_hist_norm = config_hist / (self.N * self.N)

    return config_hist, config_hist_norm
    

def load_club_list(fname):
    "Loads a list of integers from fname used for histogram state compression"
    
    clubl = []
    with open(fname, 'r') as f:
        for line in f:
            line_data = line.split(',')
            split_again = line_data[-1].split('\n')
            corrected_line = line_data[:-1]
            corrected_line.append(split_again[0])
            ints = [int(entry) for entry in corrected_line]
            clubl.append(ints)
    f.close()
    
    return clubl

#Loss and compression functions

def compress_histogram(histogram, club_list):
    """Given a histogram with N configurations, compress it to M configurations
    by binning together as per indices in club_list. Reduced histogram will be of len(club_list)
    Each row of club_list contains the indices to bin together"""
    
    compressed_histogram = np.zeros(shape=(len(club_list)))
    for i in range(len(club_list)):
        probs = [histogram[club_list[i][j]] for j in range(len(club_list[i]))]
        compressed_histogram[i] = np.sum(probs)
        
    return compressed_histogram


#statistical distance squared loss
def sd2_loss(source, target):
    bc = np.sum(np.sqrt(source)*np.sqrt(target))
    sd = np.arccos(bc)
    return sd**2

#loss function
def loss_func(x, T, target_histogram, club_list):
    'parameters now are [J_mat]. Make sure to input the temperature'
    
    print('J value now is {}'.format(x))
    J_mat = np.zeros((5,5))
    J_mat[1,2] = J_mat[2,1] = J_mat[2,3] = J_mat[3,2] = x
    
    new_model = IsingSim(N=20, J_mat = J_mat, T = T, eqSteps = 500, mcSteps = 500)
    new_model.performIsingSim()
    
    new_results = new_model.results
    
    new_histogram = new_results['Histogram']
    
    compressed_histogram = compress_histogram(target_histogram, club_list)
    compressed_histogram2 = compress_histogram(new_histogram, club_list)

    loss = sd2_loss(compressed_histogram, compressed_histogram2)
    
    print('Current loss is {}'.format(loss))
    return loss
                 
                  
