import numpy as np
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import scipy as sc
from scipy.constants import hbar

'''
class site define an object in a magnetic lattice that have atributes:

# stype, a str type variable which represent the type of atom in the magnetic lattice
# neighbors, a list type variable which contains site objects that represent the atoms that
            interact magnetically with the site.
# position, a numpy array type of dimension 2, each component of the array is a 
            float type number where, position[0] have the x coordinate,
            while position[1] have the y coordinate in the lattice.
# probability_r, a float type number that have the probability amplitude of 
            magnons in the site. 
'''
class site:
    def __init__(self, stype):
        self.stype = stype
        self.neighbors = []
        self.position = np.zeros(2)
        self.probability_r = 0

'''
class eigensystem define an object that contains 2 attributes that have the 
eigenenergies and eigenvectors of a hamiltonian at a specific point in the reciprocal space
# eigenenergies, a one dimensional np.array type variable that contains the eigenenergies of a lattice
              the dimensions of the eigenenergies attributes depend on the dimension of the
              lattice.
# eigenvectors, a N dimensional np.array type variable that have the probability amplitude 
            of every site in the unit cell of the lattice.
'''

class eigensystem:
    def __init__(self):
      self.eigenenergies = None
      self.eigenvectors = None

'''
class lattice define initiate an object that have the following attributes:
    # Ny: The number of A-B atom pairs in the vertical direction.
    # Nx: The number of horizontal atoms.
    # lattice_constant = The magnitude of the unit cell dimension in the real 
                        space.
    # magnetic_constant = A tuple with the magnetic constants of the lattice, 
                        depending the Hamiltonian function, the order and 
                        number of magnetic constants can fluctuate, for the 
                        twist hamiltonian, the tuple has de form (J, S, D, a*Lambda),
                            - J is the Heisenberg exchange.
                            - S is the spin magnitude.
                            - D is the magnitude of the DM interaction.
                            - Lambda is the magnitude of the deformation.
    # bond_vectors = A numpy array that has the three bond vectors of a honeycomb
                        lattice. 
    # kpath_ribbon = A numpy array with the values of k where the dispersion is 
                    computed.
    # unit_cell_sites = A numpy array with 2*Ny site objects, representing the 
                    unit cell sites in a ribbon lattice.
    # lattice_sites = A numpy array with 2*Ny x Nx site objects, representing the 
                    sites of the lattice.
    # ribbon_Hamiltonian = A numpy array with dimension (len(kpath), 4*Ny, 4*Ny), 
                    where the i row contain the Hamiltonian matrix corresponding
                    to the i element of the kpath.
    # ribbon_eigensystem = A numpy array of length len(kpath) where the i 
                    element has a eigensystem object corresponding to the eigensystem
                    of the i element of kpath.
'''
class lattice:
    def __init__(self, height_N, lattice_constant, ref_origin='center', Nx = 1):

        self.Ny = int(height_N)
        self.Nx = int(Nx)
        self.lattice_constant = lattice_constant
        self.magnetic_constants = None
        self.bond_vectors = np.array([[np.sqrt(3)*lattice_constant/2, lattice_constant/2],
          [-np.sqrt(3)*lattice_constant/2, lattice_constant/2],
          [0, -lattice_constant]])
        self.kpath_ribbon = None
        self.unit_cell_sites = np.zeros([2*int(self.Ny)], dtype=site)
        self.lattice_sites = np.zeros([int(self.Nx), 2*int(self.Ny)], dtype=site)
        self.ribbon_Hamiltonian = None
        self.ribbon_eigensystem = None
        self.k_probabilities = None
        self.spectral = None
        self.DOS = None




    ###########################################
    # DEFINICION DE METODOS INTERNOS
    ###########################################
    
    # Metodos de utilidad
    '''
    The set_position method set the values of the self.lattice_sites[i,j].positions
    to a numpy array with two entries such that:
        - self.lattice_sites[i,j].position = np.array([x_pos, y_pos])
    where x_pos and y_pos are the components of the real space vector position 
    of the atom in the lattice position [i,j].
    This method also set the unit lattice to be self.lattice_sites[0,:].
    '''
    def set_positions(self, origin='center'):
      self.lattice_sites[0,0] = site(0) #stype 0 = sitio a; stype 1 = sitio b
      self.lattice_sites[0,1] = site(1)
      self.lattice_sites[0,0].position = np.array([0.0,0.0])
      self.lattice_sites[0,1].position = np.array([self.lattice_constant*np.sqrt(3)/2, self.lattice_constant/2])
      if origin=='center':
        self.lattice_sites[0,0].position += -np.array([(2*self.Nx-1)*np.sqrt(3)/2,(3*self.Ny/2 - 1)])*self.lattice_constant/2
        self.lattice_sites[0,1].position += -np.array([(2*self.Nx-1)*np.sqrt(3)/2,(3*self.Ny/2 - 1)])*self.lattice_constant/2
      for i in range(2,2*self.Ny):
        self.lattice_sites[0,i] = site(i%2) 
        if i%2 == 0:
          self.lattice_sites[0,i].position[1] = self.lattice_sites[0,i-1].position[1] + 1*self.lattice_constant
          self.lattice_sites[0,i].position[0] = self.lattice_sites[0,i-1].position[0]
        else:
          n = int(i/2)
          self.lattice_sites[0,i].position[1] = self.lattice_sites[0,i-1].position[1] + 1*self.lattice_constant/2
          self.lattice_sites[0,i].position[0] = self.lattice_sites[0,i-1].position[0] + ((-1)**(n)) * self.lattice_constant*np.sqrt(3)/2
      if self.Nx>1:
        for x in range(1,self.Nx):
          for y in range(2*self.Ny):
            self.lattice_sites[x,y] = site(self.lattice_sites[x-1,y].stype)
            self.lattice_sites[x,y].position = self.lattice_sites[x-1, y].position + np.array([np.sqrt(3)*self.lattice_constant, 0])
      if origin == 'edge':
          for i in range(2*self.Ny):
              self.lattice_sites[0,i].position[1] += -self.lattice_sites[0,0].position[1]
      self.unit_cell_sites=self.lattice_sites[0,:]
    
    '''
    The print_unitcell method prints in the real space the atoms in the 
    unit cell of the nanoribbon lattice.
    '''
    
    def print_unitcell(self):
      fig, ax = plt.subplots(1,1, layout='constrained') 
      for y in range(2*self.Ny):
        abs_position = self.unit_cell_sites[y].position
        if y < 2*self.Ny-1:
          abs_nposition = self.unit_cell_sites[y+1].position
          dx, dy = abs_nposition - abs_position
          plt.arrow(abs_position[0], abs_position[1], dx, dy, zorder=1.5)
        if self.unit_cell_sites[y].stype == 0: color='b'
        else: color='r'
        plt.scatter(abs_position[0], abs_position[1], c=color, zorder=2)
      ax.set_axis_off()
      plt.axis('equal')
      plt.savefig('unit_cell.svg',format='svg',dpi=300,transparent=True)
      plt.show()

    '''
    The print_lattice method prints in the real space all of the atoms of 
    the nanoribbon lattice.
    '''
    
    def print_lattice(self):
      for x in range(self.Nx):
        for y in range(2*self.Ny):
          position = self.lattice_sites[x,y].position
          for vecino in self.lattice_sites[x,y].neighbors:
            nposition = vecino.position
            dx, dy = nposition-position
            plt.arrow(position[0], position[1], dx, dy, zorder=1.5)
          if self.lattice_sites[x,y].stype == 0: color='b'
          else: color='r'
          plt.scatter(position[0], position[1], c=color, zorder=2)
      ax = plt.gca()
      ax.axis('equal')
      plt.show()
    
    '''
    The set_neighbors method asign to every site object self.lattice_sites[i,j]
    in the lattice_sites attribute of the lattice object, a unedimensional numpy
    array which components are the site objects self.lattice_sites[i',j'] such that:
        - (self.lattice_sites[i',j'].position-self.lattice_sites[i',j']) is a 
        lattice bond vector.
    '''
    
    def set_neighbors(self):
      for x in range(self.Nx-1):
        for y in range(2*self.Ny-1):
          self.lattice_sites[x,y].neighbors.append(self.lattice_sites[x,y+1])
          self.lattice_sites[x,y+1].neighbors.append(self.lattice_sites[x,y])
          pos1 = self.lattice_sites[x,y].position
          pos2 = self.lattice_sites[x+1,y+1].position
          sep1 = pos2-pos1
          if np.isclose(np.linalg.norm(sep1),self.lattice_constant):
            self.lattice_sites[x,y].neighbors.append(self.lattice_sites[x+1,y+1])
            self.lattice_sites[x+1,y+1].neighbors.append(self.lattice_sites[x,y])  
          if y!=0:
            pos3 = self.lattice_sites[x+1,y-1].position
            sep2 = (pos3-pos1)
            if np.isclose(np.linalg.norm(sep2),self.lattice_constant):
              self.lattice_sites[x,y].neighbors.append(self.lattice_sites[x+1,y-1])
              self.lattice_sites[x+1,y-1].neighbors.append(self.lattice_sites[x,y])
      for y in range(2*self.Ny-1):
        self.lattice_sites[self.Nx-1,y].neighbors.append(self.lattice_sites[self.Nx-1,y+1])
        self.lattice_sites[self.Nx-1,y+1].neighbors.append(self.lattice_sites[self.Nx-1,y])
        
        
    '''
    The PU(Ny) method returns a paraunitary diagonal matrix of 4Ny x 4Ny dimension
    where the 2*Ny first diagonal elements have a value of 1, while the 2*Ny 
    last elements have value -1.
    ''' 
    
    def PU(self, Ny, order = 'aba+b+'):
      if order =='aba+b+':
        result = np.diag(np.concatenate([np.ones(Ny), np.ones(Ny), -np.ones(Ny), -np.ones(Ny)]))
      elif order == 'ab+a+b':
        result = np.diag(np.concatenate([np.ones(Ny), np.ones(Ny), np.ones(Ny), np.ones(Ny)]))
        for i in range(len(result[0])):
          result[i,i] =(-1)**(i)*result[i,i]
      return result

    '''
    The HermitianConjugate(X) method, returns the conjugate transpose of the 
    X matrix.
    '''

    def HermitianConjugate(self, X):
      return np.conjugate(np.transpose(X))

    '''
    The paraunitary_eig(X) method returns a tuple where the first element is 
    a 4*Ny numpy array that contains the eigenvalues of the matrix X, and the 
    second element is a 4Ny x 4Ny matrix that contains the paraunitary matrix that
    diagonalize X.
    '''

    def paraunitary_eig(self, X):
      N = X.shape[0]//2
      evals, evecs = np.linalg.eig(X)
      i_arr = np.argsort(-np.real(evals))
      j_arr = np.concatenate([i_arr[:N][::-1],i_arr[N:]])
      return evals[j_arr], evecs[:,j_arr]

    '''
    The set_ribbon_eigensystem method set the eigensystem of the ribbon lattice
    along the k_path. The ribbon eigensystem has the same size of the kpath,
    every self.ribbon_eigensystem[i] component has an eigensystem object with 
    two attributes: 
        - eigenenergies: A 2*Ny numpy array that has the ordered eigenenergies 
        of the self.Hamiltoninan[i] array.
        - eigenvectors: A [2Ny x 2Ny] numpy array containing the eigenvectors 
        of the self.Hamiltonian[i] array. The eigenvectors are ordered such 
        that self.ribbon_eigensystem[i].eigenvector[:,j] contain an 2Ny array 
        corresponding to the eigenvector asociated with the self.ribbon_eigensystem[i].eigenenergies[j]
        component.
    '''

    def set_ribbon_eigensystem(self, method='colpa'):
      self.ribbon_eigensystem = np.zeros([len(self.kpath_ribbon)], dtype=eigensystem)
      for i, k in enumerate(self.kpath_ribbon):
        if method=='colpa':
          eigen, eigvec = self.colpa_k(self.ribbon_Hamiltonian[i])
          self.ribbon_eigensystem[i] = eigensystem()
          self.ribbon_eigensystem[i].eigenenergies = eigen
          self.ribbon_eigensystem[i].eigenvectors = eigvec
        else:
          eigen = self.bogoliubov_k(self.ribbon_Hamiltonian[i])
          self.ribbon_eigensystem[i] = eigensystem()
          self.ribbon_eigensystem[i].eigenenergies = eigen
          
    '''
    The plot_graf_prob(k_index, band_index) plots the eigenstate of the 
    self.kpath_ribbon[k_index] component, and asociated to the eigenenergies[band_index]
    component. The x axis of the plot represents the real y cartesian dimension,
    while the y axis of the plot represents the occupation number of magnons.
    '''


    def plot_graf_prob(self, k_index, band_index):
      prob = np.zeros(2*self.Ny)
      U = self.ribbon_eigensystem[k_index].eigenvectors[:2*self.Ny,:2*self.Ny]
      V = self.ribbon_eigensystem[k_index].eigenvectors[2*self.Ny:4*self.Ny,:2*self.Ny]
      prob = U[:,band_index]*np.conj(U[:,band_index])+V[:,band_index]*np.conj(V[:,band_index])
      plt.plot(prob)
      ax = plt.gca()
      ax.set_xlabel('Indice de posicion Y')
      ax.set_ylabel(r'$|\Psi_{k}^{n}|^2$')
      #plt.text(1,1, 'J ='+str(self.magnetic_constants[0]))
      #plt.text(1,0.9, 'D ='+str(self.magnetic_constants[2]))
      #plt.text(1,0.8, '$\lambda$ ='+str(self.magnetic_constants[3]))
      #plt.text(1,0.7, 'n ='+str(band_index)+', $k/a\sqrt{3}$='+str(round(self.kpath_ribbon[k_index],2)))
      plt.show()
      
    '''
    The plot_ribbon_dispersion() method plots the band structure of the 
    spin waves contained in the magnetic lattice. The x axis represent the kpath
    that was followed to set the self.Hamiltonian and self.eigensystem atrributes
    of the lattice. The y axis represents the normalized energy E/(|J|S).
    It has two optional parameters: 
        - N_dest: Plots with red color instead of black only the N_dest band, where
        N_dest = 0 represent the band of lower energy.
        - ylim: sets the yaxis limits of the plot.
    '''
    
    def plot_ribbon_dispersion(self, N_dest=None, yindex=0, ylim=[0,0], linewidth=2.0):
        klength = len(self.kpath_ribbon)
        dispersion = np.zeros([klength, 4*self.Ny], dtype=complex)
        if type(N_dest)=='int':
            N_dest = [N_dest]
        for k in range(len(self.kpath_ribbon)):
          dispersion[k] = self.ribbon_eigensystem[k].eigenenergies
        for n in range(2*self.Ny):
          if N_dest != None and (n in N_dest): 
            col='r'
            order = 2.5
            width = linewidth
          else: 
            col='k'
            order = 2
            width = 1.0
          plt.plot(self.kpath_ribbon*np.sqrt(3)*self.lattice_constant, dispersion[:,n], color=col, zorder = order, linewidth=width)
        y0 = np.min(dispersion[yindex])
        y1 = np.max(dispersion[yindex])
        if ylim!=[0,0]:
            y0 = ylim[0]
            y1 = ylim[1]
        plt.ylim(y0, y1)
        plt.xlim(self.kpath_ribbon[0], self.kpath_ribbon[-1])
        ax = plt.gca()
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels(['$0$', '$\pi$', '$2\pi$'])
        ax.set_xlabel(r'$k/a\sqrt{3}$')
        ax.set_ylabel(r'$E/|J|S$')
        plt.show()

    '''
    The set_ribbon_hamiltonian(k_hamiltonian) method uses the function k_hamiltonian
    to set the self.ribbon_hamiltonian attribute to be a numpy klenght x 4Ny x 4Ny
    array and then set the values for every self.ribbon_hamiltonian[i] elements
    representing the k_hamiltonian function evaluated in kpath[i].
    '''
    
    def set_ribbon_hamiltonian(self, k_hamiltonian, defParams):
      klength = len(self.kpath_ribbon)
      self.ribbon_Hamiltonian = np.zeros([klength, 4*self.Ny, 4*self.Ny], dtype=complex)
      for i, k in enumerate(self.kpath_ribbon):
        self.ribbon_Hamiltonian[i] = k_hamiltonian(k, defParams)
      print('Hamiltoniano definido en todo el camino k')
      
    def spectral_function(self, Hamiltonian, omega, delta):
        Ny = self.Ny
        J, S, DMI, Kitaev, GAMMA, h = self.magnetic_constants
        Hsize = len(Hamiltonian[:,0])
        argument_matrix = omega*np.eye(Hsize) + 1j*delta*np.eye(Hsize) - np.dot(self.PU(Ny),Hamiltonian)
        green_matrix = np.linalg.inv(argument_matrix)
        imaginary_matrix = np.imag(green_matrix)
        result = -(imaginary_matrix)/np.pi
        
        return result
    
    def set_spectral_function(self, omega_lim, omega_size, delta):
        k_path = self.kpath_ribbon
        omega_list = np.linspace(omega_lim[0], omega_lim[1], omega_size)
        self.spectral = np.zeros((len(k_path), len(omega_list)))
        for i, k in enumerate(k_path):
            Hamiltonian = self.ribbon_Hamiltonian[i]
            for j, omega in enumerate(omega_list):
                self.spectral[i,j] = np.trace(self.spectral_function(Hamiltonian, omega, delta))
                
    def plot_spectral(self, omegalim):
        spectral_size = np.shape(self.spectral)
        J, S, DMI, Kitaev, GAMMA, h = self.magnetic_constants
        xticks = np.linspace(omegalim[0], omegalim[1], 5)
        for i, tick in enumerate(xticks):
            xticks[i] = str(tick)
        plt.imshow(np.transpose(self.spectral), origin='lower', aspect='auto', cmap='hot')
        plt.xlabel(r'$k_x \delta_x$')
        plt.ylabel(r'$\omega / |J|S$')
        ax = plt.gca();
        ax.set_xticks([0, spectral_size[0]/2, spectral_size[0]-1], labels=[r'$0$', r'$\pi$', r'$2\pi$' ]);
        ax.set_yticks([0, spectral_size[1]/4, 2*spectral_size[1]/4, 3*spectral_size[1]/4,
                       spectral_size[1]-1], labels = xticks);
    
    def kDerivative(self, n):
        k_path = self.kpath_ribbon
        klength = len(k_path)
        dispersion = np.zeros([klength, 4*self.Ny], dtype=complex)
        for k in range(len(self.kpath_ribbon)):
          dispersion[k] = self.ribbon_eigensystem[k].eigenenergies
        derivative = np.gradient(dispersion[:,n])
        return derivative
    
    def plot_kderivative(self, n):
        k_path = self.kpath_ribbon
        fig, ax = plt.subplots()
        ax.plot(k_path, self.kDerivative(n))
        plt.xlabel(r'$k_x \delta_x$')
        plt.ylabel(r'$c_g$')
        
    def set_DOS(self, omegalim):
        k_path = self.kpath_ribbon
        klength = len(k_path)
        spectral = self.spectral
        omega = np.linspace(omegalim[0], omegalim[1], len(spectral[0]))
        self.DOS = np.zeros(len(omega))
        for i, E in enumerate(omega):
            self.DOS[i] = np.trapz(spectral[:,i], x = k_path)
    
    def plot_DOS(self, omegalim):
        spectral = self.spectral
        DOS = self.DOS
        omega = np.linspace(omegalim[0], omegalim[1], len(spectral[0]))
        plt.plot(DOS, omega)
        plt.ylabel(r'$\omega / |J|S$')
        

    
            
        
            

    #########################################
    # Definicion de Hamiltonianos
    #########################################
    

    '''
    The TwistHamiltonian(kx) method is a function where the input, kx is a float
    number between 0 and 2pi representing points in the kpath. This function returns
    a 4Ny x 4Ny numpy array which represents the Hamiltonian matrix for a magnetic lattice
    that has a twist deformation which amplitude is characterized by the parameter aLambda.
    '''

    def TwistHamiltonian(self, kx, defParams):
      """
      Twist strain hamiltonian, the origin of the reference system is in the center
      Ny: numero de celdas unitarias
      """
      
      if len(self.magnetic_constants) != 6:
        print ('Cantidad incorrecta de constantes magneticas, J, S, D, Kitaev, Gamma, y h esperados, recibido: ',
        self.magnetic_constants)
      else:
        J, S, DMI, Kitaev, GAMMA, h = self.magnetic_constants
      aLambdaJ, aLambdaK = defParams
      H_kx  = np.zeros([2*self.Ny, 2*self.Ny], dtype=complex)
      H_anomalo = np.zeros([2*self.Ny, 2*self.Ny], dtype=complex)
      a = self.lattice_constant
      bond1, bond2, bond3 = self.bond_vectors
      LambdaJ = aLambdaJ/a
      LambdaK = aLambdaK/a
      A=-h
      for m in range(2*self.Ny):
          for n in range(2*self.Ny):
            y_pos = self.unit_cell_sites[m].position[1]
            D = DMI
            Gamma = GAMMA
            if self.unit_cell_sites[m].stype == 1:
              D = -DMI
              Gamma = GAMMA
              y_pos = self.unit_cell_sites[m-1].position[1]
            J_1 = J*np.exp(1-np.sqrt(1+3/4*(LambdaJ**2) * (y_pos**2 + a*(y_pos)/2)))
            J_3 = J
            K0 = Kitaev[0]*np.exp(1-np.sqrt(1+3/4*(LambdaK**2) * (y_pos**2 + a*(y_pos)/2)))
            K1 = Kitaev[1]*np.exp(1-np.sqrt(1+3/4*(LambdaK**2) * (y_pos**2 + a*(y_pos)/2)))
            K3 = Kitaev[2]
            if m==n:
              y_pos = self.unit_cell_sites[m].position[1]
              pos = bond1[0]
              if (m%(2*self.Ny)==0) or (m%(2*self.Ny) == 2*self.Ny-1):
                  H_kx[m,n] = -S*((A+3*J_3+2*D*np.sin(2*kx*pos)) + 2*K3)
              else:
                  H_kx[m,n] = -S*((A+3*J_3+2*D*np.sin(2*kx*pos)) + 2*K3)
            if m-n == 1:
                pos = bond1[0]
                kvec = [kx,0]
                f2k = S*(K0*np.exp(-1j*np.dot(kvec,bond1)) + 
                            K1*np.exp(-1j*np.dot(kvec,bond2)))
                if m%2 == 0:
                    H_kx[m,n] = 1*J_3*S
                    H_kx[n,m] = np.conj(H_kx[m,n])
                    H_anomalo[m,n] = 1j*S*Gamma
                    H_anomalo[n,m] = -np.conj(H_anomalo[m,n])
                else:
                    H_kx[m,n] = 2*S*J_1*(np.cos(kx*pos)) + f2k
                    H_kx[n,m] = np.conj(H_kx[m,n]);
                    H_anomalo[m,n] = S*(K0*np.exp(-1j*np.dot(kvec,bond1)) - 
                                   K1*np.exp(-1j*np.dot(kvec,bond2)))
                    H_anomalo[n,m] = np.conj(H_anomalo[m,n])
      Hkx = np.block([[H_kx, H_anomalo],[self.HermitianConjugate(H_anomalo), np.conj(H_kx)]])
      return Hkx


    '''
    The UniaxialHamiltonian(kx) method is a function where the input, kx is a float
    number between 0 and 2pi representing points in the kpath. This function returns
    a 4Ny x 4Ny numpy array which represents the Hamiltonian matrix for a magnetic lattice
    that has a twist deformation which amplitude is characterized by the parameter aLambda.
    '''

    def UniaxialHamiltonian(self, kx, defParam):
      """
      Twist strain hamiltonian, the origin of the reference system is in the center
      Ny: numero de celdas unitarias
      """
      
      if len(self.magnetic_constants) != 6:
        print ('Cantidad incorrecta de constantes magneticas, J, S, D, Kitaev, Gamma, h, aLambdaJ y aLambdaK esperados, recibido: ',
        self.magnetic_constants)
      else:
        J, S, DMI, Kitaev, GAMMA, h = self.magnetic_constants
      Y_size = self.unit_cell_sites[-1].position[1] - self.unit_cell_sites[0].position[1]
      defParamJ, defParamK = defParam
      c_J = defParamJ/Y_size
      c_K = defParamK/Y_size
      H_kx  = np.zeros([2*self.Ny, 2*self.Ny], dtype=complex)
      H_anomalo = np.zeros([2*self.Ny, 2*self.Ny], dtype=complex)
      a = self.lattice_constant
      bond1, bond2, bond3 = self.bond_vectors
      A=-h
      for m in range(2*self.Ny):
          for n in range(2*self.Ny):
            y_pos = self.unit_cell_sites[m].position[1]
            D = DMI
            if self.unit_cell_sites[m].stype == 1:
              D = -DMI
            Gamma = GAMMA*(1 - c_K*y_pos)
            J_1 = J*(1 - c_J*y_pos/4)
            J_3 = J*(1 - c_J*y_pos)
            K0 = Kitaev[0]*(1 - c_K*y_pos/4)
            K1 = Kitaev[1]*(1 - c_K*y_pos/4)
            K3 = Kitaev[2]*(1 - c_K*y_pos)

            if m==n:
              y_pos = self.unit_cell_sites[m].position[1]
              pos = bond1[0]
              if (m%(2*self.Ny)==0) or (m%(2*self.Ny) == 2*self.Ny-1):
                  H_kx[m,n] = -S*((A+3*J+2*D*np.sin(2*kx*pos)) + 2*Kitaev[2])
              else:
                  H_kx[m,n] = -S*((A+3*J+2*D*np.sin(2*kx*pos)) + 2*Kitaev[2])
            if m-n == 1:
                pos = bond1[0]
                kvec = [kx,0]
                f2k = S*(K0*np.exp(-1j*np.dot(kvec,bond1)) + 
                            K1*np.exp(-1j*np.dot(kvec,bond2)))
                if m%2 == 0:
                    H_kx[m,n] = 1*J_3*S
                    H_kx[n,m] = np.conj(H_kx[m,n])
                    H_anomalo[m,n] = 1j*S*Gamma
                    H_anomalo[n,m] = -np.conj(H_anomalo[m,n])
                else:
                    H_kx[m,n] = 2*S*J_1*(np.cos(kx*pos)) + f2k
                    H_kx[n,m] = np.conj(H_kx[m,n]);
                    H_anomalo[m,n] = S*(K0*np.exp(-1j*np.dot(kvec,bond1)) - 
                                   K1*np.exp(-1j*np.dot(kvec,bond2)))
                    H_anomalo[n,m] = np.conj(H_anomalo[m,n])
      Hkx = np.block([[H_kx, H_anomalo],[self.HermitianConjugate(H_anomalo), np.conj(H_kx)]])
      return Hkx

    #########################################
    # Metodos de diagonalizacion 
    ########################################
    
    '''
    The bogoliubov_k(hamiltonian) method diagonilize the hamiltonian input array using
    the method proposed by bogoliubov. It returns a onedimensional numpy array 
    with the eigenvalues of the magnon hamiltonian array.
    '''
    def bogoliubov_k(self, hamiltonian):
      Ny = self.Ny
      output = np.zeros(4*Ny)
      ParaU = self.PU(Ny)
      H = hamiltonian
      PUH = np.dot(ParaU, H)
      output = (np.sort(np.linalg.eig(PUH)[0]))
      return output

    '''
    The colpa_k(hamiltonian) method take as input a numpy square array that represent
    the hamiltonian matrix of the magnetic lattice for a value in the k path. 
    It returns a tuple (E, T), where E is a onedimensional numpy array containing the 
    ordered eigenenergies of the hamiltonian input, T is a 4Ny x 4Ny numpy array
    that diagonalize the hamiltonian input.
    '''
    
    def colpa_k(self, hamiltonian):
      N = 2*self.Ny
      ParaU = self.PU(int(N/2))
      try:
        H = self.HermitianConjugate( np.linalg.cholesky(hamiltonian) )
        L, U = self.paraunitary_eig( np.dot(np.dot(H, ParaU), self.HermitianConjugate(H)) )

        E = np.dot(ParaU, np.diag(L))
        T = np.dot(np.linalg.inv(H),np.dot(U, np.sqrt(E)))

        return np.diag(E), T
      except np.linalg.LinAlgError: #Hamiltnoniano no es definido positivo
        return -np.ones(2*N, dtype=complex), np.zeros((2*N, 2*N), dtype=complex)
    
class dislon_lattice:
    def __init__(self, mgph_param):
        self.a0=mgph_param[0]       #Lattice constant
        self.phir=mgph_param[1]     #Radial elastic force
        self.phiti=mgph_param[2]    #Transversal in plane elastic force
        self.phito=mgph_param[3]    #Transversal outplane elastic force
        self.J=mgph_param[4]
        self.S=mgph_param[5]
        self.Sz=mgph_param[6]
        self.density = None         #Mass density
        self.N = None               #Number of sites
        self.Bper=mgph_param[7]     #Perpendicular magnetoelastic constant
        self.Bpar = mgph_param[8]   #Parallel magnetoelastic constant
        self.PBC=mgph_param[9]      #Periodic Boundary Conditions
        self.h = mgph_param[10]
        self.theta = None           #Angle deviation of the ground state from the z axis
        #Dislocation info
        self.burgers = None
        self.glideN = None
        #Elastic properties
        self.poisson = None
        self.shear = None
        self.lame = None
        #Real and reciprocal basis vector
        self.A1 = self.a0*np.array([1,0,0])
        self.A2 = self.a0*np.array([0,1,0])
        self.B1 = ((2*np.pi)/(self.a0))*np.array([1, 0, 0])
        self.B2 = ((2*np.pi)/(self.a0))*np.array([0,1, 0])
        #Neighbors positions
        self.RA1 = np.array([0,0])
        self.RB1 = self.A1
        self.RB2 = self.A2
        self.RB3 = -self.A1
        self.RB4 = -self.A2
        #Hamilronian along k-path
        self.Hamiltonian = None
        #Eigensystem along k-path
        self.Eigensystem = None
        self.kpath = None
        self.phononEigensystem = None
        #k-path maker function
    def lerp(self,v0, v1, i):
        return v0 + i * (v1 - v0)
    #k-path maker function
    def getEquidistantPoints(self,p1, p2, n):
        #creating a k-path
        return [(self.lerp(p1[0],p2[0],1./n*i), self.lerp(p1[1],p2[1],1./n*i), 0) for i in range(n+1)]
    def phi_phonon(self, k):
        k1=k[0]
        k2=k[1]
        k3=k[2]
        phi = np.zeros([3,3], dtype=complex)
        modk=np.linalg.norm(k)
        matrix =np.array([[self.lame*k1**2+2*self.shear*modk**2, self.lame*k1*k2, self.lame*k1*k3],
                 [self.lame*k2*k1, self.lame*k2**2+2*self.shear*modk**2, self.lame*k2*k3],
                 [self.lame*k3*k1, self.lame*k3*k2, self.lame*k3**2+2*self.shear*modk**2]],dtype=complex)/self.density 
        phi = matrix*np.exp(1j*np.dot(k,self.A1))+matrix*np.exp(1j*np.dot(k,self.A2))+matrix*np.exp(1j*np.dot(k,-self.A1))+matrix*np.exp(1j*np.dot(k,-self.A2))
        return phi
    def KAB(self,alpha):
        #Spring Matrices with a rotation
        a=self.phir*np.cos(alpha)**2+self.phiti*np.sin(alpha)**2                                   
        b=(self.phir-self.phiti)*np.cos(alpha)*np.sin(alpha)
        c=(self.phir-self.phiti)*np.cos(alpha)*np.sin(alpha)
        d=self.phir*np.sin(alpha)**2+self.phiti*np.cos(alpha)**2
        matrix=np.array([[a,b,0],[c,d,0],[0,0,self.phito]])
        return matrix
    def Onsite_Phonon(self,k):
        term=self.KAB(self.theta1)+self.KAB(self.theta2)+self.KAB(self.theta3)+self.KAB(self.theta4)  
        return term
    def Interaction_Phonon_x_bulk(self,k):
        k=np.array(k)
        phase1=-1j*np.dot(k,self.RB1)
        phase3=-1j*np.dot(k,self.RB3)
        term=self.KAB(self.theta1)*np.exp(phase1)+self.KAB(self.theta3)*np.exp(phase3)
        return term 
    def Interaction_Phonon_y_bulk(self,k):
        k=np.array(k)
        phase2=-1j*np.dot(k,self.RB2)
        phase4=-1j*np.dot(k,self.RB4)
        term=self.KAB(self.theta2)*np.exp(phase2)+self.KAB(self.theta4)*np.exp(phase4)
        return term
   
    def Ham_Phonon(self,k):
    
        ham=self.Onsite_Phonon(k)-self.Interaction_Phonon_x_bulk(k)-self.Interaction_Phonon_y_bulk(k)
        
        return ham
    def Ham_Magnon(self,k):
        
        #https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/3/67057/files/2020/02/spin-wave_theory_using_the_Holstein-Primakoff_transformation.pdf
        ham=self.J*self.S*(-4+np.cos(np.dot(k,self.A1))+np.cos(np.dot(k,self.A2)))
        return np.matrix([[ham,0],[0,ham]])
        
    def Gamma_mgph(self,Bper,k,phonon_mode,frequency):
        #Magnon phonon interaction
        k=np.array([k[0],k[1],0])
        interaction=1j*phonon_mode[0]*k[2]+k[2]*phonon_mode[1]+(1j*k[0]+k[1])*phonon_mode[2]
        return Bper*interaction/np.sqrt(self.S*2*frequency+0.0001)
    
    def F(self, k):
        if k[0]==0:
            q = 0.0001
        if k[0]==0 and k[1]==0:
            return np.array([0,0,0])
        n = self.glideN
        b = self.burgers
        modk = np.linalg.norm(k)
        q=k[0]
        F = (n*np.dot(b, k) + b*np.dot(n, k)-(k*np.dot(n, k)*np.dot(b, k))/(modk**2 *(1-self.poisson)))/(q*modk**2)
        return F
    
    def m_dis(self, k):
        F = self.F(k)
        modF = np.linalg.norm(F)
        syst_size = self.a0*self.N
        mk = self.density*modF**2/syst_size
        return mk
    
    def Omega_dis(self, k):

        if k[0]==0 and k[1]==0:
            return 0
        F = self.F(k)
        modk = np.linalg.norm(k)
        lame = self.lame
        shear = self.shear 
        dens = self.density
        modF = np.linalg.norm(F)
        omega = np.sqrt(((lame+shear)*np.dot(k, F)**2 + shear*modk**2 *modF**2)/(dens*modF+0.001))
        
        return omega 
    
    def Z_dis(self, k):

        zk = np.sqrt(1/(2*self.m_dis(k)*self.Omega_dis(k)))
        return zk
    
    def Gdis(self, k, alpha, beta):
        #alpha and beta are polarizations, alpha/beta = 0 is x, alpha/beta = 1 is y
        #alpha/beta = 2 is z
        G = self.F(k)[alpha]*np.sin(self.a0*k[beta]) + self.F(k)[beta]*np.sin(self.a0*k[alpha])
        return G
        
    def Gamma1_mgdis(self, k, theta):
        if k[0]==0 and k[1]==0:
            return 0
        gamma = self.Z_dis(k)*self.Gdis(k, 0, 1)*2*self.Bper*np.sin(theta) + self.Z_dis(k)*self.Gdis(k, 2, 2)*2*self.Bpar*np.cos(theta)*np.sin(theta)
        return 1j*gamma/(self.a0**4*np.sqrt(2*self.S))
    
    def Gamma2_mgdis(self, k, theta):
        if k[0]==0 and k[1]==0:
            return 0
        gamma = self.Z_dis(k)*self.Gdis(k, 0, 1)*2*self.Bper*np.sin(theta) - self.Z_dis(k)*self.Gdis(k, 2, 2)*2*self.Bpar*np.cos(theta)*np.sin(theta)
        
        return 1j*gamma/(self.a0**4*np.sqrt(2*self.S))
    
    
    def Ham_Magnon_phonon_dislon(self, k):
        #Magnon Hamiltonian
        Mag_E_plus = self.J*self.S*(-4+np.cos(np.dot(k,self.A1))+np.cos(np.dot(k,self.A2))) + self.h
        Mag_E_minus = self.J*self.S*(-4+np.cos(np.dot(-k,self.A1))+np.cos(np.dot(-k,self.A2))) + self.h
        #Phonon Hamiltonian in k
        Ham=self.Ham_Phonon(k)
        (evals,evec)=self.evals_evec(Ham,"ph")
        aux_eval=evals
        aux_evec=evec
        #Phonon Hamiltoinian in -k
        Hamneg=self.Ham_Phonon(-k)
        (evals,evec)=self.evals_evec(Hamneg,"ph")
        aux_eval_neg=evals
        aux_evec_neg=evec
        #mgph interactions
        G1=self.Gamma_mgph(self.Bper,k,aux_evec[0],aux_eval[0])
        G1minus=self.Gamma_mgph(self.Bper,k,aux_evec_neg[0],aux_eval_neg[0]).conjugate()
        
        G2=self.Gamma_mgph(self.Bper,k,aux_evec[1],aux_eval[1])
        G2minus=self.Gamma_mgph(self.Bper,k,aux_evec_neg[1],aux_eval_neg[1]).conjugate()
        
        #magnon dislon interactions
        gamma1Plus = self.Gamma1_mgdis(k, self.theta)
        gamma1Minus = gamma1Plus.conjugate()
        gamma2Plus = self.Gamma2_mgdis(k, self.theta)
        gamma2Minus = gamma2Plus.conjugate()
        #dislon matrix ham element
        omega = self.Omega_dis(k)
        
        Ham_mag_ph_dis = np.array([[Mag_E_plus, 0, G1.conjugate(), G2.conjugate(), 0, 0, 0, 0, G1.conjugate(), G2.conjugate(), gamma2Minus, gamma2Minus],
                                  [0 , Mag_E_minus, 0, 0, gamma2Plus, gamma2Plus, 0, 0, 0, 0, 0, 0],
                                  [G1, 0, aux_eval[0], 0, 0, 0, 0, G1minus.conjugate(), 0, 0, 0, 0],
                                  [G2, 0, 0, aux_eval[1], 0, 0, 0, G2minus.conjugate(), 0, 0, 0, 0],
                                  [0, gamma1Minus, 0, 0, omega, 0, gamma2Minus, 0, 0, 0, 0, 0],
                                  [0, gamma1Minus, 0, 0, 0, omega, gamma2Minus, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, gamma1Plus, gamma1Plus, Mag_E_plus, 0, 0, 0, 0, 0],
                                  [0, 0, G1minus, G2minus, 0, 0, 0, Mag_E_minus, G1minus, G2minus, gamma1Minus, gamma1Minus],
                                  [G1, 0, 0, 0, 0, 0, 0, G1minus.conjugate(), aux_eval_neg[0], 0, 0, 0],
                                  [G2, 0, 0, 0, 0, 0, 0, G2minus.conjugate(), 0, aux_eval_neg[1], 0, 0],
                                  [gamma1Plus, 0, 0, 0, 0, 0, 0, gamma2Plus, 0, 0, omega, 0],
                                  [gamma1Plus, 0, 0, 0, 0, 0, 0, gamma2Plus, 0, 0, 0, omega]])
        return Ham_mag_ph_dis
    #########################################
    # Metodos de diagonalizacion 
    ########################################
    '''
    The PU(Ny) method returns a paraunitary diagonal matrix of 4Ny x 4Ny dimension
    where the 2*Ny first diagonal elements have a value of 1, while the 2*Ny 
    last elements have value -1.
    ''' 
    
    def PU(self, Ny, order = 'aba+b+'):
      if order =='aba+b+':
        result = np.diag(np.concatenate([np.ones(Ny), np.ones(Ny), -np.ones(Ny), -np.ones(Ny)]))
      elif order == 'ab+a+b':
        result = np.diag(np.concatenate([np.ones(Ny), np.ones(Ny), np.ones(Ny), np.ones(Ny)]))
        for i in range(len(result[0])):
          result[i,i] =(-1)**(i)*result[i,i]
      return result
    
    '''
    The bogoliubov_k(hamiltonian) method diagonilize the hamiltonian input array using
    the method proposed by bogoliubov. It returns a onedimensional numpy array 
    with the eigenvalues of the magnon hamiltonian array.
    '''
    def bogoliubov_k(self, hamiltonian):
      Ny = self.Ny
      output = np.zeros(4*Ny)
      ParaU = self.PU(Ny)
      H = hamiltonian
      PUH = np.dot(ParaU, H)
      output = (np.sort(np.linalg.eig(PUH)[0]))
      return output

    '''
    The colpa_k(hamiltonian) method take as input a numpy square array that represent
    the hamiltonian matrix of the magnetic lattice for a value in the k path. 
    It returns a tuple (E, T), where E is a onedimensional numpy array containing the 
    ordered eigenenergies of the hamiltonian input, T is a 4Ny x 4Ny numpy array
    that diagonalize the hamiltonian input.
    '''
    
    def colpa_k(self, hamiltonian):
      N = 2*self.Ny
      ParaU = self.PU(int(N/2))
      try:
        H = self.HermitianConjugate( np.linalg.cholesky(hamiltonian) )
        L, U = self.paraunitary_eig( np.dot(np.dot(H, ParaU), self.HermitianConjugate(H)) )

        E = np.dot(ParaU, np.diag(L))
        T = np.dot(np.linalg.inv(H),np.dot(U, np.sqrt(E)))

        return np.diag(E), T
      except np.linalg.LinAlgError: #Hamiltnoniano no es definido positivo
        return -np.ones(2*N, dtype=complex), np.zeros((2*N, 2*N), dtype=complex)
    
    def set_kpath(self,N_k):
        self.M=self.B1*0.5
        self.X=self.B1*0.5+self.B2*0.5
        self.G=np.array([0,0,0])
        path1=self.getEquidistantPoints(self.G,self.X, N_k)
        path2=self.getEquidistantPoints(self.X,self.M, N_k)
        path3=self.getEquidistantPoints(self.M,self.G, N_k)
        path2.remove(path2[0])
        path3.remove(path3[0])
        self.kpath=np.concatenate((path1,path2,path3),axis=0)
        
    def set_hamiltonian(self):
        klength = len(self.kpath)
        self.Hamiltonian = np.zeros([klength, 12, 12], dtype=complex)
        
        for i,k in enumerate(self.kpath):
            self.Hamiltonian[i] = self.Ham_Magnon_phonon_dislon(k)
            
    def set_eigensystem(self, method='colpa'):
        klength = len(self.kpath)
        self.eigensystem = np.zeros([klength], dtype=eigensystem)
        for i,k in enumerate(self.kpath):
            eigen, eigvec = self.colpa_k(self.ribbon_Hamiltonian[i])
            self.ribbon_eigensystem[i] = eigensystem()
            self.ribbon_eigensystem[i].eigenenergies = eigen
            self.ribbon_eigensystem[i].eigenvectors = eigvec
    def set_phonon_eigensystem(self, method='colpa'):
        klength = len(self.kpath)
        self.phonon_eigensystem = np.zeros(klength, dtype=eigensystem)
        for i,k in enumerate(self.kpath):
            eigen, eigvec = np.linalg.eig(self.phi_phonon(k))
            self.phonon_eigensystem[i] = eigensystem()
            self.phonon_eigensystem[i].eigenenergies = eigen
            self.phonon_eigensystem[i].eigenvectors = eigvec
    def plot_phonon_dispersion(self, N_dest=None, yindex=0, ylim=[0,0], linewidth=2.0):
        klength = len(self.kpath)
        dispersion = np.zeros([klength,3], dtype=complex)
        if type(N_dest)=='int':
            N_dest = [N_dest]
        for k in range(len(self.kpath)):
          dispersion[k] = self.phonon_eigensystem[k].eigenenergies
        for i in range(3):
            plt.plot(dispersion[:,i])
        plt.show()
    
    def mag_ph_dis_system_band(self, N_k):
        
        self.M=self.B1*0.5
        self.X=self.B1*0.5+self.B2*0.5
        self.G=np.array([0,0,0])
        path1=self.getEquidistantPoints(self.G,self.X, N_k)
        path2=self.getEquidistantPoints(self.X,self.M, N_k)
        path3=self.getEquidistantPoints(self.M,self.G, N_k)
        path2.remove(path2[0])
        path3.remove(path3[0])
        path=np.concatenate((path1,path2,path3),axis=0)
  
        x=[]
        E=[]
        x0=0
        for k in path:
            Ham=self.Ham_Magnon_phonon_dislon(k)
            [evals,evec]=self.evals_evec(Ham,"mg")#+self.Ham_auxiliar*np.identity(4),"ph") 
            index=np.argsort(evals.real)
            E.append(evals.real[index])
            x.append(x0)
            x0+=1
        E=np.array(E)
        for i in range(len(E.T)):
            plt.plot(x,E.T[i])
        plt.grid()
        plt.show()