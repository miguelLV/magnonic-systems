import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

class Magnon_Phonon:
    def __init__(self,mgph_param):
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
        self.B2 = ((2*np.pi)/(self.a0))*np.array([0,-1, 0])
        #Neighbors positions
        self.RA1 = np.array([0,0])
        self.RB1 = self.A1
        self.RB2 = self.A2
        self.RB3 = -self.A1
        self.RB4 = -self.A2
        #angle neighbors
        self.theta1=0
        self.theta2=np.pi/2.
        self.theta3=np.pi
        self.theta4=3*np.pi/2.
    #k-path maker function
    def lerp(self,v0, v1, i):
        return v0 + i * (v1 - v0)
    #k-path maker function
    def getEquidistantPoints(self,p1, p2, n):
        #creating a k-path
        return [(self.lerp(p1[0],p2[0],1./n*i), self.lerp(p1[1],p2[1],1./n*i), 0) for i in range(n+1)]
    def KAB(self,alpha,RB):
        #Spring Matrices with a rotation
        a=self.phir*np.cos(alpha)**2+self.phiti*np.sin(alpha)**2                                   
        b=(self.phir-self.phiti)*np.cos(alpha)*np.sin(alpha)
        c=(self.phir-self.phiti)*np.cos(alpha)*np.sin(alpha)
        d=self.phir*np.sin(alpha)**2+self.phiti*np.cos(alpha)**2
        matrix=np.array([[a,b,0],[c,d,0],[0,0,self.phito]])
        return matrix
    #Phonon matrix elements
    def Onsite_Phonon(self,k):
        term=self.KAB(self.theta1,self.RB1)+self.KAB(self.theta2,self.RB2)+self.KAB(self.theta3,self.RB3)+self.KAB(self.theta4,self.RB4)  
        return term
    def Interaction_Phonon_x_bulk(self,k):
        k=np.array(k)
        phase1=-1j*np.dot(k,self.RB1)
        phase3=-1j*np.dot(k,self.RB3)
        term=self.KAB(self.theta1,self.RB1)*np.exp(phase1)+self.KAB(self.theta3,self.RB3)*np.exp(phase3)
        return term 
    def Interaction_Phonon_y_bulk(self,k):
        k=np.array(k)
        phase2=-1j*np.dot(k,self.RB2)
        phase4=-1j*np.dot(k,self.RB4)
        term=self.KAB(self.theta2,self.RB2)*np.exp(phase2)+self.KAB(self.theta4,self.RB4)*np.exp(phase4)
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
        print(gamma1Plus)
        gamma1Minus = gamma1Plus.conjugate()
        gamma2Plus = self.Gamma2_mgdis(k, self.theta)
        gamma2Minus = gamma2Plus.conjugate()
        print(gamma2Plus)
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
        
    
    def Ham_Magnon_Phonon(self,n,k):
        
        #Magnon Hamiltonian
        Mag_Ham=self.Ham_Magnon(k)
        
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
        print(aux_eval_neg)
        
        #mgph interactions
        G1=self.Gamma_mgph(self.Bper,k,aux_evec[0],aux_eval[0])
        G1minus=self.Gamma_mgph(self.Bper,k,aux_evec_neg[0],aux_eval_neg[0]).conj()
        
        G2=self.Gamma_mgph(self.Bper,k,aux_evec[1],aux_eval[1])
        G2minus=self.Gamma_mgph(self.Bper,k,aux_evec_neg[1],aux_eval_neg[1]).conj()
        
        G3=self.Gamma_mgph(self.Bper,k,aux_evec[2],aux_eval[2])
        G3minus=self.Gamma_mgph(self.Bper,k,aux_evec_neg[2],aux_eval_neg[2]).conj()
        
        Mph=np.array([[Mag_Ham[0,0]  ,    G1minus,    G2minus,    G3minus,           0,        G1minus,        G2minus,        G3minus],
                      [G1minus.conj(),aux_eval[0],          0,          0,   G1.conj(),              0,              0,              0],
                      [G2minus.conj(),          0,aux_eval[1],          0,   G2.conj(),              0,              0,              0],
                      [G3minus.conj(),          0,          0,aux_eval[2],   G3.conj(),              0,              0,              0],
                      [0             ,         G1,         G2,         G3,Mag_Ham[1,1],             G1,             G2,             G3],
                      [G1minus.conj(),          0,          0,          0,   G1.conj(),aux_eval_neg[0],              0,              0],
                      [G2minus.conj(),          0,          0,          0,   G2.conj(),              0,aux_eval_neg[1],              0],
                      [G3minus.conj(),          0,          0,          0,   G3.conj(),              0,              0,aux_eval_neg[2]]])
        
        return Mph
            
            
    def evals_evec(self,ham,aux):
        if aux=="ph":
            data=np.linalg.eig(ham)
            aux_eval=np.sqrt(np.abs(data[0]))
            aux_evec=np.array(data[1])
            evals=[]
            evecs=[]
            
            for i in range(len(aux_eval)):
                evals.append(aux_eval[i])
                evecs.append(aux_evec[:,i])
            sorting=np.argsort(evals)
            Evals=[]
            Evecs=[]
            for i in sorting:
                Evals.append(evals[i])
                Evecs.append(evecs[i])
            return [np.array(Evals),np.array(Evecs)]
        else:
            #colpa method
            KK=cholesky(ham)
            sigma=np.kron(np.array([[1,0],[0,-1]]),np.identity(int(len(ham)/2.)))
            D=np.dot(KK,np.dot(sigma,KK.conj().T))
            data=np.linalg.eig(D)
            evals=np.array(data[0])
            evecs=np.array(data[1])
            sorter_index=np.argsort(evals.real)[::-1]
            sorter_index_aux=[]
            for i in range(int(len(sorter_index)/2.)):
                sorter_index_aux.append(sorter_index[i])
            for i in range(int(len(sorter_index)/2.)):
                sorter_index_aux.append(sorter_index[-i-1])
                #Calculating eigenvectors using Colpa method
            U_aux=[]
            for i in range(len(sorter_index)):
                U_aux.append(evecs[:,sorter_index_aux[i]])
            U=np.column_stack(U_aux)
            U=np.matrix(U)

            L=np.dot(U.conj().T,np.dot(D,U))
            E=np.dot(sigma,L)
            T=np.dot(np.linalg.inv(KK),np.dot(U,np.sqrt(E)))
            evals=[]
            evecs=[]
            for i in range(len(E)):#6,len(evecs)):
                evals.append(L[i,i].real)
                evecs.append(np.array(T[:,i].T)[0])
            return [np.array(evals),np.array(evecs)]
    def phonon_system_band(self,unitcell,N_k):
        
        self.M=self.B1*0.5+self.B2*0.5
        self.X=self.B1*0.5
        self.G=np.array([0,0])
        path1=self.getEquidistantPoints(self.G,self.X, N_k)
        path2=self.getEquidistantPoints(self.X,self.M, N_k)
        path3=self.getEquidistantPoints(self.M,self.G, N_k)
        path2.remove(path2[0])
        path3.remove(path3[0])
        #path5=self.getEquidistantPoints(self.M1,self.G, N_k)
        path=np.concatenate((path1,path2,path3),axis=0)
  
        #path=path1
        x=[]
        E=[]
        x0=0
        for k in path:
            Ham=self.Ham_Phonon(10,k)
            [evals,evec]=self.evals_evec(Ham,"ph")#+self.Ham_auxiliar*np.identity(4),"ph")    
            index=np.argsort(evals.real)
            E.append(evals.real[index])
            x.append(x0)
            x0+=1
        E=np.array(E)
        for i in range(len(E.T)):
            plt.plot(x,E.T[i])
        plt.grid()
        plt.show()
         
    def magnon_system_band(self,unitcell,N_k):
        
        self.M=self.B1*0.5+self.B2*0.5
        self.X=self.B1*0.5
        self.G=np.array([0,0])
        path1=self.getEquidistantPoints(self.G,self.X, N_k)
        path2=self.getEquidistantPoints(self.X,self.M, N_k)
        path3=self.getEquidistantPoints(self.M,self.G, N_k)
        path2.remove(path2[0])
        path3.remove(path3[0])
        #path5=self.getEquidistantPoints(self.M1,self.G, N_k)
        path=np.concatenate((path1,path2,path3),axis=0)
  
        #path=path1
        x=[]
        E=[]
        x0=0
        for k in path:
            Ham=self.Ham_Magnon(10,k)
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
 
    def magnon_phonon_system_band(self,unitcell,N_k):

        self.M=self.B1*0.5+self.B2*0.5
        self.X=self.B1*0.5
        self.G=np.array([0,0,0])
        path1=self.getEquidistantPoints(self.G,self.X, N_k)
        path2=self.getEquidistantPoints(self.X,self.M, N_k)
        path3=self.getEquidistantPoints(self.M,self.G, N_k)
        path2.remove(path2[0])
        path3.remove(path3[0])
        #path5=self.getEquidistantPoints(self.M1,self.G, N_k)
        path=np.concatenate((path1,path2,path3),axis=0)
  
        #path=path1
        x=[]
        E=[]
        x0=0
        for k in path:
            Ham=self.Ham_Magnon_Phonon(10,k)+np.identity(8)*0.1
            [evals,evec]=self.evals_evec(Ham,"mg")#+self.Ham_auxiliar*np.identity(4),"ph") 
            index=np.argsort(evals.real)
            E.append(evals.real[index])
            x.append(x0)
            x0+=1
        E=np.array(E)
        for i in range(len(E.T)):
            plt.plot(x,E.T[i])
        plt.ylim(0,3)
        plt.grid()
        plt.show()
        
    def mag_ph_dis_system_band(self, N_k):
        
        self.M=self.B1*0.5+self.B2*0.5
        self.X=self.B1*0.5
        self.G=np.array([0.001,0,0])
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


