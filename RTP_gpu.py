# simulation model of RTP
# symmetry breaking motility
# Seoul nat'l university
# Yongjoo Baek, Ki-won Kim and Yunsik Choe
# numerics done by Yunsik Choe
# 200920

import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
from matplotlib.animation import FuncAnimation     # animation
import sys
import torch

cuda = torch.device('cuda')
cpu = torch.device('cpu')
torch.set_num_threads(1)



class RTP_lab:     # OOP
    """basic model to simulate active Run and Tumble Particles interacting with passive object"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,alpha=1, u=10, len_time=100, N_time=100,N_X=1, N_ptcl=40000, v=0, mu=1, muw = 0.005, model=3):
        
        self.initial_state = (alpha,u,len_time,N_time,N_X, N_ptcl,v,mu,muw)    # recording initial state
        # coefficients
        self.set_coeff(alpha, u, len_time, N_time, N_X,N_ptcl,v,mu, muw) 
        
        # model=2 constantly moving passive object, model=3 interacting passive object
        self.model=model
        
        # check the validity
        self.check_coeff()  
        
        # initializing configuration of state
        self.set_zero()
        
        print('RTP model initialized')
            
            
    # setting coefficients
    def set_coeff(self,alpha=1, u=10, len_time=200, N_time=100, N_X = 1, N_ptcl=40000, v=0, mu=1, muw = 0.005):
        self.alpha = alpha                       # rate of tumble (/time dimension)
        self.u = u                               # velocity of particle
        self.len_time = len_time                 # time turation
        self.N_time = N_time                     # number of simulation in unit time
        self.N_X = N_X
        self.N_ptcl = N_ptcl
        self.N_simul = np.int64(self.len_time*self.N_time)
        self.delta_time = 1/self.N_time
        self.D_eff = self.u**2/self.alpha
        
        self.mu = mu
        
        self.F=4              # just give potential of fixed value for now
        self.l=30
        self.L= 300
        self.d=10
        self.k=20


        
        self.muw = muw
    
    
    # check coefficients for linearization condition
    def check_coeff(self):
        if self.alpha*self.delta_time <=0.01:
            pass
        else:
            print('alpha*delta_time = ',self.alpha*self.delta_time,' is not small enough. Increase the N_time for accurate simulation')   # for linearization
          
        
        
        
        
        
    # wall potential and gradient force
    
    def periodic(self,x):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        return mod_x
    
    def V(self,x):                                         # V as function of x-X (relative position w r t object)
        y = self.periodic(x-self.X)
        return self.F*(y+self.l/2)*(-self.l/2<=y)*(y<0)-self.F*(y-self.l/2)*(0<=y)*(y<self.l/2)
        
        

    def partial_V(self,x):
        y = self.periodic(x-self.X)
        
        output = 0*y
        
        
        
        # indexing
        A =  (-self.l/2-self.d/self.k<=y)*(y<=-self.l/2+self.d/self.k)      # left boundary
        B = (-self.l/2+self.d/self.k<y)*(y<-self.d/self.k)                  # left side of object
        
        C = (self.d/self.k<y)*(y<self.l/2-self.d/self.k)                   # right side of object
        D = (self.l/2-self.d/self.k<=y)*(y<=self.l/2+self.d/self.k)         # right boundary
        
        E = (-self.d/self.k<=y)*(y<=self.d/self.k)# center boundary
        
        output[A]+=self.F/(1+torch.exp(-self.k*(y[A]+self.l/2)))
        
        output[B]+=self.F
        output[C]-=self.F
        output[D]-=self.F/(1+torch.exp(self.k*(y[D]-self.l/2)))
        output[E]+=self.F* (torch.exp(-self.k*(y[E])) - torch.exp(self.k*(y[E]))  )/(   torch.exp(-self.k*(y[E]))  +  torch.exp(self.k*(y[E]))  )
        
        return output
        
        

    
    
        
        
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.time = np.linspace(0,self.len_time,num=self.N_simul)
        self.s = (2*torch.bernoulli(0.5*torch.ones(self.N_ptcl,self.N_X))-1).to(device=cuda,dtype=torch.float64)          # random direction at initial time
        self.x = torch.empty(self.N_ptcl,self.N_X).uniform_(-self.L/2, self.L/2).to(device=cuda,dtype=torch.float64)
        self.X = torch.zeros(self.N_X).to(device=cuda,dtype=torch.float64)
        self.v = torch.zeros(self.N_X).to(device=cuda,dtype=torch.float64)
    
    def tumble(self):             # random part of sle p dynamics
        p = 1-self.delta_time*self.alpha/2 # +1 no tumble, -1 tumble
        tumble = (2*torch.bernoulli(p*torch.ones(self.N_ptcl,self.N_X))-1 ).to(device=cuda,dtype=torch.float64)
        return tumble
    
    def dynamics(self,x,s):         # dynamics of x and s to give time difference of x and s
        dxa = self.u*self.delta_time
        ds = self.tumble()
        dx = dxa*s  -self.mu*self.partial_V(x)*self.delta_time
        if self.model==3:
            v = self.muw*torch.sum(self.partial_V(x),axis=0)
        elif self.model ==2:
            v = self.v




        return (v, dx, ds)
        
    def time_evolve(self):
        (v, dx, ds) = self.dynamics(self.x, self.s)
        self.v = v
       
        self.x += dx                     # active particles movement
        self.s *= ds                     # direction of movement changed if the tumble is true
        self.X += v*self.delta_time           # passive object movement
        
        self.x = self.periodic(self.x)
        self.X = self.periodic(self.X)
        
        

            
            
            
            
            
            
    # simulation protocol, Fint
    def VX_distribution(self, N_ptcl=400,N_X=1000,a = 0.7, muFu = 0.8, initialize = 'off'):
        
        # initialization
        if initialize == 'on':
            self.N_X = N_X
            self.N_ptcl = N_ptcl
            self.u = a*self.l*self.alpha/2
            self.F = muFu*self.u/self.mu
            
            self.set_zero()
            
        
        else:
            self.F = muFu*self.u/self.mu
            
            self.time_evolve()
                
            VX = self.v
            VU = VX/self.u
            
        
            data = pd.DataFrame({'a':a, 'muFu':muFu, 'vu' : VU,'muw':self.muw, 'current':self.current})
            
            return data
        
        
        
    # animation simulation 
    #def 
        
        
        
   
        
def moments(N, L, l, a, f, muw,duration,Fs, name):
    RTP = RTP_lab(alpha=1, u=10, len_time=100, N_time=Fs,N_X=40, N_ptcl=N, v=0, mu=1, muw = muw)
    RTP.l = l
    RTP.L = L
    RTP.u = a*l*RTP.alpha/2
    RTP.F = f*RTP.u/RTP.mu
    
    RTP.set_zero()
        
    
    state = os.getcwd()+'/data/'+str(name)+'.npz'
    try:
        load = np.load(state)
        save_dict={}
        save_dict['first'] = load['first']
        save_dict['second'] = load['second']
        save_dict['third'] = load['third']
        save_dict['fourth'] = load['fourth']

        save_dict['muw'] = load['muw']
        save_dict['Fs'] = load['Fs']
        save_dict['description'] = load['description']
        save_dict['count'] = load['count']
    except IOError:
        save_dict={}
        save_dict['first'] = np.zeros(duration)
        save_dict['second'] = np.zeros(duration)
        save_dict['third'] = np.zeros(duration)
        save_dict['fourth'] = np.zeros(duration)

        save_dict['muw'] = RTP.muw
        save_dict['Fs'] = RTP.N_time
        save_dict['description'] = 'L : '+str(RTP.L)+', N : '+str(RTP.N_ptcl)+', f : '+str(f) + ', a :'+str(a)
        save_dict['count'] = 0
        
 
        
#     RTP.muw =0

#     for i in trange(RTP.N_time*10):
#         RTP.time_evolve()
        
        
    first=np.zeros(duration)
    second=np.zeros(duration)
    third=np.zeros(duration)
    fourth=np.zeros(duration)
    
    RTP.muw = muw
    
#     for i in trange(duration*5):
#         RTP.time_evolve()
    
    
    for i in trange(duration):
        RTP.time_evolve()
        
        first[i]= np.average((torch.abs(RTP.v)).to(device=cpu).numpy())
        second[i]= np.average((torch.abs(RTP.v**2)).to(device=cpu).numpy())
        third[i]= np.average((torch.abs(RTP.v**3)).to(device=cpu).numpy())
        fourth[i]= np.average((torch.abs(RTP.v**4)).to(device=cpu).numpy())
    
    count = save_dict['count']
    save_dict['first']  *= count
    save_dict['second'] *= count
    save_dict['third']  *= count
    save_dict['fourth'] *= count
    save_dict['count']+=1
    count = save_dict['count']
    save_dict['first']  += first
    save_dict['second'] += second
    save_dict['third']  += third
    save_dict['fourth'] += fourth
    save_dict['first']  /= count
    save_dict['second'] /= count
    save_dict['third']  /= count
    save_dict['fourth'] /= count

    
#     os.makedirs(os.getcwd()+'/data/'+str(name),exist_ok=True)
    np.savez(state, **save_dict)

    
    
    
def measure(ptcl, number_X,L, f_init,f_fin,f_step, t_step):
    
    for i in trange(f_step):
        f = f_init + (f_fin-f_init)*i/f_step
        state = os.getcwd()+'/data/'+"ptcl"+str(ptcl)+"X"+str(number_X)+"L"+str(L)+"f"+str(f)+'.npz'
        time_moments(ptcl, number_X,L,f,t_step,state)
       
    
def simulate(N, L, l, a, f,duration,Fs, name):
    RTP = RTP_lab(alpha=0.5, u=10, len_time=100, N_time=Fs,N_X=1, N_ptcl=N, v=0, mu=1)
    RTP.l = l
    RTP.L = L
    RTP.u = a*l*RTP.alpha/2
    RTP.F = f*RTP.u/RTP.mu
    
    RTP.set_zero()
    
    X_list = duration*[None]
    v_list = duration*[None]
    

    
    RTP.muw = 1*L/RTP.N_ptcl
    
#     for i in range(int(duration/5)):
#         RTP.time_evolve()
    
    
    for i in trange(duration):
        RTP.time_evolve()
        
        X_list[i] = RTP.X.to(device=cpu).numpy()
        v_list[i] = RTP.v.to(device=cpu).numpy()

        
        
    save_dict={}
    save_dict['X'] = X_list
    save_dict['v'] = v_list
#     save_dict['co_r'] = co_r_list
#     save_dict['co_phi'] = co_phi_list

    save_dict['muw'] = RTP.muw
#     save_dict['current'] = pd.concat(current_list)
    save_dict['Fs'] = RTP.N_time
    save_dict['description'] = 'L : '+str(RTP.L)+', N : '+str(RTP.N_ptcl)+', f : '+str(f) + 'a :'+str(a)
    

    
    state = os.getcwd()+'/data/away/210205/'+str(name)+'.npz'
    os.makedirs(os.getcwd()+'/data/away/210205',exist_ok=True)
    np.savez(state, **save_dict)
    


def N_scan(fin,ffin,N,N_ptcl):
    direc ='1218/'
    rho=1
    L=300
    direc+='N/'+str(N_ptcl)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    
    for i in trange(N):
        f = fin+(ffin-fin)*i/N
        name = direc+ str(f)
        l=30
        alpha=1
        Fs=10000
        simulate(N_ptcl, L, l, alpha, f,1*rho*L/N_ptcl, 1000000,Fs, name)

def rho_scan(fin,ffin,N,rho):
    direc ='1210/'
    N_ptcl = 10000
    L=300
    direc+='rho/'+str(rho)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    for i in trange(N):
        f = fin+(ffin-fin)*i/N
        name = direc+ str(f)
        l=30
        a=1
        Fs=10000
        simulate(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 1000000,Fs, name)
        
def L_scan(fin,ffin,N,L):
    direc ='1211/'
    N_ptcl=30*L
    rho=10
    direc+='L/'+str(L)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    for i in trange(N):
        f = fin+(ffin-fin)*i/N
        name = direc+ str(f)
        l=30
        a=1
        Fs=10000
        simulate(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 1000000,Fs, name)
        
        
def N_scan_moments(fin,ffin,N,N_ptcl):
    direc ='210125_1/'
    rho=1
    L=300
    direc+='N/'+str(N_ptcl)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    
    for i in trange(N):
        f = fin+(ffin-fin)*i/N
        name = direc+ str(f)
        l=30
        a=1
        Fs=100
        moments(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 200000,Fs, name)
        torch.cuda.empty_cache()
        
def l_scan_moments(fin,ffin,N,a,N_ptcl):
    
    direc ='210601/'
    rho=1
    L=300
    direc+='a/'+str(a)+'/N/'+str(N_ptcl)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    
    
    for i in trange(N):
        f = fin+(ffin-fin)*i/N
        name = direc+ str(f)
        l=30/a
        Fs=1000
        moments(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 300000,Fs, name)
        
def density_scan(N, f_init, f_fin, N_f,group_name):
    a=1.1
    Fs=2000
    rho = 1
    L=300
    l=30
    muw = rho*L/N
    f_axis = np.linspace(f_init,f_fin,N_f)
    binning = np.linspace(-1,1,200)

    for f in f_axis:
        name = str(N)+'_'+str(f)
        R1 = RTP_lab(N_time = Fs, N_X = 32, N_ptcl = N)
        R1.L=L
        R1.l=l/a
        R1.u = a*l*R1.alpha/2
        R1.F = f*R1.u/R1.mu
        
        state = os.getcwd()+'/data/density/210519/'+str(group_name)+'/'+str(name)+'.npz'
        os.makedirs(os.getcwd()+'/data/density/210519/'+str(group_name),exist_ok=True)

        for i in trange(50000):
            R1.time_evolve()
        n,bins,patches = plt.hist(R1.v.to(device=torch.device('cpu')).numpy().flatten()/R1.u, bins = binning, density = True)
        for i in trange(9999):
            R1.time_evolve()
            n_temp,__,_ = plt.hist(R1.v.to(device=torch.device('cpu')).numpy().flatten()/R1.u, bins = binning, density = True)
            n+=n_temp
            plt.clf()
        n/=10000  
        
        save_dict={}
        save_dict['N'] = N
        save_dict['a'] = a
        save_dict['f'] = f
        save_dict['L'] = L
        save_dict['l'] = l
        save_dict['Fs'] = Fs
        save_dict['n'] = n
        save_dict['bins'] = bins
        
        np.savez(state, **save_dict)
        torch.cuda.empty_cache()
