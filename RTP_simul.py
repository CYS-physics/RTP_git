# simulation model of RTP
# symmetry breaking motility
# Seoul nat'l university
# Yongjoo Baek, Ki-won Kim and Yunsik Choe
# numerics done by Yunsik Choe
# 200920
# private

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
# import os                            # file management
from matplotlib.animation import FuncAnimation     # animation
import sys



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
        
        self.compute = False
        
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
        #return self.F*(-self.l/2<=y)*(y<0)-self.F*(0<y)*(y<=self.l/2)
        
        output = 0*y
        
        
        
        # indexing
        A =  (-self.l/2-self.d/self.k<=y)*(y<=-self.l/2+self.d/self.k)      # left boundary
        B = (-self.l/2+self.d/self.k<y)*(y<-self.d/self.k)                  # left side of object
        
        C = (self.d/self.k<y)*(y<self.l/2-self.d/self.k)                   # right side of object
        D = (self.l/2-self.d/self.k<=y)*(y<=self.l/2+self.d/self.k)         # right boundary
        
        E = (-self.d/self.k<=y)*(y<=self.d/self.k)# center boundary
        
        output[A]+=self.F/(1+np.exp(-self.k*(y[A]+self.l/2)))
        output[B]+=self.F
        output[C]-=self.F
        output[D]-=self.F/(1+np.exp(self.k*(y[D]-self.l/2)))
        output[E]+=self.F* (  np.exp(-self.k*(y[E])) - np.exp(self.k*(y[E]))  )/(   np.exp(-self.k*(y[E]))  +  np.exp(self.k*(y[E]))  )
        
        return output
        
        
        
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.time = np.linspace(0,self.len_time,num=self.N_simul)
        self.s = np.random.choice([1,-1],np.array([self.N_ptcl,self.N_X]))             # random direction at initial time
        self.x = np.random.uniform(-self.L/2, self.L/2, size=np.array([self.N_ptcl,self.N_X]))     # starting with uniformly distributed particles
        self.X = np.zeros(self.N_X)
#         self.v = np.zeros(self.N_X)
    
    def tumble(self):             # random part of s dynamics
        tumble = np.random.choice([1,-1], size=np.array([self.N_ptcl,self.N_X]), p = [1-self.delta_time*self.alpha/2, self.delta_time*self.alpha/2]) # +1 no tumble, -1 tumble
        return tumble
    
    def dynamics(self,x,s):         # dynamics of x and s to give time difference of x and s
        dxa = self.u*self.delta_time
        ds = self.tumble()
        
        dx = dxa*s  -self.mu*self.partial_V(x)*self.delta_time
        if self.model==3:
            v = self.muw*np.sum(self.partial_V(x),axis=0)
        elif self.model ==2:
            v = self.v




        return (v, dx, ds)
        
    def time_evolve(self):
        (v, dx, ds) = self.dynamics(self.x, self.s)
        
        self.v = v
#         if self.compute:
#             self.dS1 = -np.average(self.partial_V(self.x+dx/2-v*self.delta_time/2)*(dx-v*self.delta_time),axis=0)
#             self.dS2 = (self.u/self.mu)*np.average(self.s*dx,axis=0)
        
          
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

        
    def compute_EP_v(self, n_batch = 200, a=1.1, muFu = 1,duration = 100, wait=100, name = 'test'):
        self.model=2
        
        # change coeff
        self.N_X = n_batch
        self.l = self.l/a
        self.u = a*self.l*self.alpha/2
        self.F = muFu*self.u/self.mu
        
           
        self.set_zero()
        
        self.v = np.linspace(0,self.u*1,n_batch)
        
            
        for __ in trange(int(self.N_time*wait)):   # to stable distribution
            self.time_evolve()
        self.compute=True
        
        F_v = np.zeros(n_batch)
        dS1_v = np.zeros(n_batch)
        dS2_v = np.zeros(n_batch)
            
        #error_measure = np.empty(repeat)
        
            # computation part with time evolving
        for _ in trange(self.N_time*duration):
            self.time_evolve()
            F_v += (self.L/self.N_ptcl) *np.sum(self.partial_V(self.x),axis=0)/(self.N_time*duration)              # summing the V' at each x
            dS1_v += self.dS1/(self.N_time*duration)
            dS2_v += self.dS2/(self.N_time*duration)
            
        plt.plot(self.v/self.u,F_v)
        plt.xlabel('v/u')
        plt.ylabel('Force_wall')
        plt.show()
        plt.plot(self.v/self.u,dS1_v)
        plt.xlabel('v/u')
        plt.ylabel('dS1')
        plt.show()
        plt.plot(self.v/self.u,dS2_v)
        plt.xlabel('v/u')
        plt.ylabel('dS2')
        plt.show()
        plt.plot(self.v/self.u,dS1_v+dS2_v,'r')
        plt.xlabel('v/u')
        plt.ylabel('dSt')
        plt.show()
        
        return ((self.v/self.u,F_v,dS1_v,dS2_v))
    
    def compute_EP_f(self, n_batch = 200,vu = 0, a=1.1,duration = 100, wait=100, name = 'test'):
        self.model=2
        
        # change coeff
        self.N_X = 1
        self.l = self.l/a
        self.u = a*self.l*self.alpha/2
        self.v = vu*self.u
        self.set_zero()
        f_axis = np.zeros(n_batch)
        dS1_f = np.zeros(n_batch)
        dS2_f = np.zeros(n_batch)
        
        for i in trange(n_batch):
            self.set_zero()
            self.F = (i/n_batch)*self.u/self.mu
            f_axis[i] = i/n_batch

        
        
            
            for __ in range(int(self.N_time*wait)):   # to stable distribution
                self.time_evolve()
            self.compute=True

            

            #error_measure = np.empty(repeat)

                # computation part with time evolving
            for _ in range(self.N_time*duration):
                self.time_evolve()

                dS1_f[i] += self.dS1/(self.N_time*duration)
                dS2_f[i] += self.dS2/(self.N_time*duration)

       
        plt.plot(f_axis,dS1_f)
        plt.xlabel('f')
        plt.ylabel('dS1')
        plt.show()
        plt.plot(f_axis,dS2_f)
        plt.xlabel('f')
        plt.ylabel('dS2')
        plt.show()
        plt.plot(f_axis,dS1_f+dS2_f)
        plt.xlabel('f')
        plt.ylabel('dSt')
        plt.show()             
        
        return((f_axis, dS1_f, dS2_f))


        
        
        
    # moment calculation        
def time_moments(ptcl, number_X,L, f, t_step,  state):
    RTP3 = RTP_lab(alpha=1,len_time=200, N_time=10000, mu=1, muw = 0.01 )
    RTP3.L=L
    list_of_df = []
    
    measure = 100
        
    try:
        load = np.load(state)
        RTP3.N_ptcl = load['N_ptcl']
        RTP3.u = load['u']
        RTP3.F = load['F']
            
        RTP3.X = load['X']
        #RTP3.v = load['v']
        #RTP3.current = load['current']

        RTP3.s = load['s']
        RTP3.x = load['x']
        simul_iter = load['iter']
    except:
        RTP3.VX_distribution(a=0.8,muFu = f,N_ptcl=ptcl,N_X=number_X, initialize='on')
        simul_iter=0
        
    for iter in range (t_step):
        simul_iter += 1
        first = 0
        first_r = 0
        second = 0
        fourth = 0
        
        c_first = 0
        c_first_r = 0
        c_second = 0
        c_fourth = 0

        
        for _ in range(measure):
            data = RTP3.VX_distribution(a=0.8,N_ptcl=ptcl,N_X=number_X,muFu=f)
            
            first += np.average(data.vu)
            first_r += np.average(np.abs(data.vu))
            second += np.average(data.vu**2)
            fourth += np.average(data.vu**4)
            
            c_first += np.average(data.current)
            c_first_r += np.average(np.abs(data.current))
            c_second += np.average(data.current**2)
            c_fourth += np.average(data.current**4)
        
        first/=measure
        first_r/=measure
        second/=measure
        fourth/=measure
        second_r = second-first_r**2
        #binder = 1-(fourth/(3*second**2))
        
        c_first/=measure
        c_first_r/=measure
        c_second/=measure
        c_fourth/=measure
        c_second_r = c_second-c_first_r**2
        #c_binder = 1-(c_fourth/(3*c_second**2))
    
        moment = pd.DataFrame({'muFu':[f],'one':[first],'one_r':[first_r], 'two':[second], 'two_r':[second_r] ,'four':[fourth], 'c_one':[c_first],'c_one_r':[c_first_r], 'c_two':[c_second], 'c_two_r':[c_second_r] ,'c_four':[c_fourth],'simul_iter':simul_iter})
        list_of_df.append(moment)
                              
    moments = pd.concat(list_of_df)
    
    # save state
    save_dict={}
    save_dict['N_ptcl'] = RTP3.N_ptcl
    save_dict['u'] = RTP3.u
    save_dict['F'] = RTP3.F
    save_dict['X'] = RTP3.X
    save_dict['v'] = RTP3.v
    save_dict['current'] = RTP3.current
    save_dict['time'] = RTP3.time
    save_dict['s'] = RTP3.s
    save_dict['x'] = RTP3.x
    save_dict['iter'] = simul_iter
    
    np.savez(state, **save_dict)

    
    
    
    title = str(ptcl)+'ptcl'+str(number_X)+'L'+str(L)+'.csv'

    if not os.path.exists(title):
        moments.to_csv(title, mode='w')
    else:
        moments.to_csv(title, mode='a',header=False)
        
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
        
    RTP.muw = muw

    first=np.zeros(duration)
    second=np.zeros(duration)
    third=np.zeros(duration)
    fourth=np.zeros(duration)
    
#     for i in range(int(duration/5)):
#         RTP.time_evolve()
    
    
#     for i in trange(duration*5):
#         RTP.time_evolve()
        
    for i in trange(duration):
        RTP.time_evolve()
        
        first[i] = np.average(np.abs(RTP.v))
        second[i]  = np.average(np.abs(RTP.v)**2)
        third[i]  = np.average(np.abs(RTP.v)**3)
        fourth[i]  = np.average(np.abs(RTP.v)**4)
    
    
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
    
    
def trajectory(N, L, l, a, f, muw,duration,Fs, name):
    RTP = RTP_lab(alpha=1, u=10, len_time=100, N_time=Fs,N_X=40, N_ptcl=N, v=0, mu=1, muw = muw)
    RTP.l = l
    RTP.L = L
    RTP.u = a*l*RTP.alpha/2
    RTP.F = f*RTP.u/RTP.mu
    
    RTP.set_zero()
        
    
    state = os.getcwd()+'/data/'+str(name)+'.npz'
    
    save_dict={}

    save_dict['muw'] = RTP.muw
    save_dict['Fs'] = RTP.N_time
    save_dict['description'] = 'L : '+str(RTP.L)+', N : '+str(RTP.N_ptcl)+', f : '+str(f) + ', a :'+str(a)
    save_dict['count'] = 0
        
        
    RTP.muw = muw

    v_traj = np.zeros((duration,RTP.N_X))
    
#     for i in range(int(duration/5)):
#         RTP.time_evolve()
    
    
#     for i in trange(duration*5):
#         RTP.time_evolve()
        
    for i in trange(duration):
        RTP.time_evolve()
        
        v_traj[i] = RTP.v
    
 
    save_dict['v_traj'] = v_traj
#     os.makedirs(os.getcwd()+'/data/'+str(name),exist_ok=True)
    np.savez(state, **save_dict)
    
    
    
def measure(ptcl, number_X,L, f_init,f_fin,f_step, t_step):
    
    for i in trange(f_step):
        f = f_init + (f_fin-f_init)*i/f_step
        state = os.getcwd()+'/data/'+"ptcl"+str(ptcl)+"X"+str(number_X)+"L"+str(L)+"f"+str(f)+'.npz'
        time_moments(ptcl, number_X,L,f,t_step,state)
       
    
def simulate(N, L, l, a, f,duration,Fs, name):
#     state = os.getcwd()+'/data/dS/210622/'+str(name)+'.npz'
#     os.makedirs(os.getcwd()+'/data/dS/210622/'+str(N),exist_ok=True)
    
    RTP = RTP_lab(alpha=1, u=10, len_time=100, N_time=Fs,N_X=10, N_ptcl=N, v=0, mu=1, muw = 1)
    RTP.l = l
    RTP.L = L
    RTP.u = a*l*RTP.alpha/2
    RTP.F = f*RTP.u/RTP.mu
    
    RTP.set_zero()
    
    X_list = duration*[None]
    v_list = duration*[None]
    dS1_list = duration*[None]
    dS2_list = duration*[None]
    

    
    RTP.muw = 1*L/RTP.N_ptcl
    
#     for i in range(int(duration/5)):
#         RTP.time_evolve()
    
    for i in trange(duration):
        RTP.time_evolve()
        
#         X_list[i] = RTP.X
        v_list[i] = RTP.v
        dS1_list[i] = RTP.dS1
        dS2_list[i] = RTP.dS2

    v = np.array(v_list).flatten()
    dS1 = np.array(dS1_list).flatten()
    dS2 = np.array(dS2_list).flatten()
    dSt = dS1+dS2
    
    vu_axis = np.linspace(-1,1,200)
    dS1_axis = np.linspace(-0.002,0.003,200)
    dS2_axis = np.linspace(0.205,0.228,200)
    dSt_axis = np.linspace(0.205,0.228,200)
    
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('N : '+str(N)+', f : '+str(f),fontsize='20')
    ax1 = fig.add_subplot(131,title='dS1')
    plt.hist2d(v/RTP.u,dS1,bins = [vu_axis,dS1_axis])
    ax2 = fig.add_subplot(132,title='dS2')
    plt.hist2d(v/RTP.u,dS2,bins = [vu_axis,dS2_axis])
    ax3 = fig.add_subplot(133,title='dSt')
    plt.hist2d(v/RTP.u,dSt,bins =  [vu_axis,dSt_axis])
    os.makedirs('data/fig/dS/20210628/'+str(N),exist_ok=True)
    plt.savefig('data/fig/dS/20210628/'+str(N)+'/'+str(f)+'.png',dpi=300)
        
#     save_dict={}
#     save_dict['X'] = X_list
#     save_dict['v'] = v_list
#     save_dict['dS1'] = dS1_list
#     save_dict['dS2'] = dS2_list

#     save_dict['u'] = RTP.u

#     save_dict['muw'] = RTP.muw
# #     save_dict['current'] = pd.concat(current_list)
#     save_dict['Fs'] = RTP.N_time
#     save_dict['description'] = 'L : '+str(RTP.L)+', N : '+str(RTP.N_ptcl)+', f : '+str(f) + 'a :'+str(a)

    
    
#     np.savez(state, **save_dict)
    
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
    direc ='211007/'
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
        moments(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 50000,Fs, name)
    
def simul_scan(f_init, f_fin, N, N_ptcl):
    for i in trange(N):
        f=f_init+i*(f_fin-f_init)/N
        simulate(N_ptcl,300,30,1,f,1000000,1000,str(N_ptcl)+'/'+str(f))
        
        
def L_scan_moments(fin,ffin,N,L):
    
    direc ='211215/'
#     rho=10
    rho=20


    L=L
    N_ptcl = 50*L
    a=0.9
    direc+='a/'+str(a)+'/L/'+str(L)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    
    for i in trange(N):
        f = fin+(ffin-fin)*i/N
        name = direc+ str(f)
        l=30/a
        Fs=3000
        moments(N_ptcl, L, l, a, f,0.01*rho*L/N_ptcl, 300000,Fs, name)
        
def v_traj_scan(fin,ffin,N_f,L):
    
    direc ='211120_traj/'
    rho=20
    L=L
    N_ptcl = 100*L
    a=0.9
    direc+='a/'+str(a)+'/L/'+str(L)+'/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    
    for i in trange(N_f):
        f = fin+(ffin-fin)*i/N_f
        name = direc+ str(f)
        l=30/a
        Fs=2000
        trajectory(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 300000,Fs, name)
        
        
def density_scan(N, f_init, f_fin, N_f,group_name):
    a=1.1
    Fs=1000
    rho = 1
    L=300
    l=30
    muw = rho*L/N
    vu_axis = np.linspace(-1,1,200)
    dS1_axis = np.linspace(-0.002,0.003,200)
    dS2_axis = np.linspace(0.22,0.228,200)
    dSt_axis = np.linspace(0.22,0.228,200)

    for i in trange(N_f):
        f = f_fin+(f_fin-f_init)*i/N_f
        name = str(N)+'_'+str(f)
        R1 = RTP_lab(N_time = Fs, N_X = 32, N_ptcl = N)
        R1.L=L
        R1.l=l/a
        R1.muw=muw
        R1.u = a*l*R1.alpha/2
        R1.F = f*R1.u/R1.mu
        
        state = os.getcwd()+'/data/density/210624/'+str(group_name)+'/'+str(name)+'.npz'
        os.makedirs(os.getcwd()+'/data/density/210624/'+str(group_name),exist_ok=True)

        for i in trange(200000):
            R1.time_evolve()
        n_dS1,x_dS1,y_dS1,_ = plt.hist2d(R1.v/R1.u,R1.dS1, bins = [vu_axis,dS1_axis])
        n_dS2,x_dS2,y_dS2,_ = plt.hist2d(R1.v/R1.u,R1.dS2, bins = [vu_axis,dS2_axis])
        n_dSt,x_dSt,y_dSt,_ = plt.hist2d(R1.v/R1.u,R1.dS1+R1.dS2, bins = [vu_axis,dSt_axis])

        for i in trange(49999):
            R1.time_evolve()
            n_dS1_temp,__,_,___ = plt.hist2d(R1.v/R1.u,R1.dS1, bins = [vu_axis,dS1_axis])
            n_dS2_temp,__,_,___ = plt.hist2d(R1.v/R1.u,R1.dS2, bins = [vu_axis,dS2_axis])
            n_dSt_temp,__,_,___ = plt.hist2d(R1.v/R1.u,R1.dS1+R1.dS2, bins = [vu_axis,dSt_axis])

            plt.clf()
            n_dS1+=n_dS1_temp
            n_dS2+=n_dS2_temp
            n_dSt+=n_dSt_temp

        n_dS1/=50000
        n_dS2/=50000
        n_dSt/=50000
        
        save_dict={}
        save_dict['N'] = N
        save_dict['a'] = a
        save_dict['f'] = f
        save_dict['L'] = L
        save_dict['l'] = l
        save_dict['Fs'] = Fs
        save_dict['n_dS1'] = n_dS1
        save_dict['n_dS2'] = n_dS2
        save_dict['n_dSt'] = n_dSt
        
        save_dict['vu_axis'] = vu_axis

        save_dict['y_dS1'] = y_dS1
        save_dict['y_dS2'] = y_dS2
        save_dict['y_dSt'] = y_dSt
        
        np.savez(state, **save_dict)
        
        
def dS_scan(fin,ffin,N,a,N_ptcl):
    
    direc ='210620/'
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
        
def EP_scan(N_ptcl,vu_fix,name):
    direc =os.getcwd()+'/data/EP_scan/210701/'
    os.makedirs(direc,exist_ok=True)
    Fs=1000
    R1=RTP_lab(N_time=Fs,N_ptcl=N_ptcl, model=2)
    
    (f_axis,dS1, dS2) = R1.compute_EP_f(n_batch=40,vu=vu_fix,wait=100,duration=30)
    # 40, 100, 30

    
    
    state = direc+str(name)+'.npz'
    save_dict={}
    save_dict['N_ptcl'] = N_ptcl
    save_dict['vu_fix'] = R1.v/R1.u
    save_dict['f_axis'] = f_axis
    save_dict['dS1'] = dS1
    save_dict['dS2'] = dS2

    
    np.savez(state, **save_dict)
    
    
def EP_v_scan(N_ptcl,f_init, f_fin,N_f):
    direc =os.getcwd()+'/data/EP_V_scan/210706/'+str(N_ptcl)+'/'
    os.makedirs(direc,exist_ok=True)
    Fs=1000
    
    for i in trange(N_f):
        f_fix = f_init+(f_fin-f_init)*i/N_f
        name = f_fix
        R1 = RTP_lab(N_time=Fs,N_ptcl=N_ptcl, model=2)

        (vu_axis, F_v,dS1_v,dS2_v)= R1.compute_EP_v(n_batch = 50, wait=100, duration=30, muFu=f_fix)
        # 40, 100, 30



        state = direc+str(name)+'.npz'
        save_dict={}
        save_dict['N_ptcl'] = N_ptcl
        save_dict['f'] = f_fix
        save_dict['vu_axis'] = vu_axis
        save_dict['F_v'] = F_v
        save_dict['dS1_v'] = dS1_v
        save_dict['dS2_v'] = dS2_v


        np.savez(state, **save_dict)
    
    
    
def f_density(N_ptcl, f_init, f_fin, N,name):
    direc =os.getcwd()+'/data/f_density/210701/'
    state = direc+str(name)
    os.makedirs(direc,exist_ok=True)  
    Fs=1000
    a=1.1
    l=30/a
    rho=1
    L=300
    muw = 1*rho*L/N_ptcl
    RTP = RTP_lab(alpha=1, u=10, len_time=100, N_time=Fs,N_X=40, N_ptcl=N_ptcl, v=0, mu=1, muw = muw)
    RTP.l = l
    RTP.L = L
    RTP.u = a*l*RTP.alpha/2
        
        
    list_v_density = []
    list_dS_density =[]
    
    for i in trange(N):
        v_list = []
        dSt_list = []
        f = f_init+i*(f_fin-f_init)/N
        
        
        RTP.F = f*RTP.u/RTP.mu
        RTP.set_zero()
        
        for _ in trange(100000):
            RTP.time_evolve()
        RTP.compute=True
        for _ in range(1000):
            RTP.time_evolve()
            v_list.append(RTP.v)
            dSt_list.append(RTP.dS1+RTP.dS2)
        
        
        
        v_density = pd.DataFrame({'f':f,'v_list':v_list})
        list_v_density.append(v_density)
        dS_density = pd.DataFrame({'f':f,'dSt_list':dSt_list})
        list_dS_density.append(dS_density)
                              
    v_densities = pd.concat(list_v_density)
    dS_densities = pd.concat(list_dS_density)
    
    v_densities.to_pickle(state+'_v.pkl')
    dS_densities.to_pickle(state+'_dS.pkl')
    

    
# def autocorr(x):
#     (a,b) = x.shape
#     corr = np.zeros((a,b))
    
#     z_max = 0
#     for i in trange(a):
#         result = np.correlate(x[i],x[i],mode='full')
#         z = result[result.size//2:]
#         z_max = np.max(z_max,float(z.max()))
#         corr[i]=z
    
#     return corr/z_max
    
    
def anomalous(f,duration, N_ptcl):
    plt.clf()
#     a=0.7   #fc = 0.65
    a=0.9 # fc = 0.77
    Fs=2000
    
    RTP = RTP_lab(alpha=1, u=10, len_time=100, N_time=Fs,N_X=5, N_ptcl=N_ptcl, v=0, mu=1, muw = 1)
    RTP.compute = False
    RTP.l = 30
    RTP.L = 300
    RTP.u = a*RTP.l*RTP.alpha/2
    RTP.F = f*RTP.u/RTP.mu
    rho = 1
    RTP.muw = 1*rho*RTP.L/RTP.N_ptcl
    RTP.set_zero()
    
    v_traj = np.empty((RTP.N_X,duration))
    time = (np.arange(duration)+1)*RTP.delta_time
    
    for _ in trange(2000):
        RTP.time_evolve()
    for i in trange(duration):
        RTP.time_evolve()
        v_traj[:,i] = RTP.v/RTP.u
    
    
    
    # autocorr
    

    autov = np.zeros(v_traj.shape)
    
    for i in range(duration-1):
        autov[:,i] = np.average((v_traj[:,i+1:]-np.average(v_traj[:,i+1:]))*(v_traj[:,:-i-1]-np.average(v_traj[:,:-i-1])),axis=1)/np.average(v_traj**2,axis=1)
        
#     try:
#         m, c = np.polyfit(np.log(time[:int(duration/10)]), np.log(autov[:int(duration/10)]), 1) # fit log(y) = m*log(x) + c
#         y_fit = np.exp(m*np.log(time) + c) # calculate the fitted values of y
#         plt.plot(time, y_fit, ':',label='slope : ' + str(m))
#         # your code that will (maybe) throw
#     except np.linalg.LinAlgError as e:
#         pass

    plt.subplot(1,3,1)
    
    for i in range(len(autov)):
        plt.scatter(time, autov[i],s=1)


    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(0.01,2)
    plt.xlabel('t')
    plt.ylabel('corr')
    plt.grid()
#     plt.legend()
    plt.title('f :'+str(f))


    plt.subplot(1,3,2)
    
    for i in range(len(autov)):
        plt.scatter(time, autov[i],s=1)


    plt.yscale('log')
#     plt.xscale('log')
    plt.ylim(0.01,2)
    plt.xlim(0,1)
    plt.xlabel('t')
    plt.ylabel('corr')
    plt.grid()
#     plt.legend()
    plt.title('f :'+str(f))
    
    
    # diffusion
    plt.subplot(1,3,3)
    disp = np.cumsum(v_traj,axis=1)
    msd = np.zeros(v_traj.shape)
    
    for i in range(duration-1):
        msd[:,i] = np.average((disp[:,i+1:]-disp[:,:-i-1])**2,axis=1)
        
        
#     diff = np.average(np.cumsum(v_traj,axis=1)**2,axis=0)
    
#     try:
#         m, c = np.polyfit(np.log(time[:int(duration/10)]), np.log(diff[:int(duration/10)]), 1) # fit log(y) = m*log(x) + c
#         y_fit = np.exp(m*np.log(time) + c) # calculate the 
#         plt.plot(time, y_fit, ':',label='slope : ' + str(m))

#         # your code that will (maybe) throw
#     except np.linalg.LinAlgError as e:
#         pass
    for i in range(len(v_traj)):
        plt.plot(time[:-1], msd[i,:-1])
    
    plt.xlabel('t')
    plt.ylabel('<x^2>')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
#     plt.legend()
    plt.title('f :'+str(f))
    plt.savefig('image/f='+str(f)+'.png')