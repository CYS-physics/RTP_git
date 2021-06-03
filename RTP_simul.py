# simulation model of RTP
# symmetry breaking motility
# Seoul nat'l university
# Yongjoo Baek, Ki-won Kim and Yunsik Choe
# numerics done by Yunsik Choe
# 200920
# private

import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
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
        #output[A]+=self.F/2*(1+(y[A]-self.l/2)/d)
        output[B]+=self.F
        output[C]-=self.F
        output[D]-=self.F/(1+np.exp(self.k*(y[D]-self.l/2)))
        output[E]+=self.F* (  np.exp(-self.k*(y[E])) - np.exp(self.k*(y[E]))  )/(   np.exp(-self.k*(y[E]))  +  np.exp(self.k*(y[E]))  )
        #output[D]+=self.F/2*(1-(y[D]+self.l/2)/d)
        
        return output
        
        #return self.F*(-self.l/2+d/k<=y)*(y<0)  -   self.F*(0<y)*(y<=self.l/2-d/k)    +  self.F/(1+np.exp(-k*(y+self.l/2))) * (-self.l/2-d/k<y)*(y<-self.l/2+d/k)             -    self.F/(1+np.exp(k*(y-self.l/2)))   * (self.l/2-d/k<y)*(y<self.l/2+d/k)

        #return self.F/(1+np.exp(-k*(y+self.l/2))) * (y<0)             -    self.F/(1+np.exp(k*(y-self.l/2)))   * (0<y)
           # sigmoid approximation
        

    
    
        
        
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.time = np.linspace(0,self.len_time,num=self.N_simul)
        self.s = np.random.choice([1,-1],np.array([self.N_ptcl,self.N_X]))             # random direction at initial time
        self.x = np.random.uniform(-self.L/2, self.L/2, size=np.array([self.N_ptcl,self.N_X]))     # starting with uniformly distributed particles
        self.X = np.zeros(self.N_X)
        self.v = np.zeros(self.N_X)
    
    def tumble(self):             # random part of s dynamics
        tumble = np.random.choice([1,-1], size=np.array([self.N_ptcl,self.N_X]), p = [1-self.delta_time*self.alpha/2, self.delta_time*self.alpha/2]) # +1 no tumble, -1 tumble
        return tumble
    
    def dynamics(self,x,s):         # dynamics of x and s to give time difference of x and s
        dxa = self.u*self.delta_time
        ds = self.tumble()
        
        
#         sm = (s+s*ds)/2    # s at mid-time
#         sf = s*ds          # s ar final
        
#         # RK4 method to compute wall repulsion
#         dxp1 = (-self.mu*self.partial_V(x))*self.delta_time
#         x1 = x + 0.5*(dxa*sm+dxp1)
#         dxp2 = (-self.mu*self.partial_V(x1))*self.delta_time
#         x2 = x + 0.5*(dxa*sm+dxp2)
#         dxp3 = (-self.mu*self.partial_V(x2))*self.delta_time
#         x3 = x + (dxa*sf+dxp3)
#         dxp4 = (-self.mu*self.partial_V(x3))*self.delta_time
        
#         dx = dxa*sm + 1/6 *(1*dxp1+2*dxp2+2*dxp3+1*dxp4)
        
        
        
#         v = -self.muw/self.mu*np.sum(1/6*(dxp1+ 2*dxp2 + 2*dxp3 + dxp4) ,axis=0)/self.delta_time

        dx = dxa*s  -self.mu*self.partial_V(x)*self.delta_time
        if self.model==3:
            v = self.muw*np.sum(self.partial_V(x),axis=0)
        elif self.model ==2:
            v = self.v




        return (v, dx, ds)
        
    def time_evolve(self):
        (v, dx, ds) = self.dynamics(self.x, self.s)
        self.v = v
        
#         y=self.periodic(self.x-self.X)
#         self.current =np.sum(self.s*(y>=-self.l/2-self.d/self.k)*(y<=self.l/2+self.d/self.k),axis=0)         # difference of right moving and left moving active particle numbers in contact with object, see for mail from YB Sept,20, 20
        
#         self.LR =np.sum(self.s*(y>=-self.l/2+self.d/self.k)*(y<=0)*(self.s==1),axis=0)  # left side right moving
#         self.LL =np.sum(self.s*(y>=-self.l/2+self.d/self.k)*(y<=0)*(self.s==-1),axis=0)
#         self.LD =np.sum(self.s*(y>=-self.l/2-self.d/self.k)*(y<=-self.l/2+self.d/self.k)*(self.s==1),axis=0)
#         self.RR =np.sum(self.s*(y>=0)*(y<=self.l/2-self.d/self.k)*(self.s==1),axis=0)
#         self.RL =np.sum(self.s*(y>=0)*(y<=self.l/2-self.d/self.k)*(self.s==-1),axis=0)
#         self.RD =np.sum(self.s*(y>=self.l/2-self.d/self.k)*(y<=self.l/2+self.d/self.k)*(self.s==-1),axis=0)


        # coherence measure
#         phi = 2*np.pi*(self.x/self.L)
        
#         phi_x = np.average(np.cos(phi)*(np.abs(self.partial_V(self.x))),axis=0)
#         phi_y = np.average(np.sin(phi)*(np.abs(self.partial_V(self.x))),axis=0)
#         self.co_r = np.sqrt(phi_x**2+phi_y**2)
#         self.co_phi = np.arctan2(phi_y,phi_x)
        
        
        
        
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
    
    
    
def measure(ptcl, number_X,L, f_init,f_fin,f_step, t_step):
    
    for i in trange(f_step):
        f = f_init + (f_fin-f_init)*i/f_step
        state = os.getcwd()+'/data/'+"ptcl"+str(ptcl)+"X"+str(number_X)+"L"+str(L)+"f"+str(f)+'.npz'
        time_moments(ptcl, number_X,L,f,t_step,state)
       
    
def simulate(N, L, l, a, f,duration,Fs, name):
    state = os.getcwd()+'/data/away/210215/'+str(name)+'.npz'
    os.makedirs(os.getcwd()+'/data/away/210215/'+str(N),exist_ok=True)
    
    RTP = RTP_lab(alpha=0.5, u=10, len_time=100, N_time=Fs,N_X=1, N_ptcl=N, v=0, mu=1, muw = 1)
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
        
        X_list[i] = RTP.X
        v_list[i] = RTP.v

        
        
    save_dict={}
    save_dict['X'] = X_list
    save_dict['v'] = v_list
#     save_dict['co_r'] = co_r_list
#     save_dict['co_phi'] = co_phi_list

    save_dict['muw'] = RTP.muw
#     save_dict['current'] = pd.concat(current_list)
    save_dict['Fs'] = RTP.N_time
    save_dict['description'] = 'L : '+str(RTP.L)+', N : '+str(RTP.N_ptcl)+', f : '+str(f) + 'a :'+str(a)

    
    
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
        moments(N_ptcl, L, l, a, f,1*rho*L/N_ptcl, 50000,Fs, name)
    
def simul_scan(f_init, f_fin, N, N_ptcl):
    for i in trange(N):
        f=f_init+i*(f_fin-f_init)/N
        simulate(N_ptcl,300,30,1,f,1000000,100,str(N_ptcl)+'/'+str(f))
        
        
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
        n,bins,patches = plt.hist(R1.v.flatten()/R1.u, bins = binning, density = True)
        for i in trange(9999):
            R1.time_evolve()
            n_temp,__,_ = plt.hist(R1.v.flatten()/R1.u, bins = binning, density = True)
            plt.clf()
            n+=n_temp
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
        