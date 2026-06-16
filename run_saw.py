import numpy as np
from RTP_simul import RTP_lab
import os
import sys

k_s = 100
T = int(sys.argv[1])
v = float(sys.argv[2])
seed = int(sys.argv[3])
N = 80000

np.random.seed(seed)
direc = 'data/saw2/T='+str(T)+',v='+str(v)
os.makedirs(direc,exist_ok=True)

state = os.getcwd()+'/'+direc+'/'+str(T)+'_'+str(v)+'_'+str(seed)+'.npz'

R1 = RTP_lab(v=v,model=5)
R1.N_ptcl = N
R1.L = 200
R1.k_s = k_s
R1.set_zero()
X_traj = []
X_s_traj = []
N_relax = 5*T
N_measure = 30*T
N_skip = 5
for i in range(N_relax):
    for _ in range(N_skip):
        R1.time_evolve()
    if i%T == 0:
        R1.v = -R1.v

for i in range(N_measure):
    for _ in range(N_skip):
        R1.time_evolve()
    if i%T == 0:
        R1.v = -R1.v
    X_traj.append(R1.X)
    X_s_traj.append(R1.X_s)

save_dict = {}
save_dict['time'] = np.arange(N_measure)*R1.delta_time*N_skip
save_dict['X_traj'] = np.array(X_traj)
save_dict['X_s_traj'] = np.array(X_s_traj)
save_dict['k_s'] = k_s
save_dict['v'] = v
save_dict['L'] = R1.L
save_dict['N'] = N
np.savez(state,**save_dict)
