import numpy as np
from RTP_simul import RTP_lab
import os
import sys

k_s = float(sys.argv[1])
v = float(sys.argv[2])
seed = int(sys.argv[3])
N = int(sys.argv[4])
M = float(sys.argv[5])
N_X = int(sys.argv[6]) if len(sys.argv) > 6 else 10
out_root = sys.argv[7] if len(sys.argv) > 7 else 'data/kv_under2'

np.random.seed(seed)
direc = out_root+'/M='+str(M)+'/N='+str(N)
os.makedirs(direc,exist_ok=True)

state = os.path.join(direc, str(k_s)+'_'+str(v)+'_'+str(seed)+'.npz')

R1 = RTP_lab(vs=v,model=5)
R1.N_ptcl = N
R1.N_X = N_X
R1.L = 200
R1.M = M
R1.k_s = k_s#*N/40000
R1.F = R1.F*N/40000
R1.set_zero()
X_traj = []
X_s_traj = []
N_relax = 10000
N_measure = 20000
N_skip = 5
for _ in range(N_measure):
    for _ in range(N_skip):
        R1.time_evolve()
    X_traj.append(R1.X.copy())
    X_s_traj.append(R1.X_s.copy())

save_dict = {}
save_dict['time'] = np.arange(N_measure)*R1.delta_time*N_skip
save_dict['X_traj'] = np.array(X_traj)
save_dict['X_s_traj'] = np.array(X_s_traj)
save_dict['k_s'] = k_s
save_dict['v'] = v
save_dict['L'] = R1.L
save_dict['N'] = N
save_dict['M'] = M
save_dict['N_X'] = N_X
save_dict['seed'] = seed
save_dict['N_skip'] = N_skip
save_dict['N_measure'] = N_measure
np.savez(state,**save_dict)
