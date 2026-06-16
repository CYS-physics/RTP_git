import os

k_s = 120#, 150,60,40,30,20]
N_X = 8
out_root = 'data/kv_under2'
M_list = [10,100,1000,10000]#1000000,100000,10000,1000]#10000,100000]#,10,100,1000]
v_list = [0.3,0.6,1.0,1.3,1.6,2.0,2.3,2.6,3.0,3.3,3.6,4.0,4.3,4.6,5.0,5.5]#[1.0,2.0,3.0,4.0,6.0]#4.3,4.6,5,5.3,5.6]#[3,3.3,3.6,4,4.3,4.6]#[0.3,0.6,1.3,1.6]#[2.3,2.6,3.2,3.5]#[1]#2,3,3.5,4, 5]
N_list = [320000,160000,80000]#40000,20000,10000]#320000,160000,80000]#,160000,80000]#40000,80000]#,160000,320000]#[5000,10000,20000,40000,80000,160000]
file = open('jobs.txt','a')
for M in M_list:
    for v in v_list:
        for i in range(3):
            for N in N_list:
                # direc = 'data/kv/N='+str(N)
                direc = out_root+'/M='+str(M)+'/N='+str(N)
                state = os.getcwd()+'/'+direc+'/'+str(k_s)+'_'+str(v)+'_'+str(i)+'.npz'
                if os.path.exists(state):
                    continue
                else:
                    file.write('/pds/pds21/yunsik/miniconda3/bin/python run_kv.py %f %f %i %i %f %i %s \n' %(k_s, v, i,N,M,N_X,out_root)) 
