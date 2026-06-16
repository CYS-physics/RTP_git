v_list = [4,5]#[1,2,3,3.5]#,4, 5]
T_list = [2500,5000,10000,20000,40000]
file = open('jobs.txt','a')
for T in T_list:
    for v in v_list:
        for i in range(8):
            file.write('/pds/pds21/yunsik/miniconda3/bin/python run_saw.py %i %f %i \n' %(T, v, i)) 
