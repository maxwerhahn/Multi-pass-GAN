import os
#import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# for i in range(0,1):
	# os.system('manta gen_sim_data_obstacle.py steps 200 saveuni 1 reset 1 obstacles 1 spheres 0 warmup 150')
	# os.system('manta gen_sim_data_obstacle.py steps 200 saveuni 1 reset 1 obstacles 1 spheres 0 warmup 100')
# for i in range(0,1):
	# os.system('manta gen_sim_data_obstacle.py steps 200 saveuni 1 reset 1 obstacles 1 spheres 0 warmup 20')
	
#for i in range(0,1):
	#os.system('manta gen_sim_data.py saveuni 1 reset 1 steps 200 gui 0 fac 8')
	
#for i in range(0,40):
	#os.system('manta gen_data_wind_tunnel.py saveuni 1 gui 0')
#for i in range(0,10):
#	os.system('manta gen_sim_data.py saveuni 1 reset 1 steps 200 gui 0')
#seeds = []
#for j in range(40):
#	seeds.append((np.random.randint(10000000)))
# for i in range(0,40):
	# os.system('manta manta_genSimData_growing_obs.py maxUpRes 16 minUpRes 2 resetN 1 npSeed ' + str(int(seeds[i])))
for i in range(0,4):
	os.system('manta gen_sim_grow_slices_data.py maxUpRes 8 minUpRes 2 reset 1 saveuni 1 obstacles 1')

#for i in range(0,3):
	#os.system('manta manta_genSimData3.py maxUpRes 8 minUpRes 2 resetN 1 npSeed ' + str(int(seeds[i+8])))
