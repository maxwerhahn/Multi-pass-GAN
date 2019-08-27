#******************************************************************************
#
# tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
# Copyright 2018 You Xie, Erik Franz, Mengyu Chu, Nils Thuerey, Maximilian Werhahn
#
#******************************************************************************

import time
import shutil
import sys
import math
import gc
import scipy
import numpy as np
# load manta tools
sys.path.append("../tools_wscale")
import tilecreator_t as tc
import uniio
import paramhelpers as ph
from GAN import GAN, lrelu
#from skimage.measure import compare_psnr
import fluiddataloader as FDL
import os
import tensorflow as tf
from tensorflow.python.client import timeline

# ---------------------------------------------y

# initialize parameters / command line params
outputOnly	  = int(ph.getParam( "out",			 False ))>0 		# output/generation mode, main mode switch

basePath		=	 ph.getParam( "basePath",		'../2ddata_gan/' )
randSeed		= int(ph.getParam( "randSeed",		1 )) 				# seed for np and tf initialization
load_model_test = int(ph.getParam( "load_model_test", -1 )) 			# the number of the test to load a model from. can be used in training and output mode. -1 to not load a model
load_model_no   = int(ph.getParam( "load_model_no",   -1 )) 			# nubmber of the model to load

simSizeLow  	= int(ph.getParam( "simSize", 		  64 )) 			# tiles of low res sim
tileSizeLow 	= int(ph.getParam( "tileSize", 		  16 )) 			# size of low res tiles
upRes	  		= int(ph.getParam( "upRes", 		  4 )) 				# scaling factor

#Data and Output
packedSimPath		 =	 ph.getParam( "packedSimPath",		 '/data/share/GANdata/2ddata_sim/' ) 	# path to training data
fromSim		 = int(ph.getParam( "fromSim",		 1000 )) 			# range of sim data to use, start index
toSim		   = int(ph.getParam( "toSim",		   -1   )) 			# end index
dataDimension   = int(ph.getParam( "dataDim",		 2 )) 				# dimension of dataset, can be 2 or 3. in case of 3D any scaling will only be applied to H and W (DHW)
numOut			= int(ph.getParam( "numOut",		  200 )) 			# number ouf images to output (from start of sim)
saveOut	  	= int(ph.getParam( "saveOut",		 False ))>0 		# save output of output mode as .npz in addition to images
loadOut			= int(ph.getParam( "loadOut",		 -1 )) 			# load output from npz to use in output mode instead of tiles. number or output dir, -1 for not use output data
outputImages	=int(ph.getParam( "img",  			  True ))>0			# output images
outputGif		= int(ph.getParam( "gif",  			  False ))>0		# output gif
outputRef		= int(ph.getParam( "ref",			 False ))>0 		# output "real" data for reference in output mode (may not work with 3D)
#models
frame_min		= int(ph.getParam( "frame_min",		   0 ))
genModel		 =	 ph.getParam( "genModel",		 'gen_test' ) 	# path to training data
discModel		=	 ph.getParam( "discModel",		 'disc_test' ) 	# path to training data
#Training
learning_rate   = float(ph.getParam( "learningRate",  0.0002 ))
decayLR		= int(ph.getParam( "decayLR",			 False ))>0 		# decay learning rate?
dropout   		= float(ph.getParam( "dropout",  	  1.0 )) 			# keep prop for all dropout layers during training
dropoutOutput   = float(ph.getParam( "dropoutOutput", dropout )) 		# affects testing, full sim output and progressive output during training

beta			= float(ph.getParam( "adam_beta1",	 0.5 ))			#1. momentum of adam optimizer
beta2			= float(ph.getParam( "adam_beta2",	 0.999 ))			#1. momentum of adam optimizer

weight_dld		= float(ph.getParam( "weight_dld",	1.0)) 			# ? discriminator loss factor ?
k				= float(ph.getParam( "lambda",		  1.0)) 			# influence/weight of l1 term on generator loss
k2				= float(ph.getParam( "lambda2",		  0.0)) 			# influence/weight of d_loss term on generator loss
k_f				= float(ph.getParam( "lambda_f",		  1.0)) 			# changing factor of k
k2_f			= float(ph.getParam( "lambda2_f",		  1.0)) 			# changing factor of k2
k2_l1		   = float(ph.getParam( "lambda2_l1",		   1.0))						 # influence/weight of L1 layer term on discriminator loss
k2_l2		   = float(ph.getParam( "lambda2_l2",		   1.0))						 # influence/weight of L2 layer term on discriminator loss
k2_l3		   = float(ph.getParam( "lambda2_l3",		   1.0))						 # influence/weight of L3 layer term on discriminator loss
k2_l4		   = float(ph.getParam( "lambda2_l4",		   1.0))						 # influence/weight of L4 layer term on discriminator loss
#useTempoD	   = int(ph.getParam( "useTempoD",		   True ))		#apply disc time loss or not
#useTempoL2	   = int(ph.getParam( "useTempoL2",		   False ))		#apply l2 time loss or not
kt			  = float(ph.getParam("lambda_t", 1.0))				    # tempo discriminator loss; 1.0 is good, 0.0 will disable
kt_l		  = float(ph.getParam("lambda_t_l2", 0.0))				# l2 tempo loss (as naive alternative to discriminator); 1.0 is good, 0.0 will disable
batch_size	  = int(ph.getParam( "batchSize",  	  128 ))			# batch size for pretrainig and output, default for batchSizeDisc and batchSizeGen
batch_size_disc = int(ph.getParam( "batchSizeDisc",   batch_size )) 	# batch size for disc runs when training gan
batch_size_gen  = int(ph.getParam( "batchSizeGen",	batch_size )) 	# batch size for gen runs when training gan
trainGAN		= int(ph.getParam( "trainGAN",   	  True ))>0 		# GAN trainng can be switched off to use pretrainig only
trainingIterations  = int(ph.getParam( "trainingIterations",  100000 )) 		# for GAN training
discRuns 		= int(ph.getParam( "discRuns",  	  1 )) 				# number of discrimiinator optimizer runs per it
genRuns  		= int(ph.getParam( "genRuns",  		  1 )) 				# number or generator optimizer runs per it
batch_norm		= int(ph.getParam( "batchNorm",	   False ))>0			# apply batch normalization to conv and deconv layers
pixel_norm		= int(ph.getParam( "pixelNorm",	   True ))>0			# apply batch normalization to conv and deconv layers
bn_decay		= float(ph.getParam( "bnDecay",	   0.999 ))			# decay of batch norm EMA

useVelocities   = int(ph.getParam( "useVelocities",   0  )) 			# use velocities or not
useVorticities  = int(ph.getParam( "useVorticities",   0  )) 			# use vorticities or not
useFlags   = int(ph.getParam( "useFlags",   0  )) 			# use flags or not
useK_Eps_Turb = int(ph.getParam( "useK_Eps_Turb",   0  ))
premadeTiles	= int(ph.getParam( "premadeTiles",   0  ))		 		# use pre-made tiles?

useDataAugmentation = int(ph.getParam( "dataAugmentation", 0 ))		 # use dataAugmentation or not
minScale = float(ph.getParam( "minScale",	  0.85 ))				 # augmentation params...
maxScale = float(ph.getParam( "maxScale",	  1.15 ))
rot	 = int(ph.getParam( "rot",		  2	 ))		#rot: 1: 90 degree rotations; 2: full rotation; else: nop rotation 
transposeAxis	 = int(ph.getParam( "transposeAxis",		  0	 ))		#rot: 1: 90 degree rotations; 2: full rotation; else: nop rotation 

#minAngle = float(ph.getParam( "minAngle",	 -90.0  ))
#maxAngle = float(ph.getParam( "maxAngle",	  90.0  ))
flip	 =   int(ph.getParam( "flip",		  1	 ))

#Pretraining
pretrain		= int(ph.getParam( "pretrain",		0 )) 				# train generator with L2 loss before alternating training, number of epochs
pretrain_disc	= int(ph.getParam( "pretrainDisc",   0 )) 				# train discriminator before alternating training
pretrain_gen	= int(ph.getParam( "pretrainGen",	0 ))				# train generator using pretrained discriminator before alternating training

#Test and Save
testPathStartNo = int(ph.getParam( "testPathStartNo", 0  ))
testInterval	= int(ph.getParam( "testInterval", 	  100  )) 			# interval in epochs to run tests should be lower or equal outputInterval
numTests		= int(ph.getParam( "numTests", 		  batch_size_disc )) 			# number of tests to run from test data each test interval, run as batch
outputInterval	= int(ph.getParam( "outputInterval",  100  ))			# interval in epochs to output statistics
saveInterval	= int(ph.getParam( "saveInterval",	  200  ))	 		# interval in epochs to save model
alwaysSave	  = int(ph.getParam( "alwaysSave",	  True  )) 			#
maxToKeep		= int(ph.getParam( "keepMax",		 3  )) 			# maximum number of model saves to keep in each test-run
genTestImg		= int(ph.getParam( "genTestImg",	  -1 )) 			# if > -1 generate test image every output interval
note			= ph.getParam( "note",		   "" )					# optional info about the current test run, printed in log and overview
data_fraction	= float(ph.getParam( "data_fraction",		   0.3 ))
frame_max		= int(ph.getParam( "frame_max",		   200 ))
ADV_flag		= int(ph.getParam( "adv_flag",		   True )) # Tempo parameter, add( or not) advection to pre/back frame to align
ADV_mode		= int(ph.getParam( "adv_mode",		   1 )) # Tempo parameter, mode == 0: best working - cpu sided however..., mode == 1: semi-lagr advection, mode == 2: maccormack advection
change_velocity		= int(ph.getParam( "change_velocity",		   False )) 
saveMD       = int(ph.getParam( "saveMetaData", 0 ))      # profiling, add metadata to summary object? warning - only main training for now

use_spatialdisc		= int(ph.getParam( "use_spatialdisc",		   True )) #use spatial discriminator or not
velScale = float(ph.getParam("velScale", 1.0))	# velocity scale for output

# general params for architecture
upsampling_mode = int(ph.getParam( "upsamplingMode",   2 ))	# see further below...
upsampled_data = int(ph.getParam ( "upsampledData", False))	# use upsampled data ? - only for the second or third generator
generateUni = int(ph.getParam("genUni", False))		# generate uni files ?
upsampleFirst = int(ph.getParam("upsampleFirst", True))	# upsample volume before applying the first network?
usePixelShuffle = int(ph.getParam("usePixelShuffle", False))	# pixel shuffle as upsample method in the network (instead of avg_depool)
addBicubicUpsample = int(ph.getParam("addBicubicUpsample", False))	# residual learning?
startingIter = int(ph.getParam("startingIter", 0))	# starting it for loading old models and further training
load_emas = int(ph.getParam("loadEmas", False))		# loading estimated moving averages to generate outputs (not much difference from normal models)
useVelInTDisc = int(ph.getParam("useVelInTDisc", False))	# use the low-res velocity in the temporal discriminator...
upsampleMode = int(ph.getParam("upsampleMode", 1))  # upsample operation in growing networks: 1 -> nn interpolation, 2 -> lin interp.

# parameters for growing approach
use_loss_scaling = int(ph.getParam("lossScaling", False))
stageIter = int(ph.getParam("stageIter", 25000))		# amout of iterations per growing stage, e.g. when targeting 8x upres -> stageIter iter. blend in stage, stageIter iter. stabilization stage leading to 6*stageIter epochs overall
decayIter = int(ph.getParam("decayIter", 25000))	# amout of epochs during which the learning rate decays
max_fms = int(ph.getParam("maxFms", 256))		# maximum amount of feature maps for each conv layer in the growing network
use_wgan_gp = int(ph.getParam( "use_wgan_gp",		   False ))	# use gradient penalty + wgan
use_res_net = int(ph.getParam( "use_res_net",		   False )) # res net architecture for generator
use_mb_stddev = int(ph.getParam( "use_mb_stddev",		   False ))	# mini batch standard deviation in the discriminator
use_LSGAN = int(ph.getParam( "use_LSGAN",		   False )) # least squares GAN as loss
start_fms = int(ph.getParam("startFms", 512))		# amount of feature maps at the start of the generator/ end of discriminators
filterSize = int(ph.getParam("filterSize", 3))		# filter size for each conv layer 
outNNTestNo = int(ph.getParam("outNNTestNo", 17)) # should only be used for the second/third network -> specifies the data names
first_nn_arch = int(ph.getParam("firstNNArch", False)) # true = use multiplicative gaussian noise for the discriminators
use_gdrop = int(ph.getParam("gDrop", False)) # true = use multiplicative gaussian noise for the discriminators
add_adj_idcs = int(ph.getParam("add_adj_idcs", False))	# add adjacent slices to input frames

gpu_touse = int(ph.getParam("gpu", 2)) # gpu to use

# overall training iterations = stageIter * 6 because of 3 training stages, each stage with stageIter for blending + stabiliztion
trainingIterations = stageIter * 6 + decayIter


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu_touse)

# not used for wgan-gp...
gdrop_beta              = 0.9
gdrop_lim               = 0.5
gdrop_coef              = 0.2
gdrop_exp               = 2.0

# parameters for LSGAN and WGAN (+GP)
if use_LSGAN: # or -1., 1., 0.
	a = 0.
	b = 1.
	c = 1.
if use_wgan_gp:
	if not use_LSGAN:		
		wgan_lambda     = 10
		wgan_target     = 1.0 
		wgan_epsilon	= 0.001
	else:		
		# usage of gradient penalty with lsgan (not wgan)
		wgan_lambda     = 150
		wgan_target     = 30.0
		wgan_epsilon	= 0.001

ph.checkUnusedParams()

useTempoD = False
useTempoL2 = False
if(kt > 1e-6):
	useTempoD = True
if(kt_l > 1e-6):
	useTempoL2 = True
if(kt > 1e-6 and kt_l > 1e-6):
	print("please choose right temporal loss!")
	exit(1)
# initialize
simSizeHigh 	= simSizeLow * upRes
tileSizeHigh	= tileSizeLow  * upRes

if not (dataDimension == 2 or dataDimension == 3):
	print('Unsupported data dimension {}. Only 2 and 3 are supported'.format(dataDimension))
	exit(1)

if toSim==-1:
	toSim = fromSim

if outputOnly:
	currentUpres = upRes
else:
	if upsampling_mode == 2:
		currentUpres =  min(2 ** (startingIter // (stageIter * 2)+1),8)
	elif upsampling_mode == 3 or upsampling_mode == 0 or upsampling_mode == 1:
		currentUpres = 8
		
channelLayout_low = 'd'
channelLayout_high = 'd'

lowfilename = "density_low_%04d.uni"
'''
 upsampling mode
 0: lin interp after first network - second network "upsampling" z-axis,
 1: lin interp before first network - second network "refining" along y/z-axis, 
 2: first network upsampling along x and y axis,
 3: third network refining along x and z axis
'''

if upsampled_data:
	select_random_data = 0.2 # random shuffling + selecting "select_random_data" slices 
	if upsampling_mode == 1:
		lowfilename_2 = "density_low_t%04d_2x2" % (outNNTestNo) + "_%04d.uni"
	elif upsampling_mode == 0:
	    lowfilename_2 = "density_low_t%04d_2x2x1" % (outNNTestNo) + "_%04d.uni"
	elif upsampling_mode == 3:
	    lowfilename_2 = "density_low_t%04d_1x1" % (outNNTestNo) + "_%04d.uni"
else:
	select_random_data = 0.4
	
min_data_fraction = 0.08 # minimum data fraction (mostly only used for full network (8x scaling)
	
highfilename = "density_high_%04d.uni"
mfl = ["density"]
mfh = ["density"]
	
# load output of first network in high res data of tile creator -> separate when getting input	
if upsampling_mode == 1 or upsampling_mode == 3:
	channelLayout_high = 'd,d'
	channelLayout_low = 'd'
	
if useVelocities:
	channelLayout_low += ',vx,vy,vz'
	mfl= np.append(mfl, "velocity")	
	
if add_adj_idcs:
	channelLayout_low += ',d,d'

if useK_Eps_Turb or useFlags:
	print('Flags and k/eps input fields not supported yet')		
	exit(1)
	
dirIDs = np.linspace(fromSim, toSim, (toSim-fromSim+1),dtype='int16')

if (outputOnly): 
	data_fraction = 1.0
	kt = 0.0
	kt_l = 0.0
	useTempoD = False
	useTempoL2 = False
	useDataAugmentation = 0

if ((not useTempoD) and (not useTempoL2)): # should use the full sequence, not use multi_files
	tiCr = tc.TileCreator(tileSizeLow=tileSizeLow, simSizeLow=simSizeLow , dim =dataDimension, dim_t = 1, channelLayout_low = channelLayout_low, upres=upRes, premadeTiles=premadeTiles, channelLayout_high = channelLayout_high)
	floader = FDL.FluidDataLoader( print_info=3, base_path=packedSimPath, base_path_y = packedSimPath, numpy_seed = randSeed ,filename=lowfilename, filename_index_min = frame_min, oldNamingScheme=False, filename_y = None, filename_index_max=frame_max, indices=dirIDs, data_fraction=data_fraction, multi_file_list=mfl, multi_file_list_y=mfh)
	if upsampled_data:
		mfl_2 = ["density"]
		mfh_2 = mfl_2
		floader_2 = FDL.FluidDataLoader( print_info=0,base_path=packedSimPath, numpy_seed = randSeed, filename=lowfilename_2, filename_index_min = frame_min, oldNamingScheme=False, filename_index_max=frame_max, indices=dirIDs, data_fraction=data_fraction, multi_file_list=mfl_2)
else:		
	lowparalen = len(mfl)
	highparalen = len(mfh)
	mfl_tempo= np.append(mfl, mfl)
	mfl= np.append(mfl_tempo, mfl)
	mol = np.append(np.zeros(lowparalen), np.ones(lowparalen))
	mol = np.append(mol, np.ones(lowparalen)*2)
	
	mfh_tempo = np.append(mfh, mfh)
	mfh= np.append(mfh_tempo, mfh)
	moh = np.append(np.zeros(highparalen), np.ones(highparalen))
	moh = np.append(moh, np.ones(highparalen)*2)
	if upsampling_mode == 2:
		tiCr = tc.TileCreator(tileSizeLow=tileSizeLow, densityMinimum=0.002, channelLayout_high=channelLayout_high, simSizeLow=simSizeLow , dim =dataDimension, dim_t = 3, channelLayout_low = channelLayout_low, upres=currentUpres, premadeTiles=premadeTiles)
	elif upsampling_mode == 3 or upsampling_mode == 1:
		tiCr = tc.TileCreator(tileSizeLow=tileSizeLow, densityMinimum=0.002, channelLayout_high=channelLayout_high, simSizeLow=simSizeLow, dim =dataDimension, dim_t = 3, channelLayout_low = channelLayout_low, upres=currentUpres, premadeTiles=premadeTiles)
	elif upsampling_mode == 0:
		tiCr = tc.TileCreator(tileSizeLow=[tileSizeHigh, tileSizeLow], densityMinimum=0.002, channelLayout_high=channelLayout_high, simSizeLow=[simSizeHigh, simSizeLow] , dim =dataDimension, dim_t = 3, channelLayout_low = channelLayout_low, upres=[1,upRes], premadeTiles=premadeTiles)
	
	# set scale/transpose of data	
	if upsampling_mode == 1 or upsampling_mode == 3:
		scale_y = [1,1,1,1]
		scale = [currentUpres,1,1,1]
		if upsampling_mode == 3:
			transpose_axis = 1
		else:
			transpose_axis = 2
	elif upsampling_mode == 0:
		scale_y = [1,1,1,1]
		scale = [currentUpres,currentUpres,1,1]
		transpose_axis = 2
	else:
		scale_y = [1,1,1,1]
		scale = [currentUpres,1,1,1]
		transpose_axis = 0
	
	# load intermediate density fields (explicit curriculum learning)
	if currentUpres == upRes:
		currentHighFileName = highfilename
	else:
		currentHighFileName = "density_low_%i"%(currentUpres)+"_%04d.uni"
	
	# load output data of first or second network (for the second and third NN)
	if upsampled_data:	
		mfl_2 = ["density"]
		lowparalen_2 = len(mfl_2)		
		mfl_tempo_2= np.append(mfl_2, mfl_2)
		mfl_2= np.append(mfl_tempo_2, mfl_2)
		mol_2 = np.append(np.zeros(lowparalen_2), np.ones(lowparalen_2))
		mol_2 = np.append(mol_2, np.ones(lowparalen_2)*2)		
		floader_2 = FDL.FluidDataLoader( print_info=0, base_path=packedSimPath, add_adj_idcs = add_adj_idcs, base_path_y = packedSimPath, numpy_seed = randSeed, conv_slices = True, conv_axis = transpose_axis, select_random = select_random_data, density_threshold = 0.005, axis_scaling_y = scale_y, axis_scaling = scale, filename=lowfilename, oldNamingScheme=False, filename_y=lowfilename_2, filename_index_max=frame_max,filename_index_min = frame_min, indices=dirIDs, data_fraction=max(data_fraction*2/currentUpres,min_data_fraction), multi_file_list=mfl_2, multi_file_idxOff=mol_2, multi_file_list_y=mfh , multi_file_idxOff_y=moh) # data_fraction=0.1
		print('loaded low_res')
	
	# load low and high res data 

	floader = FDL.FluidDataLoader( print_info=0, base_path=packedSimPath, add_adj_idcs = add_adj_idcs, base_path_y = packedSimPath, numpy_seed = randSeed, conv_slices = True, conv_axis = transpose_axis, select_random = select_random_data, density_threshold = 0.005, axis_scaling_y = scale_y, axis_scaling = scale, filename=lowfilename, oldNamingScheme=False, filename_y=currentHighFileName, filename_index_max=frame_max,filename_index_min = frame_min, indices=dirIDs, data_fraction=max(data_fraction*2/currentUpres,min_data_fraction), multi_file_list=mfl, multi_file_idxOff=mol, multi_file_list_y=mfh , multi_file_idxOff_y=moh) # data_fraction=0.1
if useDataAugmentation:
	tiCr.initDataAugmentation(rot=rot, minScale=minScale, maxScale=maxScale ,flip=flip)

x, y, _  = floader.get()

floader = []
gc.collect()

#	density
n_inputChannels = 1

if useVelocities:
	n_inputChannels += 3
if useVorticities:
	n_inputChannels += 3	
if add_adj_idcs:				
	n_inputChannels += 2
	
if outputOnly:
	if upsampled_data:
		x_2, _, _ = floader_2.get()
	else:
		x_2 = None
	x_3d = x
	x_3d[:,:,:,:,1:4] = velScale * x_3d[:,:,:,:,1:4] # scale velocity channels
	y_3d = y
else:
	if upsampled_data:	
		_, x_2, _ = floader_2.get()

if not outputOnly:
	print('start converting to 2d slices')
	if upsampling_mode == 2 and not upsampled_data:
		x = x.reshape(-1, 1, simSizeLow, simSizeLow, n_inputChannels * 3)
		y = y.reshape(-1, 1, simSizeLow * currentUpres, simSizeLow * currentUpres, 3)	
	else:
		if upsampling_mode == 3 or upsampling_mode == 1:		
			x = x.reshape(-1, 1, simSizeLow, simSizeLow, 12)
			x_2 = x_2.reshape(-1, 1, simSizeHigh, simSizeHigh, 1)			
			y = y.reshape(-1, 1, simSizeHigh, simSizeHigh, 1)
			y = np.concatenate((y, x_2), axis = 4).reshape((-1, 1, simSizeHigh, simSizeHigh, 6))
		elif upsampling_mode == 0:			
			x = x.reshape(-1, 1, simSizeHigh, simSizeLow, 4)
			x_2 = x_2.reshape(-1, 1, simSizeHigh, simSizeLow, 1)			
			y = y.reshape(-1, 1, simSizeHigh, simSizeHigh, 3)
			x = np.concatenate((x,x_2), axis = 4).reshape((-1, 1, simSizeHigh, simSizeLow, 15))
			if 0:
				x = x.reshape(-1, 1, simSizeHigh, simSizeLow, 12)
				x_2 = x_2.reshape(-1, 1, simSizeHigh, simSizeLow, 3)
				for i in range(12):
					if i % 4 == 0:
						x[:,:,:,:,i:i+1] = x_2[:,:,:,:,i//4:i//4+1]
				y = y.reshape(-1, 1, simSizeHigh, simSizeHigh, 3)			
	print('done loading')

if not outputOnly:	
	tiCr.addData(x,y)
	
print("Random seed: {}".format(randSeed))
np.random.seed(randSeed)
tf.set_random_seed(randSeed)

# ---------------------------------------------

# 2D: tileSize x tileSize tiles; 3D: tileSize x tileSize x tileSize chunks
if upsampling_mode == 1 or upsampling_mode == 3 or upsampling_mode == 2:
	n_input = tileSizeLow  ** 2
elif upsampling_mode == 0:
	n_input = tileSizeHigh * tileSizeLow
	
n_output = tileSizeHigh ** 2

if dataDimension == 3:
	n_input  *= tileSizeLow
	n_output *= (tileSizeLow*upRes)

if upsampling_mode == 0:
	n_input *=( n_inputChannels+1)
else: 
	n_input *= n_inputChannels
			
# init paths
if not load_model_test == -1:
	if not os.path.exists(basePath + 'test_%04d/' % load_model_test):
		print('ERROR: Test to load does not exist.')
	if not load_emas:
		load_path = basePath + 'test_%04d/model_%04d.ckpt' % (load_model_test, load_model_no)
		load_path_ema = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test, load_model_no)
	else:
		load_path = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test, load_model_no)
	if outputOnly:
		out_path_prefix = 'out_%04d-%04d' % (load_model_test,load_model_no)
		test_path,_ = ph.getNextGenericPath(out_path_prefix, 0, basePath + 'test_%04d/' % load_model_test)

	else:
		test_path,_ = ph.getNextTestPath(testPathStartNo, basePath)

else:
	test_path,_ = ph.getNextTestPath(testPathStartNo, basePath)

# logging & info
sys.stdout = ph.Logger(test_path)
print('Note: {}'.format(note))
print("\nCalled on machine '"+ sys.platform[1] +"' with: " + str(" ".join(sys.argv) ) )
print("\nUsing parameters:\n"+ph.paramsToString())
ph.writeParams(test_path+"params.json") # export parameters in human readable format

if outputOnly:
	print('*****OUTPUT ONLY*****')

if not outputOnly:
	os.makedirs(test_path+"/zbu_src")
	uniio.backupFile(__file__, test_path+"/zbu_src/")
	uniio.backupFile("../tools_wscale/tilecreator_t.py", test_path+"/zbu_src/")
	uniio.backupFile("../tools_wscale/GAN.py", test_path+"/zbu_src/") 
	uniio.backupFile("../tools_wscale/fluiddataloader.py", test_path+"/zbu_src/")

# ---------------------------------------------
# TENSORFLOW SETUP

import scipy.misc

def save_img(out_path, img):
	img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
	scipy.misc.imsave(out_path, img)

def save_img_3d(out_path, img): # y ↓ x →， z ↓ x →, z ↓ y →，3 in a column
	data = np.concatenate([np.sum(img, axis=0), np.sum(img, axis=1), np.sum(img, axis=2)], axis=0)
	save_img(out_path, data)
	
#input for gen
bn=batch_norm
#training or testing for batch norm
train = tf.placeholder(tf.bool)
percentage = tf.placeholder(tf.float32)
gdrop_str_d = tf.placeholder(tf.float32)
if useTempoD:
	gdrop_str_t = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32,[None,n_input], name = "x")
x_disc =tf.placeholder(tf.float32,[None,n_input], name = "x_disc")

#real input for disc
kk = tf.placeholder(tf.float32)
kk2 = tf.placeholder(tf.float32)
kkt = tf.placeholder(tf.float32)
kktl = tf.placeholder(tf.float32)
#keep probablity for dropout
keep_prob = tf.placeholder(tf.float32)

print("x: {}".format(x.get_shape()))

# --- main graph setup ---

if use_loss_scaling:
	loss_scaling_init = 64.0
	loss_scaling_inc = 0.0005
	loss_scaling_dec = 1.0

	ls_vars = []
	g_grads = []
	d_grads = []
	t_grads = []

	ls_vars.append(tf.Variable(initial_value=np.float32(loss_scaling_init), name='g_loss_scaling_var_1'))
	ls_vars.append(tf.Variable(initial_value=np.float32(loss_scaling_init), name='d_loss_scaling_var_1'))
	ls_vars.append(tf.Variable(initial_value=np.float32(loss_scaling_init), name='t_loss_scaling_var_1'))
	
	# Apply dynamic loss scaling for the given expression.
	def apply_loss_scaling(value, ls_var):
		return value * tf.exp(ls_var * np.float32(np.log(2.0)))

	# Undo the effect of dynamic loss scaling for the given expression.
	def undo_loss_scaling( value, ls_var):
		return value * tf.exp(-ls_var * np.float32(np.log(2.0)))
		
	def calc_gradients(loss, vars, optimizer, ls_var):
		# Register device and compute gradients.
		with tf.name_scope('grads'):
			loss = apply_loss_scaling(tf.cast(loss, tf.float32), ls_var)
			grads = optimizer.compute_gradients(loss, vars, gate_gradients=tf.train.Optimizer.GATE_NONE) # disable gating to reduce memory usage
			grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads] # replace disconnected gradients with zeros
		return grads
		
	# Construct training op to update the registered variables based on their gradients.
	def apply_updates(grads, optimizer, ls_var, ind, sess = None):
		with tf.name_scope('apply_grads%d'% ind):
			total_grads = len(grads)
			
			with tf.name_scope('Scale'):
				coef = tf.constant(np.float32(1.0/total_grads), name='coef')
				coef = undo_loss_scaling(coef, ls_var)
				grads = [(g * coef, v) for g, v in grads]

			# Check for overflows.
			with tf.name_scope('CheckOverflow'):
				grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

			# Update weights and adjust loss scaling.
			with tf.name_scope('UpdateWeights'):
				ops = tf.cond(grad_ok,
					lambda: tf.group(tf.assign_add(ls_var, loss_scaling_inc), optimizer.apply_gradients(grads)),
					lambda: tf.group(tf.assign_sub(ls_var, loss_scaling_dec)))
							
			return ops

		
# build the tensorflow graph for tensor(value) re-sampling (at pos)
# value shape (batch, ..., res_x2, res_x1, channels)
# pos shape (batch, ..., res_x2, res_x1, dim)
def tensorResample(value, pos, flags = None, name='Resample'):
	# if(dataDimension == 2):
	#	value = tf.reshape(value, shape =(-1, tilesz, tilesz, int( n_input / (tilesz * tilesz)) ) )
	#	pos = tf.reshape(pos, shape=(-1, tilesz, tilesz, 2 ))
	# else:
	#	value = tf.reshape(value, shape = (-1, tilesz, tilesz, tilesz, int( n_input / (tilesz * tilesz * tilesz)) ))
	#	pos = tf.reshape( pos, shape = (-1, tilesz, tilesz, tilesz, 3 ))
	with tf.name_scope(name) as scope:
		pos_shape = pos.get_shape().as_list()
		dim = len(pos_shape) - 2  # batch and channels are ignored
		assert (dim == pos_shape[-1])
		
		floors = tf.cast(tf.floor(pos-0.5), tf.int32)
		ceils = floors+1
		
		if 0:
			# clamp min
			floors = tf.maximum(floors, tf.zeros_like(floors))
			ceils = tf.maximum(ceils, tf.zeros_like(ceils))

			# clamp max
			floors = tf.minimum(floors, tf.constant(value.get_shape().as_list()[1:dim + 1], dtype=tf.int32) - 1)
			ceils = tf.minimum(ceils, tf.constant(value.get_shape().as_list()[1:dim + 1], dtype=tf.int32) - 1)
						
		_broadcaster = tf.ones_like(ceils)
		cell_value_list = []
		cell_weight_list = []
		for axis_x in range(int(pow(2, dim))):  # 3d, 0-7; 2d, 0-3;...
			if axis_x != 4:
				condition_list = [bool(axis_x & int(pow(2, i))) for i in range(dim)]
				condition_ = (_broadcaster > 0) & condition_list
				axis_idx = tf.cast(
					tf.where(condition_, ceils, floors),
					tf.int32)

				# only support linear interpolation...
				axis_wei = 1.0 - tf.abs((pos-0.5) - tf.cast(axis_idx, tf.float32))  # shape (..., res_x2, res_x1, dim)			
				axis_wei = tf.reduce_prod(axis_wei, axis=-1, keep_dims=True)
				
				cell_weight_list.append(axis_wei)  # single scalar(..., res_x2, res_x1, 1)
				first_idx = tf.ones_like(axis_wei, dtype=tf.int32)
				first_idx = tf.cumsum(first_idx, axis=0, exclusive=True)
				cell_value_list.append(tf.concat([first_idx, axis_idx], -1))
		#print(value.get_shape())
		#print(cell_value_list[0].get_shape())
		values_new = tf.gather_nd(value, cell_value_list[0]) * cell_weight_list[0]  # broadcasting used, shape (..., res_x2, res_x1, channels )
		for cell_idx in range(1, len(cell_value_list)):
			values_new = values_new + tf.gather_nd(value, cell_value_list[cell_idx]) * cell_weight_list[cell_idx]
		return values_new # shape (..., res_x2, res_x1, channels)


def lerp(x, y, t):
	return tf.add(x, (y - x) * tf.clip_by_value(t,0.0,1.0))

def gaussian_noise_layer(input_layer, strength):  
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=1.0, dtype=tf.float32) 
    return input_layer + noise * (strength * tf.sqrt(tf.cast(input_layer.get_shape().as_list()[3], tf.float32)))
	
# set up GAN structure
def resBlock(gan, inp, s1, s2, reuse, use_batch_norm, name, filter_size=3):
	# note - leaky relu (lrelu) not too useful here

	# convolutions of resnet block
	filter = [filter_size,filter_size]
	filter1 = [1,1]
	filter = [filter_size,filter_size]
	gc1,_ = gan.convolutional_layer(  s1, filter, tf.nn.relu, stride=[1], name="g_cA_"+name, in_layer=inp, reuse=reuse, batch_norm=use_batch_norm, train=train) #->16,64
	if pixel_norm:
		gc1 = gan.pixel_norm(gc1)
	gc2,_ = gan.convolutional_layer(  s2, filter, None, stride=[1], name="g_cB_"+name,               reuse=reuse, batch_norm=use_batch_norm, train=train) #->8,128
	# shortcut connection
	gs1,_ = gan.convolutional_layer( s2, filter1 , None       , stride=[1], name="g_s_"+name, in_layer=inp, reuse=reuse, batch_norm=use_batch_norm, train=train) #->16,64
	resUnit1 = tf.nn.relu( tf.add( gc2, gs1 )  )
	if pixel_norm:
		resUnit1 = gan.pixel_norm(resUnit1)
	
	return resUnit1
	
def growBlockGen(gan, inp, upres, fms, use_batch_norm, train, reuse, output = False):
	with tf.variable_scope("genBlock%d"%(upres), reuse=reuse) as scope:
		if upsampling_mode == 2:
			if not usePixelShuffle:
				inDepool = gan.avg_depool(mode = upsampleMode)
			else:
				inDepool = gan.pixel_shuffle(inp, upres = 2, stage = "%d"%(upres))
		elif upsampling_mode == 1 or upsampling_mode == 3:
			inDepool = inp
		elif upsampling_mode == 0:
			inDepool = gan.max_depool(height_factor = 1,width_factor=2)
		filter = [filterSize,filterSize]
		if first_nn_arch:
			# deeper network in lower levels for higher low-res receptive field - only for the first network
			if upres == 2:
				outp = resBlock(gan, inDepool, fms, fms, reuse, use_batch_norm, "first" , filter_size = filter[0]) #%(upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "second", filter_size = filter[0]) #%(upres,upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "third" , filter_size = filter[0]) #%(upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "fourth", filter_size = filter[0]) #%(upres,upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "fifth", filter_size = filter[0]) #%(upres,upres)
			elif upres == 4:
				outp = resBlock(gan, inDepool, fms*2, fms, reuse, use_batch_norm, "first" , filter_size = filter[0]) #%(upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "second" , filter_size = filter[0]) #%(upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "third", filter_size = filter[0]) #%(upres,upres)
			if upres == 8:
				outp = resBlock(gan, inDepool, fms*2, fms, reuse, use_batch_norm, "first" , filter_size = filter[0]) #%(upres)
				outp = resBlock(gan, outp, fms, fms, reuse, use_batch_norm, "second", filter_size = filter[0]) #%(upres,upres)
		else:
			if use_res_net:
				# 	two res blocks per growing block
				outp = resBlock(gan, inDepool, fms, fms, reuse, use_batch_norm, "first" , filter_size = filter[0]) #%(upres)
				outp = resBlock(gan, outp, fms//2, fms//2, reuse, use_batch_norm, "second", filter_size = filter[0]) #%(upres,upres)
			else:
				inp,_ = gan.convolutional_layer(  fms, filter, lrelu, stride=[1], name="g_cA%d"%(upres), in_layer=inDepool, reuse=reuse, batch_norm=use_batch_norm, train=train) #->16,64
				if pixel_norm:
					inp = gan.pixel_norm(inp)
				outp,_ = gan.convolutional_layer(  fms, filter, lrelu, stride=[1], name="g_cB%d"%(upres), in_layer=inp, reuse=reuse, batch_norm=use_batch_norm, train=train) #->8,128			
				if pixel_norm:
					outp = gan.pixel_norm(outp)
		#	density output for blending 
		if not output:
			outpDens, _ = GAN(outp, bn_decay=bn_decay).convolutional_layer(  1, [1,1], None, stride=[1], name="g_cdensOut%d"%(upres), in_layer=outp, reuse=reuse, batch_norm=False, train=train, gain = 1)
			return outp, outpDens
		# 	else if network is for testing/generating  -> ignore blending 
		return outp
		
def growing_gen(_in, percentage, reuse=False, use_batch_norm=False, train=None, currentUpres = 2, output = False):
	global rbId
	print("\n\tGenerator (growing-sliced-resnett3-deep)")
	
	with tf.variable_scope("generator", reuse=reuse) as scope:
		if dataDimension == 2:
			if upsampling_mode == 2:
				_in = tf.reshape(_in, shape=[-1, tileSizeLow, tileSizeLow, n_inputChannels]) #NHWC
			elif upsampling_mode == 1 or upsampling_mode == 3:
				_in = tf.reshape(_in, shape=[-1, tileSizeHigh, tileSizeHigh, n_inputChannels + 1]) #NHWC
			elif upsampling_mode == 0:
				_in = tf.reshape(_in, shape=[-1, tileSizeHigh, tileSizeLow, n_inputChannels + 1]) #NHWC
		elif dataDimension == 3:
			_in = tf.reshape(_in, shape=[-1, tileSizeLow, tileSizeLow, tileSizeLow, n_inputChannels]) #NDHWC
		
		gan = GAN(_in, bn_decay=bn_decay)	
		
		#	inital conv layers
		filter = [filterSize,filterSize]
		if first_nn_arch:				
			x_g = _in		
		elif use_res_net:
			x_g = resBlock(gan, _in, 16, min(max_fms, start_fms//2)//8, reuse, False, "1", filter_size = filter[0])
			x_g = resBlock(gan, x_g, min(max_fms, start_fms//2)//4, min(max_fms, start_fms//2)//2, reuse, False, "2", filter_size = filter[0])
		else:
			x_g,_ = gan.convolutional_layer(  32, filter, lrelu, stride=[1], name="g_cA%d"%(1), in_layer=_in, reuse=reuse, batch_norm=use_batch_norm, train=train) #->16,64
			if pixel_norm:
				x_g = gan.pixel_norm(x_g)
			x_g,_ = gan.convolutional_layer(  min(start_fms//2, max_fms), filter, lrelu, stride=[1], name="g_cB%d"%(1), in_layer=x_g, reuse=reuse, batch_norm=use_batch_norm, train=train) #->8,128	
			if pixel_norm:
				x_g = gan.pixel_norm(x_g)
		#	density output for blending
		if not output:
			_oldDens, _ = GAN(x_g, bn_decay=bn_decay).convolutional_layer( 1, [1,1], None, stride=[1], name="g_cdensOut%d"%(1), in_layer=x_g, reuse=reuse, batch_norm=False, train=train, gain = 1)
			
		for j in range(1,currentUpres+1):
			num_fms = min(int(start_fms / (2**j)),max_fms)
			if not output or j == currentUpres:
				x_g, _dens = growBlockGen(gan, x_g, int(2**(j)), num_fms, use_batch_norm, train, reuse)	
			else:
				x_g = growBlockGen(gan, x_g, int(2**(j)), num_fms, use_batch_norm, train, reuse, output)	
					
			# residual learning		
			if addBicubicUpsample:
				if not output or j == currentUpres:
					if upsampling_mode == 2:
						_dens = _dens + GAN(tf.slice(_in, [0,0,0,0], [-1,tileSizeLow, tileSizeLow, 1])).avg_depool(mode = 2, scale = [int(2**(j))])
					elif upsampling_mode == 1 or upsampling_mode == 3:
						_dens = _dens + tf.slice(_in, [0,0,0,0], [-1,tileSizeHigh, tileSizeHigh, 1])						
					elif upsampling_mode == 0:
						_dens = _dens + tf.image.resize_images(tf.slice(_in, [0,0,0,4], [-1, tileSizeHigh, tileSizeLow, 1]), [tileSizeHigh, tileSizeLow * int(2**(j))], 2)

			print("\tDOFs: %d , %f m " % ( gan.getDOFs() , gan.getDOFs()/1000000.) )
				
			with tf.variable_scope("growingPart%i"%j, reuse=reuse) as scope:
				if not output:
					if upsampling_mode == 2:
						_oldDens = GAN(_oldDens).avg_depool(mode = 1)	
					elif upsampling_mode == 0:
						_oldDens = GAN(_oldDens).avg_depool(scale = [1, 2], mode = 1)	
					
					if upsampling_mode == 2:
						_oldDens = tf.reshape(lerp(_oldDens, _dens, percentage - (j - 1)), shape = [-1,tileSizeLow * (2**j),tileSizeLow * (2**j),1])
					elif upsampling_mode == 1 or upsampling_mode == 3:
						_oldDens = tf.reshape(lerp(_oldDens, _dens, percentage - (j - 1)), shape = [-1,tileSizeHigh, tileSizeHigh,1])
					elif upsampling_mode == 0:
						_oldDens = tf.reshape(lerp(_oldDens, _dens, percentage - (j - 1)), shape = [-1,tileSizeHigh, tileSizeLow * (2**j),1])					
				elif j == currentUpres:
					_oldDens = _dens
					
		print(_oldDens.get_shape().as_list())
		resF = tf.reshape( _oldDens, shape=[-1, n_output] ) # + GAN(_in).avg_depool(mode = upsampleMode)
		print("\tDOFs: %d , %f m " % ( gan.getDOFs() , gan.getDOFs()/1000000.) )
		return resF

def gn(x, gstr):
	if use_gdrop:
		return gaussian_noise_layer(x, gstr) 
	else:
		return x
		
def growBlockDisc(gan, inp,  upres, fms, use_batch_norm, train, reuse, name, gstr = 0.0):
	with tf.variable_scope(name+("Block%d"%(upres)), reuse=reuse) as scope:		
		if name == "t" and useVelInTDisc:
			filter = [filterSize+2,filterSize]
		else:			
			if first_nn_arch:
					filter = [4, 4]
			else:
				filter = [filterSize, filterSize]
					
		if first_nn_arch:
			if upres == 2:
				x1,_ = gan.convolutional_layer(  fms*3, filter, lrelu, stride=[1], name=str(name)+"_cA%d"%(upres), in_layer=gn(inp,gstr), reuse=reuse, batch_norm=use_batch_norm, train=train, in_channels=fms)
			else:
				x1,_ = gan.convolutional_layer(  fms*2, filter, lrelu, stride=[1], name=str(name)+"_cA%d"%(upres), in_layer=gn(inp,gstr), reuse=reuse, batch_norm=use_batch_norm, train=train, in_channels=fms)
			x2,_ = gan.convolutional_layer( min(min( fms*2, max_fms),start_fms//2), filter, lrelu, stride=[1], name=str(name)+"_cB%d"%(upres), in_layer=gn(x1,gstr), reuse=reuse, batch_norm=use_batch_norm, train=train)	
		else:
			x1,_ = gan.convolutional_layer(  fms, filter, lrelu, stride=[1], name=str(name)+"_cA%d"%(upres), in_layer=gn(inp,gstr), reuse=reuse, batch_norm=use_batch_norm, train=train, in_channels=fms)
			x2,_ = gan.convolutional_layer( min(min( fms*2, max_fms),start_fms//2), filter, lrelu, stride=[1], name=str(name)+"_cB%d"%(upres), in_layer=gn(x1,gstr), reuse=reuse, batch_norm=use_batch_norm, train=train, in_channels = fms)
		
		if upsampling_mode == 2:
			outp = gan.avg_pool()
		elif upsampling_mode == 1 or upsampling_mode == 3:
			outp = x2
		elif upsampling_mode == 0:
			outp = tf.nn.avg_pool(x2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="VALID") 
	
		return outp, x1, x2
		
############################################discriminator network###############################################################
def growing_disc(in_high_, in_low_, percentage,reuse=False, use_batch_norm=False, train=None,currentUpres = 2, gstr = 0.0):
	#in_low: low res reference input, same as generator input (condition)
	#in_high: real or generated high res input to classify
	#reuse: variable reuse
	#use_batch_norm: bool, if true batch norm is used in all but the first con layers
	#train: if use_batch_norm, tf bool placeholder
	#use_batch_norm = False
	print("\n\tDiscriminator (conditional binary classifier)")
	with tf.variable_scope("spatial-disc", reuse=reuse) as scope:	
		shape = tf.shape(in_low_)
		in_high_ = tf.reshape(in_high_, shape=[-1,tileSizeHigh,tileSizeHigh, 1])
		if upsampling_mode == 2:
			#if add_adj_idcs:
			#	in_low_2 = tf.slice(tf.reshape(in_low_, shape = [-1,tileSizeLow,tileSizeLow,n_inputChannels]),[0,0,0,4],[shape[0],tileSizeLow,tileSizeLow,2])
			#	in_low_2 = GAN(tf.reshape(in_low_2, shape = [-1, tileSizeLow, tileSizeLow, 2])).avg_depool(scale = [upRes], mode = upsampleMode)
			in_low_ = tf.slice(tf.reshape(in_low_, shape = [-1,tileSizeLow,tileSizeLow,n_inputChannels]),[0,0,0,0],[shape[0],tileSizeLow,tileSizeLow,1])
			in_low_ = GAN(tf.reshape(in_low_, shape = [-1, tileSizeLow, tileSizeLow, 1])).avg_depool(scale = [upRes], mode = upsampleMode)
		elif upsampling_mode == 1 or upsampling_mode == 3:
			#if add_adj_idcs:
			#	in_low_2 = tf.slice(tf.reshape(in_low_, shape = [-1,tileSizeLow,tileSizeLow,n_inputChannels]),[0,0,0,4],[shape[0],tileSizeLow,tileSizeLow,2])
			#	in_low_2 = GAN(tf.reshape(in_low_2, shape = [-1, tileSizeLow, tileSizeLow, 2])).avg_depool(scale = [upRes], mode = upsampleMode)
			in_low_ = tf.slice(tf.reshape(in_low_, shape = [-1,tileSizeLow,tileSizeLow,n_inputChannels]),[0,0,0,0],[shape[0],tileSizeLow,tileSizeLow,1])
			in_low_ = GAN(tf.reshape(in_low_, shape = [-1, tileSizeLow, tileSizeLow, 1])).avg_depool(scale = [upRes], mode = upsampleMode)
		elif upsampling_mode == 0:
			#if add_adj_idcs:
			#	in_low_2 = tf.slice(tf.reshape(in_low_, shape = [-1,tileSizeLow,tileSizeLow,n_inputChannels]),[0,0,0,4],[shape[0],tileSizeLow,tileSizeLow,2])
			#	in_low_2 = GAN(tf.reshape(in_low_2, shape = [-1, tileSizeLow, tileSizeLow, 2])).avg_depool(scale = [upRes], mode = upsampleMode)
			in_low_ = tf.slice(tf.reshape(in_low_, shape = [-1,tileSizeHigh,tileSizeLow,n_inputChannels + 1]),[0,0,0,0],[-1,tileSizeHigh,tileSizeLow,1])
			in_low_ = GAN(tf.reshape(in_low_, shape = [-1, tileSizeHigh, tileSizeLow, 1])).avg_depool(scale = [1, upRes], mode = 0)			
		#if add_adj_idcs:
		#	in_high_ = tf.concat([in_low_, in_low_2, in_high_], axis=3)
		#else:
		in_high_ = tf.concat([in_low_, in_high_], axis=3)
			
		feature_layers = []	
		#	from dens, input to growing disc
		gan = GAN(in_high_, bn_decay=bn_decay)		
		x_,_ = gan.convolutional_layer(int( start_fms / upRes), [1,1],activation_function = None  , in_layer =in_high_, stride=[1], name="d_cfromDensity%d"%(upRes), reuse=reuse, batch_norm=False, train=train)			
		feature_layers.append(lerp(tf.zeros_like(x_), x_, percentage - (currentUpres - 1) ))
		
		inHigh =  in_high_
		gan2 = GAN(inHigh, bn_decay=bn_decay)
		for j in range(currentUpres, 0,-1):
			print("\tDOFs: %d " % gan.getDOFs())	
			num_fms = int( min(start_fms / (2**(j)),max_fms))
			# 	from density field 
			if upsampling_mode == 2:
				inHigh = GAN(inHigh).avg_pool()#
			elif upsampling_mode == 0:
				inHigh = tf.nn.avg_pool(inHigh, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="VALID") #inHigh = GAN(inHigh).avg_pool()#
				
			x_, x1, x2 = growBlockDisc(gan, x_, int(2**(j)), int(num_fms), False, train, reuse,"d", gstr = gstr)	
			
			fromDensFms =min(min( num_fms*2, max_fms),start_fms//2)
			_oldDens,_ = gan2.convolutional_layer( fromDensFms, [1,1], None, stride=[1], name="d_cfromDensity%d"%(2**(j-1)), in_layer=inHigh, reuse=reuse, batch_norm=False, train=train)			
			with tf.variable_scope(("blend%i"%j), reuse=reuse) as scope:	
				if upsampling_mode == 2:		
					x_ = tf.reshape(lerp( _oldDens, x_, percentage - (j - 1)), shape = [-1,tileSizeLow * (2**(j-1)),tileSizeLow * (2**(j-1)), fromDensFms])
				elif upsampling_mode == 1 or upsampling_mode == 3:
					x_ = tf.reshape(lerp( _oldDens, x_, percentage - (j - 1)), shape = [-1,tileSizeHigh, tileSizeHigh, fromDensFms])
				elif upsampling_mode == 0:
					x_ = tf.reshape(lerp( _oldDens, x_, percentage - (j - 1)), shape = [-1,tileSizeHigh, tileSizeLow * (2**(j-1)), fromDensFms])
			# incase feature loss is to be used
			feature_layers.append(lerp(tf.zeros_like(x1), x1, percentage - (j - 1)))
			feature_layers.append(lerp(tf.zeros_like(x2), x2, percentage - (j - 1)))
		if use_mb_stddev:
			x_ = gan.minibatch_stddev_layer(x_)
		# 	last processing layers (mirroring generator)
		filter = [filterSize,filterSize]
				
		if not first_nn_arch:			
			x1,_ = gan.convolutional_layer(  32, filter, lrelu, stride=[1], name=str('d')+"_cA%d"%(1), in_layer=tf.reshape(gn(x_,gstr), shape=tf.shape(x_)), reuse=reuse, batch_norm=use_batch_norm, train=train)
			x2,_ = gan.convolutional_layer(  4, filter, None, stride=[1], name=str('d')+"_cB%d"%(1), in_layer=tf.reshape(gn(x1,gstr), shape=tf.shape(x1)), reuse=reuse, batch_norm=use_batch_norm, train=train)
		else:
			x1 = x_
			
		feature_layers.append(lerp(tf.zeros_like(x1), x1, percentage ))
		
		shape = gan.flatten()
		
		gan.fully_connected_layer(1, None, name="d_l6%d"%1, gain = 1)
		sigmoid = gan.y()		
		print("\tDOFs: %d " % gan.getDOFs())	

		return sigmoid, feature_layers
		
############################################ Tempo discriminator network ############################################################
def growing_disc_tempo(in_high_,percentage, n_t_channels=3, reuse=True, use_batch_norm=False, train=None,currentUpres = 2, gstr = 0.0):
	# in_high: real or generated high res input to classify, shape should be batch, dim_z, dim_y, dim_x, channels
	# reuse: variable reuse
	# use_batch_norm: bool, if true batch norm is used in all but the first con layers
	# train: if use_batch_norm, tf bool placeholder
	#use_batch_norm = False
	print("\n\tDiscriminator for Tempo (unconditional binary classifier)")
	with tf.variable_scope("tempo-disc", reuse=reuse) as scope:
		if useVelInTDisc:
			in_high_ = tf.reshape(in_high_, shape=[-1,tileSizeHigh,tileSizeHigh, 12])
		else:
			in_high_ = tf.reshape(in_high_, shape=[-1,tileSizeHigh,tileSizeHigh, 3])
				
		gan = GAN(in_high_, bn_decay=bn_decay)		
		x,_ = gan.convolutional_layer(int( start_fms / upRes) , [1,1],activation_function = None  , in_layer =in_high_, stride=[1], name="t_cfromDensity%d"%(upRes), reuse=reuse, batch_norm=False, train=train)			

		inHigh =  in_high_
		gan2 = GAN(inHigh, bn_decay=bn_decay)
		for j in range(currentUpres, 0,-1):
			num_fms = int( min(start_fms / (2**(j)),max_fms))
			# 	from dens	
			if upsampling_mode == 2:
				inHigh = GAN(inHigh).avg_pool()
			elif upsampling_mode == 0:
				inHigh = tf.nn.avg_pool(inHigh, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="VALID") #inHigh = GAN(inHigh).avg_pool()#

			x, x1, x2 = growBlockDisc(gan, x, int(2**(j)), int(num_fms), False, train, reuse,"t", gstr = gstr)	
				
			_oldDens, _ = gan2.convolutional_layer( min(min( num_fms*2, max_fms),start_fms//2), [1,1], None, stride=[1], name="t_cfromDensity%d"%(2**(j-1)), in_layer=inHigh, reuse=reuse, batch_norm=False, train=train)			
			
			with tf.variable_scope(("blend%i"%j), reuse=reuse) as scope:
				if upsampling_mode == 2:				
					x = tf.reshape(lerp(_oldDens, x, percentage - (j - 1)), shape = [-1,tileSizeLow * (2**(j-1)),tileSizeLow * (2**(j-1)), min(min( num_fms*2, max_fms),start_fms//2)])
				elif upsampling_mode == 1 or upsampling_mode == 3:
					x = tf.reshape(lerp(_oldDens, x, percentage - (j - 1)), shape = [-1,tileSizeHigh,tileSizeHigh, min(min( num_fms*2, max_fms),start_fms//2)])
				elif upsampling_mode == 0:
					x = tf.reshape(lerp(_oldDens, x, percentage - (j - 1)), shape = [-1,tileSizeHigh,tileSizeLow * (2**(j-1)), min(min( num_fms*2, max_fms),start_fms//2)])
		# 	last processing layers (mirroring generator)
		if use_mb_stddev:
			x = gan.minibatch_stddev_layer(x, 1)
		filter = [filterSize,filterSize]
		
		if not first_nn_arch:	
			x1,_ = gan.convolutional_layer(  32, filter, lrelu, stride=[1], name=str('t')+"_cA%d"%(1), in_layer=tf.reshape(gn(x,gstr), shape=tf.shape(x)), reuse=reuse, batch_norm=use_batch_norm, train=train) 
			x2,_ = gan.convolutional_layer(  4, filter, None, stride=[1], name=str('t')+"_cB%d"%(1), in_layer=tf.reshape(gn(x1,gstr), shape=tf.shape(x1)), reuse=reuse, batch_norm=use_batch_norm, train=train) 
		else:
			x2 = x
			
		shape = gan.flatten()
		
		gan.fully_connected_layer(1, None, name="t_l6%d"%1, gain = 1)
		sigmoid = gan.y()		
		print("\tDOFs: %d " % gan.getDOFs())	

		return sigmoid
			
############################################gen_test###############################################################
def gen_test(_in, reuse=False, use_batch_norm=False, train=None):
	global rbId
	print("\n\tGenerator-test")
	with tf.variable_scope("generator-test", reuse=reuse) as scope:
		if dataDimension == 2:
			_in = tf.reshape(_in, shape=[-1, tileSizeLow, tileSizeLow, n_inputChannels]) #NHWC
			patchShape = [2,2]
		elif dataDimension == 3:
			_in = tf.reshape(_in, shape=[-1, tileSizeLow, tileSizeLow, tileSizeLow, n_inputChannels]) #NDHWC
			patchShape = [2,2,2]
		rbId = 0
		gan = GAN(_in)

		gan.max_depool()
		i2np,_ = gan.deconvolutional_layer(32, patchShape, None, stride=[1,1], name="g_D1", reuse=reuse, batch_norm=False, train=train, init_mean=0.99) #, strideOverride=[1,1] )
		gan.max_depool()
		inp,_  = gan.deconvolutional_layer(1                   , patchShape, None, stride=[1,1], name="g_D2", reuse=reuse, batch_norm=False, train=train, init_mean=0.99) #, strideOverride=[1,1] )
		return 	tf.reshape( inp, shape=[-1, n_output] )

############################################disc_test###############################################################
def disc_test(in_low, in_high, reuse=False, use_batch_norm=False, train=None):
	print("\n\tDiscriminator-test")
	with tf.variable_scope("discriminator_test", reuse=reuse):
		if dataDimension == 2:
			#in_low,_,_ = tf.split(in_low,n_inputChannels,1)
			shape = tf.shape(in_low)
			in_low = tf.slice(in_low,[0,0],[shape[0],int(n_input/n_inputChannels)])
			in_low = GAN(tf.reshape(in_low, shape=[-1, tileSizeLow, tileSizeLow, 1])).max_depool(height_factor = upRes,width_factor = upRes) #NHWC
			in_high = tf.reshape(in_high, shape=[-1, tileSizeHigh, tileSizeHigh, 1])
			filter=[4,4]
			stride2 = [2]
		elif dataDimension == 3:
			shape = tf.shape(in_low)
			in_low = tf.slice(in_low,[0,0],[shape[0],int(n_input/n_inputChannels)])
			in_low = GAN(tf.reshape(in_low, shape=[-1, tileSizeLow, tileSizeLow, tileSizeLow, 1])).max_depool(depth_factor = upRes,height_factor = upRes,width_factor = upRes) #NDHWC
			in_high = tf.reshape(in_high, shape=[-1, tileSizeHigh, tileSizeHigh, tileSizeHigh, 1]) # dim D is not upscaled
			filter=[4,4,4]
			stride2 = [2]

		#merge in_low and in_high to [-1, tileSizeHigh, tileSizeHigh, 2]
		gan = GAN(tf.concat([in_low, in_high], axis=-1), bn_decay=bn_decay) #64
		d1,_ = gan.convolutional_layer(32, filter, lrelu, stride=stride2, name="d_c1", reuse=reuse) #32
		shape=gan.flatten()
		gan.fully_connected_layer(1, None, name="d_l5")
		if dataDimension == 2:
			d2 = tf.constant(1., shape = [batch_size, tileSizeLow,tileSizeLow,64])
			d3 = tf.constant(1., shape = [batch_size, int(tileSizeLow/2),int(tileSizeLow/2),128])	
			d4 = tf.constant(1., shape = [batch_size, int(tileSizeLow/2),int(tileSizeLow/2),256])
		elif dataDimension == 3:
			d2 = tf.constant(1., shape = [batch_size, tileSizeLow,tileSizeLow,tileSizeLow,64])
			d3 = tf.constant(1., shape = [batch_size, int(tileSizeLow/2),int(tileSizeLow/2),int(tileSizeLow/2),128])	
			d4 = tf.constant(1., shape = [batch_size, int(tileSizeLow/2),int(tileSizeLow/2),int(tileSizeLow/2),256])
		print("\tDOFs: %d " % gan.getDOFs())
		return gan.y(), d1, d2, d3, d4

#change used models for gen and disc here #other models in NNmodels.py
#gen_model = locals()[genModel]
#disc_model = locals()[discModel]
#disc_time_model = disc_binclass_cond_tempo # tempo dis currently fixed

gen_model = growing_gen
disc_model = growing_disc
disc_time_model = growing_disc_tempo

#set up GAN structure
bn=batch_norm
#training or testing for batch norm
train = tf.placeholder(tf.bool)

lr_global_step = tf.Variable(0, trainable=False)
learning_rate_scalar = learning_rate
if useTempoD:
	learning_rates_t = []
learning_rates_g = []
if use_spatialdisc:
	learning_rates_d= []
for i in range(3):
	curr_lr = learning_rate
	if decayLR:
		if useTempoD:
			learning_rates_t.append(tf.train.polynomial_decay(curr_lr, lr_global_step, decayIter, learning_rate_scalar*0.05, power=1.1))
		if use_spatialdisc:
			learning_rates_d.append(tf.train.polynomial_decay(curr_lr, lr_global_step,  decayIter, learning_rate_scalar*0.05, power=1.1))
		learning_rates_g.append(tf.train.polynomial_decay(curr_lr, lr_global_step,  decayIter, learning_rate_scalar*0.05, power=1.1))
	else:
		learning_rates_g.append(curr_lr)
		curr_lr /= 2
		if i == 1:
			if useTempoD:
				learning_rates_t.append(curr_lr/2)
			if use_spatialdisc:
				learning_rates_d.append(curr_lr/2)

k2_ls = []
for i in range( int(round(math.log(upRes,2)))*2+2):	
	k2_ls.append(1.0)
	
x = tf.placeholder(tf.float32,[None,n_input], name = "x")
x_disc =tf.placeholder(tf.float32,[None,n_input], name = "x_disc")
y = tf.placeholder(tf.float32,[None,None], name = "y")

#randSize = tf.constant(0.85, dtype =tf.float32) + tf.random_uniform([]) * tf.constant(0.3, dtype =tf.float32)
#x = tf.reshape(tf.image.crop_and_resize(tf.reshape(x, shape=[-1,tileSizeLow,tileSizeLow,n_inputChannels]),[[0,0,1,1]],tf.zeros(tf.shape(x)[0],dtype = tf.int32),crop_size = [tileSizeLow, tileSizeLow]), shape=[-1,n_input])
print("x: {}".format(x.get_shape()))

#useTempoD = False

if useTempoD:
	y_t = tf.placeholder(tf.float32,[None,None], name = "yt")
	x_t = tf.placeholder(tf.float32,[None,n_input], name = "xt")	
	if ADV_flag:
		y_pos = tf.placeholder(tf.float32,[None, None], name = "yp")
		
if not outputOnly: #setup for training	
	if upsampling_mode == 2 or upsampling_mode == 0:
		x_in = x
	elif upsampling_mode == 1 or upsampling_mode == 3:
		x_in_2 = tf.slice(tf.reshape(y, shape = [-1, tileSizeHigh, tileSizeHigh, 2]), [0,0,0,1], [-1,tileSizeHigh, tileSizeHigh, 1])		
		x_in = tf.concat((x_in_2, tf.image.resize_images(tf.slice(tf.reshape(x, shape = [-1, tileSizeLow, tileSizeLow, n_inputChannels]),[0,0,0,0], [-1, tileSizeLow, tileSizeLow, n_inputChannels]), tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1)), axis = 3)
	
	gen_y = gen_model(x_in, use_batch_norm=bn, reuse = False, train=train, currentUpres = int(round(math.log(upRes, 2))), percentage = percentage)
	
	if upsampling_mode == 1 or upsampling_mode == 3:
		gen_y = tf.reshape(gen_y, shape=[-1, tileSizeHigh, tileSizeHigh, 1])
	
	upresFac = tf.pow(tf.constant(2,dtype=tf.int32), tf.cast( tf.ceil(percentage),dtype=tf.int32))
	if upsampling_mode == 2:
		currentTileSizeX = upresFac * tf.constant(tileSizeLow, dtype = tf.int32)
	elif upsampling_mode == 1 or upsampling_mode == 3:
		currentTileSizeX = tf.constant(tileSizeHigh, dtype = tf.int32)
		
	if upsampling_mode == 2:
		y_in = tf.reshape(tf.image.resize_images(tf.reshape(y, shape= [-1, currentTileSizeX, currentTileSizeX,1]), tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1), shape = [tf.shape(gen_y)[0], n_output])
	elif upsampling_mode == 0:
		y_in = tf.reshape(y, shape = [tf.shape(gen_y)[0], n_output])
	elif upsampling_mode == 1 or upsampling_mode == 3:
		y_in = tf.reshape(tf.slice(tf.reshape(y, shape = [-1, tileSizeHigh, tileSizeHigh, 2]), [0,0,0,0], [-1,tileSizeHigh, tileSizeHigh, 1]), shape= [-1, tileSizeHigh, tileSizeHigh, 1])
	
	if use_spatialdisc:
		disc, f_y = disc_model(y_in, x_disc, use_batch_norm=bn, train=train, reuse = False, currentUpres = int(round(math.log(upRes, 2))), percentage = percentage, gstr = gdrop_str_d)
		gen, f_g = disc_model(gen_y, x_disc, use_batch_norm=bn, reuse = True, currentUpres = int(round(math.log(upRes, 2))), train=train, percentage = percentage, gstr = gdrop_str_d)
	if genTestImg > -1: sampler = gen_y
		
else: #setup for generating output with trained model
	if upsampling_mode == 2:
		x_in = x
	elif upsampling_mode == 0:
		x_in = x
	elif upsampling_mode == 1 or upsampling_mode == 3:
		x_in = tf.concat((tf.reshape(y, shape = [-1, tileSizeHigh, tileSizeHigh, 1]), tf.image.resize_images(tf.slice(tf.reshape(x, shape = [-1, tileSizeLow, tileSizeLow, n_inputChannels]),[0,0,0,0],[-1,tileSizeLow,tileSizeLow, n_inputChannels]), tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1)), axis = 3)
	
	# TODO: full pipeline (using two generators, e.g.)
	sampler = gen_model(x_in, use_batch_norm=bn, reuse = tf.AUTO_REUSE, currentUpres = int(round(math.log(upRes, 2))), train=False, percentage = percentage, output = True)

sys.stdout.flush()

if not outputOnly:
	#for discriminator [0,1] output
	if use_spatialdisc:
		if use_LSGAN or use_wgan_gp:
			d_sig_y= tf.reduce_mean(-disc)
			d_sig_g= tf.reduce_mean(gen)
		else:
			d_sig_y= tf.reduce_mean(tf.nn.sigmoid(disc))
			d_sig_g= tf.reduce_mean(tf.nn.sigmoid(gen))
						 
		disc_loss_layer = tf.reduce_mean(tf.nn.l2_loss(f_y[0] - f_g[0])) * k2_ls[0]
		for i in range(1,len(f_g)):
			disc_loss_layer += tf.reduce_mean(tf.nn.l2_loss(f_y[i] - f_g[i])) * k2_ls[i]
		
		# loss of the discriminator with real input
		if use_LSGAN:
			d_loss_y =  0.5* tf.reduce_mean( tf.square(disc - b))
			d_loss_g =  0.5* tf.reduce_mean( tf.square(gen - a))
		else:
			if use_wgan_gp:
				d_loss_y =  tf.reduce_mean(-disc)
				d_loss_g =  tf.reduce_mean(gen)
			else:
				d_loss_y= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc, labels=tf.ones_like(disc)))
				d_loss_g= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen, labels=tf.zeros_like(gen)))
		
		disc_loss = d_loss_y * weight_dld + d_loss_g
		#loss of the discriminator with input from generator
		
	gen_l2_loss = tf.nn.l2_loss(y_in - gen_y)
	l1_loss = tf.reduce_mean(tf.abs(y_in - gen_y)) #use mean to normalize w.r.t. output dims. tf.reduce_mean(tf.abs(y - gen_part))

	if use_LSGAN:
		g_loss_d = 0.5* tf.reduce_mean(tf.square(gen - c))
	else: 	
		if use_wgan_gp:
			g_loss_d = tf.reduce_mean(-gen)
		else:
			g_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen, labels=tf.ones_like(gen)))
	
	if use_spatialdisc:
		# setup gradient loss
		if use_wgan_gp:
			if upsampling_mode == 2:
				lerp_factor = tf.random_uniform([tf.shape(gen_y)[0], 1], 0.0, 1.0)
			elif upsampling_mode == 1 or upsampling_mode == 3:
				lerp_factor = tf.random_uniform([tf.shape(gen_y)[0], 1, 1, 1], 0.0, 1.0)
			elif upsampling_mode == 0:
				lerp_factor = tf.random_uniform([tf.shape(y_in)[0], 1], 0.0, 1.0)
				
			y_gp_d = lerp_factor * y_in + (1 - lerp_factor) * gen_y			
			d_out, _ = disc_model(y_gp_d, x_disc, use_batch_norm=bn, reuse = True, currentUpres = int(round(math.log(upRes, 2))), train=train, percentage = percentage)
			d_out_loss = tf.reduce_mean(d_out)
			
			grads_d = tf.gradients(d_out_loss, [y_gp_d, x_disc])[0]	
			grads_d = tf.sqrt(tf.reduce_sum(tf.square(grads_d + 1e-4), axis = 1))		
			grad_penalty_d = tf.reduce_mean(wgan_lambda * tf.square(grads_d - wgan_target))
		
			epsilon_penalty_d = tf.reduce_mean(tf.square(disc))
			disc_loss += epsilon_penalty_d * wgan_epsilon
			disc_loss += grad_penalty_d
						
	gen_loss_complete = g_loss_d + l1_loss*kk + disc_loss_layer*kk2

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	gen_update_ops = update_ops[:]
	
	#variables to be used in the different otimization steps
	vars = tf.trainable_variables()
	g_var = [var for var in vars if "g_" in var.name]
	if use_spatialdisc:
		dis_update_ops = update_ops[:]
		d_var = [var for var in vars if "d_" in var.name]

	# set up decaying learning rate, if enabled

	if (useTempoD or useTempoL2):# temporal loss here
		# currently, the update_op gathering is very ungly and very sensitive to the operation order. 
		ori_gen_loss_complete = gen_loss_complete
		# TODO: make it flexible!
		n_t = 3
		device_str = '/device:GPU:0'
		if(dataDimension == 3): # have to use a second GPU!
			device_str = '/device:GPU:1'
		with tf.device(device_str): 
			if upsampling_mode == 2:
				x_t_in = x_t
			elif upsampling_mode == 0:
				x_t_in = x_t
			elif upsampling_mode == 1 or upsampling_mode == 3:
				x_t_in = tf.concat((tf.slice(tf.reshape(y_t, shape = [-1, tileSizeHigh, tileSizeHigh, 2]), [0,0,0,1], [-1,tileSizeHigh, tileSizeHigh, 1]), tf.image.resize_images(tf.slice(tf.reshape(x_t, shape = [-1, tileSizeLow, tileSizeLow, n_inputChannels]),[0,0,0,0],[-1,tileSizeLow,tileSizeLow,n_inputChannels]), tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1)), axis = 3)
				
			gen_ts = gen_model(x_t_in,  reuse = True, use_batch_norm=bn, train=train, currentUpres = int(round(math.log(upRes, 2))), percentage = percentage)
		
		if(ADV_flag):		
			# current upres factor 
			upresFac = tf.pow(tf.constant(2,dtype=tf.int32), tf.cast( tf.ceil(percentage),dtype=tf.int32))
			if upsampling_mode == 2:
				currentTileSizeX = upresFac * tf.constant(tileSizeLow, dtype = tf.int32)
			elif upsampling_mode == 3 or upsampling_mode == 0 or upsampling_mode == 1:		
				currentTileSizeX = tf.constant(tileSizeHigh, dtype = tf.int32)
				
			if upsampling_mode == 2 or upsampling_mode == 3 or upsampling_mode == 1:
				vel_t = tf.reshape(tf.slice(tf.reshape(x_t, shape = [tf.shape(x_t)[0], tileSizeLow, tileSizeLow, 4]),[0,0,0,1],[-1, tileSizeLow, tileSizeLow, 3]), shape = [tf.shape(x_t)[0], tileSizeLow, tileSizeLow, 3])
			elif upsampling_mode == 0:
				vel_t = tf.reshape(tf.slice(tf.reshape(x_t, shape = [tf.shape(x_t)[0], tileSizeHigh, tileSizeLow, 5]),[0,0,0,1],[-1, tileSizeHigh, tileSizeLow, 3]), shape = [tf.shape(x_t)[0], tileSizeHigh, tileSizeLow, 3])
			
			if dataDimension == 2:	
				gen_ts = tf.reshape(gen_ts, shape=[tf.shape(x_t)[0], tileSizeHigh, tileSizeHigh,1])					
				gen_ts = tf.image.resize_images(gen_ts, [currentTileSizeX, currentTileSizeX], method = 1)
				if not ADV_mode:
					pos_array = tf.reshape(y_pos, shape=[-1, currentTileSizeX, currentTileSizeX, 2])
					g_resampled = tensorResample(gen_ts, pos_array)			
				else:
					# flags used as a placeholder...
					flags = tf.zeros_like(gen_ts)				
					# startBz: currently batch size cant vary and has to be constant for ADV_mode == 2
					g_resampled = GAN(gen_ts).advect(gen_ts, vel_t, flags, 0.5, ADV_mode, 1.0, startBz = (batch_size_disc // 3) * 3)	
				
				# resize input slice to fit the temporal discriminator, only necessary if upsampling_mode == 2
				if upsampling_mode == 2:
					g_resampled = tf.image.resize_images(g_resampled, tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1) 
				if useVelInTDisc:
					g_resampled = tf.concat((g_resampled,tf.cast(upresFac,dtype=tf.float32)*tf.image.resize_images(vel_t, tf.constant([tileSizeHigh, tileSizeHigh], dtype = tf.int32), method = 0)), axis = 3)
			
		g_resampled = tf.reshape(g_resampled, shape = [-1, n_t, n_output])
		g_resampled = tf.transpose(g_resampled, perm=[0, 2, 1]) # batch, n_output, channels

		if (useTempoD):
			disT_update_ops = []
			# real input for disc
			if (ADV_flag):
				if upsampling_mode == 2:
					y_resampled = tf.reshape(y_t, shape=[tf.shape(y_t)[0], currentTileSizeX, currentTileSizeX, 1])
				elif upsampling_mode == 1 or upsampling_mode == 3:
					y_resampled = tf.slice(tf.reshape(y_t, shape=[tf.shape(y_t)[0], currentTileSizeX, currentTileSizeX, 2]), [0,0,0,0], [-1, currentTileSizeX,currentTileSizeX,1])
				elif upsampling_mode == 0:
					y_resampled = tf.reshape(y_t, shape=[tf.shape(y_t)[0], tileSizeHigh, tileSizeHigh, 1])
				
				if not ADV_mode:
					y_resampled = tensorResample(y_resampled, pos_array)
				else:
					flags = tf.zeros_like(y_resampled)
					y_resampled = GAN(y_resampled).advect(y_resampled, vel_t, flags, 0.5, ADV_mode, 1.0, startBz = (batch_size_disc // 3) * 3)		
				
				if upsampling_mode == 2:
					y_resampled = tf.image.resize_images(y_resampled, tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1)
				if useVelInTDisc:
					y_resampled = tf.concat((y_resampled, tf.cast(upresFac,dtype=tf.float32)*tf.image.resize_images(vel_t, tf.constant([tileSizeHigh, tileSizeHigh], dtype = tf.int32), method = 0)), axis = 3)
			else:
				y_resampled = y_t
			y_resampled = tf.reshape(y_resampled, shape = [-1, n_t, n_output])
			y_resampled = tf.transpose(y_resampled, perm=[0, 2, 1]) # batch, n_output, channels
							
			gen_s = disc_time_model(g_resampled, use_batch_norm=bn, train=train, reuse = False, currentUpres = int(round(math.log(upRes, 2))), percentage = percentage, gstr = gdrop_str_t)
			
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			
			for update_op in update_ops:
				if ("/g_" in update_op.name) and ("generator" in update_op.name) and (not ( update_op in gen_update_ops )):
					gen_update_ops.append(update_op)
					disT_update_ops.append(update_op)
					
				if ("/t_" in update_op.name) and ("tempo-disc" in update_op.name):
					gen_update_ops.append(update_op)
											
				
			disc_s = disc_time_model(y_resampled, use_batch_norm=bn, train=train, reuse = True, currentUpres = int(round(math.log(upRes, 2))), percentage = percentage, gstr = gdrop_str_t)				
			
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			for update_op in update_ops:
				if ("/t_" in update_op.name) and ("tempo-disc" in update_op.name) and (not ( update_op in disT_update_ops )):
					disT_update_ops.append(update_op)
							
			if use_LSGAN or use_wgan_gp:			
				t_sig_y = tf.reduce_mean(-disc_s)
				t_sig_g = tf.reduce_mean(gen_s)
			else:							
				t_sig_y = tf.reduce_mean(tf.nn.sigmoid(disc_s))
				t_sig_g = tf.reduce_mean(tf.nn.sigmoid(gen_s))
			
			vars = tf.trainable_variables()
			t_var = [var for var in vars if "t_" in var.name]
			if use_LSGAN:
				t_loss_y = 0.5* tf.reduce_mean( tf.square(disc_s - b))
				t_loss_g = 0.5* tf.reduce_mean( tf.square(gen_s - a))
			else:
				if use_wgan_gp:
					t_loss_y =  tf.reduce_mean(-disc_s)
					t_loss_g =  tf.reduce_mean(gen_s)
				else:
					# loss of the discriminator with real input
					t_loss_y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s, labels=tf.ones_like(disc_s)))
					# loss of the discriminator with input from generator
					t_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_s, labels=tf.zeros_like(gen_s)))
			
			t_disc_loss = t_loss_y * weight_dld + t_loss_g
			
			if use_wgan_gp:								
				t_lerp_factor = tf.random_uniform([tf.shape(g_resampled)[0],1,1], 0.0, 1.0)
				y_gp_t = t_lerp_factor * y_resampled + (1 - t_lerp_factor) * g_resampled
				t_out = disc_time_model(y_gp_t, use_batch_norm=bn, train=train, reuse = True, currentUpres = int(round(math.log(upRes, 2))), percentage = percentage)				
				t_out_loss = tf.reduce_mean(t_out)
								
				grads_t = tf.gradients(t_out_loss, [y_gp_t])[0]	
				grads_t = tf.sqrt(tf.reduce_sum(tf.square(grads_t + 1e-4), axis = 1))		
				grad_penalty_t = tf.reduce_mean(wgan_lambda * tf.square(grads_t - wgan_target))
						
				epsilon_penalty_t = tf.reduce_mean(tf.square(disc_s))
				t_disc_loss += epsilon_penalty_t * wgan_epsilon
				t_disc_loss += grad_penalty_t
				
			if use_LSGAN:
				g_loss_t = 0.5 * tf.reduce_mean(tf.square(gen_s - c))		
			else: 	
				if use_wgan_gp:
					g_loss_t = tf.reduce_mean(-gen_s)			
				else:	
					g_loss_t= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_s, labels=tf.ones_like(gen_s)))
			
			gen_loss_complete = gen_loss_complete + kkt * g_loss_t
			
	# setup optimizer for different training/growing stages -> only update the parameters which receive gradients
	if useTempoD:
		t_disc_optimizer = []
		t_disc_optimizer_adam = []
	if use_spatialdisc:
		disc_optimizer = []
		disc_optimizer_adam = []
	gen_optimizer = []
	gen_optimizer_adam = []
	gen_emas = []
	for z in range(int(round(math.log(upRes,2)))):						
		if useTempoD:		
			with tf.control_dependencies(disT_update_ops):
				curr_t_var = []
				if z == 2:
					curr_t_var = t_var	
				else:
					for i in range(0, z+2):
						curr_t_var.extend([var for var in t_var if ("%i"%(2**(i))) in var.name and var not in curr_t_var])
				if use_loss_scaling:									
					t_disc_optimizer_adam.append(tf.train.AdamOptimizer(learning_rates_t[z], beta1=beta, beta2 = beta2,name = "t_adam_%i"%(2**(z+1))))
					t_grads.append(calc_gradients(t_disc_loss, curr_t_var, t_disc_optimizer_adam[z], ls_vars[2]))
					t_disc_optimizer.append(apply_updates(t_grads[z], t_disc_optimizer_adam[z], ls_vars[2], z))
				else:
					t_disc_optimizer.append(tf.train.AdamOptimizer(learning_rates_t[z], beta1=beta, beta2 = beta2,name = "t_adam_%i"%(2**(z+1))).minimize(t_disc_loss, var_list=curr_t_var))
			
		if use_spatialdisc:			
			with tf.control_dependencies(dis_update_ops):
				curr_d_var = []
				if z == 2:
					curr_d_var = d_var
				else:
					for i in range(0, z+2):
						curr_d_var.extend([var for var in d_var if ("%i"%(2**(i))) in var.name and var not in curr_d_var])
				#optimizer for discriminator, uses combined loss, can only change variables of the disriminator
				if use_loss_scaling:				
					disc_optimizer_adam.append(tf.train.AdamOptimizer(learning_rates_d[z], beta1=beta, beta2 = beta2,name = "d_adam_%i"%(2**(z+1))))
					d_grads.append(calc_gradients(disc_loss, curr_d_var, disc_optimizer_adam[z], ls_vars[1]))
					disc_optimizer.append(apply_updates(d_grads[z], disc_optimizer_adam[z], ls_vars[1], z))
				else:
					disc_optimizer_adam.append(tf.train.AdamOptimizer(learning_rates_d[z], beta1=beta, beta2 = beta2,name = "d_adam_%i"%(2**(z+1))))
					disc_optimizer.append(disc_optimizer_adam[z].minimize(disc_loss, var_list=curr_d_var))

		with tf.control_dependencies(gen_update_ops): #gen_update_ops):
			curr_g_var = []
			if z == 2:
				curr_g_var = g_var
			else:
				for i in range(0, z+2):
					curr_g_var.extend([var for var in g_var if ("%i"%(2**(i))) in var.name and var not in curr_g_var])
			if use_loss_scaling:			
				gen_optimizer_adam.append(tf.train.AdamOptimizer(learning_rates_g[z], beta1=beta, beta2 = beta2,name = "g_adam_%i"%(2**(z+1))))
				gen_optimizer_adam[z] = tf.contrib.opt.MovingAverageOptimizer(gen_optimizer_adam[z], 0.999)
				# optimizer for generator, can only change variables of the generator,
				g_grads.append(calc_gradients(gen_loss_complete, curr_g_var, gen_optimizer_adam[z], ls_vars[0]))
				gen_optimizer.append(apply_updates(g_grads[z], gen_optimizer_adam[z], ls_vars[0], z))
			else:
				gen_optimizer_adam.append(tf.contrib.opt.MovingAverageOptimizer(tf.train.AdamOptimizer(learning_rates_g[z], beta1=beta, beta2 = beta2,name = "g_adam_%i"%(2**(z+1)))), 0.999)
				gen_optimizer.append(gen_optimizer_adam[z].minimize(gen_loss_complete, var_list=curr_g_var))
			
# create session and saver
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.InteractiveSession(config = config)
saver = tf.train.Saver(max_to_keep=maxToKeep)
# second saver for estimation of moving average
if not outputOnly:
	saver_2 = gen_optimizer_adam[2].swapping_saver(max_to_keep=maxToKeep)
# init vars or load model
if load_model_test == -1:
	sess.run(tf.global_variables_initializer())
else:
	saver.restore(sess, load_path)
	if not outputOnly:
		saver_2.restore(sess, load_path_ema)
	print("Model restored from %s." % load_path)

if not outputOnly :
	# create a summary to monitor cost tensor
	#training losses
	if use_spatialdisc:
		lossTrain_disc  = tf.summary.scalar("discriminator-loss train",     disc_loss)
		lossTrain_gen  = tf.summary.scalar("generator-loss train",     g_loss_d)

	#testing losses
	if use_spatialdisc:
		lossTest_disc_disc   = tf.summary.scalar("discriminator-loss test real", d_loss_y)
		lossTest_disc_gen   = tf.summary.scalar("discriminator-loss test generated", d_loss_g)
		lossTest_disc = tf.summary.scalar("discriminator-loss test", disc_loss)
		lossTest_gen   = tf.summary.scalar("generator-loss test", g_loss_d)

	#discriminator output [0,1] for real input
	if use_spatialdisc:
		outTrain_disc_real = tf.summary.scalar("discriminator-out train", d_sig_y)
		outTrain_disc_gen = tf.summary.scalar("generator-out train", d_sig_g)

	#discriminator output [0,1] for generated input
	if use_spatialdisc:
		outTest_disc_real = tf.summary.scalar("discriminator-out test", d_sig_y)
		outTest_disc_gen = tf.summary.scalar("generator-out test", d_sig_g)
	
	if(useTempoD): # all temporal losses
		# training losses， discriminator, generator
		lossTrain_disc_t = tf.summary.scalar("T discriminator-loss train", t_disc_loss)
		lossTrain_gen_t = tf.summary.scalar("T generator-loss train", g_loss_t)
		
		# testing losses, discriminator( positive, negative ), generator
		lossTest_disc_disc_t = tf.summary.scalar("T discriminator-loss test real", t_loss_y)
		lossTest_disc_gen_t = tf.summary.scalar("T discriminator-loss test generated", t_loss_g)
		lossTest_disc_t = tf.summary.scalar("T discriminator-loss test", t_disc_loss)
		lossTest_gen_t = tf.summary.scalar("T generator-loss test", g_loss_t)

		# discriminator output [0,1] for real input, during training
		outTrain_disc_real_t = tf.summary.scalar("T discriminator-out train", t_sig_y)
		# discriminator output [0,1] for generated input
		outTrain_disc_gen_t = tf.summary.scalar("T generator-out train", t_sig_g)

		# discriminator output [0,1] for real input, during testing
		outTest_disc_real_t = tf.summary.scalar("T discriminator-out test",  t_sig_y)
		# discriminator output [0,1] for generated input
		outTest_disc_gen_t = tf.summary.scalar("T generator-out test",  t_sig_g)
	
	if (useTempoL2):  # all temporal losses
		lossTrain_gen_t_l = tf.summary.scalar("T generator-loss train l2", tl_gen_loss)
		lossTest_gen_t_l = tf.summary.scalar("T generator-loss test l2", tl_gen_loss)

	merged_summary_op = tf.summary.merge_all()
	summary_writer    = tf.summary.FileWriter(test_path, sess.graph)

save_no = 0
image_no = 0
if not outputOnly:
	os.makedirs(test_path+'test_img/')
	if pretrain>0 or pretrain_gen > 0 or pretrain_disc>0:
		os.makedirs(test_path+'pretrain_test_img/')

def addVorticity(Vel):
	if dataDimension == 2:
		vorout = np.zeros_like(Vel)
		for l in range(vorout.shape[0]):
			for i in range(1, vorout.shape[-3]-1):
				for j in range(1, vorout.shape[-2]-1):
					vorout[l][0][i][j][2] = 0.5 * ((Vel[l][0][i+1][j][1] - Vel[l][0][i-1][j][1]) - (Vel[l][0][i][j+1][0] - Vel[l][0][i][j-1][0]))
	else:
		vorout = np.zeros_like(Vel)
		for l in range(vorout.shape[0]):
			for i in range(1, vorout.shape[-4]-1):
				for j in range(1, vorout.shape[-3]-1):
					for k in range(1, vorout.shape[-2]-1):		
						vorout[l][i][j][k][0] = 0.5 * ((Vel[l][i][j+1][k][2] - Vel[l][i][j-1][k][2]) - (Vel[l][i][j][k+1][1] - Vel[l][i][j][k-1][1]))
						vorout[l][i][j][k][1] = 0.5 * ((Vel[l][i][j][k+1][0] - Vel[l][i][j][k-1][0]) - (Vel[l][i+1][j][k][2] - Vel[l][i-1][j][k][2]))
						vorout[l][i][j][k][2] = 0.5 * ((Vel[l][i+1][j][k][1] - Vel[l][i-1][j][k][1]) - (Vel[l][i][j+1][k][0] - Vel[l][i][j-1][k][0]))
	return vorout

def modifyVel(Dens,Vel):
	if dataDimension == 2:
		velout = Vel
		for l in range(velout.shape[0]):
			for i in range(0, velout.shape[-3]-1):
				for j in range(0, velout.shape[-2]-1):
					#add what you want
					pass
	else:
		velout = Vel
		for l in range(velout.shape[0]):
			for i in range(0, velout.shape[-4]-1):
				for j in range(0, velout.shape[-3]-1):
					for k in range(0, velout.shape[-2]-1):
						#add what you want
						pass
	return velout

def getTempoinput(batch_size = 1, isTraining = True, useDataAugmentation = False, useVelocities = False, useVorticities = False, n_t = 3, dt=0.5, useFlags = False, useK_Eps_Turb = False):
	global currentUpres

	batch_xts, batch_yts, batch_y_pos = tiCr.selectRandomTempoTiles(batch_size, isTraining, useDataAugmentation, n_t, dt)
	real_batch_sz = batch_xts.shape[0]
	
	if useVelocities and useVorticities:
		if( dataDimension == 2):
			batch_xts = np.reshape(batch_xts,[real_batch_sz,1,tileSizeLow,tileSizeLow,-1])
		else:
			batch_xts = np.reshape(batch_xts,[real_batch_sz,tileSizeLow,tileSizeLow,tileSizeLow,-1])
			
		Velinput = batch_xts[:,:,:,:,1:4]			
		Vorinput = addVorticity(Velinput)
		batch_xts = np.concatenate((batch_xts, Vorinput), axis = 4)

	batch_yts = np.reshape(batch_yts,[real_batch_sz, -1])
	batch_y_pos = np.reshape(batch_y_pos,[real_batch_sz, -1])
	batch_xts = np.reshape(batch_xts,[real_batch_sz, -1])
	return batch_xts, batch_yts, batch_y_pos
	
def getinput(index = 1, randomtile = True, isTraining = True, batch_size = 1, useDataAugmentation = False, modifyvelocity = False, useVelocities = False, useVorticities = False, useFlags = False, useK_Eps_Turb = False):
	global currentUpres
	
	if randomtile == False:
		if outputOnly:
			batch_xs, batch_ys = tiCr.getFrameTiles(index) # TODO 120 is hard coded!!
		else:
			batch_xs, batch_ys = tiCr.getFrameTiles(index) # TODO 120 is hard coded!!
	else:
		batch_xs, batch_ys = tiCr.selectRandomTiles(selectionSize = batch_size, augment=useDataAugmentation)	
		
	if useVelocities and useVorticities:
		Velinput = batch_xs[:,:,:,:,1:4] # hard coded, use tiCr.c_lists[DATA_KEY_LOW][C_KEY_VELOCITY][0]		
		Vorinput = addVorticity(Velinput)
		batch_xs = np.concatenate((batch_xs, Vorinput), axis = 4)
	if useVelocities and modifyvelocity:
		Densinput = batch_xs[:,:,:,:,0:1]
		if useFlags:
			Flaginput = batch_xs[:,:,:,:,4:5]
		if useK_Eps_Turb:
			Turbinput = batch_xs[:,:,:,:,5:7]
		Velinput = batch_xs[:,:,:,:,1:4]
		Veloutput = modifyVel(Densinput, Velinput)
		batch_xs = np.concatenate((Densinput, Veloutput))
		if useFlags:
			batch_xs = np.concatenate((batch_xs, Flaginput), axis = 4)
		if useK_Eps_Turb:
			batch_xs = np.concatenate((batch_xs, Turbinput), axis = 4)
			
	# test for removing density in zero-density input areas.... deactivated for now
	if not min(np.random.randint(0,20),1) and 1:
		batch_xs[:,:,:,:,0:1] = 0
		if add_adj_idcs:
			batch_xs[:,:,:,:,4:6] = 0
		batch_xs[:,:,:,:,1:4]*= (1.0  + np.random.rand()*1.5)
		batch_ys[:,:,:,:,:] = 0
		
	batch_xs = np.reshape(batch_xs, (-1, n_input))
	batch_ys = np.reshape(batch_ys, (batch_size, -1))
	return batch_xs, batch_ys

#evaluate the 
def generateTestImage(sim_no = fromSim, frame_no = 5000, outPath = test_path,imageindex = 0, modifyvelocity = False, currentUpres = upRes, inputPer = 1.0):
	if premadeTiles:
		#todo output for premadetiles
		pass
	else:
		batch_xs, batch_ys = getinput(randomtile = False, index = (sim_no-fromSim)*frame_max + frame_no, modifyvelocity = modifyvelocity, useVelocities = useVelocities, useVorticities = useVorticities,  useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)

		if upsampling_mode == 2:
			batch_xtsT = np.reshape(batch_xs, (-1,1,tileSizeLow,tileSizeLow,n_inputChannels))
		elif upsampling_mode == 1 or upsampling_mode == 3:
			batch_xtsT = np.reshape(batch_xs, (-1,1,tileSizeLow,tileSizeLow,n_inputChannels))
		elif upsampling_mode == 0:
			batch_xtsT = np.reshape(batch_xs, (-1,1,tileSizeHigh, tileSizeLow, n_inputChannels + 1))
			
		if upsampling_mode == 2:
			batch_ytsT = np.reshape(batch_ys, (-1,1, tileSizeLow * currentUpres, tileSizeLow*currentUpres,1))
		elif upsampling_mode == 1 or upsampling_mode == 3:
			batch_ytsT = np.reshape(batch_ys, (-1,1,tileSizeHigh,tileSizeHigh,2))
		elif upsampling_mode == 0:
			batch_ytsT = np.reshape(batch_ys, (-1,1,tileSizeHigh,tileSizeHigh,1))
		
		if outputOnly:
			tc.savePngsBatch(batch_xtsT, batch_ytsT, tiCr, outPath, tileSize = tileSizeLow, upRes = currentUpres)
		else:
			if upsampling_mode == 2:
				tc.savePngsBatch(batch_xtsT, batch_ytsT, tiCr, outPath, tileSize = tileSizeLow, upRes = currentUpres)
			elif upsampling_mode != 0:
				tc.savePngsBatch(batch_xtsT, batch_ytsT, tiCr, outPath, tileSize = tileSizeLow, upRes = upRes)
			else:
				tc.savePngsBatch(
				scipy.ndimage.zoom(batch_xtsT[:,:,:,:,0:4],[1, 1, tileSizeLow / tileSizeHigh, 1, 1], order=0, mode = 'constant', cval = 0.0),
				np.concatenate((batch_ytsT,scipy.ndimage.zoom(batch_xtsT[:,:,:,:,4:5],[1, 1, 1, tileSizeHigh // tileSizeLow, 1], order=0, mode = 'constant', cval = 0.0)), axis = 4), tiCr, outPath, tileSize = tileSizeLow, upRes = upRes)
				
		resultTiles = []
		for tileno in range(batch_xtsT.shape[0]):
			batch_xs_in = np.reshape(batch_xtsT[tileno],[-1, n_input])
			
			if upsampling_mode == 2 or upsampling_mode == 0:
				batch_ys_in = np.reshape(batch_ytsT[tileno],[1, -1])
			elif upsampling_mode == 1 or upsampling_mode == 3:
				batch_ys_in = np.reshape(batch_ytsT[tileno],[-1, n_output * 2])
				
			results = sess.run(sampler, feed_dict={x: batch_xs_in, y: batch_ys_in, percentage : inputPer,keep_prob: dropoutOutput, train: False})
			resultTiles.extend(results)		
		
		resultTiles = np.reshape(np.array(resultTiles), (-1, n_output))
		
		if dataDimension == 2: # resultTiles may have a different size
			imgSz = int(resultTiles.shape[1]**(1.0/2) + 0.5)
			resultTiles = np.reshape(resultTiles,[resultTiles.shape[0],imgSz,imgSz, 1])
		else:
			imgSz = int(resultTiles.shape[1]**(1.0/3) + 0.5)
			resultTiles = np.reshape(resultTiles,[resultTiles.shape[0],imgSz,imgSz,imgSz])
		tiles_in_image=[int(simSizeHigh/tileSizeHigh),int(simSizeHigh/tileSizeHigh)]
		tc.savePngsGrayscale(resultTiles,outPath, imageCounter=imageindex, tiles_in_image=tiles_in_image)
		if outputOnly:
			batch_ys_in = np.reshape(batch_ys,[-1, n_output])
			batch_ys_in = np.reshape(batch_ys_in,[1,simSizeHigh,simSizeHigh,1])
			tc.savePngsGrayscale(np.reshape(batch_ys_in,[batch_xs_in.shape[0],int(imgSz),int(imgSz), 1]),outPath, imageCounter=frame_max+imageindex, tiles_in_image=tiles_in_image)

# for two different networks, first upsamples two dimensions, last one upsamples one dim
def generate3DUniForNewNetwork(imageindex = 0, outPath = '../', inputPer = 3.0, head = None):
	global lowDens, n_inputChannels
	start = time.time()
	dim_output = []
	intermed_res1 = []
	if upsampling_mode == 1 or upsampling_mode == 3:
		if transposeAxis == 1:
			batch_xs_tile = scipy.ndimage.zoom(x_3d[imageindex],[1, upRes, 1, 1], order=1, mode = 'constant', cval = 0.0)
		elif transposeAxis == 2:
			batch_xs_tile = scipy.ndimage.zoom(x_3d[imageindex],[1, 1, upRes, 1], order=1, mode = 'constant', cval = 0.0)
		elif transposeAxis == 3:
			batch_xs_tile = scipy.ndimage.zoom(x_3d[imageindex],[1, 1, upRes, 1], order=1, mode = 'constant', cval = 0.0)
		else:
			batch_xs_tile = scipy.ndimage.zoom(x_3d[imageindex],[upRes, 1, 1, 1], order=1, mode = 'constant', cval = 0.0)
	elif upsampling_mode == 0:
		batch_xs_tile = scipy.ndimage.zoom(x_3d[imageindex],[1, upRes, upRes, 1], order=1, mode = 'constant', cval = 0.0)
	elif upsampling_mode == 2:
		batch_xs_tile = x_3d[imageindex]
		# z y x -> 2d conv on y - x 
	if add_adj_idcs:		
		n_inputChannels -= 2
	if upsampling_mode == 2:
		if upsampleFirst:
			# TODO test bicubic interp
			if transposeAxis == 1:
				batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,upRes,1,1] , order = 1), [-1, simSizeHigh, simSizeLow, n_inputChannels])	
			elif transposeAxis == 2:
				batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			elif transposeAxis == 3:
				batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			else:
				batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[upRes,1,1,1] , order = 1), [-1, simSizeLow, simSizeLow, n_inputChannels])			
		else:
			batch_xs_in = np.reshape(batch_xs_tile, [-1, simSizeLow, simSizeLow, n_inputChannels])			
	elif upsampling_mode == 0:
		batch_xs_in = np.concatenate((x_2[imageindex],batch_xs_tile), axis = 3).transpose((2,1,0,3)).reshape(([-1, simSizeHigh, simSizeLow, n_inputChannels]))
		batch_xs_in[:,:,:,1:3] *= upRes
		temp_vel = np.copy(batch_xs_in[:,:,:,1:2])
		batch_xs_in[:,:,:,1:2] = np.copy(batch_xs_in[:,:,:,3:4])
		batch_xs_in[:,:,:,3:4] = np.copy(temp_vel)		
	elif upsampling_mode == 3 or upsampling_mode == 1:
		batch_xs_in = batch_xs_tile
		
	if upsampling_mode != 0:
		if transposeAxis == 1:
			batch_xs_in = np.reshape(batch_xs_in.transpose(1,0,2,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)
			if x_2 is not None:
				batch_ys_in = x_2[imageindex].transpose(1,0,2,3).reshape([-1, simSizeHigh, simSizeHigh, 1])
		elif transposeAxis == 2:
			batch_xs_in = np.reshape(batch_xs_in.transpose(2,1,0,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel)
			if x_2 is not None:
				batch_ys_in = x_2[imageindex].transpose(2,1,0,3).reshape([-1, simSizeHigh, simSizeHigh, 1])
		elif transposeAxis == 3:	
			batch_xs_in = np.reshape(batch_xs_in.transpose(2,0,1,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			temp_vel2 = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel2)
			if x_2 is not None:
				batch_ys_in = x_2[imageindex].transpose(2,0,1,3).reshape([-1, simSizeHigh, simSizeHigh, 1])
		else:
			if x_2 is not None:
				batch_ys_in = x_2[imageindex]
				
	if add_adj_idcs:		
		batch_xs_in = np.concatenate((batch_xs_in, np.zeros_like(batch_xs_in[:,:,:,0:1])),  axis= 3)		
		batch_xs_in = np.concatenate((batch_xs_in, np.zeros_like(batch_xs_in[:,:,:,0:1])),  axis= 3)		
				
		for i in range(batch_xs_in.shape[0]):		
			if i == 0:		
				batch_xs_in[i:i+1,:,:,n_inputChannels:n_inputChannels+1] = np.zeros_like(batch_xs_in[i:i+1,:,:,0:1])		
				batch_xs_in[i:i+1,:,:,n_inputChannels+1:n_inputChannels+2] = batch_xs_in[i+1:i+2,:,:,0:1]				
			elif i == batch_xs_in.shape[0]-1:		
				batch_xs_in[i:i+1,:,:,n_inputChannels:n_inputChannels+1] = batch_xs_in[i-1:i,:,:,0:1]			
				batch_xs_in[i:i+1,:,:,n_inputChannels+1:n_inputChannels+2]= np.zeros_like(batch_xs_in[i-1:i,:,:,0:1])		
			else:		
				batch_xs_in[i:i+1,:,:,n_inputChannels:n_inputChannels+1] = batch_xs_in[i-1:i,:,:,0:1]				
				batch_xs_in[i:i+1,:,:,n_inputChannels+1:n_inputChannels+2] = batch_xs_in[i+1:i+2,:,:,0:1]			
		n_inputChannels +=2
		
	end = time.time()
	print("time for interp low res: {0:.6f}".format(end-start))
	batch_sz_out = 2
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	start = time.time()
	for j in range(0,batch_xs_in.shape[0]//batch_sz_out):
		if upsampling_mode == 2:
			results = sess.run(sampler, feed_dict={x: batch_xs_in[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_input), percentage : inputPer, keep_prob: dropoutOutput, train: False})
		else:
			results = sess.run(sampler, feed_dict={x: batch_xs_in[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_input), y: batch_ys_in[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_output) ,percentage : inputPer, keep_prob: dropoutOutput, train: False})
		intermed_res1.extend(results)	
		
		# accurate performance of the generator...
		if 0:
			fetched_timeline = timeline.Timeline(run_metadata.step_stats)
			chrome_trace = fetched_timeline.generate_chrome_trace_format()
			with open('timeline_8x_%04d.json'%(j), 'w') as f:
				f.write(chrome_trace)
	end = time.time()
	
	print("time for network: {0:.6f}".format(end-start))
			
	if upsampling_mode == 2:
		if upsampleFirst:
			dim_output = np.array(intermed_res1).reshape(simSizeHigh, simSizeHigh, simSizeHigh)
		else:
			dim_output = np.array(intermed_res1).reshape(simSizeLow, simSizeHigh, simSizeHigh)
	elif upsampling_mode == 3 or upsampling_mode == 0 or upsampling_mode == 1:
		dim_output = np.array(intermed_res1).reshape(simSizeHigh, simSizeHigh, simSizeHigh)
	
	save_img_3d( outPath + 'source_{:04d}.png'.format(imageindex+frame_min), dim_output/80)	
			
	if transposeAxis == 1:
		dim_output = dim_output.transpose(1,0,2)
	elif transposeAxis == 2:
		dim_output = dim_output.transpose(2,1,0)	
	elif transposeAxis == 3:
		dim_output = dim_output.transpose(1,2,0)	
	
	# output for images of slices (along every dimension + diagonally)
	if 1:
		for i in range(simSizeHigh // 2 - 3, simSizeHigh // 2 + 3):
			if np.average(dim_output[i]) > 0.0001:
				save_img(outPath + 'slice_xy_{:04d}_{:04d}.png'.format(i,(imageindex+frame_min)), dim_output[i]) #.transpose(2,1,0)
				save_img(outPath + 'slice_yz_{:04d}_{:04d}.png'.format(i,(imageindex+frame_min)), dim_output.transpose(2,1,0)[i])
				save_img(outPath + 'slice_xz_{:04d}_{:04d}.png'.format(i,(imageindex+frame_min)), dim_output.transpose(1,0,2)[i])
	if (imageindex + frame_min) == 110:
		for i in range(0, tileSizeHigh):
			if np.average(dim_output[i]) > 0.0001:
				save_img(outPath + 'slice_xy_{:04d}_{:04d}.png'.format((imageindex+frame_min),i), dim_output[i]) #.transpose(2,1,0)
				save_img(outPath + 'slice_yz_{:04d}_{:04d}.png'.format((imageindex+frame_min),i), dim_output.transpose(2,1,0)[i])
				save_img(outPath + 'slice_xz_{:04d}_{:04d}.png'.format((imageindex+frame_min),i), dim_output.transpose(1,0,2)[i])
	if 0:
		new_arr = np.zeros_like(dim_output[0])
		for i in range(0, 512):
			for j in range(0, 512):
				new_arr[i, j] = dim_output[i,i,j]
		save_img(outPath + 'slice_z_{:04d}.png'.format((imageindex+frame_min)), new_arr)
		new_arr = np.zeros_like(dim_output[0])
		for i in range(0, 512):
			for j in range(0, 512):
				new_arr[i, j] = dim_output[i,j,j]
		save_img(outPath + 'slice_x_{:04d}.png'.format((imageindex+frame_min)), new_arr)
		new_arr = np.zeros_like(dim_output[0])
		for i in range(0, 512):
			for j in range(0, 512):
				new_arr[i, j] = dim_output[i,j,i]
		save_img(outPath + 'slice_y_{:04d}.png'.format((imageindex+frame_min)), new_arr)
	start = time.time()
	
	if head is None:
		head, _ = uniio.readUni(packedSimPath + "sim_%04d/density_high_%04d.uni"%(fromSim, 0))
	head['dimX'] = simSizeHigh
	head['dimY'] = simSizeHigh
	head['dimZ'] = simSizeHigh
	if not upsampleFirst:
		head['dimZ'] = simSizeLow	
	if generateUni:
		cond_out = dim_output < 0.0005
		dim_output[cond_out] = 0
		if upsampling_mode == 2:
			if upsampleFirst:
				uniio.writeUni(packedSimPath + '/sim_%04d/density_low_t%04d_2x2_%04d.uni'%(fromSim, load_model_test, imageindex+frame_min), head, dim_output)
			else:
				uniio.writeUni(packedSimPath + '/sim_%04d/density_low_t%04d_2x2x1_%04d.uni'%(fromSim,load_model_test, imageindex+frame_min), head, dim_output)
		elif upsampling_mode == 1:
			uniio.writeUni(packedSimPath + '/sim_%04d/density_low_t%04d_1x1_%04d.uni'%(fromSim,load_model_test, imageindex+frame_min), head, dim_output)
		elif upsampling_mode == 0:
			uniio.writeUni(packedSimPath + '/sim_%04d/density_low_t%04d_1x1x1_%04d.uni'%(fromSim,load_model_test, imageindex+frame_min), head, dim_output)
		elif upsampling_mode == 3:
			uniio.writeUni(packedSimPath + '/sim_%04d/density_low_0x0_%04d.uni'%(fromSim, imageindex+frame_min), head, dim_output)
	end = time.time()
	print("time for writing uni file: {0:.6f}".format(end-start))
		
# copy adam variables between growing stages
def copyAdamVariables(curr_upres):
	vars = tf.global_variables()
	copy_vars = []
	paste_vars = vars
	tempName = "adam_%i"%curr_upres
	tempName1 = "adam_%i"%curr_upres
	for var in vars:		
		if "adam_%i"%curr_upres in var.name or  "adam_%i_1"%curr_upres in var.name:
			copy_vars.append(var)	
	output_vars = []
	for var in copy_vars:
		lengthName = len(var.name)
		tempName1 = var.name[0:lengthName-8] + "adam_%i"%(curr_upres*2) + ":0"
		tempName2 = var.name[0:lengthName-10] + "adam_%i_1"%(curr_upres*2) + ":0"
		for var_paste in paste_vars:
			if tempName1 == var_paste.name or tempName2 == var_paste.name:
				var_paste.assign(var)
				output_vars.append(var_paste)
	init_new_vars = tf.variables_initializer(output_vars)
	sess.run(init_new_vars)
	
def saveModel(cost, exampleOut=-1, imgPath = test_path):
	global save_no
	saver.save(sess, test_path + 'model_%04d.ckpt' % save_no)
	saver_2.save(sess, test_path + 'model_ema_%04d.ckpt' % save_no)
	msg = 'Saved Model %04d with cost %f.' % (save_no, cost)
	if exampleOut > -1:
		#pass
		generateTestImage(imageindex = save_no, outPath = imgPath)
	save_no += 1
	return msg

# write summary to test overview
loaded_model = ''
if not load_model_test == -1:
	loaded_model = ', Loaded %04d, %04d' % (load_model_test , load_model_no)
with open(basePath + 'test_overview.log', "a") as text_file:
	if not outputOnly:
		text_file.write(test_path[-10:-1] + ': {}D, \"{}\"\n'.format(dataDimension, note))
		text_file.write('\t{} Epochs, gen: {}, disc: {}'.format(trainingIterations, gen_model.__name__, disc_model.__name__) + loaded_model + '\n')
		text_file.write('\tgen-runs: {}, disc-runs: {}, lambda: {}, dropout: {:.4f}({:.4f})'.format(genRuns, discRuns, k, dropout, dropoutOutput) + '\n')
	else:
		text_file.write('Output:' + loaded_model + ' (' + test_path[-28:-1] + ')\n')
		text_file.write('\ttile size: {}, seed: {}, dropout-out: {:.4f}'.format(tileSizeLow, randSeed, dropoutOutput) + '\n')
	
# ---------------------------------------------
# ---------------------------------------------
# START TRAINING
training_duration = 0.0
cost = 0.0

if not outputOnly and trainGAN:
	try:
		print('\n*****TRAINING STARTED*****\n')
		print('(stop with ctrl-c)')
		avgCost_disc = 0
		avgCost_gen = 0
		avgL1Cost_gen = 0
		avgOut_disc = 0
		avgOut_gen = 0

		avgTestCost_disc_real = 0
		avgTestCost_disc_gen = 0
		avgTestCost_gen = 0
		avgTestOut_disc_real = 0
		avgTestOut_disc_gen = 0
		tests = 0
		startTime = time.time()
		startEpoch = startingIter
		intervalTime = startTime
		lastOut = 1
		lastSave = 1
		lastCost = 1e10
		saved = False
		saveMsg = ''
		kkin = k
		kk2in = k2
		disc_cost = 0
		gen_cost = 0
		
		avgTemCost_gen = 0
		avgTemCost_gen_l = 0
		avgTemCost_disc = 0
		kktin = kt
		kktin_l = kt_l

		avgOut_disc_t = 0
		avgOut_gen_t = 0
		avgTestCost_disc_real_t = 0
		avgTestOut_disc_real_t = 0
		avgTestCost_disc_gen_t = 0
		avgTestOut_disc_gen_t = 0
		avgTestCost_gen_t = 0
		avgTestCost_gen_t_l = 0
		
		t_fakeScores_train = 0.0
		d_fakeScores_train = 0.0
		gdrop_strength_d = 0.0
		gdrop_strength_t = 0.0
		check_gdrop_interval = 1
		
		# initalize blend value, etc.
		currBlendPer = 1.0
		logging_each = 1
		interpolate_Perc = True
		start_interpol = stageIter * int(math.floor(startingIter // (stageIter * 2))) * 2 + stageIter
		interpol_c = int(math.floor(startingIter // (stageIter * 2)) * stageIter)
		if (startingIter // stageIter) % 2 == 0: 
			interpol_c += (startingIter-interpol_c) % stageIter
			interpol_c += stageIter
		else:
			interpolate_Perc = False
		lrgs = 0
		stride = 3
		
		for it in range(startingIter, trainingIterations): # LR counter, start decay at half time... (if enabled) 
		
			if 0 and decayLR: # only for debugging of LR: this only outputs the current rate, not needed for training
				lrt3,lrt3g = sess.run([disc_optimizer_adam._lr_t, lr_global_step], feed_dict={lr_global_step: lrgs})
				print('\nDebug info: current learning rate, step {} * 1000 = {} '.format(lrt3g, lrt3*1000. ))
				
			# disable incrementing of interpol_c	
			if it - start_interpol == 0:
				interpolate_Perc = False
				
			# load new data when completing a growing stage  (e.g,: 2x -> 4x) only applicable for upsampling_mode == 2
			if it - start_interpol == stageIter and currentUpres < upRes:	
				startTime = time.time()		
				startEpoch = it
				copyAdamVariables(currentUpres)
				saveModel(0.0)
				currentUpres *= 2
				start_interpol = it + stageIter
				interpolate_Perc = True		
				
				if upsampling_mode == 2:
					tiCr = []
					if upsampled_data:
						floader_2 = []
					floader = []
					gc.collect()
					tiCr = tc.TileCreator(tileSizeLow=tileSizeLow, densityMinimum=0.01, channelLayout_high=channelLayout_high, simSizeLow=simSizeLow , dim =dataDimension, dim_t = 3, channelLayout_low = channelLayout_low, upres=currentUpres, premadeTiles=premadeTiles)
					
					scale_y = [1,1,1,1]
					scale = [currentUpres,1,1,1]
					# can be reduced if higher data is loaded for example
					select_random_data = 1.0
					# hard coded so far
					if upsampled_data:
						floader_2 = FDL.FluidDataLoader( print_info=0, base_path=packedSimPath, base_path_y = packedSimPath, numpy_seed = randSeed, add_adj_idcs = add_adj_idcs, conv_slices = True, conv_axis = transpose_axis, select_random = select_random_data, density_threshold = 0.002, axis_scaling_y = scale_y, axis_scaling = scale, filename=lowfilename, oldNamingScheme=False, filename_y=lowfilename_2, filename_index_max=frame_max+stride * int(round(math.log(currentUpres, 2))-1),filename_index_min = frame_min+stride * int(round(math.log(currentUpres, 2))-1), indices=dirIDs, data_fraction=max(data_fraction*2/currentUpres,min_data_fraction), multi_file_list=mfl_2, multi_file_idxOff=mol_2, multi_file_list_y=mfh , multi_file_idxOff_y=moh) # data_fraction=0.1
					
					if currentUpres == upRes:
						currentHighFileName = highfilename
					else:
						currentHighFileName = "density_low_%i"%(currentUpres)+"_%04d.uni"
					floader = FDL.FluidDataLoader( print_info=0, base_path=packedSimPath, base_path_y = packedSimPath, numpy_seed = randSeed, add_adj_idcs = add_adj_idcs, conv_slices = True, conv_axis = transpose_axis, select_random = select_random_data, density_threshold = 0.002, axis_scaling_y = scale_y, axis_scaling = scale, filename=lowfilename, oldNamingScheme=False, filename_y=currentHighFileName, filename_index_max=frame_max+stride * int(round(math.log(currentUpres, 2))-1),filename_index_min = frame_min+stride * int(round(math.log(currentUpres, 2))-1), indices=dirIDs, data_fraction=data_fraction #max(data_fraction*2/currentUpres,min_data_fraction)
					, multi_file_list=mfl, multi_file_idxOff=mol, multi_file_list_y=mfh , multi_file_idxOff_y=moh) # data_fraction=0.1
					print('loaded different files')
					if useDataAugmentation:
						tiCr.initDataAugmentation(rot=rot, minScale=minScale, maxScale=maxScale ,flip=flip)

					if upsampled_data:	
						_, x_2, _ = floader_2.get()
					xTmp, yTmp, _  = floader.get()

					floader = []
					floader_2 = []
					gc.collect()

					if not outputOnly:
						print('start converting to 2d slices')
						xTmp = xTmp.reshape(-1, 1, simSizeLow, simSizeLow, n_inputChannels * 3)
						yTmp = yTmp.reshape(-1, 1, simSizeLow*currentUpres, simSizeLow*currentUpres, 3)						
					x_2 = []
					gc.collect()
					if not outputOnly:	
						tiCr.addData(xTmp,yTmp)
					xTmp = []
					yTmp = []
					gc.collect()
				print("--------------------------NEW UPRES: %d--------------------------" % (currentUpres))
				
			if interpolate_Perc: 
				interpol_c += 1
				currBlendPer = (interpol_c)/stageIter
			else:
				currBlendPer = int(round((interpol_c)/stageIter))
				
			# set limites of blend percentage (lower limit could be ignored)
			if currBlendPer < 1.0:
				currBlendPer = 1.0
			if currBlendPer > 3.0:
				currBlendPer = 3.0
				
			index = int(round(math.log(currentUpres, 2))-1)

			if it >= stageIter*6 and decayLR:
				lrgs += 1
						
			run_options = None; run_metadata = None
			if saveMD:
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
			# TRAIN MODEL
			#discriminator variables; with real and generated input
				
			if use_spatialdisc:
				for runs in range(discRuns):
					batch_xs, batch_ys = getinput(batch_size = batch_size_disc, useDataAugmentation = useDataAugmentation, useVelocities = useVelocities, useVorticities = useVorticities,  useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					
					if use_LSGAN:
						_, disc_cost,  summary,disc_sig,gen_sig = sess.run([disc_optimizer[index], disc_loss, lossTrain_disc,d_sig_y,d_sig_g ], feed_dict={x: batch_xs, percentage: currBlendPer, x_disc: batch_xs, y: batch_ys, keep_prob: dropout, train: True, lr_global_step: lrgs, gdrop_str_d: gdrop_strength_d}     , options=run_options, run_metadata=run_metadata )
					else:
						_, disc_cost,  summary,disc_sig,gen_sig = sess.run([disc_optimizer[index], disc_loss, lossTrain_disc,d_sig_y,d_sig_g ], feed_dict={x: batch_xs, percentage: currBlendPer, x_disc: batch_xs, y: batch_ys, keep_prob: dropout, train: True, lr_global_step: lrgs}     , options=run_options, run_metadata=run_metadata )
					
					avgCost_disc += disc_cost
					summary_writer.add_summary(summary, it)
					if saveMD: summary_writer.add_run_metadata(run_metadata, 'dstep%d' % it)

			# temporal discriminator
			if(useTempoD):
				for runs in range(discRuns):
					batch_xts, batch_yts, batch_y_pos = getTempoinput(batch_size_disc, n_t = 3, dt=0.5, useVelocities = useVelocities, useVorticities = useVorticities, useDataAugmentation = useDataAugmentation, useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					dict_train = {x_t:batch_xts, percentage: currBlendPer, y_t:batch_yts, keep_prob: dropout, train: True}
					if(ADV_flag): dict_train[y_pos] = batch_y_pos
					if use_LSGAN: dict_train[gdrop_str_t] = gdrop_strength_t
					
					_, t_disc_cost, summary, t_disc_sig, t_gen_sig = sess.run(	[t_disc_optimizer[index], t_disc_loss, lossTrain_disc_t, t_sig_y, t_sig_g], feed_dict=dict_train)
					avgTemCost_disc += t_disc_cost				
					summary_writer.add_summary(summary, it)
			gdrop_strength_d = 0.0
			gdrop_strength_t = 0.0
			#generator variables
			for runs in range(genRuns):
				batch_xs, batch_ys = getinput(batch_size = batch_size_disc, useDataAugmentation = useDataAugmentation, useVelocities = useVelocities, useVorticities = useVorticities,  useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
				kkin = k_f*kkin
				kk2in = k2_f * kk2in
				# TODO a decay for weights, kktin = kt_f * kktin (kt_f<1.0)
				
				train_dict = {x: batch_xs, x_disc: batch_xs, y: batch_ys, percentage: currBlendPer, keep_prob: dropout, train: True, kk: kkin,
							  kk2: kk2in, lr_global_step: lrgs, gdrop_str_t: gdrop_strength_t, gdrop_str_d: gdrop_strength_d}
				if use_spatialdisc:
					getlist = [gen_optimizer[index], gen_loss_complete,disc_loss_layer, g_loss_d, l1_loss, lossTrain_gen, d_sig_g, t_sig_g]
				else:
					getlist = [gen_optimizer[index], gen_loss_complete,disc_loss_layer, gen_l1_loss, d_sig_g, t_sig_g]
				if(useTempoD or useTempoL2):
					batch_xts, batch_yts, batch_y_pos = getTempoinput(batch_size_disc, n_t = 3, dt=0.5, useVelocities = useVelocities, useVorticities = useVorticities, useDataAugmentation=useDataAugmentation, useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					train_dict[x_t] = batch_xts
					train_dict[y_t] = batch_yts
					if(ADV_flag):
						train_dict[y_pos] = batch_y_pos
					if(useTempoD): 
						train_dict[kkt] = kktin
						getlist.append(g_loss_t)
					if(useTempoL2): 
						train_dict[kktl] = kktin_l
						getlist.append(tl_gen_loss)

				result_list = sess.run(getlist, feed_dict=train_dict, options=run_options, run_metadata=run_metadata)
					
				if (useTempoD and (not useTempoL2)):
					if use_spatialdisc:
						_,_,layer_cost, gen_cost,  gen_l1_cost, summary, d_curr_fake,t_curr_fake, gen_tem_cost = result_list 
					else:
						_,_,layer_cost, gen_l1_cost, d_curr_fake,t_curr_fake, gen_tem_cost = result_list
					gen_tem_cost_l = 0
				elif ((not useTempoD) and useTempoL2):
					if use_spatialdisc:
						_,_,layer_cost, gen_cost,  gen_l1_cost, summary, d_curr_fake,t_curr_fake, gen_tem_cost_l = result_list
					else:
						_,_,layer_cost, gen_l1_cost, d_curr_fake,t_curr_fake, gen_tem_cost_l = result_list
					gen_tem_cost = 0
				elif (useTempoD and useTempoL2):
					if use_spatialdisc:
						_,_,layer_cost, gen_cost,  gen_l1_cost, summary, d_curr_fake,t_curr_fake, gen_tem_cost, gen_tem_cost_l = result_list
					else:
						_,_,layer_cost, gen_l1_cost, d_curr_fake,t_curr_fake, gen_tem_cost, gen_tem_cost_l = result_list
					
				else:
					if use_spatialdisc:
						_,_,layer_cost, gen_cost,  gen_l1_cost, summary, d_curr_fake,t_curr_fake = result_list
					else:
						_,_,layer_cost, gen_l1_cost, d_curr_fake, t_curr_fake = result_list
					gen_tem_cost = 0
					gen_tem_cost_l = 0
				avgL1Cost_gen += gen_l1_cost
				avgTemCost_gen += gen_tem_cost
				avgTemCost_gen_l += gen_tem_cost_l
				if use_spatialdisc:
					avgCost_gen += gen_cost
					summary_writer.add_summary(summary, it)
				if saveMD: summary_writer.add_run_metadata(run_metadata, 'gstep%d' % it)

			# save model
			if ((disc_cost+gen_cost < lastCost) or alwaysSave) and (lastSave >= saveInterval):
				lastSave = 1
				lastCost = disc_cost+gen_cost
				saveMsg = saveModel(lastCost)
				saved = True
			else:
				lastSave += 1
				saved = False

			if it % check_gdrop_interval == 0:
				if use_LSGAN:
					d_fakeScores_train = d_fakeScores_train * gdrop_beta + (1.0 - d_curr_fake) * (1.0 - gdrop_beta)
					gdrop_strength_d = gdrop_coef * (max(d_fakeScores_train - gdrop_lim, 0.0) ** gdrop_exp)
				if use_LSGAN:
					t_fakeScores_train = t_fakeScores_train * gdrop_beta + (1.0 - t_curr_fake) * (1.0 - gdrop_beta)
					gdrop_strength_t = gdrop_coef * (max(t_fakeScores_train - gdrop_lim, 0.0) ** gdrop_exp)
				
			# test model
			if (it + 1) % testInterval == 0:
				if use_spatialdisc:
					# gather statistics from training
					# not yet part of testing!
					batch_xs, batch_ys = getinput(batch_size = numTests, useVelocities = useVelocities, useVorticities = useVorticities, useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					disc_out, summary_disc_out, gen_out, summary_gen_out = sess.run([d_sig_y, outTrain_disc_real, d_sig_g, outTrain_disc_gen], feed_dict={x: batch_xs,percentage:currBlendPer, x_disc: batch_xs, y: batch_ys, keep_prob: dropout, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t})
					summary_writer.add_summary(summary_disc_out, it)
					summary_writer.add_summary(summary_gen_out, it)
					avgOut_disc += disc_out
					avgOut_gen += gen_out

					# testing starts here...
					# get test data
					batch_xs, batch_ys = getinput(batch_size = numTests, isTraining=False, useVelocities = useVelocities, useVorticities = useVorticities,  useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					#disc with real imput
					disc_out_real, summary_test_out, disc_test_cost_real, summary_test = sess.run([d_sig_y, outTest_disc_real, d_loss_y, lossTest_disc_disc], feed_dict={x: batch_xs,percentage:currBlendPer, x_disc: batch_xs, y: batch_ys, keep_prob: dropoutOutput, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t})
					summary_writer.add_summary(summary_test, it)
					summary_writer.add_summary(summary_test_out, it)
					avgTestCost_disc_real += disc_test_cost_real
					avgTestOut_disc_real += disc_out_real
					#disc with generated input
					disc_out_gen, summary_test_out, disc_test_cost_gen, summary_test = sess.run([d_sig_g, outTest_disc_gen, d_loss_g, lossTest_disc_gen], feed_dict={x: batch_xs, y: batch_ys,percentage:currBlendPer, x_disc: batch_xs, keep_prob: dropoutOutput, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t})
					summary_writer.add_summary(summary_test, it)
					summary_writer.add_summary(summary_test_out, it)
					avgTestCost_disc_gen += disc_test_cost_gen
					avgTestOut_disc_gen += disc_out_gen
				
				if(useTempoD): # temporal logs
					# T disc output with training data
					batch_xts, batch_yts, batch_y_pos = getTempoinput(numTests, useVelocities = useVelocities, useVorticities = useVorticities, n_t = 3, dt=0.5, useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					test_dict = {x_t: batch_xts, y_t: batch_yts,percentage:currBlendPer, keep_prob: dropout, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t}
					if(ADV_flag):
						test_dict[y_pos] = batch_y_pos
					t_disc_out, summary_disc_out_t, t_gen_out, summary_gen_out_t = sess.run(
						[t_sig_y, outTrain_disc_real_t, t_sig_g, outTrain_disc_gen_t],
						feed_dict=test_dict)
					summary_writer.add_summary(summary_disc_out_t, it)
					summary_writer.add_summary(summary_gen_out_t, it)
					avgOut_disc_t += t_disc_out
					avgOut_gen_t += t_gen_out

					# test data
					batch_xts, batch_yts, batch_y_pos = getTempoinput(numTests, isTraining=False, useVelocities = useVelocities, useVorticities = useVorticities, n_t = 3, dt=0.5, useFlags = useFlags, useK_Eps_Turb = useK_Eps_Turb)
					# disc with real input
					test_dict = {x_t: batch_xts,percentage:currBlendPer, y_t: batch_yts, keep_prob: dropout, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t}
					if(ADV_flag):
						test_dict[y_pos] = batch_y_pos
					t_disc_out_real, summary_test_out_t, t_disc_test_cost_real, summary_test_t = sess.run(
						[t_sig_y, outTest_disc_real_t, t_loss_y, lossTest_disc_disc_t],
						feed_dict=test_dict)
					summary_writer.add_summary(summary_test_t, it)
					summary_writer.add_summary(summary_test_out_t, it)
					avgTestCost_disc_real_t += t_disc_test_cost_real
					avgTestOut_disc_real_t += t_disc_out_real
					# disc with generated input
					test_dict = {x_t: batch_xts,percentage:currBlendPer, y_t: batch_yts, keep_prob: dropout, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t}
					if(ADV_flag):
						test_dict[y_pos] = batch_y_pos
					t_disc_out_gen, summary_test_out_t, t_disc_test_cost_gen, summary_test_t = sess.run(
						[t_sig_g, outTest_disc_gen_t, t_loss_g, lossTest_disc_gen_t],
						feed_dict=test_dict)
					summary_writer.add_summary(summary_test_t, it)
					summary_writer.add_summary(summary_test_out_t, it)
					avgTestCost_disc_gen_t += t_disc_test_cost_gen
					avgTestOut_disc_gen_t += t_disc_out_gen
					
				#gen
				train_dict = {x: batch_xs, y: batch_ys, x_disc: batch_xs, percentage:currBlendPer, keep_prob: dropoutOutput, train: False, gdrop_str_d: gdrop_strength_d, gdrop_str_t: gdrop_strength_t}
				if (useTempoD or useTempoL2):  # add tempo logs
					train_dict[x_t] = batch_xts
					train_dict[y_t] = batch_yts
					# train_dict[y_t] = batch_yts; # should be useless ?
					if(ADV_flag):
						train_dict[y_pos] = batch_y_pos
					if (useTempoD):
						train_dict[kkt] = kktin
						if use_spatialdisc:
							gen_test_cost, summary_test, gen_tem_cost, summary_test_gen \
								= sess.run([g_loss_d, lossTest_gen, g_loss_t, lossTest_gen_t], feed_dict=train_dict)
						else:
							gen_tem_cost, summary_test_gen \
								= sess.run([g_loss_t, lossTest_gen_t], feed_dict=train_dict)
						avgTestCost_gen_t += gen_tem_cost
					if (useTempoL2):
						train_dict[kktl] = kktin_l
						if use_spatialdisc:
							gen_test_cost, summary_test, gen_tem_cost, summary_test_gen \
								= sess.run([g_loss_d, lossTest_gen, tl_gen_loss, lossTest_gen_t_l], feed_dict=train_dict)
						else:
							gen_tem_cost, summary_test_gen \
								= sess.run([tl_gen_loss, lossTest_gen_t_l], feed_dict=train_dict)
						avgTestCost_gen_t_l += gen_tem_cost
					summary_writer.add_summary(summary_test_gen, it)

				else:
					if use_spatialdisc:
						gen_test_cost, summary_test = sess.run([g_loss_d, lossTest_gen], feed_dict=train_dict)
				if use_spatialdisc:	
					summary_writer.add_summary(summary_test, it)
					avgTestCost_gen += gen_test_cost

				tests += 1

			# output statistics
			if (it + 1) % outputInterval == 0:
				#training average costs
				avgCost_disc /= (outputInterval * discRuns)
				avgCost_gen /= (outputInterval * genRuns)
				avgL1Cost_gen /= (outputInterval * genRuns)
				#test average costs
				if not (tests == 0):
					avgOut_disc /= tests
					avgOut_gen /= tests
					avgTestCost_disc_real /= tests
					avgTestCost_disc_gen /= tests
					avgTestCost_gen /= tests
					avgTestOut_disc_real /= tests
					avgTestOut_disc_gen /= tests
					
				if(useTempoD):
					avgTemCost_gen /= (outputInterval * genRuns)
					avgTemCost_disc /= (outputInterval * discRuns)
					if( not tests == 0):
						avgOut_disc_t /= tests
						avgOut_gen_t /= tests
						avgTestCost_disc_real_t /= tests
						avgTestOut_disc_real_t /= tests
						avgTestCost_disc_gen_t /= tests
						avgTestOut_disc_gen_t /= tests
						avgTestCost_gen_t /= tests
						
				if (useTempoL2):
					avgTemCost_gen_l /= (outputInterval * genRuns)
					if (not tests == 0):
						avgTestCost_gen_t_l /= tests
						
				print('\nEpoch {:05d}/{}, Cost:'.format((it + 1), trainingIterations))
				print('\tdisc: loss: train_loss={:.6f} - test-real={:.6f} - test-generated={:.6f}, out: train={:.6f} - test={:.6f}'.
					format(avgCost_disc, avgTestCost_disc_real, avgTestCost_disc_gen, avgOut_disc, avgTestOut_disc_real))
				print('\tT D : loss[ -train (total={:.6f}), -test (real&1={:.6f}) (generated&0={:.6f})]'.
					format(avgTemCost_disc, avgTestCost_disc_real_t, avgTestCost_disc_gen_t))
				print('\t	sigmoidout[ -test (real&1={:.6f}) (generated&0={:.6f})'.
					format(avgTestOut_disc_real_t, avgTestOut_disc_gen_t))
				print('\t gen: loss: train={:.6f} - L1(*k)={:.3f} - test={:.6f}, DS out: train={:.6f} - test={:.6f}'
					.format(avgCost_gen, avgL1Cost_gen * k, avgTestCost_gen, avgOut_gen, avgTestOut_disc_gen))
				print('\t gen: loss[ -train (total Temp(*k)={:.6f}) -test (total Temp(*k)={:.6f})], DT out: real={:.6f} - gen={:.6f}'
					.format(avgTemCost_gen * kt, avgTestCost_gen_t * kt, avgOut_disc_t, avgOut_gen_t))
				if use_spatialdisc:
					print('\tSdisc: out real train: real=%f'%(disc_sig))
					print('\tSdisc: out gen train: gen=%f'%(gen_sig))
					print('\t layer_cost: %f'%(layer_cost))
				if(useTempoD):
					print('\tTdisc: out real train: disc=%f' % (t_disc_sig))
					print('\tTdisc: out gen train: gen=%f' % (t_gen_sig))
					print('\t tempo_cost: %f' % (gen_tem_cost))
				print('\t blending percentage: %f'%(currBlendPer))
				print('\t D_gdrop_str: %f'%(gdrop_strength_d))
				print('\t T_gdrop_str: %f'%(gdrop_strength_t))
				print('\t l1_cost: %f'%(gen_l1_cost))
				print('\t l2 tempo loss[ -train (total Temp(*k)={:.6f}) -test (total Temp(*k)={:.6f})]'
					.format(avgTemCost_gen_l * kt_l, avgTestCost_gen_t_l * kt_l))
				
				epochTime = (time.time() - startTime) / (it - startEpoch + 1)
				print('\t{} epochs took {:.2f} seconds. (Est. next: {})'.format(outputInterval, (time.time() - intervalTime), time.ctime(time.time() + outputInterval * epochTime)))
				remainingTime = (trainingIterations - it) * epochTime
				print('\tEstimated remaining time: {:.2f} minutes. (Est. end: {})'.format(remainingTime / 60.0, time.ctime(time.time() + remainingTime)))
				if saved:
					print('\t' + saveMsg) # print save massage here for clarity
				if genTestImg > -1:
					#if genTestImg <= lastOut:
					generateTestImage(outPath = test_path+'test_img/', imageindex = image_no, inputPer = currBlendPer, currentUpres = currentUpres)
					image_no +=1
				sys.stdout.flush()
				intervalTime = time.time()
				avgCost_disc = 0
				avgCost_gen = 0
				avgL1Cost_gen = 0
				avgOut_disc = 0
				avgOut_gen = 0
				avgTestCost_disc_real = 0
				avgTestCost_disc_gen = 0
				avgTestCost_gen = 0
				avgTestOut_disc_real = 0
				avgTestOut_disc_gen = 0
				tests = 0
				lastOut = 0
				
				if(useTempoD):
					avgTemCost_gen = 0
					avgTemCost_disc = 0
					avgOut_disc_t = 0
					avgOut_gen_t = 0
					avgTestCost_disc_real_t = 0
					avgTestOut_disc_real_t = 0
					avgTestCost_disc_gen_t = 0
					avgTestOut_disc_gen_t = 0
					avgTestCost_gen_t = 0
					
				if (useTempoL2):
					avgTemCost_gen_l = 0
					avgTestCost_gen_t_l = 0

			lastOut +=1

	except KeyboardInterrupt:
		print("training interrupted")
		sys.stdout.flush()
		with open(basePath + 'test_overview.log', "a") as text_file:
			text_file.write('\ttraining interrupted after %d epochs' % (it + 1) + '\n')

	print('\n*****TRAINING FINISHED*****')
	training_duration = (time.time() - startTime) / 60.0
	print('Training needed %.02f minutes.' % (training_duration))
	print('To apply the trained model, set "outputOnly" to True, and insert numbers for "load_model_test", and "load_model_no" ')
	sys.stdout.flush()
	with open(basePath + 'test_overview.log', "a") as text_file:
		text_file.write('\ttraining duration: %.02f minutes' % training_duration + '\n')


### OUTPUT MODE ###

elif outputOnly: #may not work if using tiles smaller than full sim size
	print('*****OUTPUT ONLY*****')
	#print("{} tiles, {} tiles per image".format(100, 1))
	#print("Generating images (batch size: {}, batches: {})".format(1, 100))

	head_0, _ = uniio.readUni(packedSimPath + "sim_%04d/density_low_%04d.uni"%(fromSim, 0))
	for layerno in range(frame_min,frame_max):
		print(layerno)
		generate3DUniForNewNetwork(imageindex = layerno - frame_min, outPath = test_path, head = head_0)
	if outputGif: #write gif
		print("Writing gif")
		#pg.array_to_gif(test_path, output_complete[:numOut], tileSizeHigh)
		pg.pngs_to_gif(test_path,end_idx = img_count)
	print('Test finished, %d pngs written to %s.' % (numOut, test_path) )

