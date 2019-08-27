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
import os
import faulthandler
faulthandler.enable()
import tensorflow as tf
# load manta tools
sys.path.append("../tools_wscale")
import tilecreator_t as tc
import uniio
import fluiddataloader as FDL
import paramhelpers as ph
from GAN import GAN, lrelu

# initialize parameters / command line params
outputOnly	  = int(ph.getParam( "out",			 False ))>0 		# output/generation mode, main mode switch

basePath		=	 ph.getParam( "basePath",		'../2ddata_gan/' )
randSeed		= int(ph.getParam( "randSeed",		1 )) 				# seed for np and tf initialization
load_model_test_1 = int(ph.getParam( "load_model_test_1", -1 )) 			# the number of the test to load a model from. can be used in training and output mode. -1 to not load a model
load_model_test_2 = int(ph.getParam( "load_model_test_2", -1 )) 			# the number of the test to load a model from. can be used in training and output mode. -1 to not load a model
load_model_test_3 = int(ph.getParam( "load_model_test_3", -1 )) 			# the number of the test to load a model from. can be used in training and output mode. -1 to not load a model
load_model_no_1   = int(ph.getParam( "load_model_no_1",   -1 )) 			# nubmber of the model to load
load_model_no_2   = int(ph.getParam( "load_model_no_2",   -1 )) 			# nubmber of the model to load
load_model_no_3   = int(ph.getParam( "load_model_no_3",   -1 )) 			# nubmber of the model to load

simSizeLow  	= int(ph.getParam( "simSize", 		  64 )) 			# tiles of low res sim
tileSizeLow 	= int(ph.getParam( "tileSize", 		  16 )) 			# size of low res tiles
upRes	  		= int(ph.getParam( "upRes", 		  4 )) 				# scaling factor

#Data and Output
packedSimPath		 =	 ph.getParam( "packedSimPath",		 '/data/share/GANdata/2ddata_sim/' ) 	# path to training data
fromSim		 = int(ph.getParam( "fromSim",		 1000 )) 			# range of sim data to use, start index
frame_min		= int(ph.getParam( "frame_min",		   0 ))
genModel		 =	 ph.getParam( "genModel",		 'gen_test' ) 	# path to training data
discModel		=	 ph.getParam( "discModel",		 'disc_test' ) 	# path to training data
#Training
batch_norm		= int(ph.getParam( "batchNorm",	   False ))>0			# apply batch normalization to conv and deconv layers
pixel_norm		= int(ph.getParam( "pixelNorm",	   True ))>0			# apply batch normalization to conv and deconv layers

useVelocities   = int(ph.getParam( "useVelocities",   0  )) 			# use velocities or not
useVorticities  = int(ph.getParam( "useVorticities",   0  )) 			# use vorticities or not
useFlags   = int(ph.getParam( "useFlags",   0  )) 			# use flags or not
useK_Eps_Turb = int(ph.getParam( "useK_Eps_Turb",   0  ))

transposeAxis	 = int(ph.getParam( "transposeAxis",		  0	 ))		#

#Test and Save
testPathStartNo = int(ph.getParam( "testPathStartNo", 0  ))
frame_max		= int(ph.getParam( "frame_max",		   200 ))
change_velocity		= int(ph.getParam( "change_velocity",		   False )) 

upsampling_mode = int(ph.getParam( "upsamplingMode",   2 ))
upsampled_data = int(ph.getParam ( "upsampledData", False))
generateUni = int(ph.getParam("genUni", False))
usePixelShuffle = int(ph.getParam("usePixelShuffle", False))
addBicubicUpsample = int(ph.getParam("addBicubicUpsample", False))
add_adj_idcs1 = int(ph.getParam("add_adj_idcs1", False))
add_adj_idcs2 = int(ph.getParam("add_adj_idcs2", False))
add_adj_idcs3 = int(ph.getParam("add_adj_idcs3", False))

load_emas = int(ph.getParam("loadEmas", False))
firstNNArch = int(ph.getParam("firstNNArch", True))
upsampleMode = int(ph.getParam("upsampleMode", 1))

# parameters for growing approach
use_res_net1 = int(ph.getParam( "use_res_net1",		   False ))
use_res_net2 = int(ph.getParam( "use_res_net2",		   False ))
use_res_net3 = int(ph.getParam( "use_res_net3",		   False ))
use_mb_stddev = int(ph.getParam( "use_mb_stddev",		   False )) 
start_fms_1 = int(ph.getParam("startFms1", 512))
max_fms_1 = int(ph.getParam("maxFms1", 256))
filterSize_1 = int(ph.getParam("filterSize1", 3))
start_fms_2 = int(ph.getParam("startFms2", 512))
max_fms_2 = int(ph.getParam("maxFms2", 256))
filterSize_2 = int(ph.getParam("filterSize2", 3))
start_fms_3 = int(ph.getParam("startFms3", 512))
max_fms_3 = int(ph.getParam("maxFms3", 256))
filterSize_3 = int(ph.getParam("filterSize3", 3))

velScale = float(ph.getParam("velScale", 1.0))
gpu_touse = int(ph.getParam("gpu", 0))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu_touse)

ph.checkUnusedParams()

# initialize
simSizeHigh 	= simSizeLow * upRes
tileSizeHigh	= tileSizeLow  * upRes



channelLayout_low = 'd'
channelLayout_high = 'd'

lowfilename = "density_low_%04d.uni"

toSim = fromSim
dirIDs = np.linspace(fromSim, toSim, (toSim-fromSim+1),dtype='int16')
	
highfilename = "density_high_%04d.uni"
mfl = ["density"]
mfh = ["density"]
	
# load output of first network in high res data of tile creator -> separate when getting input	

if useVelocities:
	channelLayout_low += ',vx,vy,vz'
	mfl = np.append(mfl, "velocity")	

data_fraction = 1.0
kt = 0.0
kt_l = 0.0
useTempoD = False
useTempoL2 = False
useDataAugmentation = 0

# load data
floader = FDL.FluidDataLoader( print_info=3, base_path=packedSimPath, base_path_y = packedSimPath, numpy_seed = randSeed ,filename=lowfilename, filename_index_min = frame_min, oldNamingScheme=False, filename_y = None, filename_index_max=frame_max, indices=dirIDs, data_fraction=data_fraction, multi_file_list=mfl, multi_file_list_y=mfh)

x, y, _  = floader.get()

x_3d = x
x_3d[:,:,:,:,1:4] = velScale * x_3d[:,:,:,:,1:4] # scale velocity channels

# 2D: tileSize x tileSize tiles; 3D: tileSize x tileSize x tileSize chunks
n_input = tileSizeLow  ** 2
n_output = tileSizeHigh ** 2

n_inputChannels = 1

if useVelocities:
	n_inputChannels += 3
if useVorticities:
	n_inputChannels += 3
	
n_input *= n_inputChannels

if not load_model_test_1 == -1:
	if not os.path.exists(basePath + 'test_%04d/' % load_model_test_1):
		print('ERROR: Test to load does not exist.')
	if not load_emas:
		load_path_1 = basePath + 'test_%04d/model_%04d.ckpt' % (load_model_test_1, load_model_no_1)
		load_path_ema_1 = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test_1, load_model_no_1)
	else:
		load_path_1 = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test_1, load_model_no_1)
	if outputOnly:
		out_path_prefix = 'out_%04d-%04d' % (load_model_test_1,load_model_no_1)
		test_path,_ = ph.getNextGenericPath(out_path_prefix, 0, basePath + 'test_%04d/' % load_model_test_1)
	else:
		test_path,_ = ph.getNextTestPath(testPathStartNo, basePath)

if not load_model_test_2 == -1:
	if not os.path.exists(basePath + 'test_%04d/' % load_model_test_2):
		print('ERROR: Test to load does not exist.')
	if not load_emas:
		load_path_2 = basePath + 'test_%04d/model_%04d.ckpt' % (load_model_test_2, load_model_no_2)
		load_path_ema_2 = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test_2, load_model_no_2)
	else:
		load_path_2 = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test_2, load_model_no_2)
	if outputOnly:
		out_path_prefix = 'out_%04d-%04d' % (load_model_test_2,load_model_no_2)
		test_path,_ = ph.getNextGenericPath(out_path_prefix, 0, basePath + 'test_%04d/' % load_model_test_2)
	else:
		test_path,_ = ph.getNextTestPath(testPathStartNo, basePath)

if not load_model_test_3 == -1:
	if not os.path.exists(basePath + 'test_%04d/' % load_model_test_2):
		print('ERROR: Test to load does not exist.')
		print('Using two networks')
	else:
		print('Using three networks')
		if not load_emas:
			load_path_3 = basePath + 'test_%04d/model_%04d.ckpt' % (load_model_test_3, load_model_no_3)
			load_path_ema_3 = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test_3, load_model_no_3)
		else:
			load_path_3 = basePath + 'test_%04d/model_ema_%04d.ckpt' % (load_model_test_3, load_model_no_3)
		if outputOnly:
			out_path_prefix = 'out_%04d-%04d' % (load_model_test_3,load_model_no_3)
			test_path,_ = ph.getNextGenericPath(out_path_prefix, 0, basePath + 'test_%04d/' % load_model_test_3)
		else:
			test_path,_ = ph.getNextTestPath(testPathStartNo, basePath)

				
# create session and saver
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config = config)

def save_img(out_path, img):
	img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
	scipy.misc.imsave(out_path, img)

def save_img_3d(out_path, img): # y ↓ x →， z ↓ x →, z ↓ y →，3 in a column
	data = np.concatenate([np.sum(img, axis=0), np.sum(img, axis=1), np.sum(img, axis=2)], axis=0)
	save_img(out_path, data)

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
	
def growBlockGen(gan, inp, upres, fms, use_batch_norm, train, reuse, output = False, firstGen = True, filterSize = 3, first_nn_arch = False, use_res_net = True):
	with tf.variable_scope("genBlock%d"%(upres), reuse=reuse) as scope:
		if firstGen:
			if not usePixelShuffle:
				inDepool = gan.avg_depool(mode = upsampleMode)
			else:
				inDepool = gan.pixel_shuffle(inp, upres = 2, stage = "%d"%(upres))
		else:
			inDepool = inp
			
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
			#	"recursive" output
				inp,_ = gan.convolutional_layer(  fms, filter, lrelu, stride=[1], name="g_cA%d"%(upres), in_layer=inDepool, reuse=reuse, batch_norm=use_batch_norm, train=train) #->16,64
				
				if pixel_norm:
					inp = gan.pixel_norm(inp)
				outp,_ = gan.convolutional_layer(  fms, filter, lrelu, stride=[1], name="g_cB%d"%(upres), in_layer=inp, reuse=reuse, batch_norm=use_batch_norm, train=train) #->8,128
				
				if pixel_norm:
					outp = gan.pixel_norm(outp)
		#	density output for blending 
		if not output:
			outpDens, _ = GAN(outp, bn_decay=0.0).convolutional_layer(  1, [1,1], None, stride=[1], name="g_cdensOut%d"%(upres), in_layer=outp, reuse=reuse, batch_norm=False, train=train, gain = 1)
			return outp, outpDens
		return outp
		
def growing_gen(_in, percentage, reuse=False, use_batch_norm=False, train=None, currentUpres = 2, output = False, firstGen = True, filterSize = 3, startFms = 256, maxFms = 256, add_adj_idcs = False, first_nn_arch = False, use_res_net = True):
	global rbId
	print("\n\tGenerator (growing-sliced-resnett3-deep)")
	
	with tf.variable_scope("generator", reuse=reuse) as scope:
		n_channels = n_inputChannels
		if add_adj_idcs:
			n_channels += 2
		if firstGen:
			_in = tf.reshape(_in, shape=[-1, tileSizeLow, tileSizeLow, n_channels]) #NHWC
		else:
			_in = tf.reshape(_in, shape=[-1, tileSizeHigh, tileSizeHigh, n_channels+1]) #NHWC
			
		gan = GAN(_in, bn_decay=0.0)	
		
		#	inital conv layers
		filter = [filterSize,filterSize]
		
		if first_nn_arch:				
			x_g = _in
		else:
			if use_res_net:
				x_g = resBlock(gan, _in, 16, min(maxFms, startFms//2)//8, reuse, False, "1", filter_size = filter[0])
				x_g = resBlock(gan, x_g, min(maxFms, startFms//2)//4, min(maxFms, startFms//2)//2, reuse, False, "2", filter_size = filter[0])
			else:
				x_g,_ = gan.convolutional_layer(  32, filter, lrelu, stride=[1], name="g_cA%d"%(1), in_layer=_in, reuse=reuse, batch_norm=use_batch_norm, train=train) #->16,64
				if pixel_norm:
					x_g = gan.pixel_norm(x_g)
				x_g,_ = gan.convolutional_layer(  min(startFms//2, maxFms), filter, lrelu, stride=[1], name="g_cB%d"%(1), in_layer=x_g, reuse=reuse, batch_norm=use_batch_norm, train=train) #->8,128	
				if pixel_norm:
					x_g = gan.pixel_norm(x_g)
		#	density output for blending

		for j in range(1,currentUpres+1):
			num_fms = min(int(startFms / (2**j)),maxFms)
			if not output or j == currentUpres:
				x_g, _dens = growBlockGen(gan, x_g, int(2**(j)), num_fms, use_batch_norm, train, reuse, False, firstGen, filterSize, first_nn_arch, use_res_net)	
			else:
				x_g = growBlockGen(gan, x_g, int(2**(j)), num_fms, use_batch_norm, train, reuse, output, firstGen, filterSize, first_nn_arch, use_res_net)	
					
			# residual learning		
			if addBicubicUpsample:
				if j == currentUpres:
					if firstGen:
						_dens = _dens + GAN(tf.slice(_in, [0,0,0,0], [-1,tileSizeLow, tileSizeLow, 1])).avg_depool(mode = 2, scale = [int(2**(j))])
					else:
						_dens = _dens + tf.slice(_in, [0,0,0,0], [-1,tileSizeHigh, tileSizeHigh, 1])						

			print("\tDOFs: %d , %f m " % ( gan.getDOFs() , gan.getDOFs()/1000000.) )					
					
		resF = tf.reshape( _dens, shape=[-1, n_output] ) # + GAN(_in).avg_depool(mode = upsampleMode)
		print("\tDOFs: %d , %f m " % ( gan.getDOFs() , gan.getDOFs()/1000000.) )
		return resF

gen_model = growing_gen

x = tf.placeholder(tf.float32,[None,n_input], name = "x")
y = tf.placeholder(tf.float32,[None,None], name = "y")

train = tf.placeholder(tf.bool)
# output percentage for full 8x model...
percentage = tf.placeholder(tf.float32)

# first generator
x_in = x
if not load_model_test_1 == -1:
	with tf.variable_scope("gen_1", reuse=True) as scope:
		sampler = gen_model(x_in, use_batch_norm=batch_norm, reuse = tf.AUTO_REUSE, currentUpres = int(round(math.log(upRes, 2))), train=False, percentage = percentage, output = True, firstGen = True, filterSize = filterSize_1, startFms = start_fms_1, maxFms = max_fms_1, add_adj_idcs = add_adj_idcs1, first_nn_arch = firstNNArch, use_res_net=use_res_net1)
	
# second generator
if not load_model_test_2 == -1:
	x_in_2 = tf.concat((tf.reshape(y, shape = [-1, tileSizeHigh, tileSizeHigh, 1]), tf.image.resize_images(tf.reshape(x, shape = [-1, tileSizeLow, tileSizeLow, n_inputChannels]), tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1)), axis = 3)
	with tf.variable_scope("gen_2", reuse=True) as scope:
		sampler_2 = gen_model(x_in_2, use_batch_norm=batch_norm, reuse = tf.AUTO_REUSE, currentUpres = int(round(math.log(upRes, 2))), train=False, percentage = percentage, output = True, firstGen = False, filterSize = filterSize_2, startFms = start_fms_2, maxFms = max_fms_2, add_adj_idcs = add_adj_idcs2, first_nn_arch = False, use_res_net=use_res_net2)
	
# second generator
if not load_model_test_3 == -1:
	x_in_3 = tf.concat((tf.reshape(y, shape = [-1, tileSizeHigh, tileSizeHigh, 1]), tf.image.resize_images(tf.reshape(x, shape = [-1, tileSizeLow, tileSizeLow, n_inputChannels]), tf.constant([tileSizeHigh, tileSizeHigh], dtype= tf.int32), method=1)), axis = 3)
	with tf.variable_scope("gen_3", reuse=True) as scope:
		sampler_3 = gen_model(x_in_3, use_batch_norm=batch_norm, reuse = tf.AUTO_REUSE, currentUpres = int(round(math.log(upRes, 2))), train=False, percentage = percentage, output = True, firstGen = False, filterSize = filterSize_3, startFms = start_fms_3, maxFms = max_fms_3, add_adj_idcs = add_adj_idcs3, first_nn_arch = False, use_res_net=use_res_net3)
	
if not load_model_test_1 == -1:
	gen1vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gen_1")
	gen1dict = dict((var.name[6:len(var.name)-2],var) for var in gen1vars)
	saver = tf.train.Saver(var_list = gen1dict)
	saver.restore(sess, load_path_1)
	print("Model 1 restored from %s." % load_path_1)

if not load_model_test_2 == -1:
	gen2vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gen_2")
	gen2dict = dict((var.name[6:len(var.name)-2],var) for var in gen2vars)
	saver = tf.train.Saver(var_list = gen2dict)
	saver.restore(sess, load_path_2)
	print("Model 2 restored from %s." % load_path_2)

if not load_model_test_3 == -1:
	gen3vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gen_3")
	gen3dict = dict((var.name[6:len(var.name)-2],var) for var in gen3vars)
	saver = tf.train.Saver(var_list = gen3dict)
	saver.restore(sess, load_path_3)
	print("Model 3 restored from %s." % load_path_3)


# for two different networks, first upsamples two dimensions, last one upsamples one dim
def generate3DUniForNewNetwork(imageindex = 0, outPath = '../', inputPer = 3.0, head = None):
	start = time.time()
	dim_output = []
	intermed_res1 = []
	
	batch_xs_tile = x_3d[imageindex]
	
	if not load_model_test_1 == -1:
		# z y x -> 2d conv on y - x (or different combination of axis, depending on transposeAxis)
		# and switch velocity channels depending on orientation
		if transposeAxis == 1:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,upRes,1,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeHigh, simSizeLow, n_inputChannels])
			batch_xs_in = np.reshape(batch_xs_in.transpose(1,0,2,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)	
		elif transposeAxis == 2:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			batch_xs_in = np.reshape(batch_xs_in.transpose(2,1,0,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel)
		elif transposeAxis == 3:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			batch_xs_in = np.reshape(batch_xs_in.transpose(2,0,1,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			temp_vel2 = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel2)
		else:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[upRes,1,1,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeLow, n_inputChannels])					
				
		if add_adj_idcs1:		
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
					
		# start generating output of first network
		batch_sz_out = 8
		run_metadata = tf.RunMetadata()
		
		start = time.time()
		for j in range(0,batch_xs_in.shape[0]//batch_sz_out):
			#	x in shape (z,y,x,c)
			# 	-> 512 x 512 x 512
			results = sess.run(sampler, feed_dict={x: batch_xs_in[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_input), percentage : inputPer, train: False})
			intermed_res1.extend(results)	
			
			# exact timing of network performance...
			if 0:
				fetched_timeline = timeline.Timeline(run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('timeline_8x_%04d.json'%(j), 'w') as f:
					f.write(chrome_trace)
		end = time.time()
		
		print("time for first network: {0:.6f}".format(end-start))
			
		dim_output = np.copy(np.array(intermed_res1).reshape(simSizeHigh, simSizeHigh, simSizeHigh)).transpose(2,1,0)
			
		save_img_3d( outPath + 'source_1st_{:04d}.png'.format(imageindex+frame_min), dim_output/80)	

	if not load_model_test_2 == -1:
		if transposeAxis == 3:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,upRes,1,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeHigh, simSizeLow, n_inputChannels])
			batch_xs_in = np.reshape(batch_xs_in.transpose(1,0,2,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)	
		elif transposeAxis == 0:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			batch_xs_in = np.reshape(batch_xs_in.transpose(2,1,0,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel)
		elif transposeAxis == 1:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			batch_xs_in = np.reshape(batch_xs_in.transpose(2,0,1,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			temp_vel2 = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel2)
		else:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[upRes,1,1,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeLow, n_inputChannels])					
				
		if add_adj_idcs2:		
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
					
		intermed_res1 = []
		batch_sz_out = 2
		
		start = time.time()
		for j in range(0,batch_xs_in.shape[0]//batch_sz_out):
			#	x in shape (z,y,x,c)
			# 	-> 64 x 256 x 256
			results = sess.run(sampler_2, feed_dict={x: batch_xs_in[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_input), y: dim_output[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_output) ,percentage : inputPer, train: False})
			intermed_res1.extend(results)	
			
			# exact timing of network performance...
			if 0:
				fetched_timeline = timeline.Timeline(run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('timeline_8x_%04d.json'%(j), 'w') as f:
					f.write(chrome_trace)
		end = time.time()
		
		print("time for second network: {0:.6f}".format(end-start))
		
		dim_output = np.array(intermed_res1).reshape(simSizeHigh, simSizeHigh, simSizeHigh).transpose(1,2,0)
					
		save_img_3d( outPath + 'source_2nd_{:04d}.png'.format(imageindex+frame_min), dim_output/80)	
		
	if not load_model_test_3 == -1:
		if transposeAxis == 0:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,upRes,1,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeHigh, simSizeLow, n_inputChannels])
			batch_xs_in = np.reshape(batch_xs_in.transpose(1,0,2,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,2:3] = np.copy(temp_vel)	
		elif transposeAxis == 3:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			batch_xs_in = np.reshape(batch_xs_in.transpose(0,2,1,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,2:3] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel)
		elif transposeAxis == 2:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[1,1,upRes,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeHigh, n_inputChannels])	
			batch_xs_in = np.reshape(batch_xs_in.transpose(1,2,0,3),(-1, simSizeLow, simSizeLow, n_inputChannels))
			temp_vel = np.copy(batch_xs_in[:,:,:,3:4])
			temp_vel2 = np.copy(batch_xs_in[:,:,:,13])
			batch_xs_in[:,:,:,3:4] = np.copy(batch_xs_in[:,:,:,2:3])
			batch_xs_in[:,:,:,2:3] = np.copy(batch_xs_in[:,:,:,1:2])
			batch_xs_in[:,:,:,1:2] = np.copy(temp_vel)
		else:
			batch_xs_in = np.reshape(scipy.ndimage.zoom(batch_xs_tile,[upRes,1,1,1] , order = 1, mode = 'constant', cval = 0.0), [-1, simSizeLow, simSizeLow, n_inputChannels])					
				
		if add_adj_idcs3:		
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
					
		intermed_res1 = []
		batch_sz_out = 2
		
		start = time.time()
		for j in range(0,batch_xs_in.shape[0]//batch_sz_out):
			#	x in shape (z,y,x,c)
			# 	-> 64 x 256 x 256
			results = sess.run(sampler_3, feed_dict={x: batch_xs_in[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_input), y: dim_output[j*batch_sz_out:(j+1)*batch_sz_out].reshape(-1, n_output) ,percentage : inputPer, train: False})
			intermed_res1.extend(results)	
			
			# exact timing of network performance...
			if 0:
				fetched_timeline = timeline.Timeline(run_metadata.step_stats)
				chrome_trace = fetched_timeline.generate_chrome_trace_format()
				with open('timeline_8x_%04d.json'%(j), 'w') as f:
					f.write(chrome_trace)
		end = time.time()
		
		print("time for third network: {0:.6f}".format(end-start))
		
		dim_output = np.array(intermed_res1).reshape(simSizeHigh, simSizeHigh, simSizeHigh)							
							
		save_img_3d( outPath + 'source_3rd_{:04d}.png'.format(imageindex+frame_min), dim_output/80)	
	
	if not load_model_no_2 == -1:
		dim_output = dim_output.transpose(2,0,1)
	if not load_model_no_1 == -1:
		dim_output = dim_output.transpose(2,1,0)
		
	# output for images of slices (along every dimension)
	if 1:
		for i in range(simSizeHigh // 2 - 1, simSizeHigh // 2 + 1):
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

	if head is None:
		head, _ = uniio.readUni(packedSimPath + "sim_%04d/density_low_%04d.uni"%(fromSim, 0))
	head['dimX'] = simSizeHigh
	head['dimY'] = simSizeHigh
	head['dimZ'] = simSizeHigh
		
	if generateUni:
		# set low density to zero to save storage space...
		cond_out = dim_output < 0.0005
		dim_output[cond_out] = 0
		uniio.writeUni(packedSimPath + '/sim_%04d/source_%04d.uni'%(fromSim, imageindex+frame_min), head, dim_output)
		print('stored .uni file')
	return		
	
print('*****OUTPUT ONLY*****')
#print("{} tiles, {} tiles per image".format(100, 1))
#print("Generating images (batch size: {}, batches: {})".format(1, 100))

if not load_model_test_3 == -1 and not load_model_test_2 == -1 and not load_model_test_1 == -1:
	print("At least one network has to be loaded.")
	exit(1)


head_0, _ = uniio.readUni(packedSimPath + "sim_%04d/density_low_%04d.uni"%(fromSim, 0))
for layerno in range(frame_min,frame_max):
	print(layerno)
	generate3DUniForNewNetwork(imageindex = layerno - frame_min, outPath = test_path, head = head_0)
	
print('Test finished, %d pngs written to %s.' % (frame_max - frame_min, test_path) )












