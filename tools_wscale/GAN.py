#******************************************************************************
#
# tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
# Copyright 2018 You Xie, Erik Franz, Mengyu Chu, Nils Thuerey, Maximilian Werhahn
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0 
# http://www.apache.org/licenses/LICENSE-2.0
#
#******************************************************************************

import numpy as np
import tensorflow as tf
import sys
import math
from keras import backend as kb
class GAN(object):
	#---------------------------------------------------------------------------------
	def __init__(self, _image, bn_decay=0.999):
		self.layer = _image 
		self.batch_size = tf.shape(_image)[0]
		self.DOFs = 0
		# stack 
		self.preFlatShapes = []
		self.weight_stack = []
		self.layer_num = 0
		self.layer_num_gen = 0
		self.layer_num_disc = 0
		
		self.bn_decay=bn_decay
		
		self.dtypeF = tf.float32 # tf.float32
		self.dtypeI = tf.int32 # tf.int16 # 
		
		print("Input: {}".format(self.layer.get_shape()))
	
	#---------------------------------------------------------------------------------
	# thanks to http://robromijnders.github.io/tensorflow_basic/
	def weight_image(self):
		W = self.weight_stack[-1]
		# compute size of the image
		s = W.get_shape()
		out_channels = 1
		if int(s[3]) % 3 == 0:
			out_channels = 3
		print("Shape {}".format(s))
		weight_patches = int(s[2]) * int(s[3]) / out_channels # e.g. the number of [3,3] patches in a CNN
		side_length = int(math.ceil(math.sqrt(weight_patches))) # image side length (in patches)
		image_patches = side_length * side_length # max number of patches that fit in the image
		# split into per filter weights
		ws = []
		ws_dim3 = tf.split(3, s[3] / out_channels, W) # e.g. [ [3,3,3,1], [3,3,3,1], ... ]
		for w in ws_dim3:
			# split these further
			ws.extend(tf.split(2, s[2], w))  # e.g. [ [3,3,1,1], [3,3,1,1], ... ]
		# pad image
		padding = image_patches - weight_patches
		for i in range(padding):
			ws.append(tf.zeros([s[0], s[1], 1, out_channels]))
		# build rows of image
		rows = []
		for i in range(side_length):
			start = i * side_length
			end = start + side_length
			rows.append(tf.concat(axis=0, values=ws[start:end]))
		# combine rows to image
		image = tf.concat(axis=1, values=rows) # [sidelength * ]
		s = [int(image.get_shape()[0]), int(image.get_shape()[1])]
		image = tf.reshape(image, [1, s[0], s[1], out_channels])
		image = tf.image.resize_images(image, [int(s[1] * 50), int(s[0] * 50)], 1)
		image_tag = "l" + str(self.layer_num) + "_weight_image"
		tf.image_summary(image_tag, image)
		print("Image Summary: save weights as image")
		
	#---------------------------------------------------------------------------------
	# outChannels: int
	# _patchShape: 2D: [H,W]; 3D: [D,H,W]
	# stride 3D: if 1D: [DHW], if 2D:[D,HW], if 3D:[D,H,W]
	# returns both normalized and linearized versions
	def convolutional_layer(self, outChannels, _patchShape, activation_function=tf.nn.tanh, stride=[1], name="conv",reuse=False, batch_norm=False, train=None, in_layer=None, in_channels = None, gain = np.sqrt(2)):
		if in_layer==None:
			in_layer = self.layer
		with tf.variable_scope(name, reuse = reuse):
			self.layer_num += 1
			# set the input and output dimension
			if in_channels is not None :
				inChannels = int(in_channels)
			else:
				inChannels = int(in_layer.get_shape()[-1])
			#outChannels = int(inChannels * _filterSpread)
			# create a weight matrix
			if len(_patchShape) == 2:
				W = self.weight_variable([_patchShape[0], _patchShape[1], inChannels, outChannels], name=name, gain = gain)
				self.layer = self.conv2d(in_layer, W, stride)
				self.DOFs += _patchShape[0]* _patchShape[1]* inChannels* outChannels
			elif len(_patchShape) == 3:
				W = self.weight_variable([_patchShape[0], _patchShape[1], _patchShape[2], inChannels, outChannels], name=name, gain = gain)
				self.layer = self.conv3d(in_layer, W, stride)
				self.DOFs += _patchShape[0]* _patchShape[1]* _patchShape[2]* inChannels* outChannels
				#batch_norm = False
				
			self.weight_stack.append(W)
			# create a bias vector
			b = self.bias_variable([outChannels], name=name)
			self.layer = self.layer + b
			self.DOFs += outChannels
			
			if batch_norm:
				#self.layer = self.conv_batch_norm(self.layer, train=train)
				self.layer = tf.contrib.layers.batch_norm(self.layer, decay=self.bn_decay, scale=True, scope=tf.get_variable_scope(), reuse=reuse, fused=False, is_training=train)
			layer_lin = self.layer
			if activation_function:
				self.layer = activation_function(self.layer)
			# user output
			if activation_function:
				print("Convolutional Layer \'{}\' {} ({}) : {}, BN:{}".format(name, W.get_shape(), activation_function.__name__,self.layer.get_shape(),batch_norm))
			else:
				print("Convolutional Layer \'{}\' {} ({}) : {}, BN:{}".format(name, W.get_shape(), 'None',self.layer.get_shape(),batch_norm))
			return self.layer, layer_lin
	
	#---------------------------------------------------------------------------------
	# s1: outChannels of intermediate conv layer
	# s2: outChannels of final and skip conv layer
	# filter: 2D: [H,W]; 3D: [D,H,W]
	# returns both normalized and linearized versions
	def residual_block(self, s1,s2, filter, activation_function=tf.nn.tanh, name="RB", reuse=False, batch_norm=False, train=None, in_layer=None):
		# note - leaky relu (lrelu) not too useful here
		if in_layer==None:
			in_layer = self.layer
		# convolutions of resnet block
		if len(filter) == 2:
			filter1 = [1,1]
		elif len(filter) == 3:
			filter1 = [1,1,1]
			
		print("Residual Block:")
		A,_ = self.convolutional_layer(s1, filter, activation_function, stride=[1], name=name+"_A", in_layer=in_layer, reuse=reuse, batch_norm=batch_norm, train=train)
		B,_ = self.convolutional_layer(s2, filter, None               , stride=[1], name=name+"_B",                    reuse=reuse, batch_norm=batch_norm, train=train)
		# shortcut connection
		s,_ = self.convolutional_layer(s2, filter1, None              , stride=[1], name=name+"_s", in_layer=in_layer, reuse=reuse, batch_norm=batch_norm, train=train)
		
		self.layer = tf.add( B, s)
		layer_lin = self.layer
		if activation_function:
			self.layer = activation_function(self.layer )
			
		return self.layer, layer_lin
	
	
	#---------------------------------------------------------------------------------
	# 2 x 2 max pool operation
	def max_pool(self, window_size=[2], window_stride=[2]):
		if len(self.layer.get_shape()) == 4:
			self.layer = tf.nn.max_pool(self.layer, ksize=[1, window_size[0], window_size[0], 1], strides=[1, window_stride[0], window_stride[0], 1], padding="VALID")
		elif len(self.layer.get_shape()) == 5:
			self.layer = tf.nn.max_pool3d(self.layer, ksize=[1, window_size[0], window_size[0], window_size[0], 1], strides=[1, window_stride[0], window_stride[0], window_stride[0], 1], padding="VALID")
		# user output
		print("Max Pool {}: {}".format(window_size, self.layer.get_shape()))
		return self.layer
	
	#---------------------------------------------------------------------------------
	def avg_pool(self, window_size=[2], window_stride=[2]):
		if len(self.layer.get_shape()) == 4:
			self.layer = tf.nn.avg_pool(self.layer, ksize=[1, window_size[0], window_size[0], 1], strides=[1, window_stride[0], window_stride[0], 1], padding="VALID")
		elif len(self.layer.get_shape()) == 5:
			self.layer = tf.cast(tf.nn.avg_pool3d(tf.cast(self.layer, tf.float32), ksize=[1, window_size[0], window_size[0], window_size[0], 1], strides=[1, window_stride[0], window_stride[0], window_stride[0], 1], padding="VALID"), self.dtypeF)
		# user output
		print("Avg Pool {}: {}".format(window_size, self.layer.get_shape()))
		return self.layer
	
	#---------------------------------------------------------------------------------
	# TODO: center velocities
	def SemiLagrange (self, source, vel, flags, res, pos):
		vel_shape = tf.shape(vel)
		dim = 2 #tf.size(vel_shape) - 2 # batch and channels are ignored
			
		pos = tf.subtract( tf.add( tf.cast(pos, tf.float32), tf.constant(0.0)), vel)
				
		floors = tf.cast(tf.floor(pos - 0.5), tf.int32)
		ceils = floors + 1

		# clamp min
		floors = tf.maximum(floors, tf.zeros_like(floors))
		ceils = tf.maximum(ceils, tf.zeros_like(ceils))

		# clamp max
		floors = tf.minimum(floors, tf.shape(source)[1:dim + 1] - 1)
		ceils = tf.minimum(ceils, tf.shape(source)[1:dim + 1] - 1)

		_broadcaster = tf.ones_like(ceils)
		cell_value_list = []
		cell_weight_list = []
		for axis_x in range(int(pow(2, dim))):  # 3d, 0-7; 2d, 0-3;...
			condition_list = [bool(axis_x & int(pow(2, i))) for i in range(dim)]
			condition_ = (_broadcaster > 0) & condition_list
			axis_idx = tf.cast(
				tf.where(condition_, ceils, floors),
				tf.int32)

			# only support linear interpolation...
			axis_wei = 1.0 - tf.abs((pos - 0.5) - tf.cast(axis_idx, tf.float32))  # shape (..., res_x2, res_x1, dim)
			axis_wei = tf.reduce_prod(axis_wei, axis=-1, keepdims=True)
			cell_weight_list.append(axis_wei)  # single scalar(..., res_x2, res_x1, 1)
			first_idx = tf.ones_like(axis_wei, dtype=self.dtypeI)
			first_idx = tf.cumsum(first_idx, axis=0, exclusive=True)
			cell_value_list.append(tf.concat([first_idx, axis_idx], -1))
		#print(value.get_shape())
		#print(cell_value_list[0].get_shape())
		source_fwd = tf.gather_nd(source, cell_value_list[0]) * cell_weight_list[
			0]  # broadcasting used, shape (..., res_x2, res_x1, channels )
		for cell_idx in range(1, len(cell_value_list)):
			source_fwd = source_fwd + tf.gather_nd(source, cell_value_list[cell_idx]) * cell_weight_list[cell_idx]
		return source_fwd  # shape (..., res_x2, res_x1, channels)

	def MacCormackCorrect(self, flags, source, forward, backward, strength = 1.0, threshold_flags = 0.2):
		flags = tf.reshape(flags, shape=[-1, tf.shape(source)[1], tf.shape(source)[2], 1])
		cond_flags = tf.less(flags, tf.constant(threshold_flags)) # adapt threshold
		return tf.where(cond_flags, forward + strength * 0.5 * (source - backward), forward)
				
	# checkFlag(x,y,z) (flags((x),(y),(z)) & (FlagGrid::TypeFluid|FlagGrid::TypeEmpty))
	
	def doClampComponent(self, grid_res, flags, intermed_adv, source, forward, pos, vel, startBz = 15, threshold_flags = 0.2):
		
		min = tf.ones_like(source) * sys.maxsize
		max = -tf.ones_like(source) * sys.maxsize - 1
		min_i = min
		max_i = max
		# forward 
		currPos = tf.cast(tf.cast(pos, tf.float32) - vel, tf.int32)
		# clamp lookup to grid
		i0 = tf.clip_by_value(tf.slice(currPos,[0,0,0,0],[-1,-1,-1,1]), 0, grid_res[1]-1)
		j0 = tf.clip_by_value(tf.slice(currPos,[0,0,0,1],[-1,-1,-1,1]), 0, grid_res[2]-1)
		
		# indices_0 = tf.Variable([], dtype = tf.int32, trainable = False)
		# indices_1 = tf.Variable([], dtype = tf.int32, trainable = False)
		# indices_2 = tf.Variable([], dtype = tf.int32, trainable = False)
		# indices_3 = tf.Variable([], dtype = tf.int32, trainable = False)
		# i = tf.constant(0)
		
		# indices_0 = tf.concat([tf.ones_like(i0[0])*0, i0[0], j0[0], tf.zeros_like(i0[0])], axis = 2)
		# indices_1 = tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i] + 1, j0[i], tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1)
		# indices_2 = tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i], j0[i] + 1, tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1)
		# indices_3 = tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i] + 1, j0[i] + 1, tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1)
			
		# def cond(i, indices_0, indices_1, indices_2, indices_3,i0,j0):
			# return tf.less(i, grid_res[0])
				
		# #while_condition = lambda i, indices_0, indices_1, indices_2, indices_3: tf.less(i, grid_res[0])
		
		# def body(i,indices_0,indices_1,indices_2,indices_3,i0,j0):
		
			# indices_t = [tf.concat([tf.ones_like(i0[i])*i, i0[i], j0[i], tf.zeros_like(i0[i])], axis = 2)]
			# indices_t1 = [tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i] + 1, j0[i], tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1)]
			# indices_t2 = [tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i], j0[i] + 1, tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1)]
			# indices_t3 = [tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i] + 1, j0[i] + 1, tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1)]
			
			# indices_0 = tf.concat([tf.reshape(indices_0, shape=tf.shape(indices_t)), indices_t], 0)		
			# indices_1 = tf.concat([tf.reshape(indices_1, shape=tf.shape(indices_t1)), indices_t1], 0)		
			# indices_2 = tf.concat([tf.reshape(indices_2, shape=tf.shape(indices_t2)), indices_t2], 0)		
			# indices_3 = tf.concat([tf.reshape(indices_3, shape=tf.shape(indices_t3)), indices_t3], 0)		
			
			# return tf.add(i, 1),indices_0,indices_1,indices_2,indices_3, i0, j0

		# # do the loop:
		# r,_,_,_,_,_,_ = tf.while_loop(cond, body, [i,indices_0,indices_1,indices_2,indices_3,i0,j0],shape_invariants=[i.get_shape(),tf.TensorShape(None),tf.TensorShape(None),tf.TensorShape(None),tf.TensorShape(None),i0.get_shape(),j0.get_shape()])
		
		#indices_0 = tf.convert_to_tensor(indices_0, dtype = tf.int32)
		# indices_0 = tf.concat([tf.ones_like(i0), i0, j0, tf.zeros_like(i0)], axis = 3)
		# indices_1 = tf.clip_by_value(tf.concat([tf.ones_like(i0), i0 + 1, j0, tf.zeros_like(i0)], axis = 3), 0, grid_res[2]-1)
		# indices_2 = tf.clip_by_value(tf.concat([tf.ones_like(i0), i0, j0 + 1, tf.zeros_like(i0)], axis = 3), 0, grid_res[2]-1)
		# indices_3 = tf.clip_by_value(tf.concat([tf.ones_like(i0), i0 + 1, j0 + 1, tf.zeros_like(i0)], axis = 3), 0, grid_res[2]-1)
		
		indices_0 = []
		indices_1 = []
		indices_2 = []
		indices_3 = []
		
		for i in range(startBz):
			indices_0.append( tf.concat([tf.ones_like(i0[i])*i, i0[i], j0[i], tf.zeros_like(i0[i])], axis = 2))
			indices_1.append( tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i] + 1, j0[i], tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1))
			indices_2.append( tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i], j0[i] + 1, tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1))
			indices_3.append( tf.clip_by_value(tf.concat([tf.ones_like(i0[i])*i, i0[i] + 1, j0[i] + 1, tf.zeros_like(i0[i])], axis = 2), 0, grid_res[2]-1))
			
		source_1 = tf.expand_dims(tf.gather_nd(source, indices_0), axis = 3)
		source_2 = tf.expand_dims(tf.gather_nd(source, indices_1), axis = 3)
		source_3 = tf.expand_dims(tf.gather_nd(source, indices_2), axis = 3)
		source_4 = tf.expand_dims(tf.gather_nd(source, indices_3), axis = 3)
		# const int k0 = clamp(currPos.z, 0, (orig.is3D() ? (gridSize.z-1) : 1) ); # for 3D
		
		flags_1 = tf.expand_dims(tf.gather_nd(flags, indices_0), axis = 3)
		cond_flags_1 = tf.less(flags_1, tf.constant(threshold_flags)) 
		
		flags_2 = tf.expand_dims(tf.gather_nd(flags, indices_1), axis = 3)
		cond_flags_2 = tf.less(flags_2, tf.constant(threshold_flags)) 
		
		flags_3 = tf.expand_dims(tf.gather_nd(flags, indices_2), axis = 3)
		cond_flags_3 = tf.less(flags_3, tf.constant(threshold_flags)) 
		
		flags_4 = tf.expand_dims(tf.gather_nd(flags, indices_3), axis = 3)
		cond_flags_4 = tf.less(flags_4, tf.constant(threshold_flags)) 
				
		tmp_min = min
		tmp_max = max
		cond_min = tf.greater( min, source_1)
		cond_max = tf.less( max, source_1)
		min = tf.where(cond_min, source_1, min)
		max = tf.where(cond_max, source_1, max)
		min = tf.where(cond_flags_1, min, tmp_min)
		max = tf.where(cond_flags_1, max, tmp_max)
		
		tmp_min = min
		tmp_max = max
		cond_min = tf.greater( min, source_2)
		cond_max = tf.less( max, source_2)
		min = tf.where(cond_min, source_2, min)
		max = tf.where(cond_max, source_2, max)
		min = tf.where(cond_flags_2, min, tmp_min)
		max = tf.where(cond_flags_2, max, tmp_max)
		
		tmp_min = min
		tmp_max = max
		cond_min = tf.greater( min, source_3)
		cond_max = tf.less( max, source_3)
		min = tf.where(cond_min, source_3, min)
		max = tf.where(cond_max, source_3, max)
		min = tf.where(cond_flags_3, min, tmp_min)
		max = tf.where(cond_flags_3, max, tmp_max)
		
		tmp_min = min
		tmp_max = max
		cond_min = tf.greater( min, source_4)
		cond_max = tf.less( max, source_4)
		min = tf.where(cond_min, source_4, min)
		max = tf.where(cond_max, source_4, max)
		min = tf.where(cond_flags_4, min, tmp_min)
		max = tf.where(cond_flags_4, max, tmp_max)
		
		# find min/max around source pos
		# if(checkFlag(i0,j0,k0)) { min, max = getMinMax(min, max, orig(i0,j0,k0));  haveFl=true; }
		# if(checkFlag(i1,j0,k0)) { min, max = getMinMax(min, max, orig(i1,j0,k0));  haveFl=true; }
		# if(checkFlag(i0,j1,k0)) { min, max = getMinMax(min, max, orig(i0,j1,k0));  haveFl=true; }
		# if(checkFlag(i1,j1,k0)) { min, max = getMinMax(min, max, orig(i1,j1,k0));  haveFl=true; }

		# for 3D
		# if(orig.is3D()) {
		# if(checkFlag(i0,j0,k1)) { getMinMax(minv, maxv, orig(i0,j0,k1)); haveFl=true; }
		# if(checkFlag(i1,j0,k1)) { getMinMax(minv, maxv, orig(i1,j0,k1)); haveFl=true; }
		# if(checkFlag(i0,j1,k1)) { getMinMax(minv, maxv, orig(i0,j1,k1)); haveFl=true; }
		# if(checkFlag(i1,j1,k1)) { getMinMax(minv, maxv, orig(i1,j1,k1)); haveFl=true; } } 
		
		# if(!haveFl) return fwd;
		# if(cmpMinMax(min,max,dst)) dst = fwd;
		cond_complete = tf.logical_or( tf.logical_or(tf.logical_or(tf.less(intermed_adv, min) , tf.greater(intermed_adv, max)), tf.equal(min, min_i)) , tf.equal(max, max_i))

		return tf.where(cond_complete, forward, intermed_adv)
	

	def MacCormackClamp(self, flags, vel, intermed_adv, source, forward, pos, startBz = 15):		
		grid_res = tf.shape(vel)		
		return self.doClampComponent(grid_res, flags, intermed_adv, source, forward, pos, vel, startBz);
			
	# def getMacCormackPosBatch(macgrid_batch, dt, cube_len_output=-1):  

		# vel_pos_high_inter = getSemiLagrPosBatch(macgrid_input, dtArray, self.tileSizeHigh[1]).reshape((real_batch_sz, -1))
		
		# vel_pos_high_inter = getSemiLagrPosBatch(macgrid_input, -dtArray, self.tileSizeHigh[1]).reshape((real_batch_sz, -1))

	#	velocity has to be centered, not in MAC-form
	def advect(self, source, vel, flags, dt, order, strength=0.0, name = "Advection", startBz = 15):
		res = tf.shape(source)
		#assert (tf.shape(vel) == res and tf.shape(flags) == res)
		vel_shape = tf.shape(source)#get_shape().as_list()
		print(vel_shape)
		dim = tf.size(vel_shape) - 2  # batch and channels are ignored
		
		# create index array
		# TODO precompute array
		pos_x = tf.range(start = 0.5, limit = tf.cast(res[1],dtype = tf.float32)+tf.constant(0.5,dtype = tf.float32), dtype = tf.float32)
		pos_x = tf.tile(pos_x, [res[2]])
		if dim == 3:
			pos_x = tf.tile(pos_x, [res[3]])
			pos_x = tf.reshape(pos_x, [1, vel_shape[1], vel_shape[2], vel_shape[3], 1])	
			pos_y = tf.transpose(pos_x, [0,2,1,3,4])
			pos_z = tf.transpose(pos_x, [0,3,2,1,4])
			pos = tf.stack([pos_z, pos_y, pos_x])
		else:
			pos_x = tf.reshape(pos_x, [1, vel_shape[1], vel_shape[2], 1])	
			pos_y = tf.transpose(pos_x, [0,2,1,3])
			pos = tf.cast(tf.concat([pos_y, pos_x], axis = 3), dtype = tf.float32)+tf.constant(0.5,dtype = tf.float32)

		upResFactor = tf.maximum(tf.cast((res[1] / tf.shape(vel)[1]),dtype =tf.float32),tf.cast((res[2] / tf.shape(vel)[2]),dtype =tf.float32))
		vel = tf.slice(vel, [0,0,0,0], [-1,-1,-1,2])
		vel = tf.concat((tf.slice(vel,[0,0,0,1], [-1,-1,-1,1]), tf.slice(vel,[0,0,0,0], [-1,-1,-1,1])), axis = 3)
		vel = tf.image.resize_images(vel,[res[1], res[2]], 0)
		#vel = tf.contrib.image.transform(vel, [1.0/upResFactor, 0, 0.5/upResFactor, 0, 1.0/upResFactor, 0.5/upResFactor, 0, 0], 'BILINEAR', output_shape=[tf.shape(source)[1], tf.shape(source)[2]])
		vel *= upResFactor
		vel_y = tf.contrib.image.transform(tf.slice(vel,[0,0,0,0],[-1,-1,-1,1]), [1, 0, 0, 0, 1, 1, 0, 0], 'NEAREST')
		vel_x = tf.contrib.image.transform(tf.slice(vel,[0,0,0,1],[-1,-1,-1,1]), [1, 0, 1, 0, 1, 0, 0, 0], 'NEAREST')
		vel = tf.constant(0.5, dtype=tf.float32) * (vel + tf.concat((vel_y, vel_x), axis = 3))
		#vel = tf.contrib.image.transform(vel, [1.0/upResFactor, 0, 0.5/upResFactor, 0, 1.0/upResFactor, 0.5/upResFactor, 0, 0], 'BILINEAR', output_shape=[tf.shape(source)[1], tf.shape(source)[2]])
		
		#vel_y = tf.contrib.image.transform(tf.slice(vel, [0,0,0,1],[-1,-1,-1,1]), [1/upResFactor, 0, 0.5/upResFactor, 0, 1/upResFactor, 0.0, 0, 0], 'BILINEAR', output_shape=[tf.shape(source)[1], tf.shape(source)[2]])
		#vel = tf.concat([vel_x,vel_y], axis = 3) 
		# build time step array		
		dt_arr_1 = tf.cast(tf.ones_like(pos), tf.float32) * dt		
		dt_arr = tf.concat([dt_arr_1, tf.cast(tf.zeros_like(pos), tf.float32), dt_arr_1 * -1.0], axis = 0)
		dt_arr = tf.tile(dt_arr, [vel_shape[0]//3,1,1,1])
		pos = tf.tile(pos, [tf.shape(source)[0],1,1,1])
		vel *= dt_arr
		# advect quantity: source
		with tf.variable_scope(name): 
			forward_adv = self.SemiLagrange(source, vel, flags, res, pos)
			if order == 2:
				backward_adv = self.SemiLagrange(forward_adv, -vel , flags, res, pos)					
				intermed_correct = self.MacCormackCorrect(flags, source, forward_adv, backward_adv, strength)
				out_adv = self.MacCormackClamp(flags, vel, intermed_correct, source, forward_adv, pos, startBz)
				return out_adv
		return forward_adv
	
	#---------------------------------------------------------------------------------
	# make layer flat
	# e.G. [1, 4, 4, 2] -> [1, 32]
	def flatten(self):
		# get unflat shape
		layerShape = self.layer.get_shape()
		self.preFlatShapes.append(layerShape)
		# compute flat size
		flatSize = int(layerShape[1]) * int(layerShape[2]) * int(layerShape[3])
		if len(layerShape) == 5:
			flatSize *= int(layerShape[4])
		# make flat
		self.layer = tf.reshape(self.layer, [-1, flatSize])
		# user output
		print("Flatten: {}".format(self.layer.get_shape()))
		return flatSize
	
	#---------------------------------------------------------------------------------
	def fully_connected_layer(self, _numHidden, _act, name="full", gain = np.sqrt(2)):
		with tf.variable_scope(name):
			self.layer_num += 1
			# get previous layer size
			numInput = int(self.layer.get_shape()[1])
			# build layer variables
			W = self.weight_variable([numInput, _numHidden], name=name, gain =gain)
			b = self.bias_variable([_numHidden], name=name)
			self.DOFs += numInput*_numHidden + _numHidden
			# activate
			self.layer = tf.matmul(self.layer, W) + b
			if _act:
				self.layer = _act(self.layer)  # ??
			# user output
			if _act:
				print("Fully Connected Layer \'{}\': {}".format(name, self.layer.get_shape()))
			else:
				print("Linear Layer \'{}\': {}".format(name, self.layer.get_shape()))
			return self.layer
	
	#---------------------------------------------------------------------------------
	# make layer 3D (from previously stored)
	# e.G. [1, 32] -> [1, 4, 4, 2]
	def unflatten(self):
		unflatShape = self.preFlatShapes.pop()
		if len(unflatShape) == 4:
			unflatShape = [-1, int(unflatShape[1]), int(unflatShape[2]), int(unflatShape[3])]
		elif len(unflatShape) == 5:
			unflatShape = [-1, int(unflatShape[1]), int(unflatShape[2]), int(unflatShape[3]), int(unflatShape[4])]
		self.layer = tf.reshape(self.layer, unflatShape)
		print("Unflatten: {}".format(self.layer.get_shape()))
		return self.layer

	# pixelnorm, used in progressive growing of gans https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
	def pixel_norm(self, in_layer, epsilon=1e-8):
		self.layer = in_layer * tf.rsqrt(tf.reduce_mean(tf.square(in_layer), axis=3, keep_dims=True) + epsilon)
		return self.layer
		
	def minibatch_stddev_layer(self, x, group_size=4):
		group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or less than) group_size.
		s = x.shape                                             # [NCHW]  Input shape.
		y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
		y = tf.cast(y, self.dtypeF)                              # [GMCHW] Cast to FP32.
		y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
		y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
		y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
		y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
		y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
		y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
		self.layer = tf.concat([x, y], axis=3) 
		return self.layer         					            # [NCHW]  Append as new fmap.


# 2D: 
# max_pool2d( inputs, kernel_size, stride=2, padding='VALID', data_format=DATA_FORMAT_NHWC, outputs_collections=None, scope=None)
# try 
# tf.contrib.layers.max_pool3d , https://www.tensorflow.org/api_docs/python/tf/contrib/layers/max_pool3d
# max_pool3d( inputs, kernel_size, stride=2, padding='VALID', data_format=DATA_FORMAT_NDHWC, outputs_collections=None, scope=None)
# -> no fractions / depooling!

	#---------------------------------------------------------------------------------
	# inverse of 2 x 2 max pool , note window size&stride given as [x,y] pair
	# does not support 3D
	def max_depool(self, in_layer=None, depth_factor =2, height_factor=2, width_factor=2):
		if in_layer==None:
			in_layer = self.layer

		#if 1: # alt with deconv
			#lo, li = self.deconvolutional_layer(1, [1,1], None, stride=[2,2], name="g_D1", reuse=reuse, batch_norm=use_batch_norm, train=train) 
			#return lo
		'''
		if len(self.layer.get_shape()) == 4:
			outWidth = in_layer.get_shape()[2] * window_stride[0] + window_size[0] - window_stride[0]
			outHeight = in_layer.get_shape()[1] * window_stride[1] + window_size[1] -  window_stride[1]
			self.layer = tf.image.resize_images(in_layer, [int(outHeight), int(outWidth)], 1) #1 = ResizeMethod.NEAREST_NEIGHBOR
			print("Max Depool {}: {}".format(window_size, self.layer.get_shape()))
		'''
		if len(self.layer.get_shape()) == 4:
			#self.layer = tf.contrib.keras.backend.resize_images(self.layer, height_factor, width_factor, 'channels_last')
			self.layer = kb.resize_images(self.layer, height_factor, width_factor, 'channels_last')
			print("Max Depool : {}".format(self.layer.get_shape()))	
		if len(self.layer.get_shape()) == 5:
			#self.layer = tf.contrib.keras.backend.resize_volumes(self.layer, depth_factor, height_factor, width_factor, 'channels_last')
			self.layer = kb.resize_volumes(self.layer, depth_factor, height_factor, width_factor, 'channels_last')
			print("Max Depool : {}".format(self.layer.get_shape()))
		return self.layer
		
	#---------------------------------------------------------------------------------
	# resizes H and W dimensions of NHWC or NDHWC (scale only H and W for 3D DHW data)
	# https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images
	def avg_depool(self, window_size=[1, 1], window_stride=[2,2], mode = 0, scale = [2]):
		is3D = False
		if len(self.layer.get_shape()) == 5: # 3D data, merge D into C to have a 4D tensor (like 2D data)
			is3D = True
			self.layer = tf.transpose(self.layer, [0,2,3,1,4]) # NDHWC -> NHWDC
			s=self.layer.get_shape() # NHWDC
			self.layer = tf.reshape(self.layer, [-1, int(s[1]), int(s[2]), int(s[3])*int(s[4])]) # NHWDC -> NHW(D*C)
		if len(scale) == 1:
			outWidth = self.layer.get_shape()[2] * scale[0]#window_stride[0] + window_size[0] - window_stride[0]
			outHeight = self.layer.get_shape()[1] * scale[0]#window_stride[1] + window_size[1] -  window_stride[1]
		elif len(scale) == 2:
			outWidth = self.layer.get_shape()[2] * scale[1]#window_stride[0] + window_size[0] - window_stride[0]
			outHeight = self.layer.get_shape()[1] * scale[0]#window_stride[1] + window_size[1] -  window_stride[1]
		self.layer = tf.cast(tf.image.resize_images(tf.cast(self.layer, tf.float32), [int(outHeight), int(outWidth)], mode), self.dtypeF) #0 = ResizeMethod.BILINEAR
		
		if is3D: # recover D dimension
			self.layer = tf.reshape(self.layer, [-1, int(outHeight), int(outWidth), int(s[3]), int(s[4])])
			self.layer = tf.transpose(self.layer, [0,3,1,2,4]) # -> NDHWC
			s=self.layer.get_shape() # NDHWC
			self.layer = tf.reshape(self.layer, [-1, int(s[1]), int(s[2]), int(s[3])*int(s[4])])# NDHWC ->	 NDH(W*C)
			self.layer = tf.cast(tf.image.resize_images(tf.cast(self.layer, tf.float32), [int(s[1]*scale[0]), int(s[2])], mode), self.dtypeF) #0 = ResizeMethod.BILINEAR		
			self.layer = tf.reshape(self.layer, [-1, int(s[1]*scale[0]), int(s[2]), int(s[3]), int(s[4])])# NDHWC

		print("Avg Depool {}: {}".format(window_size, self.layer.get_shape()))
		return self.layer
	
	def pixel_shuffle(self, input_layer = None, upres = 2, stage = "1"):		
		if input_layer == None:
			input_layer = self.layer	
			
		input_layer,_ = self.convolutional_layer(  input_layer.get_shape().as_list()[3] * 4, [1,1], None, stride=[1], name="g_cPS"+stage, in_layer=input_layer, reuse=tf.AUTO_REUSE, batch_norm=False, train=True) #->16,64				
		self.layer = tf.depth_to_space(input_layer, upres, name = "Pixel_Shuffle")
		return self.layer
	
	#---------------------------------------------------------------------------------
	# outChannels: int
	# _patchShape: 2D: [H,W]; 3D: [D,H,W]
	# stride 3D: if 1D: [DHW], if 2D:[D,HW], if 3D:[D,H,W]
	def deconvolutional_layer(self, outChannels, _patchShape, activation_function=tf.nn.tanh, stride=[1], name="deconv",reuse=False, batch_norm=False, train=None, init_mean=0., strideOverride=None):
		if init_mean==1.:
			name = name+"_EXCLUDE_ME_"
		with tf.variable_scope(name):
			self.layer_num += 1
			shape = self.layer.get_shape()
			# spread channels
			inChannels = int(self.layer.get_shape()[-1])
			#outChannels = int(inChannels / _filterSpread) # must always come out even

			dcStride = stride
			if strideOverride is not None:
				dcStride = strideOverride

			if len(_patchShape) == 2:
				if len(stride) == 1:
					stride = [stride[0],stride[0]]
				# create a weight matrix
				W = self.weight_variable([_patchShape[0], _patchShape[1], outChannels, inChannels], name=name, init_mean=init_mean)
				self.layer = self.deconv2d(self.layer, W, [self.batch_size, int(shape[1]*stride[0]), int(shape[2]*stride[1]), outChannels], dcStride)
				self.DOFs += _patchShape[0]* _patchShape[1]* outChannels* inChannels
			if len(_patchShape) == 3:
				if len(stride) == 1:
					stride = [stride[0],stride[0],stride[0]]
				elif len(stride) == 2:
					stride = [stride[0],stride[1],stride[1]]
				# create a weight matrix
				W = self.weight_variable([_patchShape[0], _patchShape[1], _patchShape[2], outChannels, inChannels], name=name, init_mean=init_mean)
				self.layer = self.deconv3d(self.layer, W, [self.batch_size, int(shape[1]*stride[0]), int(shape[2]*stride[1]), int(shape[3]*stride[2]), outChannels], dcStride)
				self.DOFs += _patchShape[0]* _patchShape[1]* _patchShape[2]* outChannels* inChannels
				#batch_norm = False
				
			# create a bias vector
			b = self.bias_variable([outChannels], name=name)
			self.layer = self.layer + b
			self.DOFs += outChannels
			
			if len(_patchShape) == 2:
				self.layer = tf.reshape(self.layer, [-1, int(shape[1]*stride[0]), int(shape[2]*stride[1]), outChannels])
			if len(_patchShape) == 3:
				self.layer = tf.reshape(self.layer, [-1, int(shape[1]*stride[0]), int(shape[2]*stride[1]), int(shape[3]*stride[2]), outChannels])
				
			if batch_norm:
				#self.layer = self.conv_batch_norm(self.layer, train=train)
				self.layer = tf.contrib.layers.batch_norm(self.layer, decay=self.bn_decay, scale=True, scope=tf.get_variable_scope(), reuse=reuse, fused=False, is_training=train)
			layer_lin = self.layer
			if activation_function:
				self.layer = activation_function(self.layer)
			# user output
			if activation_function:
				print("Deconvolutional Layer \'{}\' {} ({}): {}, BN:{}".format(name, W.get_shape(), activation_function.__name__, self.layer.get_shape(),batch_norm))
			else:
				print("Deconvolutional Layer \'{}\' {} ({}): {}, BN:{}".format(name, W.get_shape(), 'None', self.layer.get_shape(),batch_norm))
			return self.layer, layer_lin
	
	#---------------------------------------------------------------------------------
	#adds noise to the current layer
	#channels: number of noise channels to add, uses channels of current layer if < 1
	def noise(self, channels=-1):
		shape=tf.shape(self.layer)
		if channels > 0:
			shape[-1] = channels
		noise = tf.random_normal(shape=shape, mean=0.0, stddev=0.04, dtype=self.dtypeF)
		self.layer = tf.concat([self.layer, noise], axis=-1)
		print("Noise {}: {}".format(noise.get_shape(), self.layer.get_shape()))
		return self.layer

	#---------------------------------------------------------------------------------
	#adds the given tensor to self.layer on axis -1(channels)
	def concat(self, layer):
		self.layer = tf.concat(values=[self.layer, layer], axis=-1)
		print("Concat {}: {}".format(layer.get_shape(), self.layer.get_shape()))
		return self.layer
	#---------------------------------------------------------------------------------
	#applys the given operation to self.layer
	def apply(self, op):
		self.layer = op(self.layer)
		print("Apply \'{}\': {}".format(op.__name__, self.layer.get_shape()))
		return self.layer
	#---------------------------------------------------------------------------------
	def dropout(self, keep_prob):
		self.layer = tf.nn.dropout(self.layer, keep_prob)
		print("Dropout: {}".format(self.layer.get_shape()))
		return self.layer
	
	#---------------------------------------------------------------------------------
	def y(self):
		return self.layer

	#---------------------------------------------------------------------------------
	def getDOFs(self):
		return self.DOFs

	#---------------------------------------------------------------------------------
	# generate random valued weight field
	def weight_variable(self, shape, name="w", gain=np.sqrt(2), use_he = False, in_lay = None, use_wscale = True):
		#use tf.get_variable() instead of tf.Variable() to be able to reuse variables
		
		if in_lay is None: in_lay = np.prod(shape[:-1])
		std = gain / np.sqrt(in_lay) # He init
		if use_wscale:
			wscale = tf.constant(np.float32(std), name='wscale', dtype = self.dtypeF)
			v = tf.get_variable("weight", shape, initializer=tf.initializers.random_normal(dtype = self.dtypeF), dtype = self.dtypeF) * wscale
		else:
			v = tf.get_variable("weight", shape, initializer=tf.keras.initializers.he_normal(dtype = self.dtypeF), dtype = self.dtypeF)
		#else:
		#	v = tf.get_variable("weight", shape, initializer=tf.random_normal_initializer(stddev=s, mean=init_mean))

		#print("\t{}".format(v.name))
		#print("\t{}".format(v.name)) # NT_DEBUG
		#print("\t{}".format( tf.get_variable_scope() )); 
		#exit(1)
		return v

	#---------------------------------------------------------------------------------
	# gemerate biases for the nodes
	def bias_variable(self, shape, name="b"):
		return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.1, dtype = self.dtypeF), dtype = self.dtypeF)

	#---------------------------------------------------------------------------------
	def conv2d(self, x, W, stride=[1]):
		if len(stride) == 1: #[HW]
			strides = [1, stride[0], stride[0], 1]
		elif len(stride) == 2: #[H,W]
			strides = [1, stride[0], stride[1], 1]
		return tf.nn.conv2d(x, W, strides=strides, padding="SAME")
		
	def conv3d(self, x, W, stride=[1]):
		if len(stride) == 1: #[DHW]
			strides = [1, stride[0], stride[0], stride[0], 1]
		elif len(stride) == 2: #[D,HW] for use when striding time and space separately
			strides = [1, stride[0], stride[1], stride[1], 1]
		elif len(stride) == 3: #[D,H,W]
			strides = [1, stride[0], stride[1], stride[2], 1]
		return tf.nn.conv3d(x, W, strides=strides, padding="SAME")

	#---------------------------------------------------------------------------------
	def deconv2d(self, x, W, output_shape, stride=[1]):
		if len(stride) == 1:
			strides = [1, stride[0], stride[0], 1]
		elif len(stride) == 2:
			strides = [1, stride[0], stride[1], 1]
		return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding="SAME")
		
	def deconv3d(self, x, W, output_shape, stride=[1]):
		if len(stride) == 1:
			strides = [1, stride[0], stride[0], stride[0], 1]
		elif len(stride) == 2: # for use when striding time and space separately
			strides = [1, stride[0], stride[1], stride[1], 1]
		elif len(stride) == 3:
			strides = [1, stride[0], stride[1], stride[2], 1]
		return tf.nn.conv3d_transpose(x, W, output_shape=output_shape, strides=strides, padding="SAME")

	def variable_summaries(self, var, name):
		"""Attach a lot of summaries to a Tensor."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean/' + name, mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
			tf.summary.scalar('sttdev/' + name, stddev)
			tf.summary.scalar('max/' + name, tf.reduce_max(var))
			tf.summary.scalar('min/' + name, tf.reduce_min(var))
			tf.summary.histogram(name, var)
			
	
#from https://github.com/bamos/dcgan-completion.tensorflow
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)	
