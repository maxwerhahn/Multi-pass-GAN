
import numpy as np
import time, shutil, os, sys

sys.path.append("../tools")
import tilecreator_t as tc
import fluiddataloader as fdl

image_counter = 0
def save_img(data, save_vel=False, shape=[1,1]):
	global image_counter
	tc.savePngsGrayscale(data[tc.DATA_KEY_HIGH],'../tiletest/test_img/high_',imageCounter=image_counter, tiles_in_image=shape, plot_vel_x_y=False)
	tc.savePngsGrayscale(data[tc.DATA_KEY_LOW], '../tiletest/test_img/low_', imageCounter=image_counter, channels = [4], tiles_in_image=shape, plot_vel_x_y=False)
	image_counter += 1

dim = 2
fromSim = 1000
toSim = 1000
print('##### init') 

useVelocities = True

channelLayout_low = 'd'
mfl = ["density"]
mfh = ["density"]

if useVelocities:
	channelLayout_low += ',vx,vy,vz'
	mfl= np.append(mfl, "velocity")
	channelLayout_low += ',f'
	mfl= np.append(mfl, "flags")

#premadeTiles: True- cut tiles with a regular pattern when loading data, False- cut random tiles on demand
TC = tc.TileCreator(tileSizeLow=16, simSizeLow=64, dim=dim, dim_t=1, upres=4,premadeTiles=False, densityMinimum=0.05, channelLayout_low=channelLayout_low) #
#clear test images
#if os.path.exists('../test_img/'):
#	shutil.rmtree('../test_img/')
#os.makedirs('../test_img/')
print('##### load')
dirIDs = np.linspace(fromSim, toSim, (toSim-fromSim+1),dtype='int16')
lowfilename = "density_low_%04d.uni"
highfilename = "density_high_%04d.uni"

FDL = fdl.FluidDataLoader( print_info=1, base_path='../data/', filename=lowfilename, oldNamingScheme=False,  filename_y=highfilename, filename_index_max=120, indices=dirIDs, data_fraction=1, multi_file_list=mfl, multi_file_list_y=mfh)
x, y, xFilenames  = FDL.get()

print(x.shape)
if 1:
	xt = []
	for z in range (1,4):
		Obsinput = x[:,:,:,:,int(5*z-1):int(5*z)] 
		densVel = x[:,:,:,:,int((z-1)*5):int(z*5-1)] 
		for i in range(0, x.shape[0]):
			for j in range(0, 64):
				for k in range(0, 64):
					if Obsinput[i][0][j][k][0] <= 3 and Obsinput[i][0][j][k][0] >= 1.5:
						Obsinput[i][0][j][k][0] = 1.0
					else:
						Obsinput[i][0][j][k][0] = 0.0

		if(len(xt) == 0):
			xt = np.concatenate((densVel, Obsinput), axis = 4)
		else:			
			xt = np.concatenate((xt,np.concatenate((densVel, Obsinput), axis = 4)), axis=4)
	x = xt
y = y[:,:,:,0:256,:]
TC.addData(x,y)
print(x.shape)
print('##### sample')
#get batch, data format: [batchSize, z, y, x, channels]
#low, high = TC.getRandomFrame()#TC.selectRandomTiles(128)
#
#test output, all tiles in one image; average z axis, factor for visibility
#tc.savePngsGrayscale(tiles=[np.average(high, axis=0)*8], path='../test_img/high_', imageCounter=0, tiles_in_image=[1,1])
#tc.savePngsGrayscale([np.average(low, axis=0)*8], '../test_img/low_', tiles_in_image=[1,1])

#test parser
if 1:
	print('\n\tparser test 0')
	TC.parseChannels('d	,  D, vx,vy,vZ ,d, v1z,v1y ,v1x')
	print('\n\tparser test 1')
	#duplicate
	try:
		TC.parseChannels('d,d,vx,vy,vz,d,v1z,v1y,v1x,vx')
	except tc.TilecreatorError as e:
		print(e)
	#missing
	print('\n\tparser test 2')
	try:
		TC.parseChannels('d,d,vx,vy,vz,d,v1z,v1y')
	except tc.TilecreatorError as e:
		print(e)
	print('\n\tparser test 3')
	try:
		TC.parseChannels('d,d,vx,vy,vz,d,v1z,v1x')
	except tc.TilecreatorError as e:
		print(e)
	# unsupportet
	print('\n\tparser test 4')
	try:
		TC.parseChannels('d,d,vx,vy,vz,d,b')
	except tc.TilecreatorError as e:
		print(e)
	print('\n\tparser test 5')
	try:
		TC.parseChannels('d,d,vx,vy,vz,dd')
	except tc.TilecreatorError as e:
		print(e)

#test batch creation with complete augmentation
TC.initDataAugmentation(rot=2, minScale=0.85, maxScale=1.15 ,flip=True)
if 0:
	batch = 32
	startTime = time.time()
	low, high = TC.selectRandomTiles(batch, augment=True)
	endTime=(time.time()-startTime)
	print('{} tiles batch creation time: {:.4f}, per tile: {:.4f}'.format(batch, endTime, endTime/batch))
	#print(low.shape, high.shape)
	if dim == 3:
		high = np.average(high, axis=1)*8
		low = np.average(low, axis=1)*8
	if dim == 2:
		high.shape = (batch, 64, 64, 1)
		low.shape = (batch, 16, 16, 1)
	tc.savePngsGrayscale(tiles=high, path='../tiletest/test_img/batch_high_', imageCounter=0, tiles_in_image=[4,8])
	tc.savePngsGrayscale(low, '../tiletest/test_img/batch_low_', tiles_in_image=[4,8])
	
# test load data
if 0:
	#low, high = TC.getFrame(20)
	TC.clearData()
	low, high = np.ones((64,64,4)), np.ones((256,256,1))
	try:
		TC.addData(low, high)
	except tc.TilecreatorError as e:
		print(e)

#test tile concat
if 0:
	frame = 20
	low, high = TC.getDatum(frame)
	
	high_tiles = TC.createTiles(high, TC.tile_shape_high)
	tc.savePngsGrayscale(np.reshape(high_tiles,(len(high_tiles), 64,64, 1)),'../tiletest/test_img/high_',imageCounter=0, tiles_in_image=[4,4])
	
	high_tiles = TC.createTiles(high, TC.tile_shape_high, strides=32)
	tc.savePngsGrayscale(np.reshape(high_tiles,(len(high_tiles), 64,64, 1)),'../tiletest/test_img/high_',imageCounter=1, tiles_in_image=[7,7])
	
	high_frame = TC.concatTiles(high_tiles, [1,7,7])
	tc.savePngsGrayscale(np.reshape(high_frame,(1, 448,448, 1)),'../tiletest/test_img/high_',imageCounter=2, tiles_in_image=[1,1])
	high_frame = TC.concatTiles(high_tiles, [1,7,7], [0,16,16,0])
	tc.savePngsGrayscale(np.reshape(high_frame,(1, 224,224, 1)),'../tiletest/test_img/high_',imageCounter=3, tiles_in_image=[1,1])
	
	
#test all augmentation methods 2D
if 1 and dim==2:
	frame = 20
	data = {}
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	#low, high = TC.frame_inputs[10], TC.frame_outputs[10]
	save_img(data, True)
	
	save_img(TC.flip(data, [1]), True) #flip y
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.flip(data, [2]), True)
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.flip(data, [1,2]), True)
	
	#rot
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.rotate(data), True)
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.rotate(data), True)
	
	#scale
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.scale(data, 0.8), True)
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.scale(data, 1.2), True)
	
	#rot90
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.rotate90(data, [2,1]), True)
	data[tc.DATA_KEY_LOW], data[tc.DATA_KEY_HIGH] = TC.getDatum(frame)
	save_img(TC.rotate90(data, [1,2]), True)
	
#test FLIP
if 0 and dim==3:
#	low_f, high_f = TC.flip(low, high, [0]) #flip z, won't show as we average over z (3D) or z is only 1 (2D)
#	tc.savePngsGrayscale(tiles=[np.average(high_f, axis=0)*8], tileSize=high_f.shape[2], path='../test_img/high_', imageCounter=1, tiles_in_image=[1,1])
	low_f, high_f = TC.flip(low, high, [1]) #flip y
	tc.savePngsGrayscale(tiles=[np.average(high_f, axis=0)*8], path='../test_img/high_', imageCounter=2, tiles_in_image=[1,1])
	tc.savePngsGrayscale([np.average(low_f, axis=0)*8], '../test_img/low_', imageCounter=2, tiles_in_image=[1,1])
#	low_f, high_f = TC.flip(low, high, [2]) #flip x
#	tc.savePngsGrayscale(tiles=[np.average(high_f, axis=0)*8], path='../test_img/high_', imageCounter=3, tiles_in_image=[1,1])
#	low_f, high_f = TC.flip(low, high, [1,2]) #flip y and x
#	tc.savePngsGrayscale(tiles=[np.average(high_f, axis=0)*8], path='../test_img/high_', imageCounter=4, tiles_in_image=[1,1])

#test ROT
if 0 and dim==3:
	print('testing rotation performance, this may take a while...')
	batch = 10
	theta = [ np.pi / 180 * 45,
			  np.pi / 180 * 0,
			  np.pi / 180 * 0 ]
	startTime = time.time()
	for i in range(batch):
		low_r, high_r = TC.rotate(low, high, theta)
	endTime=(time.time()-startTime)/batch
	print('matrix rot time: {:.8f}'.format(endTime))
	tc.savePngsGrayscale(tiles=[np.average(high_r, axis=0)*8], path='../test_img/high_', imageCounter=1, tiles_in_image=[1,1])
	startTime = time.time()
	for i in range(batch):
		low_r, high_r = TC.rotate4(low, high, theta)
	endTime=(time.time()-startTime)/batch
	print('matrix 4D rot time: {:.8f}'.format(endTime))
	tc.savePngsGrayscale(tiles=[np.average(high_r, axis=0)*8], path='../test_img/high_', imageCounter=2, tiles_in_image=[1,1])
	startTime = time.time()
	for i in range(batch):
		low_r, high_r = TC.rotate_simple(low, high, -45)
	endTime=(time.time()-startTime)/batch
	print('simple rot time: {:.8f}'.format(endTime))
	tc.savePngsGrayscale(tiles=[np.average(high_r, axis=0)*8], path='../test_img/high_', imageCounter=3, tiles_in_image=[1,1])
	
if 0 and dim==3:
	theta = [ np.pi / 180 * 45,
			  np.pi / 180 * 0,
			  np.pi / 180 * 0 ]
	low_r, high_r = TC.rotate(low, high, theta)
	tc.savePngsGrayscale(tiles=[np.average(high_r, axis=0)*8], path='../test_img/high_', imageCounter=1, tiles_in_image=[1,1], plot_vel_x_y=True)
	tc.savePngsGrayscale([np.average(low_r, axis=0)*8], '../test_img/low_', imageCounter=1, tiles_in_image=[1,1], plot_vel_x_y=True)
#	theta = [ np.pi / 180 * 0,
#			  np.pi / 180 * 45,
#			  np.pi / 180 * 0 ]
#	low_r, high_r = TC.rotate(low, high, theta)
#	tc.savePngsGrayscale(tiles=[np.average(high_r, axis=0)*8], path='../test_img/high_', imageCounter=2, tiles_in_image=[1,1])
#	tc.savePngsGrayscale([np.average(low_r, axis=0)*8], '../test_img/low_', imageCounter=2, tiles_in_image=[1,1])
	

#test SCALE
if 0 and dim==3:
	low_s, high_s = TC.scale(low, high, 0.8)
	tc.savePngsGrayscale(tiles=[np.average(high_s, axis=0)*8], path='../test_img/high_', imageCounter=1, tiles_in_image=[1,1])
	low_s, high_s = TC.scale(low, high, 1.2)
	tc.savePngsGrayscale(tiles=[np.average(high_s, axis=0)*8], path='../test_img/high_', imageCounter=2, tiles_in_image=[1,1])

