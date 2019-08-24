import os

# 4x multi pass
if 0:
	## first nn: test 4
	os.system('python multipassGAN-4x.py randSeed 174213111 upRes 4 startIndex 0 out 1 pretrain 0 pretrainDisc 0 tileSize 64 trainingIterations 60000 lambda 5.0 lambda2 0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1005 toSim 1005 outputInterval 200 genTestImg 1 dropout 0.5 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 useK_Eps_Turb 0 useFlags 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan_sliced/ packedSimPath ../data/ lambda_t 1.0 lambda_t_l2 0.0 frame_min 110 frame_max 111 data_fraction 0.01 adv_flag 1 dataAugmentation 1 premadeTiles 0 rot 1 load_model_test 4 load_model_no 1199 sliceMode 1 genUni 1 interpMode 1 upsamplingMode 2 upsampledData 0 velScale 1.0')
	## second nn: test 48
	os.system('python multipassGAN-4x.py randSeed 174213111 upRes 4 startIndex 0 out 1 pretrain 0 pretrainDisc 0 tileSize 64 trainingIterations 60000 lambda 5.0 lambda2 0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1005 toSim 1005 outputInterval 200 genTestImg 1 dropout 0.5 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 useK_Eps_Turb 0 useFlags 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan_sliced/ packedSimPath ../data/ lambda_t 1.0 lambda_t_l2 0.0 frame_min 110 frame_max 111 data_fraction 0.01 adv_flag 1 dataAugmentation 1 premadeTiles 0 rot 1 load_model_test 48 load_model_no 799 sliceMode 1 genUni 1 interpMode 1 upsamplingMode 1 upsampledData 1 velScale 1.0')

# 8x multi pass
if 1:
	## first nn: test 242
	## second nn: test 244 (or 246), whatever works best
	sim_no = 3006
	min_frame = 100
	max_frame = 101

	# either this command:
	os.system('python multipassGAN-out.py randSeed 174213111 upRes 8 use_res_net1 1 use_res_net2 1 use_res_net3 0 pixelNorm 1 batchNorm 0 out 1 tileSize 64 simSize 64 fromSim %04d useVelocities 1 useVorticities 0 useK_Eps_Turb 0 useFlags 0 genModel gen_resnet discModel disc_binclass basePath ../ packedSimPath ../../ frame_max %04d frame_min %04d load_model_test_1 59 load_model_no_1 405 firstNNArch 1 load_model_test_2 409 load_model_no_2 499 load_model_test_3 411 load_model_no_3 749 velScale 1.0 genUni 1 upsampledData 1 upsamplingMode 3 add_adj_idcs1 1 add_adj_idcs2 0 add_adj_idcs3 0 maxFms2 192 startFms2 192 filterSize2 5 maxFms3 192 startFms3 96 filterSize3 5 maxFms1 256 startFms1 256 filterSize1 3 upsampleMode 0 usePixelShuffle 0 loadEmas 0 addBicubicUpsample 1 gpu 0 transposeAxis 0' % (sim_no,max_frame,min_frame))

	#for i in range(20):

	# or these two (also stores intermediate results)
	
	#os.system('python multipassGAN-8x.py randSeed 174213111 upRes 8 use_res_net 1 firstNNArch 1 add_adj_idcs 1 pixelNorm 1 batchNorm 0 out 1 pretrain 0 pretrainDisc 0 tileSize 64 simSize 64 trainingIterations 60000 lambda 5.0 lambda2 0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim %04d toSim %04d outputInterval 200 genTestImg 1 dropout 0.5 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 useK_Eps_Turb 0 useFlags 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../ packedSimPath ../../ lambda_t 1.0 lambda_t_l2 0.0 frame_max %04d frame_min %04d data_fraction 0.05 adv_flag 1 dataAugmentation 1 premadeTiles 0 rot 1 load_model_test 59 load_model_no 405 velScale 1.0 genUni 1 upsampledData 0 upsamplingMode 2 maxFms 256 startFms 256 filterSize 3 usePixelShuffle 0 loadEmas 0 upsampleMode 0 addBicubicUpsample 1 gpu 0 transposeAxis 0' % (sim_no, sim_no,max_frame,min_frame))
	
	#os.system('python multipassGAN-8x.py randSeed 174213111 upRes 8 use_res_net 1 outNNTestNo 59 pixelNorm 1 batchNorm 0 out 1 pretrain 0 pretrainDisc 0 tileSize 64 simSize 64 trainingIterations 60000 lambda 5.0 lambda2 0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim %04d toSim %04d outputInterval 200 genTestImg 1 dropout 0.5 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 useK_Eps_Turb 0 useFlags 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../ packedSimPath ../../ lambda_t 1.0 lambda_t_l2 0.0 frame_max %04d frame_min %04d data_fraction 0.05 adv_flag 1 dataAugmentation 1 premadeTiles 0 rot 1 load_model_test 409 load_model_no 499 velScale 1.0 genUni 1 upsampledData 1 upsamplingMode 1 maxFms 256 startFms 192 filterSize 5 usePixelShuffle 0 loadEmas 0 addBicubicUpsample 1 gpu 0 transposeAxis 2' % (sim_no, sim_no,max_frame,min_frame))
	
	#os.system('python multipassGAN-8x.py randSeed 174213111 upRes 8 use_res_net 0 outNNTestNo 409 pixelNorm 1 batchNorm 0 out 1 pretrain 0 pretrainDisc 0 tileSize 64 simSize 64 trainingIterations 60000 lambda 5.0 lambda2 0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim %04d toSim %04d outputInterval 200 genTestImg 1 dropout 0.5 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 useK_Eps_Turb 0 useFlags 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../ packedSimPath ../../ lambda_t 1.0 lambda_t_l2 0.0 frame_max %04d frame_min %04d data_fraction 0.05 adv_flag 1 dataAugmentation 1 premadeTiles 0 rot 1 load_model_test 411 load_model_no 749 velScale 1.0 genUni 1 upsampledData 1 upsamplingMode 3 maxFms 256 startFms 96 filterSize 5 usePixelShuffle 0 loadEmas 0 addBicubicUpsample 1 gpu 0 transposeAxis 1' % (sim_no, sim_no,max_frame,min_frame))

