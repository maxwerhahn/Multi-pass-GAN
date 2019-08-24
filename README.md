# Multi-pass-GAN  

Public source code for the SCA paper "A Multi-Pass GAN for Fluid Flow Super-Resolution". Authors: Maximilian Werhahn, You Xie, MengYu Chu, Nils Thuerey. Technical University of Munich.  

Paper: https://arxiv.org/pdf/1906.01689.pdf,  
Video: https://www.youtube.com/watch?v=__WE22dB6AA  

## Requirements  
  
tensorflow >= 1.10  
mantaflow for datagen  

## Directories  
`../datagen/`:			data generation via mantaflow  
`../GAN/`:					output + training + network files  
`../tools_wscale/`:	helper functions, data loader, etc.  

## Compilation  
First, compile mantaflow with numpy support (as usual), follow 
http://mantaflow.com/install.html.
One difference is, in the CMake settings, numpy shoule be enabled: 
"cmake .. -DGUI=ON -DOPENMP=ON -DNUMPY=ON".
Note that if mantaflow is installed on a remote server, GUI is not supported, i.e.:
"cmake .. -DGUI=OFF -DOPENMP=ON -DNUMPY=ON".

## Data Generation  
Either use the file `../datagen/gen_mul_data.py` or similar commands for the file `../datagen/gen_sim_grow_slices_data.py` to generate a dataset. It will be stored in `../data3d_growing/sim_%04d`.

## Training  
Call `../GAN/example_run_training.py`
