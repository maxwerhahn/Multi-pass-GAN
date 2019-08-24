#******************************************************************************
#
# tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
# Copyright 2018 You Xie, Erik Franz, Mengyu Chu, Nils Thuerey, Maximilian Werhahn
#
# Varying density data gen, 2d/3d
# 2 modes:
#
# - double sim mode (mode==1)
# 		Runs hi-res, then downsamples to coarse ("sm") sim in intervals, 
# 		by default every frame
#
# - wavelet turbulence mode (mode==2)
# 		Runs low-res, then upsamples and adds WLT
#
#******************************************************************************

from manta import *
import os, shutil, math, sys, time
from datetime import datetime
import numpy as np
sys.path.append("../tools")
import paramhelpers as ph

# Main params  ----------------------------------------------------------------------#
steps    = 120
simNo    = 1000  # start ID
showGui  = 0
basePath = 'V:/data3d_sliced_growing2/'
npSeedstr   = "-1"
dim         = 3

# Solver params  
res         = 64
resetN      = 1
saveEveryK  = 2
targettimestep = 0.5
wup = 10 + np.random.randint(6)

# cmd line
basePath        =     ph.getParam( "basepath",        basePath        )
npSeedstr       =     ph.getParam( "seed"    ,        npSeedstr       )
npSeed          =     int(npSeedstr)
resetN			= int(ph.getParam( "reset"   ,        resetN))
dim   			= int(ph.getParam( "dim"     ,        dim))
simMode			= int(ph.getParam( "mode"    ,        1 ))  # 1 = double sim, 2 = wlt
savenpz 		= int(ph.getParam( "savenpz",         False))>0
saveuni 		= int(ph.getParam( "saveuni",         True))>0
saveppm 		= int(ph.getParam( "saveppm" ,        False))>0
showGui 		= int(ph.getParam( "gui"     ,        showGui))
res     		= int(ph.getParam( "res"     ,        res))
steps     		= int(ph.getParam( "steps"   ,        steps))
timeOffset   	= int(ph.getParam( "warmup"  ,        wup))    # skip certain no of steps at beginning
scaleFactorMin = int(ph.getParam( "minUpRes"  ,        2))
scaleFactorMax = int(ph.getParam( "maxUpRes"  ,        8))
useObstacles = int(ph.getParam( "obstacles"  ,        False))>0
ph.checkUnusedParams()
doRecenter  = False   # re-center densities , disabled for now

setDebugLevel(1)
if not basePath.endswith("/"): basePath = basePath+"/"

# Init solvers -------------------------------------------------------------------#
sm_gs = vec3(res,res,res) 
xl_gs = sm_gs * float(scaleFactorMax)
if (dim==2):  xl_gs.z = sm_gs.z = 1  # 2D

# solvers
sms = [] 
for i in range(int(math.log(scaleFactorMax,2))):
	gs = sm_gs * (2**i)
	if dim == 2: gs.z = 1
	sms.append(Solver(name='smaller'+ str(i), gridSize = gs, dim=dim))
	sms[i].timestep = targettimestep / saveEveryK
# wlt Turbulence output fluid
xl = Solver(name='larger', gridSize = xl_gs, dim=dim)
xl.timestep = targettimestep / saveEveryK
timings = Timings()

fff=0.5
buoyFac = vec3(-1.5 + np.random.rand() * 3.0, 2.5 + np.random.rand() * 3.5, -1.5 + np.random.rand() * 3.0) * 2.0
if buoyFac.y < 2.5:
	buoyFac = vec3(0,0,0) # make sure we have some sims without buoyancy
#buoyFac = 0.125
buoy    = vec3(0.0,-0.0005,0.0) * buoyFac / saveEveryK
xl_buoys = []
for i in range(int(math.log(scaleFactorMax,2))):
	xl_buoys.append(buoy) # * vec3(1./scaleFactorMax * (2**i)))
	print("Buoyancy: " + format(xl_buoys[i]) +", factor " + str(buoyFac))
print("Buoyancy: " + format(buoy) +", factor " + str(buoyFac))
# xl_buoy = buoy * vec3(1./scaleFactor)

if savenpz or saveuni or saveppm: 
	folderNo = simNo
	simPath,simNo = ph.getNextSimPath(simNo, basePath)

	# add some more info for json file
	ph.paramDict["simNo"] = simNo
	ph.paramDict["type"] = "smoke"
	ph.paramDict["dt"] = 0.5
	ph.paramDict["buoyX"] = buoy.x
	ph.paramDict["buoyY"] = buoy.y
	ph.paramDict["buoyZ"] = buoy.z
	ph.paramDict["seed"] = npSeed
	ph.paramDict["name"] = "gen6combined"
	ph.paramDict["version"] = printBuildInfo()
	ph.paramDict["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
	ph.writeParams(simPath + "description.json") # export sim parameters 

	sys.stdout = ph.Logger(simPath)
	print("Called on machine '"+ sys.platform[1] +"' with: " + str(" ".join(sys.argv) ) )
	print("Saving to "+simPath+", "+str(simNo))
	# optional , backupFile(__file__, simPath)  

if(npSeed<0): 
	npSeed = np.random.randint(0, 2147483647 )
print("Random seed %d" % npSeed)
np.random.seed(npSeed)

# Simulation Grids  -------------------------------------------------------------------#
xl_flags   = xl.create(FlagGrid)
xl_vel     = xl.create(MACGrid)
xl_density = xl.create(RealGrid)
xl_velTmp  = xl.create(MACGrid)
xl_tmp     = xl.create(RealGrid)
xl_phiObs = xl.create(LevelsetGrid)

# for domain centering
xl_velRecenter = xl.create(MACGrid)
phiObs = sms[0].create(LevelsetGrid)
velRecenters = []
flags    = []
vel      = []
velTmp   = []
density  = []
tmp = []
obstacles = []
xl_obstacles = []

bWidth=0

for i in range(int(math.log(scaleFactorMax,2))):
	flags.append(sms[i].create(FlagGrid))
	flags[i].initDomain(boundaryWidth=bWidth * 2**i)
	
if useObstacles: 
	obstacles = []
	xl_obstacles = []
	# init obstacles
	min_p = res//8
	max_p = res//8*7
	min_v = 2
	max_v = res//32 * 6
	num = np.random.randint(7,15)
	for i in range(num):
		rect = np.random.randint(0,2)
		rand_p = vec3(np.random.randint(min_p,max_p), np.random.randint(min_p,max_p) + res//16*3, np.random.randint(min_p,max_p))
		rand_s = vec3(np.random.randint(min_v,max_v), np.random.randint(min_v,max_v), np.random.randint(min_v,max_v))

		if rect:		
			obstacles.append(Box( parent=sms[0],p0=rand_p - rand_s / 2, p1 = rand_p + rand_s / 2))		
			xl_obstacles.append(Box( parent=xl,p0=scaleFactorMax * (rand_p - rand_s / 2), p1 = scaleFactorMax * (rand_p + rand_s / 2)))
		else:
			rand_s = np.random.randint(min_v,max_v-1)
			obstacles.append(Sphere(parent=sms[0], center=rand_p, radius=rand_s))	
			xl_obstacles.append(Sphere(parent=xl, center=scaleFactorMax*rand_p, radius=scaleFactorMax * rand_s))
	
	for i in range (0, len(obstacles)):
		phiObs.join(obstacles[i].computeLevelset())
		
	for i in range (0, len(xl_obstacles)):
		xl_phiObs.join(xl_obstacles[i].computeLevelset())
		
	setObstacleFlags(flags=flags[0], phiObs=phiObs) 
	setObstacleFlags(flags=xl_flags, phiObs=xl_phiObs) 

for i in range(int(math.log(scaleFactorMax,2))):
	density.append(sms[i].create(RealGrid))
	velRecenters.append(sms[i].create(MACGrid))
	vel.append(sms[i].create(MACGrid))
	tmp.append(sms[i].create(RealGrid))
	flags[i].fillGrid()

xl_flags   = xl.create(FlagGrid)
xl_vel     = xl.create(MACGrid)
xl_velTmp  = xl.create(MACGrid)
xl_density = xl.create(RealGrid)
xl_flags.initDomain(boundaryWidth=bWidth*scaleFactorMax)
xl_flags.fillGrid()

boundaries = np.random.randint(4)
print(boundaries)
for i in range(int(math.log(scaleFactorMax,2))):
##	if boundaries == 1:
#		setOpenBound(flags[i],    bWidth * 2**i,'xy',FlagOutflow|FlagEmpty) 
#	elif boundaries == 2:
#		setOpenBound(flags[i],    bWidth* 2**i,'xX',FlagOutflow|FlagEmpty) 
#	else:
	setOpenBound(flags[i],    bWidth * 2**i,'xXyYzZ',FlagOutflow|FlagEmpty) 
		
#if boundaries == 1:
#	setOpenBound(xl_flags, bWidth*scaleFactorMax,'xy',FlagOutflow|FlagEmpty) 
#elif boundaries == 2:
#	setOpenBound(xl_flags, bWidth*scaleFactorMax,'xX',FlagOutflow|FlagEmpty) 
#else:
setOpenBound(xl_flags, bWidth*scaleFactorMax,'xXyYzZ',FlagOutflow|FlagEmpty)

# wavelet turbulence octaves

wltnoise = NoiseField( parent=xl, loadFromFile=True)
# scale according to lowres sim , smaller numbers mean larger vortices
wltnoise.posScale = vec3( int(1.0*sm_gs.x) ) * 0.5
wltnoise.timeAnim = 0.05

wltnoise2 = NoiseField( parent=xl, loadFromFile=True)
wltnoise2.posScale = wltnoise.posScale * 2.0
wltnoise2.timeAnim = 0.02

wltnoise3 = NoiseField( parent=xl, loadFromFile=True)
wltnoise3.posScale = wltnoise2.posScale * 2.0
wltnoise3.timeAnim = 0.03

# inflow sources ----------------------------------------------------------------------#

# init random density
sources  = []
noise    = []  # xl
sourSm   = []
noiSm    = []  # sm
inflowSrc = [] # list of IDs to use as continuous density inflows

noiseN = 18
#noiseN = 1
nseeds = np.random.randint(10000,size=noiseN)

cpos = vec3(0.5,0.25,0.5)

randoms = np.random.rand(noiseN, 10)
random_scales = np.random.rand(noiseN, 1)
for nI in range(noiseN):
	if random_scales[nI] > 0.15:
		random_scales[nI] = 1.0 + np.random.rand()*0.25
	else:
		random_scales[nI] = np.random.rand()*0.9 + 0.35
	
	rand_val = 0.5 + np.random.rand()*0.75
	noise.append( xl.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
	#noise[nI].posScale = vec3( res * (0.02 + 0.055 * np.random.rand()) * (randoms[nI][7] + 1) ) * (float(scaleFactorMax))
	noise[nI].posScale = vec3( res * 0.1 * (randoms[nI][7] + 1) ) * ( float(scaleFactorMax))
	noise[nI].clamp = True
	noise[nI].clampNeg = -np.random.rand() * 0.1
	noise[nI].clampPos = rand_val + (1.25-rand_val) * np.random.rand()
	noise[nI].valScale = 1.0
	#noise[nI].valOffset = 0.5 * randoms[nI][9] * (1.0 + np.random.rand())
	noise[nI].valOffset = 0.5 * randoms[nI][9]
	noise[nI].timeAnim = 0.25 + np.random.rand() * 0.1
	noise[nI].posOffset = vec3(1.5)
	
	for i in range(int(math.log(scaleFactorMax,2))):
		noiSm.append([])
		noiSm[i].append( sms[i].create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
		noiSm[i][nI].timeAnim = noise[nI].timeAnim / ( float(scaleFactorMax) * 2**i)
		noiSm[i][nI].posOffset = vec3(1.5) 
		noiSm[i][nI].posScale = noise[nI].posScale #vec3( res * 0.1 * (randoms[nI][7] + 1) ) * ( float(2**i)) # noise[nI].posScale
		noiSm[i][nI].clamp    = noise[nI].clamp    
		noiSm[i][nI].clampNeg = noise[nI].clampNeg 
		noiSm[i][nI].clampPos = noise[nI].clampPos 
		noiSm[i][nI].valScale = noise[nI].valScale 
		noiSm[i][nI].valOffset= noise[nI].valOffset
		
	# random offsets
	coff = vec3(0.45,0.3,0.45) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
	radius_rand = 0.05 + 0.05 * randoms[nI][3]
	if radius_rand > 0.14:
		radius_rand *= (1.0 + np.random.rand() * 0.325)
		
	upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )

	if 1 and randoms[nI][8] > 0.5: # turn into inflow?
		if coff.y > -0.2:
			coff.y += -0.1
		coff.y *= 0.5
		inflowSrc.append(nI)

	if(dim == 2): 
		coff.z = 0.0
		upz.z = 1.0
	if np.random.randint(2):
		sources.append(xl.create(Sphere, center=xl_gs*(cpos+coff), radius=xl_gs.x*radius_rand, scale=upz))
	else:
		sources.append(xl.create(Cylinder, center=xl_gs*(cpos+coff), radius=xl_gs.x*radius_rand, z=gs*vec3(0.05 - 0.1 * np.random.rand(), 0.05 - 0.1 * np.random.rand(), 0.05 - 0.1 * np.random.rand())))
		
	for i in range(int(math.log(scaleFactorMax,2))):
		sourSm.append([])
		sourSm[i].append( sms[i].create(Sphere, center=sm_gs*(cpos+coff)*2**i, radius=sm_gs.x*radius_rand*2**i, scale=upz))
		
	print (nI, "centre", xl_gs*(cpos+coff), "radius", xl_gs.x*radius_rand, "other", upz )	
	densityInflow( flags=xl_flags, density=xl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
	for i in range(int(math.log(scaleFactorMax,2))):
		densityInflow( flags=flags[i], density=density[i], noise=noiSm[i][nI], shape=sourSm[i][nI], scale=1.0, sigma=1.0 )

# init random velocities

inivel_sources = []
inivel_vels = []
inivel_sourcesSm = []
inivel_velsSm = []
if 1: # from fluidnet
	c = 3 + np.random.randint(3) # "sub" mode
	xgs = xl_gs
	if 1:
		# 3..5 - ini vel sources
		if c==3: numFac = 2; sizeFac = 0.7;
		if c==4: numFac = 3; sizeFac = 0.5;
		if c==5: numFac = 5; sizeFac = 0.3;
		numNs = int( numFac * float(dim) )  
		for ns in range(numNs):
			p = [0.5,0.5,0.5]
			Vrand = np.random.rand(10) 
			for i in range(3):
				p[i] += (Vrand[0+i]-0.5) * 0.6
			p = Vec3(p[0],p[1],p[2])
			size = ( 0.06 + 0.14*Vrand[3] ) * sizeFac

			v = [0.,0.,0.]
			for i in range(3):
				v[i] -= (Vrand[0+i]-0.5) * 0.6 * 2. # invert pos offset , towards middle
				v[i] += (Vrand[4+i]-0.5) * 0.3      # randomize a bit, parametrized for 64 base
			v = Vec3(v[0],v[1],v[2]) 
			v = v*0.9*0.325 # tweaking
			v = v*(1. + 0.5*Vrand[7] ) # increase by up to 50% 
			v *= float(scaleFactorMax)

			print( "IniVel Pos " + format(p) + ", size " + format(size) + ", vel " + format(v) )
			sourceV = xl.create(Sphere, center=xgs*p, radius=xgs.x*size, scale=vec3(1))
			inivel_sources.append(sourceV)
			inivel_vels.append(v)
			for i in range(int(math.log(scaleFactorMax,2))):
				sourceVsm = sms[i].create(Sphere, center=sm_gs*p*2**i, radius=sm_gs.x*size*2**i, scale=vec3(1))
				inivel_sourcesSm.append([])
				inivel_sourcesSm[i].append(sourceVsm)
				inivel_velsSm.append([])
				inivel_velsSm[i].append(v)# * (1./scaleFactorMax) * 2**i)

blurSigs = []
for i in range(int(math.log(scaleFactorMax,2))):
	blurSigs.append( float(scaleFactorMax) / (2**i) / 3.544908)	 # 3.544908 = 2 * sqrt( PI )
	#xl_blurden.copyFrom( xl_density )
	#blurRealGrid( xl_density, xl_blurden, blurSigs[i])
	#interpolateGrid( target=density[i], source=xl_blurden )

	xl_velTmp.copyFrom( xl_vel )
	blurMacGrid( xl_vel, xl_velTmp, blurSigs[i])
	interpolateMACGrid( target=vel[i], source=xl_velTmp )
	vel[i].multConst( vec3(1./scaleFactorMax*(2**i)) * saveEveryK )

# wlt params ---------------------------------------------------------------------#

if simMode==2:
	wltStrength = 0.8
	if resetN==1:
		print("Warning!!!!!!!!!!!!!! Using resetN=1 for WLT doesnt make much sense, resetting to never")
		resetN = 99999

def calcCOM(dens):
	velOffsets = []
	if doRecenter:
		newCentre = calcCenterOfMass(xl_density)
		#mantaMsg( "Current moff "+str(newCentre) )
		xl_velOffset = xl_gs*float(0.5) - newCentre
		xl_velOffset = xl_velOffset * (1./ xl.timestep) 
		
		for i in range(int(math.log(scaleFactorMax,2))):
			velOffsets.append(xl_velOffset * (1./ float(scaleFactorMax) * 2 ** i))
			if(dim == 2):
				xl_velOffset.z = velOffsets[i].z = 0.0 
	else: 
		for i in range(int(math.log(scaleFactorMax,2))):
			velOffsets.append(vec3(0.0))
		xl_velOffset = vec3(0.0)

	return velOffsets, xl_velOffset

# Setup UI ---------------------------------------------------------------------#
if (showGui and GUI):
	gui=Gui()
	gui.show()
	#gui.pause()

t = 0
doPrinttime = False

# main loop --------------------------------------------------------------------#
while t < steps*saveEveryK+timeOffset*saveEveryK:
	curt = t * xl.timestep
	sys.stdout.write( "Current sim time t: " + str(curt) +" \n" )
	#density.setConst(0.); xl_density.setConst(0.); # debug reset

	if doPrinttime:
		starttime = time.time()
		print("starttime: %2f" % starttime)	

	# --------------------------------------------------------------------#
	if simMode==1: 
		velOffsets, xl_velOffset = calcCOM(xl_density)

		if 1 and len(inflowSrc)>0:
			# note - the density inflows currently move with the offsets!
			for nI in inflowSrc:
				#for i in range(int(math.log(scaleFactorMax,2))):
				#	densityInflow( flags=flags[i], density=density[i], noise=noiSm[i][nI], shape=sourSm[i][nI], scale=1.0, sigma=1.0 )
				densityInflow( flags=xl_flags, density=xl_density, noise=noise[nI], shape=sources[nI], scale=random_scales[nI][0], sigma=1.0 )
				if t < timeOffset*saveEveryK:
					sources[i].applyToGrid( grid=xl_vel , value=inivel_vels[i]*1.5)
		
		#xl_flags.applyToGrid(xl_density,TypeObstacle, value =0.0)
		# high res fluid
		advectSemiLagrange(flags=xl_flags, vel=xl_vel, grid=xl_vel, order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth)
		setWallBcs(flags=xl_flags, vel=xl_vel, phiObs=xl_phiObs)
		addBuoyancy(density=xl_density, vel=xl_vel, gravity=buoy , flags=xl_flags)
		if 1:
			for i in range(len(inivel_sources)):
				inivel_sources[i].applyToGrid( grid=xl_vel , value=inivel_vels[i] )
		if 1 and ( t< (timeOffset+5) ): 
			vorticityConfinement( vel=xl_vel, flags=xl_flags, strength=0.035 )

		solvePressure(flags=xl_flags, vel=xl_vel, pressure=xl_tmp ,  preconditioner=PcMGStatic )
		setWallBcs(flags=xl_flags, vel=xl_vel, phiObs=xl_phiObs)
		xl_velRecenter.copyFrom( xl_vel )
		#xl_velRecenter.addConst( xl_velOffset )
		if( dim == 2 ):
			xl_vel.multConst( vec3(1.0,1.0,0.0) )
			xl_velRecenter.multConst( vec3(1.0,1.0,0.0) )
		advectSemiLagrange(flags=xl_flags, vel=xl_vel, grid=xl_density, order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth)

		# low res fluid, velocity
		if( t % resetN == 0) :
			for i in range(int(math.log(scaleFactorMax,2))):
				if i == 0:
					xl_velTmp.copyFrom( xl_vel )
					blurMacGrid( xl_vel, xl_velTmp, blurSigs[i])
					interpolateMACGrid( target=vel[i], source=xl_velTmp )
					vel[i].multConst( vec3(1./scaleFactorMax) * 2**i * saveEveryK)
				#flags[i].applyToGrid(vel[i], TypeObstacle, value =vec3(0.0))
		else:
			for i in range(int(math.log(scaleFactorMax,2))):
				advectSemiLagrange(flags=flags[i], vel=vel[i], grid=vel[i], order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth)
				setWallBcs(flags=flags[i], vel=vel[i])
				addBuoyancy(density=density[i], vel=vel[i], gravity=xl_buoys[i] , flags=flags[i])
				if 1:
					for j in range(len(inivel_sourcesSm[0])):
						inivel_sourcesSm[i][j].applyToGrid( grid=vel[i] , value=inivel_velsSm[i][j] )
				if 0 and ( t< timeOffset-10 ): 
					vorticityConfinement( vel=vel[i], flags=flags[i], strength=0.05/scaleFactorMax * 2**i )
				solvePressure(flags=flags[i], vel=vel[i], pressure=tmp[i],  preconditioner=PcMGStatic )
				setWallBcs(flags=flags[i], vel=vel[i])

		#for i in range(int(math.log(scaleFactorMax,2))):
			#velRecenters[i].copyFrom(vel[i])
			#velRecenters[i].addConst( velOffsets[i] )

		#KEpsilonBcs(flags=flags,k=k,eps=eps,intensity=intensity, nu = nu,fillArea=False)
		#advectSemiLagrange(flags=flags, vel=vel, grid=k, order=1)
		#advectSemiLagrange(flags=flags, vel=vel, grid=eps, order=1)
		#KEpsilonBcs(flags=flags,k=k,eps=eps,intensity=intensity, nu = nu,fillArea=False)
		#KEpsilonComputeProduction(vel=vel, k=k, eps=eps, prod=prod, nuT=nuT, strain=strain, pscale=prodMult) 
		#KEpsilonSources(k=k, eps=eps, prod=prod)
		#KEpsilonGradientDiffusion(k=k, eps=eps, vel=vel, nuT=nuT, sigmaU=10.0);
		# low res fluid, density
		if( t % resetN == 0) :
			for i in range(int(math.log(scaleFactorMax,2))-1,-1,-1):
				xl_tmp.copyFrom( xl_density )
				blurRealGrid( xl_density, xl_tmp, blurSigs[i])
				interpolateGrid( target=density[i], source=xl_tmp )
				#flags[].applyToGrid(density[i], TypeObstacle, value = 0.0)
		else:
			for i in range(int(math.log(scaleFactorMax,2))):
				advectSemiLagrange(flags=flags[i], vel=vel[i], grid=density[i], order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth)

	# --------------------------------------------------------------------#
	elif simMode==2:
		# low res fluid, density
		if( t % resetN == 0) :
			xl_tmp.copyFrom( xl_density )
			blurRealGrid( xl_density, xl_tmp, blurSig)
			interpolateGrid( target=density, source=xl_tmp )
		
		advectSemiLagrange(flags=flags, vel=velRecenter, grid=density, order=2, clampMode=2)    
		if t<=1: velRecenter.copyFrom(vel); # special , center only density once, leave vel untouched 
		advectSemiLagrange(flags=flags, vel=velRecenter, grid=vel,     order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth )
		
		if 1 and len(inflowSrc)>0:
			# note - the density inflows currently move with the offsets!
			for nI in inflowSrc:
				densityInflow( flags=xl_flags, density=xl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
				densityInflow( flags=flags, density=density, noise=noiSm[nI], shape=sourSm[nI], scale=1.0, sigma=1.0 )
		
		setWallBcs(flags=flags, vel=vel)    
		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)
		if 1:
			for i in range(len(inivel_sourcesSm)):
				inivel_sourcesSm[i].applyToGrid( grid=vel , value=inivel_velsSm[i] )

		vorticityConfinement( vel=vel, flags=flags, strength=0.1 ) 
		
		solvePressure(flags=flags, vel=vel, pressure=tmp , cgMaxIterFac=5.0, cgAccuracy=0.001, preconditioner=PcMGDynamic )
		setWallBcs(flags=flags, vel=vel)
		
		computeEnergy(flags=flags, vel=vel, energy=tmp)
		computeWaveletCoeffs(tmp)
		
		# xl solver, update up-res'ed grids ...

		# new centre of mass , from XL density
		velOffset , xl_velOffset = calcCOM(xl_density)
		xl_velOffset = velOffset  # note - hires advection does "scaleFac" substeps below! -> same offset

		if 1 and len(inflowSrc)>0:
			velOffset *= 0.5;  xl_velOffset *= 0.5;  # re-centering reduced

		# high res sim
		
		interpolateGrid( target=xl_tmp, source=tmp )
		interpolateMACGrid( source=vel, target=xl_vel )
		
		applyNoiseVec3( flags=xl_flags, target=xl_vel, noise=wltnoise, scale=wltStrength*1.0 , weight=xl_tmp)
		# manually weight and apply further octaves
		applyNoiseVec3( flags=xl_flags, target=xl_vel, noise=wltnoise2, scale=wltStrength*0.8 , weight=xl_tmp)
		applyNoiseVec3( flags=xl_flags, target=xl_vel, noise=wltnoise3, scale=wltStrength*0.8*0.8 , weight=xl_tmp)
		
		xl_velRecenter.copyFrom( xl_vel )
		xl_velRecenter.addConst( xl_velOffset )
		if( dim == 2 ):
			xl_velRecenter.multConst( vec3(1.0,1.0,0.0) )

		for substep in range(scaleFactor): 
			advectSemiLagrange(flags=xl_flags, vel=xl_velRecenter, grid=xl_density, order=2, clampMode=2)    

		velRecenter.copyFrom(vel)
		velRecenter.addConst( velOffset )
		if( dim == 2 ):
			velRecenter.multConst( vec3(1.0,1.0,0.0) )
	else:
		print("Unknown sim mode!"); exit(1)

	if doPrinttime:
		endtime = time.time()
		print("endtime: %2f" % endtime)
		print("runtime: %2f" % (endtime-starttime))

	# --------------------------------------------------------------------#

	# save low and high res
	# save all frames
	if t>=timeOffset*saveEveryK and t%saveEveryK == 0:
		tf = (t/saveEveryK-timeOffset) 
		if savenpz:
			print("Writing NPZs for frame %d"%tf)
			copyGridToArrayReal( target=sm_arR, source=density )
			np.savez_compressed( simPath + 'density_low_%04d.npz' % (tf), sm_arR )
			copyGridToArrayVec3( target=sm_arV, source=vel )
			np.savez_compressed( simPath + 'velocity_low_%04d.npz' % (tf), sm_arV )
			copyGridToArrayReal( target=xl_arR, source=xl_density )
			np.savez_compressed( simPath + 'density_high_%04d.npz' % (tf), xl_arR )
			copyGridToArrayVec3( target=xl_arV, source=xl_vel )
			np.savez_compressed( simPath + 'velocity_high_%04d.npz' % (tf), xl_arV )
		if saveuni:
			print("Writing UNIs for frame %d"%tf)
			xl_density.save(simPath + 'density_high_%04d.uni' % (tf))
			xl_flags.save(simPath + 'flags_high_%04d.uni' % (tf))

			for i in range(int(math.log(scaleFactorMax,2))):
				if i == 0:
					density[i].save(simPath + 'density_low_%04d.uni'% (tf))
					vel[i].save(simPath + 'velocity_low_%04d.uni'% (tf) )
					flags[i].save(simPath + 'flags_low_%04d.uni' % (tf))
					phiObs.save(simPath + 'levelset_low_%04d.uni' % (tf))
				else:
					density[i].save(simPath + 'density_low_%i_%04d.uni' % (2**i,tf))
					#vel[i].save(    simPath + 'velocity_low_%i_%04d.uni' % (2**i,tf))
					#flags[i].save(simPath + 'flags_low_%i_%04d.uni' % (2**i,tf))
			
		if(saveppm):
			print("Writing ppms for frame %d"%tf)
			projectPpmFull( xl_density, simPath + 'density_high_%04d.ppm' % (tf), 0, 5.0 )
			projectPpmFull( density, simPath + 'density_low_%04d.ppm' % (tf), 0, 5.0 )
	for i in range(len(sms)):
		sms[i].step()
	xl.step()
	#gui.screenshot( 'out_%04d.jpg' % t ) 
	timings.display() 
	t = t+1


