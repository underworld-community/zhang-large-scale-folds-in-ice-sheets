# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# import modules, create output directory
import numpy as np
import math
import os

#from mpi4py import MPI

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()

import underworld as uw

from underworld import function as fn
#import underworld.visualisation as vis

import pickle

if not os.getcwd().rsplit("/")[-1] == "output":
    outputPath = os.path.join(os.path.abspath("."),"output/")
    if not os.path.exists ( outputPath ):
        os.makedirs ( outputPath )

    os.chdir(outputPath)

# +
## basic parameters

g = 9.81
#ice_density = 910.

#A = 1e-16
n = 4.

MODEL_DATA = {}
coord = fn.input()
bed = 1000.0
iceHeight = 2500.0+bed
WarmHeight = (iceHeight-bed) * 40./100.+bed
airHeight = 500.0
zmin = 0.
zmax = 20000.
MODEL_DATA['MIN_Y'] = 0.0
MODEL_DATA['MAX_Y'] = 3000.0+bed
MODEL_DATA['MAX_X'] = 500.
MODEL_DATA['MIN_X'] = 0.0
MODEL_DATA['MIN_Z'] = 0.0
MODEL_DATA['MAX_Z'] = 25000.
MODEL_DATA['RES_X'] = 4
MODEL_DATA['RES_Y'] = 32
MODEL_DATA['RES_Z'] = 512 

mi = 0.2

#elementType = "Q1/dQ0"
#elementType = "Q2/dQ1"
#elementType = "Q1/dPc1"
#elementType = "Q2/dPc1"
MODEL_DATA['ELEMENT_TYPE'] = "Q1/dQ0"
MODEL_DATA['PERIODIC_X'] = True
MODEL_DATA['PERIODIC_Y'] = False
MODEL_DATA['PERIODIC_Z'] = False
MODEL_DATA['PARTICLES_PER_CELL'] = 30

mesh = uw.mesh.FeMesh_Cartesian( elementType = ( MODEL_DATA['ELEMENT_TYPE'] ) ,
                                 elementRes  = ( MODEL_DATA['RES_X'], MODEL_DATA['RES_Y'], MODEL_DATA['RES_Z']),
                                 minCoord    = ( MODEL_DATA['MIN_X'], MODEL_DATA['MIN_Y'], MODEL_DATA['MIN_Z'] ),
                                 maxCoord    = ( MODEL_DATA['MAX_X'], MODEL_DATA['MAX_Y'], MODEL_DATA['MAX_Z'] ),
                                 periodic    = ( MODEL_DATA['PERIODIC_X'], MODEL_DATA['PERIODIC_Y'], MODEL_DATA['PERIODIC_Z'] )
                               )

velocityField    = mesh.add_variable(         dataType="double",  nodeDofCount=3 )
pressureField    = mesh.subMesh.add_variable( dataType="double",  nodeDofCount=1 )
directorField    = mesh.add_variable( dataType="double",  nodeDofCount=3 )

velocityField.data[:] = [0.,0.,0.]
directorField.data[:] = [0.,1.,0.]
pressureField.data[:] = 0.

## visualisation parameters
MODEL_DATA_FILE = outputPath + "model_data.p"
pickle.dump( MODEL_DATA, open(MODEL_DATA_FILE , "wb" ) )

# +
# Create a swarm which will define our material geometries, and will also
# track deformation and history dependence of particles.
swarm  = uw.swarm.Swarm( mesh=mesh, particleEscape=True)

swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=MODEL_DATA['PARTICLES_PER_CELL'] )
swarm.populate_using_layout( layout=swarmLayout )

# create pop control object
pop_control1 = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=MODEL_DATA['PARTICLES_PER_CELL'])

# create advector
advector1 = uw.systems.SwarmAdvector(swarm=swarm,velocityField=velocityField, order=2)

# +
# Initialise particle properties 
materialVariable = swarm.add_variable( dataType="int", count=1 )
particleDensity = swarm.add_variable ( dataType="double", count=1 )
particleInitialYPos = swarm.add_variable ( dataType="double", count=1 )

particleStrainrate = swarm.add_variable ( dataType="double", count=1 )
particleViscosity = swarm.add_variable ( dataType="double", count=1 )
particleViscosity2 = swarm.add_variable ( dataType="double", count=1 )
particleShearstress = swarm.add_variable ( dataType="double", count=1 )

particleDirector = swarm.add_variable ( dataType="double", count=3 )
# particleMeshDirector below only used to save the director if calculated as a mesh variable
#particleMeshDirector = swarm.add_variable ( dataType="double", count=3 ) 

#particleWeakzone = swarm.add_variable ( dataType="int", count=1 )

particleTemperature = swarm.add_variable ( dataType="double", count=1 ) ###
particleTemperature.data[:] = 0.

#particleSnowHeight = swarm.add_variable ( dataType="double", count=1 )
#particleSnowHeight.data[:] = 0.

particleVelocity = swarm.add_variable( dataType="double", count=3 )
particleVelocity.data[:] = (0.,0.,0.)

#iceSurf = swarm.add_variable ( dataType="int", count=1 )
#iceSurf.data[:] = 0

particleCreationTime = swarm.add_variable ( dataType="float", count=1 )
particleCreationTime.data[:] = 0.

bumpHeight = swarm.add_variable( dataType="double", count=1 )
bumpHeight.data[:]=bed
WarmiceHeight = swarm.add_variable( dataType="double", count=1 )
snowPlane = swarm.add_variable ( dataType="double", count=1 )
snowPlane.data[:]=iceHeight
# -

# ### Definition of materials

# +
materialV = 1  # viscoplastic ice
materialVC = 2 # viscoplastic ice in the channel
materialA = 0 # Air
materialR = 3   # rock

coord = fn.input()
z=swarm.data[:, 2]
x=swarm.data[:, 0]
y=swarm.data[:, 1]

particleInitialYPos.data[:] = np.expand_dims(coord.evaluate(swarm)[:,1], axis=1)

amplitude = 300.0
WaveWidth = 5000. #4500/100, 14000/200
omega = 2.0 * np.pi / WaveWidth
Warmfunc = WarmHeight + amplitude/2. * np.sin(omega *z) # * fn.math.sin(omega * coord[2])
WarmiceHeight.data[:] = np.expand_dims(Warmfunc, 1)

conditions = [ 
               (       coord[1] > iceHeight,    materialA    ),
               (       coord[1] > WarmiceHeight,    materialV    ),
               (       coord[1] > bed,    materialVC    ),
               (       True ,           materialR                   ), 
             ]

materialVariable.data[:] = fn.branching.conditional( conditions ).evaluate(swarm)

# +
for index in np.ndindex(directorField.data.shape[0]):
    maxAngle = 5./90.*np.pi/2.
    gaus = np.random.normal(0., maxAngle) #Gaussian distribution
    Warmice_function = WarmHeight + amplitude/2. * np.sin(omega * mesh.data[index][2])
    derivative = omega *amplitude/2. * np.cos(omega * mesh.data[index][2])
    tangent_angle = np.arctan(derivative) #angle of the tangent line
    #print(tangent_angle)
    TopAmplitude = 0.
    tangent_angle_top = 0.

    if iceHeight>=mesh.data[index][1]>Warmice_function:
        tangent_angle_func = tangent_angle + (mesh.data[index][1]-Warmice_function)/(iceHeight-Warmice_function) * (tangent_angle_top-tangent_angle)
        directorField.data[index][1] = np.cos(gaus-tangent_angle_func)
        directorField.data[index][2] = np.sin(gaus-tangent_angle_func)
    elif Warmice_function>=mesh.data[index][1]>bed:
        tangent_angle_func= (mesh.data[index][1]-bed)/(Warmice_function-bed) * tangent_angle
        directorField.data[index][1] = np.cos(gaus-tangent_angle_func)
        directorField.data[index][2] = np.sin(gaus-tangent_angle_func)
    else:
        directorField.data[index][1] =np.cos(gaus)
        directorField.data[index][2] =np.sin(gaus)

particleDirector.data[:]  = directorField.evaluate(swarm)

# +
## swarms to track the deformation
surfaceSwarm1 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm2 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm3 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm4 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm5 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm6 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm7 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm8 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm9 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
surfaceSwarm10 = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
#airiceSwarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)

# create advector
advector2 = uw.systems.SwarmAdvector(swarm=surfaceSwarm1,velocityField=velocityField, order=2)
advector3 = uw.systems.SwarmAdvector(swarm=surfaceSwarm2,velocityField=velocityField, order=2)
advector4 = uw.systems.SwarmAdvector(swarm=surfaceSwarm3,velocityField=velocityField, order=2)
advector5 = uw.systems.SwarmAdvector(swarm=surfaceSwarm4,velocityField=velocityField, order=2)
advector6 = uw.systems.SwarmAdvector(swarm=surfaceSwarm5,velocityField=velocityField, order=2)
advector7 = uw.systems.SwarmAdvector(swarm=surfaceSwarm6,velocityField=velocityField, order=2)
advector8 = uw.systems.SwarmAdvector(swarm=surfaceSwarm7,velocityField=velocityField, order=2)
advector9 = uw.systems.SwarmAdvector(swarm=surfaceSwarm8,velocityField=velocityField, order=2)
advector10 = uw.systems.SwarmAdvector(swarm=surfaceSwarm9,velocityField=velocityField, order=2)
advector11 = uw.systems.SwarmAdvector(swarm=surfaceSwarm10,velocityField=velocityField, order=2)
#airiceadvector = uw.systems.SwarmAdvector(swarm=airiceSwarm,velocityField=velocityField, order=2)

#surfacePoints1
surfacePoints = np.array(np.meshgrid(np.linspace(0, MODEL_DATA['MAX_X'], 50), MODEL_DATA['MAX_Y'], np.linspace(0., MODEL_DATA['MAX_Z'], 500))).T.reshape(-1, 3)

x = surfacePoints[:, 0]
z = surfacePoints[:, 2]
elevation = WarmHeight + amplitude/2. * np.sin(omega * z)

surfacePoints[:, 1] = iceHeight
surfaceSwarm1.add_particles_with_coordinates(surfacePoints)
#airiceSwarm.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (9-4)/6*(iceHeight-elevation)
surfaceSwarm2.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (8-4)/6*(iceHeight-elevation)
surfaceSwarm3.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (7-4)/6*(iceHeight-elevation)
surfaceSwarm4.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (6-4)/6*(iceHeight-elevation)
surfaceSwarm5.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (5-4)/6*(iceHeight-elevation)
surfaceSwarm6.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation
surfaceSwarm7.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = bed + 3/4 * (elevation-bed)  
surfaceSwarm8.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = bed + 2/4 * (elevation-bed)  
surfaceSwarm9.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = bed + 1/4 * (elevation-bed)  
surfaceSwarm10.add_particles_with_coordinates(surfacePoints)

#surfacePoints2
surfacePoints = np.array(np.meshgrid(np.linspace(0, MODEL_DATA['MAX_X'], 500), MODEL_DATA['MAX_Y'], np.linspace(0., MODEL_DATA['MAX_Z'], 50))).T.reshape(-1, 3)
x = surfacePoints[:, 0]
z = surfacePoints[:, 2]
elevation = WarmHeight + amplitude/2. * np.sin(omega * z)

surfacePoints[:, 1] = iceHeight
surfaceSwarm1.add_particles_with_coordinates(surfacePoints)
#airiceSwarm.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (9-4)/6*(iceHeight-elevation)
surfaceSwarm2.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (8-4)/6*(iceHeight-elevation)
surfaceSwarm3.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (7-4)/6*(iceHeight-elevation)
surfaceSwarm4.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (6-4)/6*(iceHeight-elevation)
surfaceSwarm5.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation + (5-4)/6*(iceHeight-elevation)
surfaceSwarm6.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = elevation
surfaceSwarm7.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = bed + 3/4 * (elevation-bed)  
surfaceSwarm8.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = bed + 2/4 * (elevation-bed)  
surfaceSwarm9.add_particles_with_coordinates(surfacePoints)
surfacePoints[:, 1] = bed + 1/4 * (elevation-bed)  
surfaceSwarm10.add_particles_with_coordinates(surfacePoints)
    
# to visualize the surface swarm in paraview we need a pseudo variable
surfaceParticle1 = surfaceSwarm1.add_variable ( dataType="int", count=1 )
surfaceParticle1.data[:] = 1
surfaceParticle2 = surfaceSwarm2.add_variable ( dataType="int", count=1 )
surfaceParticle2.data[:] = 2
surfaceParticle3 = surfaceSwarm3.add_variable ( dataType="int", count=1 )
surfaceParticle3.data[:] = 3
surfaceParticle4 = surfaceSwarm4.add_variable ( dataType="int", count=1 )
surfaceParticle4.data[:] = 4
surfaceParticle5 = surfaceSwarm5.add_variable ( dataType="int", count=1 )
surfaceParticle5.data[:] = 5
surfaceParticle6 = surfaceSwarm6.add_variable ( dataType="int", count=1 )
surfaceParticle6.data[:] = 6
surfaceParticle7 = surfaceSwarm7.add_variable ( dataType="int", count=1 )
surfaceParticle7.data[:] = 7
surfaceParticle8 = surfaceSwarm8.add_variable ( dataType="int", count=1 )
surfaceParticle8.data[:] = 8
surfaceParticle9 = surfaceSwarm9.add_variable ( dataType="int", count=1 )
surfaceParticle9.data[:] = 9
surfaceParticle10 = surfaceSwarm10.add_variable ( dataType="int", count=1 )
surfaceParticle10.data[:] = 10
#airiceParticle = airiceSwarm.add_variable ( dataType="int", count=1 )
#airiceParticle.data[:] = 11

# +
T0 = -30. # °C
#Tbed = -5. # °C
Tbed = -3. # °C

T_func = Tbed + (coord[1]-bed)/(WarmiceHeight-bed)*(T0-Tbed)

Tconditions = [ 
                (       coord[1] > WarmiceHeight,  T0      ),
                (       coord[1] > bed,  T_func      ),
                (       True,            Tbed  ),
              ]

particleTemperature.data[:] = fn.branching.conditional( Tconditions ).evaluate(swarm)

# +
## functions, incl flow law
R = 0.008314 # kJ / (T*mol)
QhighT = 155. # kJ/mol, activation energy, Kuiper dislocation creep
QsmallT = 64.
A0smallT = 5e5 * 3.1536e7 #Mpa-4s-1 to Pa-4a-1
A0highT = 6.96e23 * 3.1536e7
#if rank == 0:
#    print(A0smallT, A0highT) #15768000000000.0 2.1949056e+31

strainRateTensor = fn.tensor.symmetric(velocityField.fn_gradient)
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateTensor)

viscosityFnAir    = fn.misc.constant(1e9 / 3.1536e7)
minViscosityIceFn  = fn.misc.constant(1e+11 / 3.1536e7)
maxViscosityIceFn  = fn.misc.constant(1e+17 / 3.1536e7)
viscosityFnRock = fn.misc.constant(1e19 / 3.1536e7)

V1 = 0.5 * (A0smallT * fn.math.exp(-QsmallT / (R*(particleTemperature + 273.)))) ** (-1./n) * (strainRate_2ndInvariantFn**((1.-n) / float(n)))
V2 = 0.5 * (A0highT * fn.math.exp(-QhighT / (R*(particleTemperature + 273.)))) ** (-1./n) * (strainRate_2ndInvariantFn**((1.-n) / float(n)))
V1 = V1 * 1e6 #pa*a
V2 = V2 * 1e6 #pa*a
#V1 = V1 * 1e6 / 3.1536e7  #pa*a
#V2 = V2 * 1e6 / 3.1536e7

VisBaseconditions = [
                (       particleTemperature <= -11., V1 ), 
                (       True,                        V2),               
                ]

#viscosityMap1 = {
#                materialV:  viscosityFnIce ,
#                materialVC: viscosityFnIce,
#               }

#viscosityMap2 = { 
#                 materialV:  viscosityFnColdIce,
#                materialVC: viscosityFnWarmIce,
#               }

viscosityIceFn = fn.branching.conditional( VisBaseconditions )
viscosityFnIce = fn.misc.max(fn.misc.min(viscosityIceFn, maxViscosityIceFn), minViscosityIceFn)

viscosityMap = {
                materialA:  viscosityFnAir,
                materialV:  viscosityFnIce,
                materialVC: viscosityFnIce,
                materialR: viscosityFnRock,
               }
viscosityFn = fn.branching.map( fn_key=materialVariable, mapping=viscosityMap )

#viscosityFnAir2    = 0.0
viscosityFnIce2    = (1-1/3) * viscosityFnIce
viscosityMap2 = {
                materialA:  0.,
                materialV:  viscosityFnIce2,
                materialVC: viscosityFnIce2,
                materialR: 0.,
               }
viscosityFn2 = fn.branching.map( fn_key=materialVariable, mapping=viscosityMap2 )
viscosityFn3=viscosityFn-viscosityFn2
particleViscosity.data[:] = viscosityFn.evaluate(swarm)

# +
#devStressFn = 2.0 * viscosityFn * strainRateTensor
shearStressFn = strainRate_2ndInvariantFn * viscosityFn * 2.0

densityFnIce = (18.02 / (19.30447 - 7.988471e-4 * (particleTemperature+273.) + 7.563261e-6 * ((particleTemperature+273.)**2) )) * 1000.
#917.5085940523277,921.4114212256626
densityFnAir = fn.misc.constant( 0. )
densityFnRock = fn.misc.constant( 2700. )

densityMap = {
                materialA:  densityFnAir,
                materialV:  densityFnIce,
                materialVC:  densityFnIce,
                materialR: densityFnRock
             }

densityFn = fn.branching.map(fn_key=materialVariable, mapping=densityMap)

particleDensity.data[:] = densityFn.evaluate(swarm)

#surf_inclination = 0.5 * np.pi / 180. # 0.1 = Experiment D, 0.5 = Experiment B
#z_hat = (math.sin(surf_inclination), - math.cos(surf_inclination), 0.)

z_hat = (0., -1., 0.)
buoyancyFn = densityFn * z_hat * g

# +
## set boundary conditions
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
kWalls = mesh.specialSets["MinK_VertexSet"] + mesh.specialSets["MaxK_VertexSet"]
#outerkWall = mesh.specialSets["MinK_VertexSet"]
front = mesh.specialSets["MinI_VertexSet"]
back = mesh.specialSets["MaxI_VertexSet"]
base   = mesh.specialSets["MinJ_VertexSet"]
top    = mesh.specialSets["MaxJ_VertexSet"]
leftWall = mesh.specialSets["MinK_VertexSet"]
rightWall = mesh.specialSets["MaxK_VertexSet"]

allWalls = iWalls + jWalls + kWalls

meshVz = 5.
#meshVx = 2*meshVz*maxX*maxY/(2*maxY*maxZ-BumpAmplitude*(1/k*np.sin(k*TotalBumpWidth)+TotalBumpWidth))
#Sbump = BumpAmplitude*(zmax-zmin)*(BA1+BA2+BA3+BA4)/(2*Amax*(1.0 + 0.5 + 0.25 + 0.125))
#meshVx = meshVz*MODEL_DATA['MAX_X']*(MODEL_DATA['MAX_Y']-bed)/((MODEL_DATA['MAX_Y']-bed)*MODEL_DATA['MAX_Z']-Sbump)
meshVx = meshVz*MODEL_DATA['MAX_X']*(MODEL_DATA['MAX_Y']-bed)/((MODEL_DATA['MAX_Y']-bed)*MODEL_DATA['MAX_Z'])
#print(meshVx) 0.1
velocityField.data[:] = 0.
#velocityField.data[leftWall, 2] = meshVz
#velocityField.data[rightWall, 2] = -meshVz

loadA = mesh.specialSets['Empty']
for index in mesh.specialSets["MinI_VertexSet"]:
    if mesh.data[index][1]<=bed:
        loadA+=index

loadB = mesh.specialSets['Empty']
for index in mesh.specialSets["MaxI_VertexSet"]:
    if mesh.data[index][1]<=bed:
        loadB+=index
        
loadC = mesh.specialSets['Empty']
for index in mesh.specialSets["MinK_VertexSet"]:
    if mesh.data[index][1]<=bed:
        loadC+=index

loadD = mesh.specialSets['Empty']
for index in mesh.specialSets["MaxK_VertexSet"]:
    if mesh.data[index][1]<=bed:
        loadD+=index
        
for i in mesh.specialSets["MaxK_VertexSet"]:
    loc = mesh.data[i,1]
    if loc > bed:
        velocityField.data[i][2] = -meshVz

for i in mesh.specialSets["MaxI_VertexSet"]:
    loc = mesh.data[i,1]
    if loc>bed:
        velocityField.data[i][0] = meshVx

velocityBCs = uw.conditions.DirichletCondition(
                                                variable        = velocityField, 
                                                indexSetsPerDof = (iWalls+base+loadC+loadD, base, kWalls+base+loadA+loadB),
                                              )
# -

def c_axis_rotation(dt, steps = 1.):

    dt /= steps
    
    for i in range(0, int(steps)):
        
        #iceIndices = np.array(np.where(materialVariable.data == materialV + materialVC)[0])
        iceIndices = np.array(np.where(np.logical_or(materialVariable.data == materialV, 
                                                     materialVariable.data == materialVC))[0])
        #iceIndices = np.array(np.where(materialVariable.data == materialVC)[0])
        
        velGrad = velocityField.fn_gradient.evaluate(swarm).reshape(swarm.particleLocalCount, mesh.dim, mesh.dim)
        velGrad = velGrad[iceIndices]
        velGradT = velGrad.swapaxes(-1,1)

        # rate of deformation and rate of rotation
        D = 0.5 * (velGrad + velGradT)
        W = 0.5 * (velGrad - velGradT)

        particleDirector.data[iceIndices] = particleDirector.data[iceIndices] + dt * ( np.einsum("ijk,ik->ij", W, particleDirector.data[iceIndices]) - np.einsum("ijk,ik->ij", D, particleDirector.data[iceIndices]) + np.einsum("ij,ij->i",particleDirector.data[iceIndices], np.einsum("ijk,ik->ij",D,particleDirector.data[iceIndices]))[:,None] * particleDirector.data[iceIndices])

        #finally normalize the c-axes
        particleDirector.data[iceIndices] = particleDirector.data[iceIndices] / np.absolute(np.linalg.norm(particleDirector.data[iceIndices], axis=1).reshape(len(iceIndices),1))

        # we want to rotate all directors, if they point towards the negative y-direction
        # this should make it easier to display them
        b = np.where(particleDirector.data[:,1] < 0.)
        particleDirector.data[b] *= -1.

# +
## setup solver and solve

stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    voronoi_swarm=swarm,
    conditions=[
            velocityBCs,
            ],
    fn_viscosity=viscosityFn,
    _fn_viscosity2=viscosityFn2,
    #_fn_director=directorField,
    _fn_director=particleDirector,
    fn_bodyforce=buoyancyFn,
)

solver = uw.systems.Solver(stokes)

solver.set_inner_method("mg")
solver.options.scr.ksp_type="cg"
solver.set_penalty(1.0e6) # higher penalty = larger stability + (often) faster calculation
# solver.options.scr.ksp_rtol = 1.0e-3

surfaceArea = uw.utils.Integral( fn=1.0, mesh=mesh, integrationType='surface', surfaceIndexSet=top)
surfacePressureIntegral = uw.utils.Integral( fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=top)

def calibrate_pressure():

    global pressureField
    global surfaceArea
    global surfacePressureIntegral

    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate() 
    pressureField.data[:] -= p0 / area

    # print (f'Calibration pressure {p0 / area}')
# -

def flow(rotate_caxes = True):
    
    global calibrate_pressure
    global advector1
    global advector2
    global advector3
    global advector4
    global advector5
    global advector6
    global advector7
    global advector8
    global advector9
    global advector10
    global advector11
    #global airiceadvector
    global pop_control1
        
    #solver.solve(nonLinearIterate=True, nonLinearTolerance=nl_tol, callback_post_solve=calibrate_pressure)
    solver.solve(nonLinearIterate=True, nonLinearMaxIterations = 50, callback_post_solve=calibrate_pressure)

    # Retrieve the maximum possible timestep for the advection system.
    t1=5.
    dt = advector1.get_max_dt()
    if dt>t1:
        dt=t1

    # Advect using this timestep size.
    advector1.integrate(dt, update_owners=True) # the swarm
    advector2.integrate(dt, update_owners=True) # the surface swarm
    advector3.integrate(dt, update_owners=True)
    advector4.integrate(dt, update_owners=True)
    advector5.integrate(dt, update_owners=True)
    advector6.integrate(dt, update_owners=True)
    advector7.integrate(dt, update_owners=True)
    advector8.integrate(dt, update_owners=True)
    advector9.integrate(dt, update_owners=True)
    advector10.integrate(dt, update_owners=True)
    advector11.integrate(dt, update_owners=True)
    #airiceadvector.integrate(dt, update_owners=True)

    #meshVx(dt)
    #meshVx_deform(dt)
    # particle population control
    #swarm.update_particle_owners()
    pop_control1.repopulate()
    
    if rotate_caxes:
        
        c_axis_rotation(dt, steps = 100.)
        #c_axis_rotation_mesh(dt, steps = 100.)

    return (dt)


# +
maxSteps = 10000
stepsize = 20.   

step = 0
t = 0.

xdmf_mesh    = mesh.save('mesh.h5')

while step < maxSteps:

    if not step%stepsize: # if multiple of ..
        
        #print ("in step " + str(step))
        
        ignore = swarm.save('swarm_' + str(step) + '.h5')
        
        # eval swarm variables
        particleStrainrate.data[:] = strainRate_2ndInvariantFn.evaluate(swarm)
        particleViscosity.data[:] = viscosityFn.evaluate(swarm)
        particleViscosity2.data[:] = viscosityFn3.evaluate(swarm)
        particleShearstress.data[:] = shearStressFn.evaluate(swarm)
        particleVelocity.data[:]  = velocityField.evaluate(swarm)
        
        # save swarm variables as xdmf files
        xdmf_swarm = swarm.save('swarm_' + str(step) + '.h5')
        xdmf_surfswarm1 = surfaceSwarm1.save('surf_swarm1_' + str(step) + '.h5')
        xdmf_surfswarm2 = surfaceSwarm2.save('surf_swarm2_' + str(step) + '.h5')
        xdmf_surfswarm3 = surfaceSwarm3.save('surf_swarm3_' + str(step) + '.h5')
        xdmf_surfswarm4 = surfaceSwarm4.save('surf_swarm4_' + str(step) + '.h5')
        xdmf_surfswarm5 = surfaceSwarm5.save('surf_swarm5_' + str(step) + '.h5')
        xdmf_surfswarm6 = surfaceSwarm6.save('surf_swarm6_' + str(step) + '.h5')
        xdmf_surfswarm7 = surfaceSwarm7.save('surf_swarm7_' + str(step) + '.h5')
        xdmf_surfswarm8 = surfaceSwarm8.save('surf_swarm8_' + str(step) + '.h5')
        xdmf_surfswarm9 = surfaceSwarm9.save('surf_swarm9_' + str(step) + '.h5')
        xdmf_surfswarm10 = surfaceSwarm10.save('surf_swarm10_' + str(step) + '.h5')
        
        xdmf_surfaceParticle1 = surfaceParticle1.save('surfaceSwarm1_' + str(step) + '.h5')
        xdmf_surfaceParticle2 = surfaceParticle2.save('surfaceSwarm2_' + str(step) + '.h5')
        xdmf_surfaceParticle3 = surfaceParticle3.save('surfaceSwarm3_' + str(step) + '.h5')
        xdmf_surfaceParticle4 = surfaceParticle4.save('surfaceSwarm4_' + str(step) + '.h5')
        xdmf_surfaceParticle5 = surfaceParticle5.save('surfaceSwarm5_' + str(step) + '.h5')
        xdmf_surfaceParticle6 = surfaceParticle6.save('surfaceSwarm6_' + str(step) + '.h5')
        xdmf_surfaceParticle7 = surfaceParticle7.save('surfaceSwarm7_' + str(step) + '.h5')
        xdmf_surfaceParticle8 = surfaceParticle8.save('surfaceSwarm8_' + str(step) + '.h5')
        xdmf_surfaceParticle9 = surfaceParticle9.save('surfaceSwarm9_' + str(step) + '.h5')
        xdmf_surfaceParticle10 = surfaceParticle10.save('surfaceSwarm10_' + str(step) + '.h5')
        
        surfaceParticle1.xdmf('surfaceSwarm1_' + str(step) + '.xdmf', xdmf_surfaceParticle1,
                            "surfaceParticle1", xdmf_surfswarm1, "SurfSwarm1", modeltime=step)
        surfaceParticle2.xdmf('surfaceSwarm2_' + str(step) + '.xdmf', xdmf_surfaceParticle2,
                            "surfaceParticle2", xdmf_surfswarm2, "SurfSwarm2", modeltime=step)
        surfaceParticle3.xdmf('surfaceSwarm3_' + str(step) + '.xdmf', xdmf_surfaceParticle3,
                            "surfaceParticle3", xdmf_surfswarm3, "SurfSwarm3", modeltime=step)
        surfaceParticle4.xdmf('surfaceSwarm4_' + str(step) + '.xdmf', xdmf_surfaceParticle4,
                            "surfaceParticle4", xdmf_surfswarm4, "SurfSwarm4", modeltime=step)
        surfaceParticle5.xdmf('surfaceSwarm5_' + str(step) + '.xdmf', xdmf_surfaceParticle5,
                            "surfaceParticle5", xdmf_surfswarm5, "SurfSwarm5", modeltime=step)
        surfaceParticle6.xdmf('surfaceSwarm6_' + str(step) + '.xdmf', xdmf_surfaceParticle6,
                            "surfaceParticle6", xdmf_surfswarm6, "SurfSwarm6", modeltime=step)
        surfaceParticle7.xdmf('surfaceSwarm7_' + str(step) + '.xdmf', xdmf_surfaceParticle7,
                            "surfaceParticle7", xdmf_surfswarm7, "SurfSwarm7", modeltime=step)
        surfaceParticle8.xdmf('surfaceSwarm8_' + str(step) + '.xdmf', xdmf_surfaceParticle8,
                            "surfaceParticle8", xdmf_surfswarm8, "SurfSwarm8", modeltime=step)
        surfaceParticle9.xdmf('surfaceSwarm9_' + str(step) + '.xdmf', xdmf_surfaceParticle9,
                            "surfaceParticle9", xdmf_surfswarm9, "SurfSwarm9", modeltime=step)
        surfaceParticle10.xdmf('surfaceSwarm10_' + str(step) + '.xdmf', xdmf_surfaceParticle10,
                            "surfaceParticle10", xdmf_surfswarm10, "SurfSwarm10", modeltime=step)

        #xdmf_meshvar = velocityField.save('velocityField_' + str(step) + '.h5')
        #velocityField.xdmf('velocityField_' + str(step) + '.xdmf', xdmf_meshvar, "Velocity", 
        #                   xdmf_mesh, "Mesh", modeltime=step)

        xdmf_particleStrainrate = particleStrainrate.save('particleStrainrate_' + str(step) + '.h5')
        particleStrainrate.xdmf('particleStrainrate_' + str(step) + '.xdmf', xdmf_particleStrainrate, 
                                "particleStrainrate", xdmf_swarm, "Swarm", modeltime=step)

        xdmf_particleDirector = particleDirector.save('particleDirector_' + str(step) + '.h5')
        particleDirector.xdmf('particleDirector_' + str(step) + '.xdmf', xdmf_particleDirector, 
                              "particleDirector", xdmf_swarm, "Swarm", modeltime=step)

        #xdmf_particleMeshDirector = particleMeshDirector.save('particleMeshDirector_' + str(step) + '.h5')
        #particleMeshDirector.xdmf('particleMeshDirector_' + str(step) + '.xdmf', xdmf_particleMeshDirector, 
        #                      "particleMeshDirector", xdmf_swarm, "Swarm", modeltime=step)
        
        xdmf_particleViscosity = particleViscosity.save('particleViscosity_' + str(step) + '.h5')
        particleViscosity.xdmf('particleViscosity_' + str(step) + '.xdmf', xdmf_particleViscosity, 
                                "particleViscosity", xdmf_swarm, "Swarm", modeltime=step)
        
        #xdmf_particleViscosity2 = particleViscosity2.save('particleViscosity2_' + str(step) + '.h5')
        #particleViscosity2.xdmf('particleViscosity2_' + str(step) + '.xdmf', xdmf_particleViscosity2, 
        #                        "particleViscosity2", xdmf_swarm, "Swarm", modeltime=step)
        
        #xdmf_particleCreationTime = particleCreationTime.save('particleCreationTime_' + str(step) + '.h5')
        #particleCreationTime.xdmf('particleCreationTime_' + str(step) + '.xdmf', xdmf_particleCreationTime, 
        #                        "particleCreationTime", xdmf_swarm, "Swarm", modeltime=step)

        xdmf_materialVariable = materialVariable.save('particleMaterial_' + str(step) + '.h5')
        materialVariable.xdmf('particleMaterial_' + str(step) + '.xdmf', xdmf_materialVariable, 
                              "materialVariable", xdmf_swarm, "Swarm", modeltime=step)

        xdmf_particleTemperature = particleTemperature.save('particleTemperature_' + str(step) + '.h5')
        particleTemperature.xdmf('particleTemperature_' + str(step) + '.xdmf', xdmf_particleTemperature, 
                                "particleTemperature", xdmf_swarm, "Swarm", modeltime=step)
        
        xdmf_particleDensity = particleDensity.save('particleDensity_' + str(step) + '.h5')
        particleDensity.xdmf('particleDensity_' + str(step) + '.xdmf', xdmf_particleDensity, 
                                "particleDensity", xdmf_swarm, "Swarm", modeltime=step)
        
        xdmf_particleShearstress = particleShearstress.save('particleShearstress_' + str(step) + '_'+ str(t) + '.h5')
        particleShearstress.xdmf('particleShearstress_' + str(step) + '.xdmf', xdmf_particleShearstress, 
                                "particleShearstress", xdmf_swarm, "Swarm", modeltime=step)
        
        # visualizing the velocityField in paraviewe doesn't work for whatever reason (Paraview just crashes)
        # so we save it as a particle property
        #particleVelocity.data[:]  = velocityField.evaluate(swarm)
        xdmf_particleVelocity = particleVelocity.save('particleVelocity_' + str(step) + '.h5')
        particleVelocity.xdmf('particleVelocity_' + str(step) + '.xdmf', xdmf_particleVelocity, 
                      "particleVelocity", xdmf_swarm, "Swarm", modeltime=step)
        
        print (str(t) + ' years, step: ' + str(step))
    
    if step < maxSteps-1:
        t += flow(rotate_caxes = True)    
    
    step += 1
