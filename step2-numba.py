#######################################
#######################################
## 50x50Nodes.py
##
## Runge-Kutta simulation of 50 cells/ring x 50 rings with feedback
## Create image contour plots of full X matrix at given intervals
## Calculate sum through column and difference in sum across ring
##
##Store minimal amount of data in RAM
#######################################
#######################################


#######################################
# Library imports
#######################################


import time as mytime
import math
import numpy as np
from matplotlib import pyplot
from numba import jit, njit

np.random.seed(1234)
#######################################
# Constants
#######################################

VM2 = 20.0  # 10^-6 M/s
VM3 = 23.0  # 10^-6 M/s
K2 = 1.0  # 10^-6 M/s
KR = 0.8  # 10^-6 M/s
KA = 0.9  # 10^-6 M/s
kf = 1.0  # 1/s
n = 1
m = 1
p = 2
k = 0.8  # 1/s
v = 0.325  # v= v0 + v1*beta

tau = 16  # s

Dxl = 2.0  # 1/s
Dxr = 6.0  # 1/s
Dxd = 1.0  # 1/s
kt = 0.1  # 1/s

sampleFreq = 1.0  # s
duration = 5

#######################################
# Initial conditions
#######################################

ringsize = 50
rings = 200
# fileX = fromfile("X0.dat", sep=';')
# fileY = fromfile("Y0.dat", sep=';')
omega = 1.047
Am = 1.2

# X0 = np.zeros((ringsize,rings), double)
# Y0 = np.zeros((ringsize,rings), double)
# Amp = 0.55
# index=0
# for ring in range(0,3):
#    for cell in range(24,27):
#        X0[cell][ring] = Amp
#        Y0[cell][ring] = Amp 
#        index += 1          

# for ring in range(rings):
#    for cell in range(ringsize):
#        X0[cell][ring] = fileX[index]
#        Y0[cell][ring] = fileY[index] 
#        index += 1  

X0 = (np.random.rand(ringsize, rings) - 0.5) * 0.1 + 0.406
Y0 = (np.random.rand(ringsize, rings) - 0.5) * 0.2 + 2.76


#######################################
# Functions
#######################################

# Model
@njit(cache = True)
def f(X, Y):
    return VM2 * X / (K2 + X ) - VM3 * Y * np.square(X) / ((KR + Y) * (KA * KA + np.square(X) )) \
           - kf * Y

@njit(cache = True)
def dXdt(X, Y, flux):
    return v - f(X, Y) - k * X + flux

@njit(cache = True)
def dYdt(X, Y):
    return f(X, Y)


# Log data to file
def writeRing(filename, data, ring):
    f = open(filename + str(ring) + '.dat', 'w')
    for t in range(0, N):
        string = ""
        for cell in range(0, ringsize - 1):
            string += str(data[t][ring][cell]) + ';'
        string += str(data[t][ring][ringsize - 1]) + '\n'
        f.write(string)


def writeCylinder(filename, data, ring):
    f = open(filename + str(ring) + '.dat', 'w')
    for ring in range(0, rings):
        string = ""
        for cell in range(0, ringsize - 1):
            string += str(data[time % tau][ring][cell]) + ';'
        string += str(data[time % tau][ring][ringsize - 1]) + '\n'
        f.write(string)


    #######################################


# Perform simulation
#######################################


# taken from https://gist.github.com/synapticarbors/a22e1834d7cfc46eee2a26cebc6f817b
from numba import types
from numba.extending import overload_method


@overload_method(types.Array, 'take')
def array_take(arr, indices):
   if isinstance(indices, types.Array):
       def take_impl(arr, indices):
           n = indices.shape[0]
           res = np.empty(n, arr.dtype)
           for i in range(n):
               res[i] = arr[indices[i]]
           return res
       return take_impl

@njit(cache = True)
def roll(a, shift):
    
    n = a.size
    reshape = True
    
    if n == 0:
        return a
    shift %= n
    
    indexes = np.concatenate((np.arange(n - shift, n), np.arange(n - shift)))
    
    res = a.take(indexes)
    if reshape:
        res = res.reshape(a.shape)
    return res

@njit(cache = True)
def fluxCompensator( X, index, ring, t):
        xr = roll(X[index][ring], 1)
        xl = roll(X[index][ring], -1)
        flux = Dxr * (xr - X[index][ring]) + Dxl * (xl - X[index][ring])

        # vertical diffusion
        if ring == 0:
            d = np.greater(X[index][ring], X[index][ring + 1])
            flux[d] -= Dxd * (X[index][ring][d] - X[index][ring + 1][d])
        elif ring == rings - 1:
            d = np.greater(X[index][ring - 1], X[index][ring])
            flux[d] += Dxd * (X[index][ring - 1][d] - X[index][ring][d])
        else:
            d = np.greater(X[index][ring - 1], X[index][ring])
            flux[d] += Dxd * (X[index][ring - 1][d] - X[index][ring][d])
            d = np.greater(X[index][ring], X[index][ring + 1])
            flux[d] -= Dxd * (X[index][ring][d] - X[index][ring + 1][d])

        # Feedback across ring, tau elements back in time
        if t >= tau:
            xshift = roll(X[(t - tau) % tau][ring], ringsize // 2)
            flux += kt * (xshift - X[index][ring])
        return flux
    
@njit(cache = True)
def ringdynamics( X, Y, cylinderConcPerCell, index, t, tau, h, omega, Am):
    for ring in range(rings):

        # Sinus-Exitation
        if 0 <= ring <= 3:
            X[index][ring][24:27 + 1] = Am * 0.5 * (1 + math.cos(omega * t * h))
        #                   Y[index][ring][cell] = Amp*(math.cos(omega*t*h))**2
        flux = fluxCompensator( X, index, ring, t)
        
        # Perform Runge-Kutta integration on current cell
        x = X[index][ring]
        y = Y[index][ring]
        k11 = h * dXdt(x, y, flux)
        k12 = h * dYdt(x, y)

        k21 = h * dXdt(x + 0.5 * k11, y + 0.5 * k12, flux)
        k22 = h * dYdt(x + 0.5 * k11, y + 0.5 * k12)

        k31 = h * dXdt(x + 0.5 * k21, y + 0.5 * k22, flux)
        k32 = h * dYdt(x + 0.5 * k21, y + 0.5 * k22)

        k41 = h * dXdt(x + k31, y + k32, flux)
        k42 = h * dYdt(x + k31, y + k32)
        
        X[(index + 1) % tau][ring] = x + (k11 + 2 * k21 + 2 * k31 + k41) / 6
        Y[(index + 1) % tau][ring] = y + (k12 + 2 * k22 + 2 * k32 + k42) / 6

        # Calculate sum of X over all rings for each cell position
        cylinderConcPerCell[:, t] += X[index, ring]


# Set step size 
h = 0.02
N = int(duration / h) + 1

sampleFreq = sampleFreq / h
tau = int(tau / h)

cylinderConcPerCell = np.zeros((ringsize, N), float)
totalConcVector = np.zeros((2, N), float)
angle = 2 * math.pi / ringsize

# initialize X and Y matrices
# X[time][ring][cell]
X = np.zeros((tau, rings, ringsize), float)
Y = np.zeros((tau, rings, ringsize), float)
for ring in range(rings):
    X[0, ring] = X0[:, ring]
    Y[0, ring] = Y0[:, ring]

time = np.arange(0, h * N + h, h)

# Loop over all cells and time points
startTime = mytime.time()
for t in range(0, N):
    index = t % tau
    #X[(index + 1) % tau], Y[(index + 1) % tau] = ringdynamics( X, Y, cylinderConcPerCell, index, t)
    ringdynamics( X, Y, cylinderConcPerCell, index, t,  tau, h, omega, Am)
    
    # Generate progress output to standard out
    if t % 5 == 0 and t != 0:
        elapsed = mytime.time() - startTime
        ratio = 1.0 * t / N
        print('%d/%d, \t %.1f%%, \t elapsed: %.1f \t ETA: %.1f' \
              % (t, N, ratio * 100, elapsed, elapsed / ratio - elapsed))

        # Write data output to files
    if t % sampleFreq == 0:
        # Write raw data to files
        #        writeCylinder('outputX_F_Dxr_' + str(Dxr) + '_k_' + ':', X, t)
        #        writeCylinder('outputY_F_Dxr_' + str(Dxr) + '_k_' + ':', Y, t)

        # Write X matrix to image file
        fig = pyplot.figure(figsize=(5, 16))
        pyplot.clf()
        fig = pyplot.contourf(X[(index + 1) % tau, :, :])
        pyplot.xlabel('Cell')
        pyplot.ylabel('Ring')
        pyplot.colorbar()
        pyplot.title('Dxr = ' + str(Dxr) + ', Dxl = ' + str(Dxl) + ', time = ' + str(t * h) + ' s')
        pyplot.savefig('50x200_RingVsCell_F_Dxr_' + str(Dxr) + '_Dxl_' + str(Dxl)  + '_k_' + str(k) +\
                '_' + str(t).rjust(5,'0') + '.png')

        

    #######################################
    # Post processing
    #######################################

    # calculate total concentration totalConcVector
    x = 0
    y = 0
    for cell in range(0, ringsize):
        x += cylinderConcPerCell[cell, t] * math.cos(angle * cell)
        y += cylinderConcPerCell[cell, t] * math.sin(angle * cell)

    # Angle
    totalConcVector[0, t] = math.atan(y / x)
    if totalConcVector[0, t] < 0:
        totalConcVector[0, t] += 2 * math.pi
    totalConcVector[0, t] = totalConcVector[0, t] / 2 / math.pi * ringsize
    # Length
    totalConcVector[1, t] = math.sqrt(x * x + y * y)

# Calculate total diff concentration vector across cylinderConcPerCell
totalDiffConcVector = np.zeros((ringsize, N), float)
for t in range(N):
    totalDiffConcVector[:,t] = cylinderConcPerCell[:,t] - np.roll(cylinderConcPerCell[:,t], ringsize//2)


# Plot Cylinder Sum
fig = pyplot.figure(figsize=(5, 16))
pyplot.clf()
fig = pyplot.contourf(cylinderConcPerCell)
pyplot.xlabel('Iterations (' + str(1 / h) + '/s)')
pyplot.ylabel('Cell')
pyplot.colorbar()
pyplot.title('Vertical sum through cylinder\n Dxr = ' + str(Dxr) + '_Dxl = ' + str(Dxl) + \
      ', k=' + str(k))
pyplot.savefig('50x200_VerticalConcentrationSum_Dxr_' + str(Dxr) + '_Dxl_' + str(Dxl)  + '_k_' + str(k) + '.png')


fig = pyplot.figure(figsize=(12, 4))
pyplot.clf()
pyplot.plot(time[1:], totalConcVector[0, :], label='Angle [Cell Number]')
pyplot.plot(time[1:], totalConcVector[1, :], label='Length [uM]')
pyplot.legend()
pyplot.xlabel('Time [s]')
pyplot.title('Total Ca++ Concentration vector for cylinder')
pyplot.savefig('50x200_VerticalConcentrationVector_Dxr_' + str(Dxr) + '_Dxl_' + str(Dxl)  + '_k_' + str(k) +\
        '.png')

# Plot Diff Cylinder Sum
fig = pyplot.figure(figsize=(5, 16))
pyplot.clf()
fig = pyplot.contourf(totalDiffConcVector)
pyplot.xlabel('Iterations (' + str(1 / h) + '/s)')
pyplot.ylabel('Cell')
pyplot.colorbar()
pyplot.title('Vertical diff sum through cylinder\n Dxr = ' + str(Dxr) + ', Dxl = ' + str(Dxl) + \
      ', k=' + str(k))
pyplot.savefig('50x200_VerticalDiffConcentrationSum_Dxr_' + str(Dxr) + '_Dxl_' + str(Dxl)  + '_k_' + str(k) + '.png')
