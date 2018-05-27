cimport cython

from libc.math cimport cos

cdef double VM2 = 20.0 # 10^-6 M/s
cdef double VM3 = 23.0 # 10^-6 M/s
cdef double K2 = 1.0 # 10^-6 M/s
cdef double KR = 0.8 # 10^-6 M/s
cdef double KA = 0.9 # 10^-6 M/s
cdef double kf = 1.0  # 1/s

cdef double k = 0.8  # 1/s
cdef double v = 0.325  # v= v0 + v1*beta

cdef double Dxl = 2.0  # 1/s
cdef double Dxr = 6.0  # 1/s
cdef double Dxd = 1.0  # 1/s
cdef double kt = 0.1 # 1/s

cdef double sampleFreq = 1.0 # s
cdef int duration = 5

cdef double h = 0.02
cdef int N = int(duration/h) + 1
#######################################
# Initial conditions
#######################################

cdef int ringsize=50
cdef int rings=200
#fileX = fromfile("X0.dat", sep=';')
#fileY = fromfile("Y0.dat", sep=';')



#X0 = np.zeros((ringsize,rings), double)
#Y0 = np.zeros((ringsize,rings), double)
#Amp = 0.55
#index=0
#for ring in range(0,3):
#    for cell in range(24,27):
#        X0[cell][ring] = Amp
#        Y0[cell][ring] = Amp 
#        index += 1          
  
#for ring in range(rings):
#    for cell in range(ringsize):
#        X0[cell][ring] = fileX[index]
#        Y0[cell][ring] = fileY[index] 
#        index += 1  


cdef double f(double X, double Y):
    return VM2*X/(K2+X) - VM3*Y*X*X/((KR+Y)*(KA*KA+X*X)) \
            - kf*Y

cdef double dXdt(double X, double Y, double flux):
    return v - f(X,Y) - k*X + flux

cdef double dYdt(double X,double Y):
    return f(X,Y)

# @cython.boundscheck(False)
cpdef timestep(double[:,:,:] X, double[:,:,:] Y, double[:,:] cylinderConcPerCell, int index, 
               int ring, int t, int tau, double omega, double Am):
    cdef double flux
    cdef double x, y
    cdef double k11, k12, k21, k22, k31, k32, k41, kk42
    cdef double Xret, Yret
    cdef int cell
    for cell in range(0,ringsize): 
            flux = 0.0
# Sinus-Exitation
            if 0 <= ring <= 3:
                if 24 <= cell <= 27:
                    X[index][ring][cell] = Am*0.5*(1 + cos(omega*t*h))
#                   Y[index][ring][cell] = Amp*(math.cos(omega*t*h))**2
# horizontal diffusion
            if cell == 0:
                flux = Dxr*(X[index][ring][ringsize-1] - X[index][ring][cell]) \
                   + Dxl*(X[index][ring][1] - X[index][ring][cell])
            elif cell==ringsize-1:
                flux = Dxr*(X[index][ring][cell-1] - X[index][ring][cell]) \
                       + Dxl*(X[index][ring][0] - X[index][ring][cell])
            else :
                flux = Dxr*(X[index][ring][cell-1] - X[index][ring][cell]) \
                       + Dxl*(X[index][ring][cell+1] - X[index][ring][cell])
# vertical diffusion
            if ring == 0:
                if X[index][ring][cell] > X[index][ring+1][cell]:
                    flux -= Dxd*(X[index][ring][cell] - X[index][ring+1][cell])
            elif ring==rings-1:
                if X[index][ring-1][cell] > X[index][ring][cell]:
                    flux += Dxd*(X[index][ring-1][cell] - X[index][ring][cell])
            else:
                if X[index][ring-1][cell] > X[index][ring][cell]:
                    flux += Dxd*(X[index][ring-1][cell] - X[index][ring][cell])
                if X[index][ring][cell] > X[index][ring+1][cell]:
                    flux -= Dxd*(X[index][ring][cell] - X[index][ring+1][cell])

# Feedback across ring, tau elements back in time
            if t>=tau:
                flux += kt*(X[(t-tau)%tau][ring][(cell + ringsize//2)%ringsize]\
                              - X[index][ring][cell])


# Perform Runge-Kutta integration on current cell
            x = X[index][ring][cell]
            y = Y[index][ring][cell]
            k11 = h*dXdt(x,y,flux)
            k12 = h*dYdt(x,y)

            k21 = h*dXdt(x+0.5*k11, y+0.5*k12,flux)
            k22 = h*dYdt(x+0.5*k11, y+0.5*k12)

            k31 = h*dXdt(x+0.5*k21, y+0.5*k22,flux)
            k32 = h*dYdt(x+0.5*k21, y+0.5*k22)

            k41 = h*dXdt(x+k31, y+k32,flux)
            k42 = h*dYdt(x+k31, y+k32)

            Xret = x + (k11 + 2*k21 + 2*k31 + k41)/6
            Yret = y + (k12 + 2*k22 + 2*k32 + k42)/6
            X[(index+1)%tau][ring][cell] = Xret
            Y[(index+1)%tau][ring][cell] = Yret
            cylinderConcPerCell[cell][t] += X[index][ring][cell]

