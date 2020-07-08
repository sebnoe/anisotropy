# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Author: Sebastian Noe, snoe@geophysik.uni-muenchen.de

import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.linalg import null_space


# #### Generating a random elastic tensor
#
# The generation of the 6x6 (in Voigt notation) elastic tensor is rather straightforward. Each symmetry system has a certain amount of equal value entries. Variances from generation to generation are introduced with _sigma_. The diagonal elements are (kind of) forced to be higher than the sum of the entire row/column in order to ensure positive eigenvalues in the Christoffel equation. 
#
# These randomly generated elastic tensor may be unphysical. Another function to call specific materials is installed.

# +
def load_random_medium_list():
    medium= ['isotropic','cubic','VTI','tetragonal','trigonal','orthorhombic','monoclinic','triclinic']
    for i in range(0,len(medium)):
        print('#'+str(i), medium[i])
    return medium
    
def get_random_C(mode, sigma):
    C = np.zeros((6,6))
    #options for mode: 'isotropic','cubic','VTI','trigonal','tetragonal','orthorhombic','monoclinic','triclinic'
    if mode=='triclinic':
        print(mode)
        for i in range(0,6):
            if i<3:
                C[i][i] = round((np.random.randn()*sigma+100),3)*6
            else:
                C[i][i] = round((np.random.randn()*sigma+100),3)*2
            for j in range(i+1,6):
                if i<3 and j<3:
                    C[i][j] = round((np.random.randn()*sigma+100),3)*3
                else:    
                    C[i][j] = round((np.random.randn()*sigma+100),3)
                C[j][i] = C[i][j]
    
    if mode=='isotropic':
        print(mode)
        first = round((np.random.randn()*sigma+100),3)*5
        second = round((np.random.randn()*sigma+100),3)*5
        for i in range(0,3):
            C[i][i] = first + 2* second
            for j in range(i+1,3):
                C[i][j] = first
                C[j][i] = C[i][j]
        for i in range(3,6):
            C[i][i] = second

    if mode=='VTI':
        print(mode)
        coeff = np.random.randn(5)*sigma+100
        C[0][0] = coeff[0]*4
        C[1][1] = coeff[0]*4
        C[2][2] = coeff[1]*4
        C[0][2] = coeff[2]*2
        C[1][2] = coeff[2]*2
        C[3][3] = coeff[3]
        C[4][4] = coeff[3]
        C[5][5] = coeff[4]
        C[0][1] = C[0][0]-2*C[5][5]
        for i in range(0,6):
            for j in range(i,6):
                C[i][j] = round(C[i][j],3)
                C[j][i] = C[i][j]
                
    if mode=='orthorhombic':
        print(mode)
        coeff = np.zeros(9)
        for i in range(0,9):
            coeff[i] = round(np.random.randn()*sigma+100,5)
        C[0,0] = coeff[0]*4
        C[1,1] = coeff[1]*4
        C[2,2] = coeff[2]*4
        C[3,3] = coeff[3]*2
        C[4,4] = coeff[4]*2
        C[5,5] = coeff[5]*2
        C[0,1] = coeff[6]
        C[0,2] = coeff[7]
        C[1,2] = coeff[8]
        C[1,0] = C[0,1]
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        
    
    if mode=='cubic':
        print(mode)
        coeff = np.zeros(3)
        for i in range(0,3):
            coeff[i] = round(np.random.randn()*sigma+100,5)
        C[0,0] = coeff[0]*3
        C[1,1] = C[0,0]
        C[2,2] = C[0,0]
        C[3,3] = coeff[1]
        C[4,4] = C[3,3]
        C[5,5] = C[3,3]
        C[0,1] = coeff[2]
        C[0,2] = C[0,1]
        C[1,2] = C[0,1]
        C[1,0] = C[0,1]
        C[2,0] = C[0,1]
        C[2,1] = C[0,1]
        
    if mode=='tetragonal':
        print(mode)
        coeff = np.zeros(6)
        for i in range(0,6):
            coeff[i] = round(np.random.randn()*sigma+100,5)
        C[0,0] = coeff[0]*4
        C[1,1] = coeff[0]*4
        C[2,2] = coeff[1]*4
        C[3,3] = coeff[2]*2
        C[4,4] = coeff[2]*2
        C[5,5] = coeff[3]*2
        C[0,1] = coeff[4]
        C[0,2] = coeff[5]
        C[1,2] = coeff[5]
        C[1,0] = C[0,1]
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        
    if mode=='trigonal':
        print(mode)
        coeff = np.zeros(6)
        for i in range(0,6):
            coeff[i] = round(np.random.randn()*sigma+100,5)
        C[0,0] = coeff[0]*4
        C[1,1] = coeff[0]*4
        C[2,2] = coeff[1]*4
        C[3,3] = coeff[2]*2
        C[4,4] = coeff[2]*2
        C[0,1] = coeff[3]
        C[5,5] = (C[0,0] - C[0,1])/2       
        C[0,2] = coeff[4]
        C[1,2] = coeff[4]
        C[1,0] = C[0,1]
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        C[0,3] = coeff[5]
        C[1,3] = -coeff[5]
        C[3,0] = coeff[5]
        C[3,1] = -coeff[5]
        C[4,5] = coeff[5]
        C[5,4] = coeff[5]
    
    if mode=='monoclinic':
        print(mode)
        coeff = np.zeros(13)
        for i in range(0,13):
            coeff[i] = round(np.random.randn()*sigma+100,5)
        C[0,0] = coeff[0]*6
        C[1,1] = coeff[1]*6
        C[2,2] = coeff[2]*6
        C[3,3] = coeff[3]*2
        C[4,4] = coeff[4]*2
        C[5,5] = coeff[5]*2 
        C[0,1] = coeff[6]     
        C[0,2] = coeff[7]
        C[1,2] = coeff[8]
        C[1,0] = C[0,1]
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        C[0,5] = coeff[9]
        C[5,0] = coeff[9]
        C[1,5] = coeff[10]
        C[5,1] = coeff[10]
        C[2,5] = coeff[11]
        C[5,2] = coeff[11]
        C[3,4] = coeff[12]
        C[4,3] = coeff[12] 
        
    print(C)
    C = C*1e8
    return C, 3000.


# -

# #### Load specific VTI media
#
# VTI media and their respective elastic tensors are fully determined when 6 parameters are given. Vertical p-wave velocity, vertical s-wave velocity, density and the three Thomsen parameters $\epsilon$,$\delta$ and $\gamma$.
#
# In his paper, Thomsen (1986) included a long list of real rocks with their corresponding parameters. A few of those are callable with this function. 
#
# It's possible to change single parameters, e.g. look at taylor sandstone when $\epsilon$ is negative instead of positive while keeping all other parameters fixed.

# +
def load_medium_list():
    medium= ['isotropic','taylor sandstone','mesaverde clayshale','mesaverde laminated siltstone',\
            'mesaverde mudshale','mesaverde calcareous sandstone','quartz','Baird-shale']
    for i in range(0,len(medium)):
        print('#'+str(i), medium[i])
    print(' ')    
    return medium    

def get_specific_VTI(name,give_thomsen=False, density=3000.00001,eps=1e-5,gamma=1e-5,delta=1e-5,vp0=3500.00001,vs0=3500.00001/np.sqrt(3)):
    C = np.zeros((6,6))
    
    if name=='taylor sandstone':
        print(name)
        eps2 = 0.11
        gamma2 = 0.255
        delta2 = -0.035
        vp02 = 3368
        vs02 = 1829
        density2= 2500
    elif name=='mesaverde clayshale':
        print(name)
        eps2 = 0.334
        gamma2 = 0.575
        delta2 = 0.730
        vp02 = 3928
        vs02 = 2055
        density2 = 2590 
    elif name=='mesaverde laminated siltstone':
        print(name)
        eps2 = 0.091
        gamma2 = 0.046
        delta2 = 0.565
        vp02 = 4449.
        vs02 = 2585.
        density2 = 2570.
    elif name=='quartz':
        print(name)
        eps2 = -0.096
        gamma2 = -0.159
        delta2 = 0.273
        vp02 = 6096
        vs02 = 4481
        density2 = 2650.
    elif name=='mesaverde mudshale':
        print(name)
        eps2 = 0.010
        gamma2 = -0.005
        delta2 = 0.012
        vp02 = 5073
        vs02 = 2998
        density2 = 2680.
    elif name=='mesaverde calcareous sandstone':
        print(name)
        eps2 = 0.000
        gamma2 = -0.007
        delta2 = -0.264
        vp02 = 5460
        vs02 = 3219
        density2 = 2690.   
    elif name=='isotropic':
        print(name)
        eps2 = 0.
        gamma2 = 0.
        delta2 = 0.
        vp02 = 5000.
        vs02 = vp02/np.sqrt(3)
        density2 = 3000.
    elif name=='Baird-shale':
        print(name)
        eps2 = 0.42
        gamma2 = 0.36
        delta2 = 0.21
        vp02 = 2800.
        vs02 = 1750.
        density2 = 2600.    
        
    mod = False
    if eps!=1e-5:
        eps2 = eps
        mod = True
    if gamma!=1e-5:
        gamma2 = gamma
        mod = True
    if delta!=1e-5:
        delta2 = delta
        mod = True
    if vp0!=3500.00001:
        vp02 = vp0 
        mod = True
    if vs0!=3500.00001/np.sqrt(3):
        vs02 = vs0
        mod = True
    if density!=3000.00001:
        density2 = density
        mod = True
    if mod==True:
        print('modified')
              
    c33 = vp02**2 * density2
    c44 = vs02**2 * density2
    c11 = c33 * (2 * eps2 + 1)
    c66 = c44 * (2 * gamma2 + 1)
    c13 = np.sqrt( 2 * delta2 * c33 * (c33-c44) + (c33-c44)**2) - c44
    
    C[0][0] = c11
    C[0][1] = c11 - 2*c66
    C[1][0] = C[0][1]
    C[0][2] = c13
    C[2][0] = C[0][2]
    C[1][1] = c11
    C[2][1] = c13
    C[1][2] = C[2][1]
    C[2][2] = c33
    C[3][3] = c44
    C[4][4] = c44
    C[5][5] = c66
        
    if give_thomsen:
        print(' ')
        print('vp0     =',round(vp02,3))
        print('vs0     =',round(vs02,3))
        print('eps     =',round(eps2,3))
        print('delta   =',round(delta2,3))
        print('gamma   =',round(gamma2,3))
        print('density =',round(density2,3))
        print(' ')
        
    print(C*1e-9)
    return C, density2


# -

# #### Selecting propagation direction(s)
#
# This function gives the option between three different modes. In 'input' the program asks for specific x,y and z values of the propagation direction vector. The vector will be normalized. The mode 'random' gives the option to generate _N_ randomly selected unitized vectors. This samples the medium in many different directions. To make the sampling more realistic, the probability density function of the vertical angle $\theta$ follows a cosine-function, meaning that horizontal arrivals are more likely than vertical ones. This should account for the fact that the seismometer is close to the surface. All propagation directions therefore point upwards. The probability density function for the horizontal angle $\phi$ is constant.
# The mode 'planar' gives N propagation directions part of the same plane with constant angular off-set. Plane is determined by a max inclination $\theta$ the azimuth $\phi$ of the line where the plane cuts into the horizon.
#
# Depending on the direction, the (3x3) Matrix $\Gamma$ is calculated by:
#
# $
# \Gamma = L(\nu) \cdot C \cdot L^T(\nu)
# $
#
# $
# \Gamma_{ij} = L_{ik}C_{km}L_{mj}^T = L_{ik}L_{jm}C_{km} 
# $
#
# This is crucial for the setup of the Kristoffel equation: 
#
# $
# (\Gamma - \rho V^2I)\vec{u}=\vec{0} 
# $
#
# The eigenproblem will be solved in get_eigenvals().
# The function returns two lists containing all generated directions of propagation and corresponding $\Gamma$'s.

def get_direction(mode, C, N,theta_e=0.,phi_e=0.):
    nus = []
    gammas = []
    if mode=='input':
        k = [float(input('x ')),float(input('y ')),float(input('z '))]

        wavelength = 2*np.pi/(np.sqrt(k[0]**2+k[1]**2+k[2]**2))
        nu = k / (np.sqrt(k[0]**2+k[1]**2+k[2]**2))
        L = [[nu[0],0,0],[0,nu[1],0],[0,0,nu[2]],[0,nu[2],nu[1]],[nu[2],0,nu[0]],[nu[1],nu[0],0]]
        gamma = np.zeros((3,3))
        for i in range(0,3):
            for j in range(i,3):
                for k in range(0,6):
                    for n in range(0,6):
                        gamma[i][j] += L[k][i]*L[n][j]*C[k][n]
                gamma[j][i] = gamma[i][j]
        gammas.append(gamma)       
        nus.append(nu)        
    if mode =='random':
        na = 1000
        prec = 10000
        xk = np.linspace(0,np.pi/2*prec,na)
        pk = np.cos(xk/prec)
        pk = pk/sum(pk)
        custm = stats.rv_discrete(name='cosine', values=(xk, pk))
        for l in range(0,N):
            theta = np.pi/2 - custm.rvs(size=1)[0]/prec
            phi = np.random.randint(na)/na*2*np.pi            
            k = [np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]
            nu = k / (np.sqrt(k[0]**2+k[1]**2+k[2]**2))
            L = [[nu[0],0,0],[0,nu[1],0],[0,0,nu[2]],[0,nu[2],nu[1]],[nu[2],0,nu[0]],[nu[1],nu[0],0]]
            gamma = np.zeros((3,3))
            for i in range(0,3):
                for j in range(i,3):
                    for k in range(0,6):
                        for n in range(0,6):
                            gamma[i][j] += L[k][i]*L[n][j]*C[k][n]
                    gamma[j][i] = gamma[i][j]
            gammas.append(gamma)       
            nus.append(nu)
            
    if mode == 'planar':
        d2r = np.pi/180
        #theta_e = float(input('theta '))*d2r
        #phi_e = float(input('phi '))*d2r
        theta_e = theta_e*d2r
        phi = np.ones(N)*phi_e*d2r
        theta = np.linspace(-np.pi/2,np.pi/2,N)
        for i in range(0,N):
            if theta[i]<0:
                theta[i] = -theta[i]
                phi[i] = (phi[i]+np.pi)%(2*np.pi)
        
        R = np.zeros((3,3))
    
        # Rodrigues' rotation formula, matrix will be rotated around vector r by angle theta_e 
        
        r = [ np.cos(phi_e), np.sin(phi_e), 0 ]
        r_sum = 0
        for i in range(0,3):
            r_sum += r[i]**2
        for i in range(0,3):
            r[i] = r[i]/np.sqrt(r_sum)

        K = [[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]]

        K_sq = [[-r[1]**2-r[2]**2,r[0]*r[1],r[0]*r[2]], \
                [r[0]*r[1],-r[0]**2-r[2]**2,r[1]*r[2]], \
                [r[0]*r[2],r[1]*r[2],-r[0]**2-r[1]**2]]
        
        for i in range(0,3):
            for j in range(0,3):
                K[i][j] = np.sin(theta_e)*K[i][j]
                K_sq[i][j] = (1-np.cos(theta_e))*K_sq[i][j]
        
        R[0,0] = 1 + K[0][0] + K_sq[0][0]
        R[1,0] = K[1][0]+K_sq[1][0]
        R[2,0] = K[2][0]+K_sq[2][0]

        R[0,1] = K[0][1]+K_sq[0][1]
        R[1,1] = 1 + K[1][1] + K_sq[1][1]
        R[2,1] = K[2][1]+K_sq[2][1]

        R[0,2] = K[0][2]+K_sq[0][2]
        R[1,2] = K[1][2]+K_sq[1][2]
        R[2,2] = 1 + K[2][2] + K_sq[2][2]
        
        for i in range(0,N):
            nu = [np.cos(phi[i])*np.sin(theta[i]),np.sin(phi[i])*np.sin(theta[i]),np.cos(theta[i])]
            nu = np.dot(R,nu)
            L = [[nu[0],0,0],[0,nu[1],0],[0,0,nu[2]],[0,nu[2],nu[1]],[nu[2],0,nu[0]],[nu[1],nu[0],0]]
            gamma = np.zeros((3,3))
            for i in range(0,3):
                for j in range(i,3):
                    for k in range(0,6):
                        for n in range(0,6):
                            gamma[i][j] += L[k][i]*L[n][j]*C[k][n]
                    gamma[j][i] = gamma[i][j]
            gammas.append(gamma)       
            nus.append(nu)
            
    return nus, gammas    


# #### Given propagation direction
#
# This function serves the purpose of creating the Christoffel-matrix $\Gamma$ for a given set of propagation directions $nus$.

def get_gamma(nus, C):
    gammas = []
    for nu in nus:
        L = [[nu[0],0,0],[0,nu[1],0],[0,0,nu[2]],[0,nu[2],nu[1]],[nu[2],0,nu[0]],[nu[1],nu[0],0]]
        gamma = np.zeros((3,3))
        for i in range(0,3):
            for j in range(i,3):
                for k in range(0,6):
                    for n in range(0,6):
                        gamma[i][j] += L[k][i]*L[n][j]*C[k][n]
                gamma[j][i] = gamma[i][j]
        gammas.append(gamma)
    return gammas    


# #### Visualize propagation directions in stereonet
#
# Stereonet plot of all given propagation directions.
#

def plot_directions(nus):
    r2d = 180/np.pi
    fig = plt.figure(figsize=(6,6))
    plt.title('Direction of propagation')
    ax = fig.add_subplot(111, projection='stereonet')
    ax.pole(0, 0, 'black', markersize=8,marker='o')
    for i in range(0,len(nus)):
        ic,dc = get_angles([nus[i][1],nus[i][0],nus[i][2]])
        ic,dc = ic*r2d, dc*r2d+90.
        ax.pole(dc, ic, 'r', markersize=10,marker='*')
    ax.grid()
    plt.savefig('directions_plot.png')
    plt.show()


# #### Solving the eigenproblem
#
# This short function solves the eigenproblem posed by the Kristoffel equation. Note, $\Gamma$ depends on the elastic tensor and on the propagatio direction (anisotropy). The (3x3) matrix yields eigenvalues and corresponding eigenvectors. Eigenvalues are $\rho*V^2$, so the actual velocities can be easily calculated. Usually, all three eigenvalues will be different for an anisotropic medium resulting in three distinct wavefronts arriving in the synthetic seismometer. The eigenvectors are the respective polarizations. In an isotropic setting, the polarization of the p-wave is always parallel to the propagation direction which is not the case anymore with anisotropy.
# Velocities and Polarizations are returned.

def get_eigenvals(gamma, density):
    w,v = np.linalg.eig(gamma)
    vel = np.sqrt(w/density)
    return vel, v


# #### Constructing a synthetic signal
#
# We use the plane wave ansatz.
# According to this, the displacement $u$ can be written down:
#
# $
# u(x,t) = A n_i \exp(i\omega(t-\frac{x\cdot\nu}{V_i}))
# $
# where A is the amplitude, $n_i$ the polarization, $\omega=2 \pi f$ the angle frequency, $\nu$ the normalized direction of propagation and $V_i$ the velocity. Polarization and velocity are eigenvector/eigenvalue-pairs and solutions to the Kristoffel equation. Because there are three different eigenvalues, each component measures three arrivals.
# Actually, we measure acceleration rather than displacement. Applying $\partial_t^2$ we get
#
# $
# \ddot u(x,t) = -\omega^2A n_i \exp(i\omega(t-\frac{x\cdot\nu}{V_i}))
# $
#
# Rotation rates can be calculated according to 
#
# $\dot\Omega = \frac{1}{2}\nabla\times \dot u$. 
#
# Therefore we get
#
# $
# \dot\Omega(x,t) = -\frac{\omega^2A}{2V_i} (\nu\times n_i)\exp(i\omega(t-\frac{x\cdot\nu}{V_i}))
# $
#
# Note that the expression for the velocity moves out of the trigonometric function due to spatial derivatives. This will be crucial for estimations of velocities. It's polarization is $\nu\times n_i$ and will be written down as $r_i$ in later functions. 
#
# -------------------
#
# In order to produce a synthetic signal, the exponential terms are replaced by an arbitrarily chosen source function. Here, it is chosen such that acceleration and rotation rate (both second derivatives) are scaled by the first derivative of a gaussian. Normalizations of the gaussians are not considered here. The signal for each wavefront will be longer for lower frequencies and short for high frequencies. 
#
# Practically, the distance between source and receiver must be given. Here, i want the fastest wavefront to arrive after roughly five seconds and determine the length of the entire signal according to the slowest wavefront. 
#
# The six-component measurement is returned as well as the time markers.
#
#

def get_seis(v,vel,nu,f,fs):
    xr = max(vel)*5.
    tmin = xr / max(vel) * 0.3
    tmax = xr / min(vel) * 1.3
    nt = int(((tmax-tmin) * fs)+1) 
    t = np.linspace(tmin,tmax,nt)
    seis = np.zeros((6,nt))   
    A = 1.
    omega = 2*np.pi*f
    
    seis[0,:] = - v[0,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[0,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[0,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[1,:] = - v[1,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[1,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[1,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2) 
    seis[2,:] = - v[2,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[2,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[2,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2) 
    
    seis[3,:] =- np.cross(nu,v[:,0])[0]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2)\
            - np.cross(nu,v[:,1])[0]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu,v[:,2])[0]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[4,:] = - np.cross(nu,v[:,0])[1]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - np.cross(nu,v[:,1])[1]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu,v[:,2])[1]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[5,:] = - np.cross(nu,v[:,0])[2]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - np.cross(nu,v[:,1])[2]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu,v[:,2])[2]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    return seis, t


# #### Plotting the synthetic signals
#
# A quick and easy way to plot the signal is performed by this function. Inputs must be the synthetic signal, it's corresponding time axis and the propagation direction $\nu$. The latter will not influence the actual plots in any way, i just find it helpful to visualize it's value in the title of the plot.
# Translational and rotational components are plotted separately and the three spatial directions are offset by the maximum amplitude. Therefore, the amplitude-axis' ticks are omitted as they may cause confusion.

def plotseis(seis,t,nu):
    a1 = abs(seis[0:3][:].max())
    plt.figure(figsize=(8,4))
    plt.title('x: '+str(round(nu[0],2))+ ' y: '+str(round(nu[1],2))+' z: '+str(round(nu[2],2)))
    
    plt.plot(t,seis[0,:],label='x')
    plt.plot(t,seis[1,:]-a1,label='y')
    plt.plot(t,seis[2,:]-2*a1,label='z')
    plt.legend()
    plt.yticks([])
    plt.show()
    a2 = seis[3:6][:].max()
    plt.figure(figsize=(8,4))

    plt.plot(t,seis[3,:],label='rot_x')
    plt.plot(t,seis[4,:]-a2,label='rot_y')
    plt.plot(t,seis[5,:]-2*a2,label='rot_z')
    plt.legend()
    plt.yticks([])
    plt.show()


# #### Plotting the particle motions
#
# This plotting tool is almost entirely copied from Heiner's notebook. The particle motion in each plane (xy, yz, xz) is shown. For anisotropic media we expect elliptical motions. If elliptical motions cannot be observed, the three wavefronts arrive completely separated, so maybe the frequency should be lowered.

def plot_particle_motions(seis):
    a1 = seis[0:3][:].max()
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.scatter(seis[0,:],seis[1,:],1,'b')
    plt.xlabel(' x ')
    plt.ylabel(' y')
    plt.title("x-y " , size=10)
    plt.xlim([-a1,a1])
    plt.ylim([-a1,a1])
    plt.grid()

    plt.subplot(2,2,2)
    plt.scatter(seis[0,:],seis[2,:],1,'b')
    plt.xlabel(' x ')
    plt.ylabel(' z')
    plt.title("x-z " , size=10)
    plt.xlim([-a1,a1])
    plt.ylim([-a1,a1])
    plt.grid()

    plt.subplot(2,2,3)
    plt.scatter(seis[1,:],seis[2,:],1,'b')
    plt.xlabel(' y ')
    plt.ylabel(' z')
    plt.title("y-z " , size=10)
    plt.xlim([-a1,a1])
    plt.ylim([-a1,a1])
    plt.grid()

    plt.show()


# #### Estimating wave velocities
#
# In this experiment, wave velocities are estimated for numerous propagation directions for anisotropic media.
#
# ##### Why is it possible to estimate velocities from single station data? 
#
# There is no phase difference between rotation rate and acceleration. For each wavefield $m = qP, qS1$ or $qS2$, we observe following proportionalities
#
# $
# \ddot u_{x,m} = -A\omega^2n_{x,m}
# $
#
# $
# \ddot u_{y,m} = -A\omega^2n_{y,m}
# $
#
# $
# \ddot u_{z,m} = -A\omega^2n_{z,m}
# $
#
# $
# \dot \Omega_{x,m} = -\frac{A\omega^2}{2v_m}(\nu\times n_m)_x = -\frac{A\omega^2}{2v_m}(\nu_yn_{z,m}-\nu_zn_{y,m})
# $
#
# $
# \dot \Omega_{y,m} = -\frac{A\omega^2}{2v_m}(\nu\times n_m)_y = -\frac{A\omega^2}{2v_m}(\nu_zn_{x,m}-\nu_xn_{z,m})
# $
#
# $
# \dot \Omega_{z,m} = -\frac{A\omega^2}{2v_m}(\nu\times n_m)_z = -\frac{A\omega^2}{2v_m}(\nu_xn_{y,m}-\nu_yn_{x,m})
# $
#
# The seismometer could be rotated, such that in the new reference frame the direction of propagation is parallel to the x-axis $\nu=(1,0,0)^T$. Additionally, the plane perpendicular to $\nu$ can be rotated such that the shear waves are split.
# Thus, rotations are:
#
# $
# \dot\Omega_{x,m} = 0
# $
#
# $
# \dot\Omega_{y,m} = \frac{A\omega^2}{2v_m}n_{z,m}
# $
#
# $
# \dot\Omega_{z,m} = -\frac{A\omega^2}{2v_m}n_{y,m}
# $
#
# Extract velocities by forming ratios:
#
# $
# \frac{\ddot u_{y,m}}{\dot\Omega_{z,m}} = 2v_m
# $
#
# $
# \frac{\ddot u_{z,m}}{\dot\Omega_{y,m}} = 2v_m
# $
#
# $
# \frac{\sqrt{\ddot u_{y,m}^2+\ddot u_{z,m}^2}}{\sqrt{\dot\Omega_{y,m}^2+\dot\Omega_{z,m}^2}} = 2v_m
# $
#
# The first two equations will yield quasi-shear wave velocities. Their amplitudes are simply the maximum values over the entire time interval. For the last equation, one has to determine the amplitudes corresponding to the first arrival. The resulting velocity will be qP and it is only possible to determine if curls are observed.
#
#
# #### The function
#
# The main function is defined in cell 1. It can roughly be split into three parts, 1) Rotation such that $x=\nu$ 2) Rotation such that y and z exhibit profound shear wave splitting and 3) the picking of amplitudes for the qS estimation.
#
# The most important inputs are the original (unrotated) measurements of all 6 components and the direction of propagation $\nu$. This direction may already be a previously estimated vector. The time array is passed along for potential plots.
#
#
#
# ##### 1) Rotation of seisometer such that x-axis points into the direction of propagation
#
# First, the direction of propagation is written down in terms of horizontal and vertical angles ($\phi,\theta$).
# The rotation is necessary for the estimation because it is done with amplitude ratios between translational and rotational signals. Because rotation entirely takes place in the plane vertical to the propagation direction, the amplitudes of the rotation are too small when the axis doesn't lie perfectly in the plane. To perform this specific transformation of the signal we need a rotation matrix.
#
# The Rodrigues' rotation formula is used to calculate the rotation matrix. It describes a rotation around a vector $r$ by some angle $\alpha$. The idea is to find vector $r$ such the x-axis is rotated by angle $\alpha=\pi$ and falls perfectly onto $\nu$. Therefore, $r$ must be halfway between $x$ and $\nu$. 
#
# $
# r = \frac{1}{2}\begin{pmatrix}1\\0\\0\end{pmatrix}+\frac{1}{2}\begin{pmatrix}\cos(\phi)\sin(\theta)\\\sin(\phi)\sin(\theta)\\\cos(\theta)\end{pmatrix}
# $
#
# Afterwards, $r$ is normalized. Rodrigues' rotation formula to get a rotation matrix is (for $\alpha=\pi$):
#
# $
# R = I + \sin(\alpha)K+(1-\cos(\alpha))K^2 = I + 2K^2
# $
# (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
#
# where K is the cross-product matrix with the identity:
#
# $
# Kv = r\times v
# $
#
# so
#
# $
# K = \begin{pmatrix} 0&-r_z & r_y \\ r_z&  0 & -r_x \\ -r_y & r_x & 0       \end{pmatrix}
# $
# .
#
# It can be checked whether the rotation worked by looking at the plots (turn on by setting _plot_rotated_seismo=True_). The rotation rate around the x-axis should constantly be zero because it points in the direction of propagation and $\dot\Omega\perp\nu$. 
#
# ##### 2) Rotation around x-axis such that y and z each show shear wave splitting perfectly
#
# This section is mostly from Heiner's notebook. The maximum cross-correlation coefficient between y and z axis is searched for by rotating around the x-axis. The angle with the maximum ccc will be the final rotation of the seismometer. This method captures shear wave splitting because the ccc will be the highest when the respective amplitudes on both components are maximized.  
#
# ##### 3) Picking amplitudes
#
# Having correctly rotated seismograms, the shear wave velocities can be estimated by
#
# $
# V_{S1} = -\frac{1}{2}\ddot u_y/\dot\Omega_z
# $
#
# $
# V_{S2} = -\frac{1}{2}\ddot u_z/\dot\Omega_y
# $
#
# where the values of the rotation rates and accelerations are the maximum values.
#
# For the estimation of the qP-velocity an extra step has to be introduced. Whenever the translational data exceeds some threshold for the first time, the corresponding amplitudes of translational and rotational components are extracted. If rotational motions are zero, the velocity cannot be estimated.
#
# $
# V_{P} = \frac{1}{2}\sqrt{\ddot u_{y}^2+\ddot u_{z}^2}/\sqrt{\dot\Omega_{y}^2+\dot\Omega_{z}^2}
# $
#
#
# The function returns three estimated velocities, the rotated seismograms and the Rotation matrix.
#

# +
## FUNCTION APPLIES TO NOISE-LESS CASE ONLY

def estimate_velocity(seis, nu,t):
    nt = len(seis[0,:])
    r2d = 180/np.pi
    vel_e = np.zeros(2)
    
    theta, phi = get_angles(nu)
    
    R = np.zeros((3,3))
    
    # Rodrigues' rotation formula, matrix will be rotated around vector r by angle pi 
    # x-axis=direction of propagation after rotation

    
    r = [(1. + np.cos(phi)*np.sin(theta))/2,\
         (np.sin(phi)*np.sin(theta))/2,\
         np.cos(theta)/2]
    r_sum = 0
    for i in range(0,3):
        r_sum += r[i]**2
    for i in range(0,3):
        r[i] = r[i]/np.sqrt(r_sum)
        
    K_sq = [[-r[1]**2-r[2]**2,r[0]*r[1],r[0]*r[2]], \
            [r[0]*r[1],-r[0]**2-r[2]**2,r[1]*r[2]], \
            [r[0]*r[2],r[1]*r[2],-r[0]**2-r[1]**2]]
    
    R[0,0] = 1 + 2*K_sq[0][0]
    R[1,0] = 2*K_sq[1][0]
    R[2,0] = 2*K_sq[2][0]

    R[0,1] = 2*K_sq[0][1]
    R[1,1] = 1 + 2*K_sq[1][1]
    R[2,1] = 2*K_sq[2][1]

    R[0,2] = 2*K_sq[0][2]
    R[1,2] = 2*K_sq[1][2]
    R[2,2] = 1 + 2*K_sq[2][2]
    
    seis_new = np.zeros((6,nt))
    
    for k in range(0,3):
        seis_new[k,:]   = R[k,0]*seis[0,:]+R[k,1]*seis[1,:]+R[k,2]*seis[2,:]
        seis_new[k+3,:] = R[k,0]*seis[3,:]+R[k,1]*seis[4,:]+R[k,2]*seis[5,:]
    
    
    nang = 721
    xc  = np.zeros(nang)
    ang = np.linspace(0,180,nang)
    d2r = r2d**-1

    for i in range(nang):
        angle = ang[i]
        xr = np.cos(angle*d2r)*seis_new[1,:] -  np.sin(angle*d2r)*seis_new[2,:]
        yr = np.sin(angle*d2r)*seis_new[1,:] +  np.cos(angle*d2r)*seis_new[2,:]
        junk = np.corrcoef(xr, yr)
        xc[i] = junk[1,0]
        ang[i] = angle
        
    imax = np.argmax(xc)
    imax2 = np.argmin(xc)
    if abs(imax)<abs(imax2):
        imax = imax2
    amax = ang[imax]
    
    angle = amax
    xr = np.cos(angle*d2r)*seis_new[1,:] -  np.sin(angle*d2r)*seis_new[2,:]
    yr = np.sin(angle*d2r)*seis_new[1,:] +  np.cos(angle*d2r)*seis_new[2,:]
    
    seis_new[1,:] = xr
    seis_new[2,:] = yr
    
    xrr = np.cos(angle*d2r)*seis_new[4,:] -  np.sin(angle*d2r)*seis_new[5,:]
    yrr = np.sin(angle*d2r)*seis_new[4,:] +  np.cos(angle*d2r)*seis_new[5,:]
    
    seis_new[4,:] = xrr
    seis_new[5,:] = yrr
    
    R2 = np.array([[1,0,0],[0,np.cos(angle*d2r),-np.sin(angle*d2r)],[0,np.sin(angle*d2r),np.cos(angle*d2r)]])
    R = np.dot(R,R2.transpose())
        
    j1 = np.argmax(xrr)
    j1c = np.argmin(xrr)
    j2 = np.argmax(yrr)
    j2c = np.argmin(yrr)
    qS1 = abs(yr[j1]/xrr[j1])/2
    qS2 = abs(xr[j2]/yrr[j2])/2
    
    safety = nt//100
    if abs(j1-j2)<safety or abs(j1-j2c)<safety or abs(j1c-j2)<safety:    # if picks are from same peak!
        angle = 45.
        xr = np.cos(angle*d2r)*seis_new[1,:] -  np.sin(angle*d2r)*seis_new[2,:]
        yr = np.sin(angle*d2r)*seis_new[1,:] +  np.cos(angle*d2r)*seis_new[2,:]

        seis_new[1,:] = xr
        seis_new[2,:] = yr

        xrr = np.cos(angle*d2r)*seis_new[4,:] -  np.sin(angle*d2r)*seis_new[5,:]
        yrr = np.sin(angle*d2r)*seis_new[4,:] +  np.cos(angle*d2r)*seis_new[5,:]

        seis_new[4,:] = xrr
        seis_new[5,:] = yrr

        R2 = np.array([[1,0,0],[0,np.cos(angle*d2r),-np.sin(angle*d2r)],[0,np.sin(angle*d2r),np.cos(angle*d2r)]])
        R = np.dot(R,R2.transpose())
        
        j1 = np.argmax(xrr)
        j2 = np.argmax(yrr)
        qS1 = abs(yr[j1]/xrr[j1])/2
        qS2 = abs(xr[j2]/yrr[j2])/2
    

    ip = np.argmax(seis_new[0,:])
    uy = seis_new[1,ip]
    uz = seis_new[2,ip]
    ry = seis_new[4,ip]
    rz = seis_new[5,ip]
    
    qP = .5 * np.sqrt(uy**2+uz**2)/np.sqrt(ry**2+rz**2)
        
    if qS1<qS2:
        qS1, qS2 = qS2, qS1
    
    vel_e = [qP, qS1, qS2]
    
    return vel_e, seis_new, R


# -

# ##### Getting vertical and horizontal angles
#
# Pretty self-explanatory. Angles are for spherical coordinates and returned in radians. Given vector doesn't have to be unitized.

def get_angles(r):
    r = r / np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    theta = np.arccos(r[2])
    if theta == 0. or theta==np.pi:
        phi = 0
    else:
        if r[0]>0.:
            phi = np.arctan(r[1]/r[0])
        elif r[0]==0.:
            phi = r[1]/(abs(r[1]))*np.pi/2
        elif r[0]<0. and r[1]>= 0:
            phi = np.arctan(r[1]/r[0])+np.pi
        elif r[0]<0. and r[1]<0.:
            phi = np.arctan(r[1]/r[0])-np.pi
        else:
            print('Error in get_angles().')
    return theta,phi        


# #### Determine polarization of a three-component signal
#
# Definition Covariance matrix: 
#
# $
# Cov(i,j) = \int_{t_0}^{t_1} d_i d_j dt
# $
#
# d is the three-dimensional data, i and j are the respective directions and the integral is computed over a specific time interval. 
# The covariance matrix is a non-normalized correlation matrix, so the diagonal elements are auto-correlations and the off-diagonal elements cross-correlations. For discrete measurements, the integral is substituted by a sum.
#
# Per different signal, the covariance matrix will yield one more dimension (i.e. non-zero eigenvalue) and thus one more eigenvector that is equal to the polarizations of the signal. Without weighting, it's not possible to allocate polarizations to arrivals. Signals are different whenever they fulfill two requirements, different arrival times and different polarizations. 
#
# In an isotropic medium, shear waves are not split and thus two polarizations arrive at the same time. This results in a single signal 'felt' by the covariance matrix and the corresponding eigenvector will be a somewhat arbitrary vector inside the plane the two polarizations span. For the same reasoning, rotational measurements in isotropic setting yield just one of two possible rotational polarizations.
#
# Looking at rotational measurements for anisotropic media, there will be three distinct arrivals. But because rotational motions all happen inside the plane perpendicular to the propagation direction (assuming it is constant for all three wavefronts), the third polarization will not add a third dimension to the covariance matrix and thus the eigenproblem will yield a zero-eigenvalue.  
#

def get_polarizations(seis,mode):  
    nt = len(seis[0,:])    
    if mode=='nav': #Noise AVerage
        width = 10
        amp = np.sqrt(seis[0,:]**2+seis[1,:]**2+seis[2,:]**2)
        threshold = np.max(amp)*0.5
        ip = 0
        while amp[ip]<threshold:
            ip+=1
        nt = len(seis[0,:])    
        Cov = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                for it in range(ip,ip+width+1):
                    Cov[i,j] += 1/width * seis[i,it]*seis[j,it]
        w,Q = np.linalg.eigh(Cov, UPLO='U')
        loc = np.argsort(np.abs(w))[-1]
        n1 = Q[:,loc]            
        n1 *= 1/np.sqrt(np.sum(n1**2)) 

        is2 = nt-1
        while amp[is2]<threshold:
            is2 -= 1
        Cov = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                for it in range(is2-width,is2+1):
                    Cov[i,j] += 1/width * seis[i,it]*seis[j,it]
        w,Q = np.linalg.eigh(Cov, UPLO='U')
        loc = np.argsort(np.abs(w))[-1]
        n3 = Q[:,loc]            
        n3 *= 1/np.sqrt(np.sum(n3**2))

        n2 = np.cross(n3,n1)

        return [n1,n2,n3]


# #### Determine propagation direction with 6C-measurements
#
# In isotropic media, the propagation direction is simply the polarization of the first arriving wave. 
#
# In anisotropic media, this is not the case anymore. However, assuming that the propagation direction is constant for all arriving wavefields (qP, qS1, qS2), it's possible to take the cross-product of two non-parallel rotational polarizations to estimate the propagation direction.
#
# Quick derivation:
#
# $
# u \propto n
# $
#
# And from $\Omega = \frac{1}{2}\nabla\times u$ we get
#
# $
# \Omega \propto r = (\nu\times n)
# $
#
# where $\nu$ is the normalized direction of propagation ($\nu\parallel k$). Here we see $\Omega\perp\nu$ and $r\perp\nu$. 
#
# This orthogonality will be the case for each of the three wavefronts qP,qS1 and qS2. From this we get that the three rotational polarizations $r_1,r_2$ and $r_3$ all lie in the plane perpendicular to $\nu$. Straightforward, the propagation direction can be determined by
#
# $
# \nu = \frac{r_1\times r_2}{\vert\vert r_1\times r_2\vert\vert}
# $
#

# +
def get_propagation_direction(seis,f,fs,switchoption=False,control_fac=1.,turnoffp=False):
    if turnoffp:
        iqp,iqs1,_ = pick_arrivals(seis,f,fs)
        trace = seis[:,iqp+(iqs1-iqp)//2:]
    else:
        trace = seis
    
    n_rot, splitting = get_polarizations_rot(trace[3:,:],control_fac)
    switch = False
    if switchoption and not splitting:
        switch = True 
        return switch
    nr1,nr2,_ = n_rot
    nu_e = np.cross(nr1,nr2)
    if nu_e[2]<0.:
        nu_e = -nu_e
    return nu_e

def get_polarizations_rot(trace,control_fac):
    nt = len(trace[0,:])
    Cov = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            for it in range(0,nt):
                Cov[i][j] += 1/nt**2 * trace[i,it] * trace[j,it] * (nt-it)**3

    w,v = np.linalg.eig(Cov)
    for i in range(0,3):
        if w[i]<0.:
            w[i] = -w[i]
    l3 = np.argmax(w)
    l1 = np.argmin(w)
    for i in range(0,3):
        if l1!=i and l3!=i:
            l2 = i
    if w[l2]*control_fac < w[l1]:
        splitting = False
    else:
        splitting = True
    
            
    return [v[:,l3], v[:,l2],0], splitting


# -

# ### Extracting shear wave velocities
#
# Function is not in use anymore.

def extract_slower_velocities(vel):
    vel_s = []
    if vel[0]>vel[1] and vel[0]>vel[2]:
        if vel[1]>vel[2]:
            vel_s.extend([vel[1],vel[2]])
        else:
            vel_s.extend([vel[2],vel[1]])
    elif vel[1]>vel[0] and vel[1]>vel[2]:
        if vel[0] > vel[2]:
            vel_s.extend([vel[0],vel[2]])
        else:
            vel_s.extend([vel[2],vel[0]])
    else:
        if vel[0]>vel[1]:
            vel_s.extend([vel[0],vel[1]])
        else:
            vel_s.extend([vel[1],vel[0]])
    return vel_s 


# #### Isolate first peak and get it's amplitude on six-components
#
# For this to work, the P wave must arrive separately from the S waves. 
# The first time the translational wavefield expands over some threshold, the amplitudes of each component are given.
#
# It's possible to given the function translational and rotational measurements or the six strain components.

def amplitude_first_peak(seis):
    nt = len(seis[0,:])
    mode = np.zeros((4,nt))
    for i in range(0,3):
        eps = max(seis[i,:]) * 1e-3
        for it in range(0,nt):
            if abs(seis[i,it])>eps:
                mode[i+1,it] = 1
    for it in range(0,nt):
        if mode[1,it]==1 or mode[2,it]==1 or mode[3,it]==1:
            mode[0,it] = 1

    jt = 0
    it = 1
    while jt==0:
        if mode[0,it] != mode[0,it-1]:
            if mode[0,it]!=1:
                jt = it
        it +=1
    jt = np.argmax(seis[0,:jt]**2+seis[1,:jt]**2+seis[2,:jt]**2)    
    a0 = abs(seis[0,jt])
    a1 = abs(seis[1,jt])
    a2 = abs(seis[2,jt])
    a3 = abs(seis[3,jt])
    a4 = abs(seis[4,jt])
    a5 = abs(seis[5,jt])
        
    return a0, a1, a2, a3, a4, a5    


# 'Quick and dirty' way to plot translations and strain measurements 

def plotseis_strain(seis,t,nu):
    a1 = abs(seis[0:3][:].max())
    plt.figure(figsize=(8,4))
    plt.title('Accelerations x: '+str(round(nu[0],2))+ ' y: '+str(round(nu[1],2))+' z: '+str(round(nu[2],2)))
    plt.plot(t,seis[0,:],label='x')
    plt.plot(t,seis[1,:]-a1,label='y')
    plt.plot(t,seis[2,:]-2*a1,label='z')
    plt.legend()
    plt.yticks([])
    plt.show()
    
    a3 = seis[6:12][:].max()
    plt.figure(figsize=(8,4))
    plt.title('Strain rates')
    plt.plot(t,seis[6,:],label='e_xx')
    plt.plot(t,seis[7,:]-a3,label='e_yy')
    plt.plot(t,seis[8,:]-2*a3,label='e_zz')
    plt.plot(t,seis[9,:]-3*a3,label='e_xy')
    plt.plot(t,seis[10,:]-4*a3,label='e_xz')
    plt.plot(t,seis[11,:]-5*a3,label='e_yz')
    plt.legend(loc=3)
    plt.yticks([])
    plt.show()


# ### Getting a measurement for 12C
#
# Is applied for 7C-cases. All strain components are measured such that the applied noise-level depends on the incident angle.

def get_seis_strain(v, vel, nu, f, fs):
    xr = max(vel)*5.
    tmin = xr / max(vel) * 0.3
    tmax = xr / min(vel) * 1.3
    nt = int(((tmax-tmin) * fs)+1) 
    t = np.linspace(tmin,tmax,nt)
    seis = np.zeros((12,nt))   
    A = 1
    omega = 2*np.pi*f
    
    seis[0,:] = - v[0,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[0,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[0,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[1,:] = - v[1,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[1,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[1,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2) 
    seis[2,:] = - v[2,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[2,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[2,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2) 
    
    seis[3,:] =- np.cross(nu,v[:,0])[0]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2)\
            - np.cross(nu,v[:,1])[0]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu,v[:,2])[0]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[4,:] = - np.cross(nu,v[:,0])[1]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - np.cross(nu,v[:,1])[1]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu,v[:,2])[1]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[5,:] = - np.cross(nu,v[:,0])[2]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - np.cross(nu,v[:,1])[2]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu,v[:,2])[2]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    
    # xx
    seis[6,:] = - A * omega**2/vel[0]*v[0,0]*nu[0]*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/vel[1]*v[0,1]*nu[0]*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/vel[2]*v[0,2]*nu[0]*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    # yy
    seis[7,:] = - A * omega**2/vel[0]*v[1,0]*nu[1]*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/vel[1]*v[1,1]*nu[1]*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/vel[2]*v[1,2]*nu[1]*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    # zz 
    seis[8,:] = - A * omega**2/vel[0]*v[2,0]*nu[2]*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/vel[1]*v[2,1]*nu[2]*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/vel[2]*v[2,2]*nu[2]*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    # xy
    seis[9,:] = - A * omega**2/(2*vel[0])*(v[0,0]*nu[1]+v[1,0]*nu[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
               - A * omega**2/(2*vel[1])*(v[0,1]*nu[1]+v[1,1]*nu[0])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
               - A * omega**2/(2*vel[2])*(v[0,2]*nu[1]+v[1,2]*nu[0])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
     #xz 
    seis[10,:] = - A * omega**2/(2*vel[0])*(v[0,0]*nu[2]+v[2,0]*nu[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/(2*vel[1])*(v[0,1]*nu[2]+v[2,1]*nu[0])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/(2*vel[2])*(v[0,2]*nu[2]+v[2,2]*nu[0])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
     #yz 
    seis[11,:] = - A * omega**2/(2*vel[0])*(v[2,0]*nu[1]+v[1,0]*nu[2])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/(2*vel[1])*(v[2,1]*nu[1]+v[1,1]*nu[2])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/(2*vel[2])*(v[2,2]*nu[1]+v[1,2]*nu[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    
    return seis, t


# ### Rotation
#
# After rotation, the seismometer's x axis will be parallel to given $n$.

def rotate_seis_around_vector(seis,n):
    nt = len(seis[0,:])
    r2d = 180/np.pi
    theta, phi = get_angles(n)
    
    R = np.zeros((3,3))
    
    # Rodrigues' rotation formula, matrix will be rotated around vector r by angle pi 
    # x-axis=direction of propagation after rotation

    
    r = [(1. + np.cos(phi)*np.sin(theta))/2,\
         (np.sin(phi)*np.sin(theta))/2,\
         np.cos(theta)/2]
    r_sum = 0
    for i in range(0,3):
        r_sum += r[i]**2
    for i in range(0,3):
        r[i] = r[i]/np.sqrt(r_sum)
        
    K_sq = [[-r[1]**2-r[2]**2,r[0]*r[1],r[0]*r[2]], \
            [r[0]*r[1],-r[0]**2-r[2]**2,r[1]*r[2]], \
            [r[0]*r[2],r[1]*r[2],-r[0]**2-r[1]**2]]
    
    R[0,0] = 1 + 2*K_sq[0][0]
    R[1,0] = 2*K_sq[1][0]
    R[2,0] = 2*K_sq[2][0]

    R[0,1] = 2*K_sq[0][1]
    R[1,1] = 1 + 2*K_sq[1][1]
    R[2,1] = 2*K_sq[2][1]

    R[0,2] = 2*K_sq[0][2]
    R[1,2] = 2*K_sq[1][2]
    R[2,2] = 1 + 2*K_sq[2][2]
    
    seis_new = np.zeros((6,nt))
    
    for k in range(0,3):
        seis_new[k,:]   = R[k,0]*seis[0,:]+R[k,1]*seis[1,:]+R[k,2]*seis[2,:]
        seis_new[k+3,:] = R[k,0]*seis[3,:]+R[k,1]*seis[4,:]+R[k,2]*seis[5,:]
    
    return seis_new, R


# ### Rotate elastic tensor C.
#
# First, the elastic tensor in Voigt notation has to be translated into conventional $c_{ijkl}$.
# With rotation matrix R, the rotated elastic tensor $\tilde{c}$ is:
#
# $
# \tilde{c}_{ijkl} = R_{im}R_{jn}R_{ko}R_{lp}c_{mnop}
# $
#
# Rotation matrix R corresponds to plane. Angle is a in degrees. Returned elastic tensor in Voigt notation

# +
def rotate_C(C,plane,a):
    a = a * np.pi/180
    c = np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if i==j:
                        alpha = i
                    elif (i==1 and j==2) or (i==2 and j==1):
                        alpha = 3
                    elif (i==0 and j==2) or (i==2 and j==0):
                        alpha = 4
                    elif (i==1 and j==0) or (i==0 and j==1):
                        alpha = 5
                    if k==l:
                        beta = k
                    elif (k==1 and l==2) or (k==2 and l==1):
                        beta = 3
                    elif (k==0 and l==2) or (k==2 and l==0):
                        beta = 4
                    elif (k==1 and l==0) or (k==0 and l==1):
                        beta = 5
                    c[i,j,k,l] = C[alpha,beta]
    cr = rt(c,plane,a)
    C_new = np.zeros((6,6))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if i==j:
                        alpha = i
                    elif (i==1 and j==2) or (i==2 and j==1):
                        alpha = 3
                    elif (i==0 and j==2) or (i==2 and j==0):
                        alpha = 4
                    elif (i==1 and j==0) or (i==0 and j==1):
                        alpha = 5
                    if k==l:
                        beta = k
                    elif (k==1 and l==2) or (k==2 and l==1):
                        beta = 3
                    elif (k==0 and l==2) or (k==2 and l==0):
                        beta = 4
                    elif (k==1 and l==0) or (k==0 and l==1):
                        beta = 5
                    C_new[alpha,beta] = cr[i,j,k,l]
                    
    return C_new

def rt(c,plane,a):  ##Code from Heiner
    """
    Rotates a 4-th order tensor with angle a around plane = 1, 2, 3
    """
    
    r =  np.zeros([3,3])
    cr = np.zeros([3,3,3,3])

    if plane == 1:
    # Rotation around x
        r[0,0] = 1.
        r[1,1] = np.cos(a)
        r[1,2] = np.sin(a)
        r[2,1] =-np.sin(a)
        r[2,2] = np.cos(a)
    elif plane == 2:
    # Rotation around y
        r[1,1] = 1.
        r[0,0] = np.cos(a)
        r[0,2] = np.sin(a)
        r[2,0] =-np.sin(a)
        r[2,2] = np.cos(a)
        
    elif plane == 3:
    # Rotation around z
        r[2,2] = 1.
        r[0,0] = np.cos(a)
        r[1,0] = np.sin(a)
        r[0,1] =-np.sin(a)
        r[1,1] = np.cos(a)
        
    else:
        raise NotImplementedError
        
        
    # tensor rotation

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    sum = 0.
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    sum=sum+r[i,ii]*r[j,jj]*r[k,kk]*r[l,ll]*c[ii,jj,kk,ll]
                    cr[i,j,k,l]=sum
    
    
    return cr


# -

# Function that has been used to simulate an isolated arrival. Outdated

def get_seis_one_wavetype(mode,v,vel,nu,f,fs):
    
    l1 = np.argmax(vel)
    l3 = np.argmin(vel)
    for i in range(0,3):
        if l1!=i and l3!=i:
            l2 = i
    if mode=='qP':
        vel_0 = vel[l1]
        v_0 = v[:,l1]
    elif mode=='qS1':
        vel_0 = vel[l2]
        v_0 = v[:,l2]
    elif mode=='qS2':
        vel_0 = vel[l3]
        v_0 = v[:,l3]
    
    xr = max(vel)*5.
    tmin = xr / max(vel) * 0.3
    tmax = xr / min(vel) * 1.3
    nt = int(((tmax-tmin) * fs)+1) 
    t = np.linspace(tmin,tmax,nt)
    seis = np.zeros((7,nt))   
    A = 1
    omega = 2*np.pi*f
    vel = vel_0
    v = v_0
    
    seis[0,:] = - v[0]*A*omega**2*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2) 
    seis[1,:] = - v[1]*A*omega**2*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    seis[2,:] = - v[2]*A*omega**2*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
     
    
    seis[3,:] =- np.cross(nu,v)[0]*A*omega**2/(2*vel)*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    seis[4,:] =- np.cross(nu,v)[1]*A*omega**2/(2*vel)*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    seis[5,:] =- np.cross(nu,v)[2]*A*omega**2/(2*vel)*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    
    seis[6,:] = - A * omega**2/vel*v[2]*nu[2]*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2) 
    
    return seis, t


# 'Quick and dirty' way to plot 7C-seismogram

def plot_7C(seis,t):
    a1 = abs(seis[0:3][:].max())
    plt.figure(figsize=(8,4))
    plt.title('Translation')
    
    plt.plot(t,seis[0,:],label='x')
    plt.plot(t,seis[1,:]-a1,label='y')
    plt.plot(t,seis[2,:]-2*a1,label='z')
    plt.legend()
    plt.yticks([])
    plt.show()
    a2 = seis[3:6][:].max()
    plt.figure(figsize=(8,4))
    plt.title('Rotation')
    plt.plot(t,seis[3,:],label='rot_x')
    plt.plot(t,seis[4,:]-a2,label='rot_y')
    plt.plot(t,seis[5,:]-2*a2,label='rot_z')
    plt.legend()
    plt.yticks([])
    plt.show()
    
    plt.figure(figsize=(8,4))
    plt.title('Strain')
    plt.plot(t,seis[6,:],color='green',label='strain_zz')
    plt.legend()
    plt.yticks([])
    plt.show()


# ### Setting up d and g for the elastic tensor inversion.
#
# Given data vector $d$ and forward operator $g$ (both can be empty), new data and new rows for $g$ are added. 
# The Christoffelmatrix is calculated from polarizations (n_trans), estimated velocities (vel_e) and known density.
# Activated mode is 'triclinic'. Matrix g is extended with the estimated direction.

def add_new_data(mode,d,g,nu_e,vel_e,n_trans,density):
    d = list(d)
    g = list(g)
    QT = np.array(n_trans)
    Q = QT.transpose()
    if np.shape(Q)==(3,3):
        D = density * np.array([[vel_e[0]**2,0,0],[0,vel_e[1]**2,0],[0,0,vel_e[2]**2]])
        GAMMA = np.dot(np.dot(Q,D),QT)
        d.extend([GAMMA[0,0],GAMMA[1,1],GAMMA[2,2],GAMMA[0,1],GAMMA[0,2],GAMMA[1,2]])
        if mode=='triclinic':
            GT = np.array([    [nu_e[0]**2,0,0,0,0,0],\
                               [0,nu_e[1]**2,0,0,0,0],\
                               [0,0,nu_e[2]**2,0,0,0],\
                               [0,0,0,nu_e[0]*nu_e[1],0,0],\
                               [0,0,0,0,nu_e[0]*nu_e[2],0],\
                               [0,0,0,0,0,nu_e[1]*nu_e[2]],\
                               [0,0,0,nu_e[0]*nu_e[2],nu_e[0]*nu_e[1],0],\
                               [0,0,0,nu_e[1]*nu_e[2],0,nu_e[0]*nu_e[1]],\
                               [0,0,0,0,nu_e[1]*nu_e[2],nu_e[0]*nu_e[2]],\
                               [2*nu_e[0]*nu_e[2],0,0,0,nu_e[0]**2,0],\
                               [2*nu_e[0]*nu_e[1],0,0,nu_e[0]**2,0,0],\
                               [0,2*nu_e[1]*nu_e[2],0,0,0,nu_e[1]**2],\
                               [0,2*nu_e[0]*nu_e[1],0,nu_e[1]**2,0,0],\
                               [0,0,2*nu_e[1]*nu_e[2],0,0,nu_e[2]**2],\
                               [0,0,2*nu_e[0]*nu_e[2],0,nu_e[2]**2,0],\
                               [0,nu_e[2]**2,nu_e[1]**2,0,0,nu_e[1]*nu_e[2]],\
                               [nu_e[2]**2,0,nu_e[0]**2,0,nu_e[0]*nu_e[2],0],\
                               [nu_e[1]**2,nu_e[0]**2,0,nu_e[0]*nu_e[1],0,0],\
                               [0,0,2*nu_e[0]*nu_e[1],nu_e[2]**2,nu_e[1]*nu_e[2],nu_e[0]*nu_e[2]],\
                               [0,2*nu_e[0]*nu_e[2],0,nu_e[1]*nu_e[2],nu_e[1]**2,nu_e[0]*nu_e[1]],\
                               [2*nu_e[1]*nu_e[2],0,0,nu_e[0]*nu_e[2],nu_e[0]*nu_e[1],nu_e[0]**2]
                      ])
            G = GT.transpose()
        elif mode=='cubic':
            G = np.array([[nu_e[0]**2,0,nu_e[1]**2+nu_e[2]**2],\
                          [nu_e[1]**2,0,nu_e[0]**2+nu_e[2]**2],\
                          [nu_e[2]**2,0,nu_e[0]**2+nu_e[1]**2],\
                          [0,nu_e[0]*nu_e[1],nu_e[0]*nu_e[1]],\
                          [0,nu_e[0]*nu_e[2],nu_e[0]*nu_e[2]],\
                          [0,nu_e[1]*nu_e[2],nu_e[1]*nu_e[2]]
                         ])
        
        elif mode=='VTI':
            G = np.array([[nu_e[0]**2,0,0,nu_e[2]**2,nu_e[1]**2],\
                       [nu_e[1]**2,0,0,nu_e[2]**2,nu_e[0]**2],\
                       [0,nu_e[2]**2,0,nu_e[0]**2+nu_e[1]**2,0],\
                       [nu_e[0]*nu_e[1],0,0,0,-nu_e[0]*nu_e[1]],\
                       [0,0,nu_e[0]*nu_e[2],nu_e[0]*nu_e[2],0],\
                       [0,0,nu_e[1]*nu_e[2],nu_e[1]*nu_e[2],0]
                        ])
        elif mode=='tetragonal':
            G = np.array([[nu_e[0]**2,0,0,0,nu_e[2]**2,nu_e[1]**2],\
                          [nu_e[1]**2,0,0,0,nu_e[2]**2,nu_e[0]**2],\
                          [0,nu_e[2]**2,0,0,nu_e[0]**2+nu_e[1]**2,0],\
                          [0,0,nu_e[0]*nu_e[1],0,0,nu_e[0]*nu_e[1]],\
                          [0,0,0,nu_e[0]*nu_e[2],nu_e[0]*nu_e[2],0],\
                          [0,0,0,nu_e[1]*nu_e[2],nu_e[1]*nu_e[2],0]
                         ])
        elif mode=='trigonal':
            G = np.array([[nu_e[0]**2,0,0,2*nu_e[1]*nu_e[2],nu_e[2]**2,nu_e[1]**2],\
                          [nu_e[1]**2,0,0,-2*nu_e[1]*nu_e[2],nu_e[2]**2,nu_e[0]**2],\
                          [0,nu_e[2]**2,0,0,nu_e[0]**2+nu_e[1]**2,0],\
                          [nu_e[0]*nu_e[1],0,0,2*nu_e[0]*nu_e[2],0,-nu_e[0]*nu_e[1]],\
                          [0,0,nu_e[0]*nu_e[2],2*nu_e[0]*nu_e[1],nu_e[0]*nu_e[2],0],\
                          [0,0,nu_e[1]*nu_e[2],nu_e[0]**2-nu_e[1]**2,nu_e[1]*nu_e[2],0]\
                         ])
        elif mode=='orthorhombic':
            G = np.array([[nu_e[0]**2,0,0,0,0,0,0,nu_e[2]**2,nu_e[1]**2],\
                          [0,nu_e[1]**2,0,0,0,0,nu_e[2]**2,0,nu_e[0]**2],\
                          [0,0,nu_e[2]**2,0,0,0,nu_e[1]**2,nu_e[0]**2,0],\
                          [0,0,0,nu_e[0]*nu_e[1],0,0,0,0,nu_e[0]*nu_e[1]],\
                          [0,0,0,0,nu_e[0]*nu_e[2],0,0,nu_e[0]*nu_e[2],0],\
                          [0,0,0,0,0,nu_e[1]*nu_e[2],nu_e[1]*nu_e[2],0,0]\
                         ])
        g.extend(G)
    return np.array(d), np.array(g)


# Quick way to sort model vector entries $m_i$ to their corresponding elastic tensor component in Voigt notation. Activated mode is 'triclinic'. 

def sort_elastic_coeff(mode,m):
    C = np.zeros((6,6))
    if mode=='triclinic':
        C[0,0] = m[0] 
        C[1,1] = m[1]
        C[2,2] = m[2]
        C[0,1] = m[3]
        C[0,2] = m[4]
        C[1,2] = m[5]
        C[0,3] = m[6]
        C[1,4] = m[7]
        C[2,5] = m[8]
        C[0,4] = m[9]
        C[0,5] = m[10]
        C[1,3] = m[11]
        C[1,5] = m[12]
        C[2,3] = m[13]
        C[2,4] = m[14]
        C[3,3] = m[15]
        C[4,4] = m[16]
        C[5,5] = m[17]
        C[3,4] = m[18]
        C[3,5] = m[19]
        C[4,5] = m[20]
    elif mode=='cubic':
        C[0,0] = m[0]
        C[1,1] = m[0]
        C[2,2] = m[0]
        C[0,1] = m[1]
        C[0,2] = m[1]
        C[1,2] = m[1]
        C[3,3] = m[2]
        C[4,4] = m[2]
        C[5,5] = m[2]
    elif mode=='VTI':
        C[0,0] = m[0]
        C[1,1] = m[0]
        C[2,2] = m[1]
        C[0,2] = m[2]
        C[1,2] = m[2]
        C[3,3] = m[3]
        C[4,4] = m[3]
        C[5,5] = m[4]
        C[0,1] = C[1,1]-2*C[5,5]
    elif mode=='tetragonal':
        C[0,0] = m[0]
        C[1,1] = m[0]
        C[2,2] = m[1]
        C[0,1] = m[2]
        C[0,2] = m[3]
        C[1,2] = m[3]
        C[3,3] = m[4]
        C[4,4] = m[4]
        C[5,5] = m[5]   
    elif mode=='trigonal':
        C[0,0] = m[0]
        C[1,1] = m[0]
        C[2,2] = m[1]
        C[0,2] = m[2]
        C[1,2] = m[2]
        C[0,3] = m[3]
        C[1,3] = - C[0,3]
        C[3,3] = m[4]
        C[4,4] = m[4]
        C[4,5] = C[0,3]
        C[5,5] = m[5]
        C[0,1] = C[1,1]-2*C[5,5]
    elif mode=='orthorhombic':
        C[0,0] = m[0]
        C[1,1] = m[1]
        C[2,2] = m[2]
        C[0,1] = m[3]
        C[0,2] = m[4]
        C[1,2] = m[5]
        C[3,3] = m[6]
        C[4,4] = m[7]
        C[5,5] = m[8]
                       
    for i in range(0,6):
        for j in range(i,6):
            C[j,i] = C[i,j]      
    return C        


# Outdated code. 
#
# Was calculating the misfit between given measurements and simulated ones from the estimated elastic tensor C_e.

def get_misfit(seis,t,nu,C_e, f, density,fs):
    gammas_syn = get_gamma(nu, C_e)
    misfits = []
    for j in range(0,len(nu)):
        vel_syn, v_syn = get_eigenvals(gammas_syn[j], density)
        seis_syn, t_syn = get_seis(v_syn,vel_syn,nu[j],f,fs)
        misfit = 0
        nt = min(len(t_syn),len(t))
        #for i in range(0,6):
        #    plt.plot(t[j],seis[j][i,:])
        #    plt.plot(t_syn,seis_syn[i,:])
        #    plt.show()
        for k in range(0,6):
            amax = max(seis[j][k,:])
            for it in range(0,nt):
                misfit += ((seis_syn[k,it]-seis[j][k,it])/amax)**2 / nt
        misfits.append(misfit)        
    return misfits       


# Display the estimated tensor in a absolute value plot. Not used for the creation of thesis/paper-figures

def plot_tensor_estimation(C,C_e):
    param = ['C11','C22','C33','C12','C13','C23','C44','C55','C66','C45','C46','C56','C14','C15','C16','C24',\
         'C25','C26','C34','C35','C36']
    c_true = [C[0,0],C[1,1],C[2,2],C[0,1],C[0,2],C[1,2],C[3,3],C[4,4],C[5,5],C[3,4],C[3,5],C[4,5],C[0,3],C[0,4],\
              C[0,5],C[1,3],C[1,4],C[1,5],C[2,3],C[2,4],C[2,5]]
    c_est = [C_e[0,0],C_e[1,1],C_e[2,2],C_e[0,1],C_e[0,2],C_e[1,2],C_e[3,3],C_e[4,4],C_e[5,5],C_e[3,4],C_e[3,5],\
             C_e[4,5],C_e[0,3],C_e[0,4],C_e[0,5],C_e[1,3],C_e[1,4],C_e[1,5],C_e[2,3],C_e[2,4],C_e[2,5]]
    
    plt.figure(figsize=(15,10))
    plt.scatter(param,c_est,color='blue',marker='d',label='Est',s=70,edgecolors='black')
    plt.scatter(param,c_true,color='red',marker='x',label='True',s=80,edgecolors='black')
    plt.grid()
    plt.legend(loc=1)
    plt.show()


def plot_tensor_estimation2(C,C_e,C_e2):
    param = ['C11','C22','C33','C12','C13','C23','C44','C55','C66','C45','C46','C56','C14','C15','C16','C24',\
         'C25','C26','C34','C35','C36']
    c_true = [C[0,0],C[1,1],C[2,2],C[0,1],C[0,2],C[1,2],C[3,3],C[4,4],C[5,5],C[3,4],C[3,5],C[4,5],C[0,3],C[0,4],\
              C[0,5],C[1,3],C[1,4],C[1,5],C[2,3],C[2,4],C[2,5]]
    c_est = [C_e[0,0],C_e[1,1],C_e[2,2],C_e[0,1],C_e[0,2],C_e[1,2],C_e[3,3],C_e[4,4],C_e[5,5],C_e[3,4],C_e[3,5],\
             C_e[4,5],C_e[0,3],C_e[0,4],C_e[0,5],C_e[1,3],C_e[1,4],C_e[1,5],C_e[2,3],C_e[2,4],C_e[2,5]]
    c_est2 = [C_e2[0,0],C_e2[1,1],C_e2[2,2],C_e2[0,1],C_e2[0,2],C_e2[1,2],C_e2[3,3],C_e2[4,4],C_e2[5,5],C_e2[3,4],\
              C_e2[3,5],C_e2[4,5],C_e2[0,3],C_e2[0,4],C_e2[0,5],C_e2[1,3],C_e2[1,4],C_e2[1,5],C_e2[2,3],C_e2[2,4],\
              C_e2[2,5]]
    
    plt.figure(figsize=(15,10))
    plt.scatter(param,c_est,color='blue',marker='d',label='Est 7C',s=70,edgecolors='black')
    plt.scatter(param,c_est2,color='lime',label='Est 6C',s=70,edgecolors='black')
    plt.scatter(param,c_true,color='red',marker='x',label='True',s=80,edgecolors='black')
    plt.grid()
    plt.legend(loc=1)
    plt.show()


# ### Noise in the measurements
#
# This function allows the addition of noise to either 6 or 12 components (12 with strain). For 7C measurements only the zz-component will further be used in the code, however it's necessary to apply all strain data to the noise creation as otherwise the single strain trace will be perturbed independent of inclination.
#
# awgn = Average white gaussian noise.

# +
def make_some_noise(seis,SNR_dB_u,SNR_dB_r,SNR_dB_s=100.):
    c, nt = np.shape(seis)
    seis_noise = np.zeros((c,nt))
    seis_noise[0:3] = awgn(seis[0:3],SNR_dB_u)
    seis_noise[3:6] = awgn(seis[3:6],SNR_dB_r)
    if c>6:
        seis_noise[6:] = awgn(seis[6:],SNR_dB_s)
    return seis_noise

def awgn(x,SNR_dB):  #Code from Shihao
    L = x.size
    SNR = 10**(SNR_dB/10)                  # SNR to linear scale
    Esym = np.sum(abs(x)**2,axis=None)/(L) # calculate actual symbol energy
    N0 = Esym / SNR                        # find the noise spectral density
    noiseSigma = np.sqrt(N0)               # standard deviation for AWGN Noise
    n = noiseSigma * np.random.randn(x[:,0].size,x[0,:].size) #computed noise
    return x + n                           # received signal


# -

# Estimate velocities for 7C. qP-velocity will be estimated via the strain rate.
#
# $
# v_{qP} = \frac{\ddot u_z}{\epsilon_{zz}}\nu_z
# $

## Function only for noise-less case
def estimate_velocity_7C(seis, nu,t, min_vertical):
    nt = len(seis[0,:])
    r2d = 180/np.pi 
    d2r = 1/r2d
    theta, phi = get_angles(nu)
    R = np.zeros((3,3))
    r = [(1. + np.cos(phi)*np.sin(theta))/2,\
         (np.sin(phi)*np.sin(theta))/2,\
         np.cos(theta)/2]
    r_sum = 0
    for i in range(0,3):
        r_sum += r[i]**2
    for i in range(0,3):
        r[i] = r[i]/np.sqrt(r_sum)
        
    K_sq = [[-r[1]**2-r[2]**2,r[0]*r[1],r[0]*r[2]], \
            [r[0]*r[1],-r[0]**2-r[2]**2,r[1]*r[2]], \
            [r[0]*r[2],r[1]*r[2],-r[0]**2-r[1]**2]]
    
    R[0,0] = 1 + 2*K_sq[0][0]
    R[1,0] = 2*K_sq[1][0]
    R[2,0] = 2*K_sq[2][0]

    R[0,1] = 2*K_sq[0][1]
    R[1,1] = 1 + 2*K_sq[1][1]
    R[2,1] = 2*K_sq[2][1]

    R[0,2] = 2*K_sq[0][2]
    R[1,2] = 2*K_sq[1][2]
    R[2,2] = 1 + 2*K_sq[2][2]
    
    seis_new = np.zeros((6,nt))
    
    for k in range(0,3):
        seis_new[k,:]   = R[k,0]*seis[0,:]+R[k,1]*seis[1,:]+R[k,2]*seis[2,:]
        seis_new[k+3,:] = R[k,0]*seis[3,:]+R[k,1]*seis[4,:]+R[k,2]*seis[5,:]
    
    
    ip = np.argmax(seis_new[0,:])
    uz = seis[2,ip]
    ezz = seis[6,ip]
    if (theta*r2d) < min_vertical:
        qP = abs(uz*np.cos(theta) / ezz)
    else:
        uy = seis_new[1,ip]
        uz = seis_new[2,ip]
        ry = seis_new[4,ip]
        rz = seis_new[5,ip]

        qP = .5 * np.sqrt(uy**2+uz**2)/np.sqrt(ry**2+rz**2)
    
    nang = 721
    xc  = np.zeros(nang)
    ang = np.linspace(0,180,nang)
    d2r = r2d**-1

    for i in range(nang):
        angle = ang[i]
        xr = np.cos(angle*d2r)*seis_new[1,:] -  np.sin(angle*d2r)*seis_new[2,:]
        yr = np.sin(angle*d2r)*seis_new[1,:] +  np.cos(angle*d2r)*seis_new[2,:]
        junk = np.corrcoef(xr, yr)
        xc[i] = junk[1,0]
        ang[i] = angle
        
    imax = np.argmax(xc)
    imax2 = np.argmin(xc)
    if abs(imax)<abs(imax2):
        imax = imax2
    amax = ang[imax]
    
    angle = amax
    xr = np.cos(angle*d2r)*seis_new[1,:] -  np.sin(angle*d2r)*seis_new[2,:]
    yr = np.sin(angle*d2r)*seis_new[1,:] +  np.cos(angle*d2r)*seis_new[2,:]
    
    seis_new[1,:] = xr
    seis_new[2,:] = yr
    
    xrr = np.cos(angle*d2r)*seis_new[4,:] -  np.sin(angle*d2r)*seis_new[5,:]
    yrr = np.sin(angle*d2r)*seis_new[4,:] +  np.cos(angle*d2r)*seis_new[5,:]
    
    seis_new[4,:] = xrr
    seis_new[5,:] = yrr
    
    R2 = np.array([[1,0,0],[0,np.cos(angle*d2r),-np.sin(angle*d2r)],[0,np.sin(angle*d2r),np.cos(angle*d2r)]])
    R = np.dot(R,R2.transpose())
        
    j1 = np.argmax(xrr)
    j1c = np.argmin(xrr)
    j2 = np.argmax(yrr)
    j2c = np.argmin(yrr)
    qS1 = abs(yr[j1]/xrr[j1])/2
    qS2 = abs(xr[j2]/yrr[j2])/2
    
    safety = nt//100
    if abs(j1-j2)<safety or abs(j1-j2c)<safety or abs(j1c-j2)<safety:    # if picks are from same peak!
        angle = 45.
        xr = np.cos(angle*d2r)*seis_new[1,:] -  np.sin(angle*d2r)*seis_new[2,:]
        yr = np.sin(angle*d2r)*seis_new[1,:] +  np.cos(angle*d2r)*seis_new[2,:]

        seis_new[1,:] = xr
        seis_new[2,:] = yr

        xrr = np.cos(angle*d2r)*seis_new[4,:] -  np.sin(angle*d2r)*seis_new[5,:]
        yrr = np.sin(angle*d2r)*seis_new[4,:] +  np.cos(angle*d2r)*seis_new[5,:]

        seis_new[4,:] = xrr
        seis_new[5,:] = yrr

        R2 = np.array([[1,0,0],[0,np.cos(angle*d2r),-np.sin(angle*d2r)],[0,np.sin(angle*d2r),np.cos(angle*d2r)]])
        R = np.dot(R,R2.transpose())
        
        j1 = np.argmax(xrr)
        j2 = np.argmax(yrr)
        qS1 = abs(yr[j1]/xrr[j1])/2
        qS2 = abs(xr[j2]/yrr[j2])/2
    
        
    if qS1<qS2:
        qS1, qS2 = qS2, qS1
    
    vel_e = [qP, qS1, qS2]
    
    return vel_e, seis_new, R


# Get simulation of isolated arrival for 7C.

def get_seis_one_wavetype(mode,v,vel,nu,f,fs):
    
    l1 = np.argmax(vel)
    l3 = np.argmin(vel)
    for i in range(0,3):
        if l1!=i and l3!=i:
            l2 = i
    if mode=='qP':
        vel_0 = vel[l1]
        v_0 = v[:,l1]
    elif mode=='qS1':
        vel_0 = vel[l2]
        v_0 = v[:,l2]
    elif mode=='qS2':
        vel_0 = vel[l3]
        v_0 = v[:,l3]
    
    xr = max(vel)*5.
    tmin = xr / max(vel) * 0.3
    tmax = xr / min(vel) * 1.3
    nt = int(((tmax-tmin) * fs)+1) 
    t = np.linspace(tmin,tmax,nt)
    seis = np.zeros((7,nt))   
    A = 1
    omega = 2*np.pi*f
    vel = vel_0
    v = v_0
    
    seis[0,:] = - v[0]*A*omega**2*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2) 
    seis[1,:] = - v[1]*A*omega**2*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    seis[2,:] = - v[2]*A*omega**2*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
     
    
    seis[3,:] =- np.cross(nu,v)[0]*A*omega**2/(2*vel)*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    seis[4,:] =- np.cross(nu,v)[1]*A*omega**2/(2*vel)*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    seis[5,:] =- np.cross(nu,v)[2]*A*omega**2/(2*vel)*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2)
    
    seis[6,:] = - A * omega**2/vel*v[2]*nu[2]*(t-xr/vel)*np.exp(-(f**2)*(t-xr/vel)**2) 
    
    return seis, t


# Not used anymore. 
#
# Getting seismometer for different propagation directions. 

def get_seis_strain_diff_propdir(v,vel,nu,f,fs):    
    xr = max(vel)*5.
    tmin = xr / max(vel) * 0.3
    tmax = xr / min(vel) * 1.3
    nt = int(((tmax-tmin) * fs)+1) 
    t = np.linspace(tmin,tmax,nt)
    seis = np.zeros((12,nt))   
    A = 1
    omega = 2*np.pi*f
    ## here, nu is a 3x3 matrix
    seis[0,:] = - v[0,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[0,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[0,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[1,:] = - v[1,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[1,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[1,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2) 
    seis[2,:] = - v[2,0]*A*omega**2*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - v[2,1]*A*omega**2*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - v[2,2]*A*omega**2*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2) 
    
    seis[3,:] =- np.cross(nu[0],v[:,0])[0]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2)\
            - np.cross(nu[1],v[:,1])[0]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu[2],v[:,2])[0]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[4,:] = - np.cross(nu[0],v[:,0])[1]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - np.cross(nu[1],v[:,1])[1]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu[2],v[:,2])[1]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    seis[5,:] = - np.cross(nu[0],v[:,0])[2]*A*omega**2/(2*vel[0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
            - np.cross(nu[1],v[:,1])[2]*A*omega**2/(2*vel[1])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2) \
            - np.cross(nu[2],v[:,2])[2]*A*omega**2/(2*vel[2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    
    # xx
    seis[6,:] = - A * omega**2/vel[0]*v[0,0]*nu[0,0]*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/vel[1]*v[0,1]*nu[1,0]*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/vel[2]*v[0,2]*nu[2,0]*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    # yy
    seis[7,:] = - A * omega**2/vel[0]*v[1,0]*nu[0,1]*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/vel[1]*v[1,1]*nu[1,1]*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/vel[2]*v[1,2]*nu[2,1]*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    # zz 
    seis[8,:] = - A * omega**2/vel[0]*v[2,0]*nu[0,2]*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/vel[1]*v[2,1]*nu[1,2]*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/vel[2]*v[2,2]*nu[2,2]*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    # xy
    seis[9,:] = - A * omega**2/(2*vel[0])*(v[0,0]*nu[0,1]+v[1,0]*nu[0,0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
               - A * omega**2/(2*vel[1])*(v[0,1]*nu[1,1]+v[1,1]*nu[1,0])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
               - A * omega**2/(2*vel[2])*(v[0,2]*nu[2,1]+v[1,2]*nu[2,0])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
     #xz 
    seis[10,:] = - A * omega**2/(2*vel[0])*(v[0,0]*nu[0,2]+v[2,0]*nu[0,0])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/(2*vel[1])*(v[0,1]*nu[1,2]+v[2,1]*nu[1,0])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/(2*vel[2])*(v[0,2]*nu[2,2]+v[2,2]*nu[2,0])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
     #yz 
    seis[11,:] = - A * omega**2/(2*vel[0])*(v[2,0]*nu[0,1]+v[1,0]*nu[0,2])*(t-xr/vel[0])*np.exp(-(f**2)*(t-xr/vel[0])**2) \
                - A * omega**2/(2*vel[1])*(v[2,1]*nu[1,1]+v[1,1]*nu[1,2])*(t-xr/vel[1])*np.exp(-(f**2)*(t-xr/vel[1])**2)\
                - A * omega**2/(2*vel[2])*(v[2,2]*nu[2,1]+v[1,2]*nu[2,2])*(t-xr/vel[2])*np.exp(-(f**2)*(t-xr/vel[2])**2)
    
    return seis, t


# This function creates two plots from the same data. First, a 6x6 subplot where the upper right triangle-matrix shows for a histogram for multiple elastic tensor estimations. Gaussian curves are fitted with maximum-likelihood parameters (expected value and standard deviation).
#
# Second image plots the expected values of elastic parameters closer to each other. Errorbar is one standard deviation.

def triangularplot(C,C_e,nbins,width,Ninv,save,savename = ' '):
    fac = 1e-9 ##conversion to GPa
    width *= 1e9*fac
    Nplot = 1000

    param = [['C11','C12','C13','C14','C15','C16'],\
             ['C00','C22','C23','C24','C25','C26'],\
             ['C00','C00','C33','C34','C35','C36'],\
             ['C00','C00','C00','C44','C45','C46'],\
             ['C00','C00','C00','C00','C55','C56'],\
             ['C00','C00','C00','C00','C00','C66']]

    fig, ax = plt.subplots(ncols=6,nrows=6, figsize=(13,13),sharey=True)

    # get rid of potential 'nan' in estimations
    i1, i2, i3 = np.shape(C_e)
    C_e_stripped = []
    for i in range(i1):
        c = 0
        for j in range(i2):
            for k in range(i3):
                if not np.isnan(C_e[i,j,k]) and not C_e[i,j,k]==0.0: c+=1
        if c==36:
            C_e_stripped.append(C_e[i,:,:])
    print(str(Ninv - np.shape(C_e_stripped)[0])+' estimates stripped')
    C_e_stripped = np.array(C_e_stripped)
    mu = np.zeros((6,6))
    sig = np.zeros((6,6))
    for i in range(0,6):
        for j in range(i,6):
            n = len(C_e_stripped[:,i,j])
            mu[i,j] = 1/n * np.sum(C_e_stripped[:,i,j]*fac) 
            sig[i,j] = 1/n * np.sum((C_e_stripped[:,i,j]*fac-mu[i,j])**2)

    for i in range(0,6):
        for j in range(0,6):
            if i<=j:
                hist_range = [C[i,j]*fac-width,C[i,j]*fac+width]
                x = np.linspace(hist_range[0],hist_range[1],Nplot)
                y = 1/np.sqrt(2*np.pi*sig[i,j])*np.e**(-1/(2*sig[i,j])*(x-mu[i,j])**2)
                ax[i,j].hist(C_e_stripped[:,i,j]*fac,bins=nbins,color='darkorange',density=True, range=hist_range)
                ax[i,j].set_yticks([])
                ax[i,j].tick_params(labelsize=13)
                ax[i,j].axvline(C[i,j]*fac,linestyle='dotted',c='black')
                ax[i,j].axvline(mu[i,j],linestyle='dashed',c='black')
                ax[i,j].plot(x,y,color='black',linestyle='dashed',linewidth=1.2)
                ax[i,j].text(0.03,0.8,param[i][j],transform = ax[i,j].transAxes,fontsize=15)
            else:
                ax[i,j].set_axis_off()
                if i==5 and j==0:
                    ax[i,j].text(0.9,0.9,'sym.',weight='bold',transform = ax[i,j].transAxes,fontsize=25) 
    
    if save:
        plt.savefig(savename+'_tri.png')
    plt.plot()
    
    param = ['C11','C22','C33','C12','C13','C23','C44','C55','C66','C45','C46','C56','C14','C15','C16','C24',\
         'C25','C26','C34','C35','C36']
    c_true = np.array([C[0,0],C[1,1],C[2,2],C[0,1],C[0,2],C[1,2],C[3,3],C[4,4],C[5,5],C[3,4],C[3,5],C[4,5],C[0,3],C[0,4],\
              C[0,5],C[1,3],C[1,4],C[1,5],C[2,3],C[2,4],C[2,5]])*1e-9
    c_est = np.array([mu[0,0],mu[1,1],mu[2,2],mu[0,1],mu[0,2],mu[1,2],mu[3,3],mu[4,4],mu[5,5],mu[3,4],\
             mu[3,5],mu[4,5],mu[0,3],mu[0,4],mu[0,5],mu[1,3],mu[1,4],mu[1,5],mu[2,3],mu[2,4],\
             mu[2,5]])
    c_err = np.sqrt(np.array([sig[0,0],sig[1,1],sig[2,2],sig[0,1],sig[0,2],sig[1,2],sig[3,3],sig[4,4],sig[5,5],sig[3,4],\
             sig[3,5],sig[4,5],sig[0,3],sig[0,4],sig[0,5],sig[1,3],sig[1,4],sig[1,5],sig[2,3],sig[2,4],\
             sig[2,5]]))

    plt.figure(figsize=(13,8))
    plt.title('Taylor Sandstone',fontsize=20)
    plt.errorbar(param,c_est,yerr=c_err,label='Estimation',color='blue',fmt='o',ms=8,capsize=5,mfc='blue',mec='black')
    plt.scatter(param,c_true,color='red',marker='X',label='True',s=110,edgecolors='black')
    plt.grid(axis='y')
    plt.xlabel('Elastic parameter',fontsize=15)
    plt.ylabel('Value [GPa]',fontsize=15)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.legend(loc=1,fontsize=20)
    if save:
        plt.savefig(savename+'_err.png')
    plt.show()


# ### Performs the bedding determination for an estimated elastic tensor of VTI symmetry.
#
# Concept from Abramian(2019). Difficulty is having a rotated hexagonal system that also has noise. Noise is basically ignored in this setup.

# +
def getrotationVTI(C):
    c = np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if i==j:
                        alpha = i
                    elif (i==1 and j==2) or (i==2 and j==1):
                        alpha = 3
                    elif (i==0 and j==2) or (i==2 and j==0):
                        alpha = 4
                    elif (i==1 and j==0) or (i==0 and j==1):
                        alpha = 5
                    if k==l:
                        beta = k
                    elif (k==1 and l==2) or (k==2 and l==1):
                        beta = 3
                    elif (k==0 and l==2) or (k==2 and l==0):
                        beta = 4
                    elif (k==1 and l==0) or (k==0 and l==1):
                        beta = 5
                    c[i,j,k,l] = C[alpha,beta]
                    
    
    v = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                v[i,j] += c[k,i,k,j]
    vprime = v - 1./3 * np.trace(v)*np.identity(3)      

    w2,v2 = np.linalg.eigh(vprime)

    l = np.trace(vprime**3)/np.trace(vprime**2)*2

    ind2 = np.argmin([abs(w2[0]-l),abs(w2[1]-l),abs(w2[2]-l)])
    if ind2 == 0:
        ind21,ind22 = 1,2
    elif ind2 == 1:
        ind21,ind22 = 2,0
    else:
        ind21,ind22 = 0,1
        
    u2 = np.array([v2[:,ind22],v2[:,ind21],v2[:,ind2]])

    e = np.identity(3) 
    g2 = np.zeros((3,3))
    g21 = np.zeros((3,3))
    g22 = np.zeros((3,3))
    g23 = np.zeros((3,3))
    g24 = np.zeros((3,3))
    g25 = np.zeros((3,3))
   
    for i in range(0,3):
        for j in range(0,3):
            g2[i,j] = np.dot(u2[i,:],e[j,:])
    c2 = rotate_tensor_arbitrary_rot(c,g2)

    u2 = np.array([v2[:,ind21],v2[:,ind2],v2[:,ind22]])
    for i in range(0,3):
        for j in range(0,3):
            g21[i,j] = np.dot(u2[i,:],e[j,:])
    c21 = rotate_tensor_arbitrary_rot(c,g21)
    u2 = np.array([v2[:,ind2],v2[:,ind21],v2[:,ind22]])
    for i in range(0,3):
        for j in range(0,3):
            g22[i,j] = np.dot(u2[i,:],e[j,:])
    c22 = rotate_tensor_arbitrary_rot(c,g22)
    u2 = np.array([v2[:,ind21],v2[:,ind22],v2[:,ind2]])
    for i in range(0,3):
        for j in range(0,3):
            g23[i,j] = np.dot(u2[i,:],e[j,:])
    c23 = rotate_tensor_arbitrary_rot(c,g23)  
    
    u2 = np.array([v2[:,ind22],v2[:,ind2],v2[:,ind21]])
    for i in range(0,3):
        for j in range(0,3):
            g24[i,j] = np.dot(u2[i,:],e[j,:])        
    c24 = rotate_tensor_arbitrary_rot(c,g24)
    
    u2 = np.array([v2[:,ind2],v2[:,ind22],v2[:,ind21]])
    for i in range(0,3):
        for j in range(0,3):
            g25[i,j] = np.dot(u2[i,:],e[j,:])        
    c25 = rotate_tensor_arbitrary_rot(c,g25)
    
    i = np.argmin([abs(c2[1,2,1,2]-c2[0,2,0,2]),abs(c21[1,2,1,2]-c21[0,2,0,2]),abs(c22[1,2,1,2]-c22[0,2,0,2]),
                   abs(c23[1,2,1,2]-c23[0,2,0,2]),abs(c24[1,2,1,2]-c24[0,2,0,2]),abs(c25[1,2,1,2]-c25[0,2,0,2])])
    if i==1:
        c2 = c21
        g2 = g21
    elif i==2:
        c2 = c22
        g2 = g22
    elif i==3:
        c2 = c23
        g2 = g23
    elif i==4:
        c2 = c24
        g2 = g24
    elif i==5:
        c2 = c25
        g2 = g25    
        
    C2 = np.zeros((6,6))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if i==j:
                        alpha = i
                    elif (i==1 and j==2) or (i==2 and j==1):
                        alpha = 3
                    elif (i==0 and j==2) or (i==2 and j==0):
                        alpha = 4
                    elif (i==1 and j==0) or (i==0 and j==1):
                        alpha = 5
                    if k==l:
                        beta = k
                    elif (k==1 and l==2) or (k==2 and l==1):
                        beta = 3
                    elif (k==0 and l==2) or (k==2 and l==0):
                        beta = 4
                    elif (k==1 and l==0) or (k==0 and l==1):
                        beta = 5
                    C2[alpha,beta] = c2[i,j,k,l]
    return C2, g2

def tensor_product(a,b):
    c1,c2 = np.shape(a)
    c3,c4 = np.shape(b)
    z = np.zeros((c1,c2,c3,c4))
    for i in range(0,c1):
        for j in range(0,c2):
            for k in range(0,c3):
                for l in range(0,c4):
                    z[i,j,k,l] = a[i,j]*b[k,l]
    return z

def tensor_product_bar(a,b):
    c1,c2 = np.shape(a)
    c3,c4 = np.shape(b)
    z = np.zeros((c1,c2,c3,c4))
    for i in range(0,c1):
        for j in range(0,c2):
            for k in range(0,c3):
                for l in range(0,c4):
                    z[i,j,k,l] = 1./2 * (a[i,k]*b[j,l]+a[i,l]*b[j,k]) 
    return z

def tensor_norm(a):
    c1,c2,c3,c4 = np.shape(a)
    z = 0
    for i in range(0,c1):
        for j in range(0,c2):
            for k in range(0,c3):
                for l in range(0,c4):
                    z += a[i,j,k,l]*a[i,j,k,l]
    return z     

def rotate_tensor_arbitrary_rot(c,r):
    cnew = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    sum = 0.
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    sum=sum+r[i,ii]*r[j,jj]*r[k,kk]*r[l,ll]*c[ii,jj,kk,ll]
                    cnew[i,j,k,l]=sum
    return cnew


# -

# Convenient function to quickly get arrival times of three wavefronts.

def pick_arrivals(seis,f,fs):
    nt = len(seis[0,:])
    hw = int(10*fs/100.*15./f)
    amp = np.sqrt(seis[0,:]**2+seis[1,:]**2+seis[2,:]**2)
    rot = np.sqrt(seis[3,:]**2+seis[4,:]**2+seis[5,:]**2)
    threshold = max(amp)*0.4
    ip = 0
    while amp[ip]<threshold:
        ip += 1
    ip += np.argmax(amp[ip:ip+hw]) 
    
    is1 = ip + nt//10
    while rot[is1]<max(rot)*0.4:
        is1 += 1
    is1 += np.argmax(rot[is1:is1+hw])     
    
    is2 = nt-1
    while rot[is2]<max(rot)*0.4:
        is2 -= 1    
    is2 -= np.argmax(rot[is2-hw:is2])        
    return ip,is1,is2  


def pick_arrivals_3D(seis,f,fs):
    nt = len(seis[0,:])
    hw = int(10*fs/100.*15./f)
    amp = np.sqrt(seis[0,:]**2+seis[1,:]**2+seis[2,:]**2)
    threshold = max(amp)*0.4
    ip = 0
    while amp[ip]<threshold:
        ip += 1
    ip += np.argmax(amp[ip:ip+hw]) 
    
    is1 = ip + nt//10
    while amp[is1]<max(amp)*0.4:
        is1 += 1
    is1 += np.argmax(amp[is1:is1+hw])     
    
    is2 = nt-1
    while amp[is2]<max(amp)*0.4:
        is2 -= 1    
    is2 -= np.argmax(amp[is2-hw:is2])        
    return ip,is1,is2


# ### Estimate velocities with noise
#
# Separates between 6C and 7C cases. Seismometer is rotated into propagation direction and y and z are positioned such that phases are split. The 'safety' is there because sometimes the splitting is off by 45 degrees, which will be repaired by another rotation. Only ratios with weights over a certain threshold (default: 1 magnitude over median) are considered. Velocities are estimated with a weighted average over said ratios. Velocities and Weights are both returned.

def estimate_velocities_noise(seis,nu_e,f,fs):
    nt = len(seis[0,:])
    if len(seis[:,0])==6:
        ests = np.zeros((3,2))     # 0 qProt, 1 qS1rot, 2 qS2rot
                                   # 0 absolute velocity estimation, 1 corresponding weight
            
        seis,_ = rotate_seis_around_vector(seis,nu_e)
        
        iqp,iqs1,iqs2 = pick_arrivals(seis,f,fs)
        nang = 360
        xc  = np.zeros(nang)
        ang = np.linspace(-45,45,nang)
        d2r = np.pi/180

        for ii in range(nang):
            angle = ang[ii]
            xr = np.cos(angle*d2r)*seis[4,:] -  np.sin(angle*d2r)*seis[5,:]
            yr = np.sin(angle*d2r)*seis[4,:] +  np.cos(angle*d2r)*seis[5,:]
            junk = np.corrcoef(xr, yr)
            xc[ii] = junk[1,0]
            ang[ii] = angle
        
        imax = np.argmax(xc)

        angle = ang[imax]+90.
        seis[1,:],seis[2,:] = np.cos(angle*d2r)*seis[1,:] -  np.sin(angle*d2r)*seis[2,:],\
                              np.sin(angle*d2r)*seis[1,:] +  np.cos(angle*d2r)*seis[2,:]
        
        seis[4,:],seis[5,:] = np.cos(angle*d2r)*seis[4,:] -  np.sin(angle*d2r)*seis[5,:],\
                                  np.sin(angle*d2r)*seis[4,:] +  np.cos(angle*d2r)*seis[5,:]
          
        rotamp = np.max(np.sqrt(seis[4,:]**2+seis[5,:]**2))
        h = 0.2
        hw = 3
        if (np.max(np.abs(seis[4,iqs1-hw:iqs1+hw]))>h*rotamp and np.max(np.abs(seis[5,iqs1-hw:iqs1+hw]))>h*rotamp)\
            or (np.max(np.abs(seis[4,iqs2-hw:iqs2+hw]))>h*rotamp and np.max(np.abs(seis[5,iqs2-hw:iqs2+hw]))>h*rotamp):
            angle = 45.
            seis[1,:],seis[2,:] = np.cos(angle*d2r)*seis[1,:] -  np.sin(angle*d2r)*seis[2,:],\
                                  np.sin(angle*d2r)*seis[1,:] +  np.cos(angle*d2r)*seis[2,:]
            seis[4,:],seis[5,:] = np.cos(angle*d2r)*seis[4,:] -  np.sin(angle*d2r)*seis[5,:],\
                                  np.sin(angle*d2r)*seis[4,:] +  np.cos(angle*d2r)*seis[5,:]
                          
        vel_e = 1/2. * np.sqrt(seis[1,:]**2+seis[2,:]**2)/np.sqrt(seis[4,:]**2+seis[5,:]**2)
        weight = np.sqrt(seis[1,:]**2+seis[2,:]**2) * np.sqrt(seis[4,:]**2+seis[5,:]**2)
        
        wm = np.median(weight)
        velbins = [[],[],[]]
        weightbins = [[],[],[]]
        for it in range(nt):
            if weight[it]>wm*10.**(1.):  # 1 order of magnitude above median                    
                ivel = np.argmin([abs(it-iqp),abs(it-iqs1),abs(it-iqs2)])
                velbins[ivel].append(vel_e[it])
                weightbins[ivel].append(weight[it])
                
        for j in range(0,3):
            if len(velbins[j])!=0:
                w = np.sum(weightbins[j])
                for jj in range(0,len(velbins[j])):
                    ests[j,0] += 1./w * weightbins[j][jj] * velbins[j][jj]
                ests[j,1] = w  
         
        return ests
        
    
    elif len(seis[:,0])==7:
        
        ests = np.zeros((6,2))     # 0 qProt, 1 qS1rot, 2 qS2rot, 3 qPstrain, 4 qS1strain, 5 qS2strain
                                   # 0 absolute velocity estimation, 1 corresponding weight
        
        vel_e_strain = seis[2,:] / seis[6,:] * nu_e[2]
        weight_strain = (seis[2,:]**2 * seis[6,:]**2 * nu_e[2]**2)**(1/4)
        strain_trace = seis[6,:]
        seis,_ = rotate_seis_around_vector(seis,nu_e)
        
        nang = 360
        xc  = np.zeros(nang)
        ang = np.linspace(0,180,nang)
        d2r = np.pi/180

        for ii in range(nang):
            angle = ang[ii]
            xr = np.cos(angle*d2r)*seis[1,:] -  np.sin(angle*d2r)*seis[2,:]
            yr = np.sin(angle*d2r)*seis[1,:] +  np.cos(angle*d2r)*seis[2,:]
            junk = np.corrcoef(xr, yr)
            xc[ii] = junk[1,0]
            ang[ii] = angle
        
        imax = np.argmax(xc)
        imax2 = np.argmin(xc)
        if abs(imax)<abs(imax2):
            imax = imax2
        amax = ang[imax] + 45.

        angle = amax
        seis[1,:],seis[2,:] = np.cos(angle*d2r)*seis[1,:] -  np.sin(angle*d2r)*seis[2,:],\
                              np.sin(angle*d2r)*seis[1,:] +  np.cos(angle*d2r)*seis[2,:]
        
        iqp,iqs1,iqs2 = pick_arrivals(seis,f,fs)
        
        safety = 10
        
        if abs(iqs1-iqs2)<safety:
            angle = 45.
            seis[1,:],seis[2,:] = np.cos(angle*d2r)*seis[1,:] -  np.sin(angle*d2r)*seis[2,:],\
                                  np.sin(angle*d2r)*seis[1,:] +  np.cos(angle*d2r)*seis[2,:]

            iqp,iqs1,iqs2 = pick_arrivals(seis,f,fs)
                          
        vel_e = 1/2. * np.sqrt(seis[1,:]**2+seis[2,:]**2)/np.sqrt(seis[4,:]**2+seis[5,:]**2)
        weight = np.sqrt(seis[1,:]**2+seis[2,:]**2) * np.sqrt(seis[4,:]**2+seis[5,:]**2)
        
        wm = np.median(weight)
        wm_strain = np.median(weight_strain)
        velbins = [[],[],[]]
        weightbins = [[],[],[]]
        velbins_strain = [[],[],[]]
        weightbins_strain = [[],[],[]]
        for it in range(nt):
            if weight[it]>wm*10.**(1.):  # 1 order of magnitude above median                    
                ivel = np.argmin([abs(it-iqp),abs(it-iqs1),abs(it-iqs2)])
                velbins[ivel].append(vel_e[it])
                weightbins[ivel].append(weight[it])
            if weight_strain[it]>wm_strain*10.**(1.):  # 1 order of magnitude above median
                ivel = np.argmin([abs(it-iqp),abs(it-iqs1),abs(it-iqs2)])
                velbins_strain[ivel].append(vel_e_strain[it])
                weightbins_strain[ivel].append(weight_strain[it])
                
        for j in range(0,3):
            if len(velbins[j])!=0:
                w = np.sum(weightbins[j])
                for jj in range(0,len(velbins[j])):
                    ests[j,0] += 1./w * weightbins[j][jj] * velbins[j][jj]
                ests[j,1] = w 
        for j in range(0,3):
            if len(velbins_strain[j])!=0:
                w = np.sum(weightbins_strain[j])
                for jj in range(0,len(velbins_strain[j])):
                    ests[j+3,0] += 1./w * weightbins_strain[j][jj] * velbins_strain[j][jj]
                ests[j+3,1] = w  
         
        return ests
    
    else:
        print('ERROR in estimate_velocities_noise')


# Convenient function to get linearly scaled thomsen parameters.

def get_thomsen_linear(Nc):
    eps = np.zeros((8,Nc))
    delta = np.zeros((8,Nc))
    gamma = np.zeros((8,Nc))
    eps[1,:] = np.linspace(0,0.11,Nc)
    delta[1,:] = np.linspace(0,-0.035,Nc)
    gamma[1,:] = np.linspace(0,0.255,Nc)
    eps[2,:] = np.linspace(0,0.334,Nc)
    delta[2,:] = np.linspace(0,0.73,Nc)
    gamma[2,:] = np.linspace(0,0.575,Nc)
    eps[3,:] = np.linspace(0,0.091,Nc)
    delta[3,:] = np.linspace(0,0.565,Nc)
    gamma[3,:] = np.linspace(0,0.046,Nc)
    eps[4,:] = np.linspace(0,0.01,Nc)
    delta[4,:] = np.linspace(0,0.012,Nc)
    gamma[4,:] = np.linspace(0,-0.005,Nc)
    eps[5,:] = np.linspace(0,0.0,Nc)
    delta[5,:] = np.linspace(0,-0.264,Nc)
    gamma[5,:] = np.linspace(0,-0.007,Nc)
    eps[6,:] = np.linspace(0,-0.096,Nc)
    delta[6,:] = np.linspace(0,0.273,Nc)
    gamma[6,:] = np.linspace(0,-0.159,Nc)
    eps[7,:] = np.linspace(0,0.42,Nc)
    delta[7,:] = np.linspace(0,0.21,Nc)
    gamma[7,:] = np.linspace(0,0.36,Nc)
    return eps, delta, gamma
