import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bplanck_SI as bp
import read_ck as rck
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('iterations', metavar='iterations', type=int, nargs='+',
                    help='Set the number of iterations for the integrations')
args = parser.parse_args()

# 1. spaces ....................................................................
"""
The grid runs downwards, from 20 km to 0 km.
"""
data                = np.loadtxt('earth_atmosphere_model.txt')
opac                = rck.InterpolatedCorrKapp('mix')
sun_spectrum        = bp.bplanck(opac.nu_hz, 5780.1) * 0.000068
z                   = np.flip(data[0:21, 0])
rho                 = np.flip(data[0:21, 3])
cos_theta           = np.linspace(-1, 1, 20)
cos_theta_sun       = 0.5
nz                  = len(z)
nmu                 = len(cos_theta)
dz                  = z[0]-z[1]
dmu                 = 2./nmu
F                   = np.zeros(nz) #stellar flux
I                   = np.zeros((nz, nmu)) #intensity matrix
tau                 = np.zeros(nz)
dtau_between_layers = np.zeros(nz-1)

# 2. calculate optical depth and stellar flux ..................................

dtau_between_layers[10:16]=25.*0.0001*0.5*(rho[11:17]+rho[10:16])*dz

for i in range(1, nz):
    """
    Calculates the total optical depth.
    """
    tau[i]=tau[i-1]+dtau_between_layers[i-1]
    
"""
Calculate the stellar flux as it passes through the atmosphere. We use the
frequency-integrated intensity of the sun in the visual range (~380-740 nm).
"""
F[0] = np.sum(sun_spectrum[77:120]*opac.dnu[77:120])
for i in range(1, nz):
    F[i]=F[i-1]*np.exp(-dtau_between_layers[i-1]/cos_theta_sun)

# 3. Helper functions ..........................................................

def J():
    """
    This function calculates the mean intensity.
    """
    J = np.copy(F)
    for i in range(nz):
        J[i]+=0.5*dmu*np.sum(I[i, :])
    return J
    
    
def integrated_FTR(I, J, z, dtau, direction):
    """
    This function integrates the radiative transfer equation.
    direction = + 1 means down, direction = - 1 means up.
    """
    x     = I * np.exp(-dtau)
    xp    = x + (1.-(1.+dtau)*np.exp(-dtau))/dtau * J[z-direction]
    xxp   = xp + (dtau -1.+np.exp(-dtau))/dtau * J[z]
    return xxp

def piecewise_integration(J):
    """
    This function performs the FTR integration at each height z.
    """
    for i in range(nz-1):
        """ Downwards """
        for j in range(int(nmu/2)):
            dtau = abs(dtau_between_layers[i]/cos_theta[j])
            if(dtau==0):
                I[i+1, j]=I[i, j]
            else:
                I[i+1, j]=integrated_FTR(I[i, j], J, i+1, dtau, direction=+1)
        """ Upwards """
        for l in range(int(nmu/2), nmu):
            dtau = abs(dtau_between_layers[nz-2-i]/cos_theta[l])
            if (dtau==0):
                I[nz-2-i, l]=I[nz-1-i, l]
            else:
                I[nz-2-i, l]=integrated_FTR(I[nz-1-i, l], J, nz-2-i, dtau, direction=-1)

# 3. Iterate ...................................................................

for i in range(args.iterations[0]):
    piecewise_integration(J())
    
# 4. Plot ......................................................................

print('Ratio of mean intensities for cloud and no cloud cover: '+str(J()[-1]/F[0]))

fig    = plt.figure(figsize=(8,8), constrained_layout=True)
gs     = fig.add_gridspec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[2,1])
axs0   = fig.add_subplot(gs[0,0])
axs1   = fig.add_subplot(gs[1,0])

axs0.plot(z, J(), label='mean intensity')
axs0.plot(z, F, label='stellar flux')
axs0.plot(z, F[0]*np.ones(nz), color='orange', linestyle=':')

axs0.set_title(r'Mean intensity $J(z)$')
axs0.set_xlabel(r'z [m]')
axs0.set_ylabel(r'Mean intensity $\ J(z)\ [W/m^2]$')

axs0.add_patch(patches.Rectangle((4000, 0), 6000, 2000,
                                 alpha=0.3, facecolor="gray", edgecolor="none",
                                 linewidth=0, linestyle='solid'))
axs0.annotate('Cloud layer', xy=(5200, 900), color='white')
axs0.annotate('Intensity decrease: '+str(round(100*(1.-J()[-1]/F[0]), 2))+' %',
              xy=(12000, 200), color='black')
axs0.annotate('Iterations: '+str(args.iterations[0]),
              xy=(12000, 150), color='black')
axs0.legend()

axs1.plot(z, tau)
axs1.set_ylim(0, 10)
axs1.set_title(r'Optical depth $\tau(z)$')
axs1.set_xlabel(r'z [m]')

axs1.add_patch(patches.Rectangle((4000, 0), 6000, 12,
                                 alpha=0.3, facecolor="gray", edgecolor="none",
                                 linewidth=0, linestyle='solid'))
axs1.annotate('Cloud layer', xy=(5200, 9), color='white')

plt.show()


