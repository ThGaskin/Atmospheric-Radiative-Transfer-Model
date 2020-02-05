import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import read_ck as rck
import bplanck_SI as bp
import sys
np.set_printoptions(threshold=sys.maxsize)

# 1. Spaces and matrices .......................................................

"""
The space is pointing upwards, from z=0m to z=60km. Hence, mu = -1/sqrt(3)
represents the upward direction, mu = +1/sqrt(3) represents the downward direction.
"""

data                 = np.loadtxt('earth_atmosphere_model.txt')
opac                 = rck.InterpolatedCorrKapp('mix')
z_ax                 = data[:, 0]
T                    = data[:, 1]
P                    = data[:, 2]
cos_theta            = [-1./np.sqrt(3), +1./np.sqrt(3)]
dz                   = z_ax[1]-z_ax[0]
nz                   = len(z_ax)
nnu                  = len(opac.nu_hz)
ng                   = len(opac.g_weights)
nmu                  = len(cos_theta)
cos_theta_sun        = 0.5
ggrav                = 9.81
B_atmosphere         = np.zeros((nz, nnu))
kappa                = opac.kappa * 0.1 # conversion to SI
kappa_between_layers = 0.5 * ( kappa[1:,:,:] + kappa[:-1,:,:] )
dtau_between_layers  = np.zeros_like(kappa_between_layers)
dP_between_layers    = P[:-1]-P[1:]
sun_spectrum         = bp.bplanck(opac.nu_hz, 5780.1) * opac.nu_hz * 0.000068 # ster
earth_spectrum       = bp.bplanck(opac.nu_hz, T[0]) * opac.nu_hz
J                    = np.zeros((nz, nnu)) # mean intensity
F_net                = np.zeros_like(J)    # net flux
F_s                  = np.zeros_like(J)    # stellar flux
F_e                  = np.zeros_like(J)    # terrestrial flux

for z in range(nz):
    B_atmosphere[z, :]=bp.bplanck(opac.nu_hz, T[z]) * opac.nu_hz
for i in range(nz-1):
    dtau_between_layers[i, :, :] = kappa_between_layers[i, :, :] * dP_between_layers[i]/ggrav
                                   
# 2. FTR integration ...........................................................

def integrated_FTR(I, z, dtau, nu, mu, direction):
    """
    This function integrates the radiative transfer equation.
    direction = + 1 means up, direction = - 1 means down.
    """
    dt     = abs(dtau/mu)
    x      = I * np.exp(-dt)
    xp     = x + (1.-(1.+dt)*np.exp(-dt))/dt * B_atmosphere[z, nu]
    return xp + (dt-1.+np.exp(-dt))/dt * B_atmosphere[z-direction, nu]


# 3. Integration ...............................................................

def perform_step(nu):
    """
    This function calculates the stellar flux and the mean intensity J_nu(z)
    for a given wavelength nu.
    """
    I_z_mu             = np.zeros((nz, nmu))
    I_z_g_mu           = np.zeros((nz, ng, nmu))
    F_s_g              = np.zeros((nz, ng))
    F_e_g              = np.zeros_like(F_s_g)
    I_z_g_mu[0, :, 0]  = earth_spectrum[nu]
    F_s_g[-1, :]       = sun_spectrum[nu]
    F_e_g[0, :]        = earth_spectrum[nu]
    
    for z in range(1, nz):
        for g in range(ng):
            """Upwards integration"""
            I_z_g_mu[z, g, 0]      = integrated_FTR(I_z_g_mu[z-1, g, 0],
                                               z,
                                               dtau_between_layers[z-1, nu, g],
                                               nu,
                                               cos_theta[0],
                                               direction=+1)
            
            """Downwards integration"""
            I_z_g_mu[nz-z-1, g, 1] = integrated_FTR(I_z_g_mu[nz-z, g, 1],
                                                nz-z-1,
                                                dtau_between_layers[nz-z-1, nu, g],
                                                nu,
                                                cos_theta[1],
                                                direction=-1)

            """Calculate the stellar flux"""
            F_s_g[nz-z-1, g] = F_s_g[nz-z, g]*np.exp(-dtau_between_layers[nz-z-1, nu, g]/cos_theta_sun)
           
            """Calculate the terrestrial flux"""
            F_e_g[z, g] = F_e_g[z-1, g]*np.exp(-dtau_between_layers[z-1, nu, g])
   
    
    for z in range(nz):
        """Sum over g"""
        I_z_mu[z, 0]   = np.sum(I_z_g_mu[z, :, 0]*opac.g_weights)
        I_z_mu[z, 1]   = np.sum(I_z_g_mu[z, :, 1]*opac.g_weights)
        F_s[z, nu]     = np.sum(F_s_g[z, :]*opac.g_weights)
        F_e[z, nu]     = np.sum(F_e_g[z, :]*opac.g_weights)
        J[z, nu]       = 0.5*np.sum(I_z_mu[z, :])+(F_s[z, nu])
        F_net[z, nu]   = 2.*np.pi/np.sqrt(3.)*(I_z_mu[z, 0]-I_z_mu[z, 1]) \
                         -cos_theta_sun*F_s[z, nu] #we are defining positive flux in upwards direction
                         
# 4. Caluculate intensities for all wavelengths ................................
for nu in range(nnu):
    perform_step(nu)

# 5. Plot ......................................................................

"""
First plot: Mean intensities
"""

F_up              = np.copy(F_net)
F_down            = -1.*np.copy(F_net)
F_up[F_up<0.]     = 0
F_down[F_down<0.] = 0

f                 = plt.figure(1, figsize=(10, 8))
J_ax              = plt.axes([0.1, 0.2, 0.8, 0.7])
J_slider_ax       = plt.axes([0.1, 0.05, 0.8, 0.05])

J_ax.set_yscale('log')
J_ax.set_ylim(0.1, 1.05*J.max())
plt.axes(J_ax)

J_plot,          = plt.semilogx(opac.wl_micron, J[0, :],
                                color='b',
                                label='Mean intensity')

solar_intensity, = plt.semilogx(opac.wl_micron, F_s[0, :],
                                linestyle=':',
                                color='purple',
                                label='Solar flux')
                                
earth_intensity, = plt.semilogx(opac.wl_micron, F_e[0, :],
                                linestyle=':',
                                label='Terrestrial flux')
                                
J_slider = Slider(J_slider_ax, 'z [km]', 0, 60, valinit=0, valfmt='%0.0f')

def update(z):
    J_plot.set_ydata(J[int(z), :])
    solar_intensity.set_ydata(F_s[int(z), :])
    earth_intensity.set_ydata(F_e[int(z), :])
    f.canvas.draw_idle()
J_slider.on_changed(update)

plt.xlabel(r'$\lambda\; [\mu m]$')
plt.ylabel(r'$\nu\ J_\nu(z)\; [W/m^2]$')
plt.title(r'Mean intensity $J_\nu(z)$')
plt.legend()

"""
Second plot: Net flux
"""

g              = plt.figure(2, figsize=(10, 8))
F_ax           = plt.axes([0.1, 0.2, 0.8, 0.7])
F_slider_ax    = plt.axes([0.1, 0.05, 0.8, 0.05])

F_ax.set_yscale('log')
F_ax.set_ylim(0.2, 2.5e+3)
plt.axes(F_ax)

upward_flux,      = plt.semilogx(opac.wl_micron, F_up[0, :],
                                 color='green',
                                 label='Net upward flux')

downward_flux,    = plt.semilogx(opac.wl_micron, F_down[0, :],
                                 color='orange',
                                 label='Net downward flux')
                                 
F_slider = Slider(F_slider_ax, 'z [km]', 0, 60, valinit=0, valfmt='%0.0f')

def update(z):
    upward_flux.set_ydata(F_up[int(z), :])
    downward_flux.set_ydata(F_down[int(z), :])
    f.canvas.draw_idle()
F_slider.on_changed(update)

plt.xlabel(r'$\lambda\; [\mu m]$')
plt.ylabel(r'$\nu\ F_\nu(z)\; [W/m^2]$')
plt.title(r'Net flux $F_\nu(z)$')
plt.legend()

plt.show()
