import numpy as np
import math

# The SI unit version of bplanck.py

def bplanck(freq,temp):
    """
    This function computes the Planck function

                   2 h nu^3 / c^2
       B_nu(T)  = ------------------    [ J / m^2 s ster Hz ]
                  exp(h nu / kT) - 1

    Arguments:
         freq  [Hz]            = Frequency in Herz
         temp  [K]             = Temperature in Kelvin
    """
    const1 = 4.7991598e-11
    const2 = 1.4745284e-50
    x      = const1*freq/(temp+1e-99)
    xp     = np.exp(-x)
    xxp    = 1-xp
    mask   = x<1e-6
    xxp[mask] = x[mask]
    bpl    = xp*const2*(freq**3)/xxp
    return bpl

def bplanckdt(freq,temp):
    """
    This function computes the temperature derivative of the
    Planck function 
      
         dB_nu(T)     2 h^2 nu^4      exp(h nu / kT)        1 
         --------   = ---------- ------------------------  ---
            dT          k c^2    [ exp(h nu / kT) - 1 ]^2  T^2
     
    Arguments:
         freq  [Hz]            = Frequency in Herz
         temp  [K]             = Temperature in Kelvin
    """
    const1 = 4.7991598e-11
    const2 = 7.0764973e-61
    assert np.isscalar(temp), "Error: bplanckdt cannot receive a temperature array. Only a scalar allowed."
    if np.isscalar(freq):
        nu = np.array([freq])
    else:
        nu = np.array(freq)
    bpldt = np.zeros(len(nu))
    for inu in range(len(nu)):
        x   = const1*nu[inu]/(temp+1e-290)
        if(x < 300.):
            theexp     = np.exp(x)
            bpldt[inu] = const2 * nu[inu]**4 * theexp / ( (theexp-1.0)**2 * temp**2 ) + 1.e-290
        else:
            bpldt[inu] = 0.0
    if np.isscalar(freq):
        bpldt = bpldt[0]
    return bpldt

def intensity_from_tbrightlin(freq,tbrightlin):
    """
    Compute the intensity from the linear brightness temperature.
    This is simply the Rayleigh-Jeans law:

       I_nu = T_brightlin * (2*kk/cc**2) * freq^2

    """
    const3 = 3.0724719e-40
    return tbrightlin * ( const3*(freq**2) )

def tbrightlin_from_intensity(freq,intensity):
    """
    Compute the linear brightness temperature from the intensity.
    This is a simple formula:

     T_brightlin = I_nu / ( (2*kk/cc**2) * freq^2 )

    """
    const3 = 3.0724719e-40
    return intensity / ( const3*(freq**2) )

def intensity_from_tbrightfull(freq,tbrightfull):
    """
    Compute the intensity from the full brightness temperature.
    This is simply the planck function.
    """
    return bplanck(freq,tbrightfull)

def tbrightfull_from_intensity(freq,intensity):
    """
    Compute the full brightness temperature from the intensity.
    This is the inverse of the planck function.
    """
    const1 = 4.7991598e-11
    const2 = 1.4745284e-50
    return const1*freq/np.log(1.0+const2*(freq**3)/intensity)

def tbrightlin_from_tbrightfull(freq,tbrightfull):
    """
    Convert the full brightness temperature into 
    linear brightness temperature.
    """
    intensity = intensity_from_tbrightfull(freq,tbrightfull)
    return tbrightlin_from_intensity(freq,intensity)

def tbrightfull_from_tbrightlin(freq,tbrightlin):
    """
    Convert the linear brightness temperature into 
    full brightness temperature.
    """
    intensity = intensity_from_tbrightlin(freq,tbrightlin)
    return tbrightfull_from_intensity(freq,intensity)
