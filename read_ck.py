import numpy as np
import os

class CorrKappOpac(object):
    """
    Container for the correlated-kappa data of a single molecule. The data
    is read from the directories with molecule names, e.g. H2O/. The files
    that are read are:

       correlkappa_<molname>_<T>.K_<P>bar.npy     Numpy data file with the kappas
       correlkappa_H2O_metadata.txt               Text file with metadata

    This object either contains the data for a single choice of temperature 
    and pressure, or it contains the data for all available choices of 
    temperature and pressure in the molecule directory (e.g. H2O/). To read
    all data for this molecule:

       from read_ck import *
       opac = CorrKappOpac()
       opac.read_all_kappas()

    The kappa values are then in opac.all_kappas[]. The array structure is:

       opac.all_kappas[it,ip,inu,ig]    Opacity in [cm^2/g] where g = gram of
                                        molecule.

    where it and ip are the index of the temperature and pressure grids:

       opac.temperature_array[it]       Temperature in [K]
       opac.pressure_array[ip]          Pressure in [dyne/cm^2]

    and inu and ig are the frequency and g-value indices:

       opac.nu_hz[inu]                  Frequency in [Hz]
       opac.wl_micron[inu]              Wavelength in [micrometer]
       opac.g_values[ig]                The Correlated-K g value grid
       opac.g_weights[ig]               The Correlated-K g weights

    Once the opac.all_kappas are loaded into this object, you have the
    option of computing the interpolated correlated kappas for a given
    set of temperature and pressure values (PT-structures, i.e. atmosphere
    models). This is done with interpolate_onto_temp_pres_values(T,P).

    Example (using an external module called atmosphere.py to create
    a simple PT-structure of the Earth's atmosphere):

       from atmosphere import *
       from read_ck import *
       nz       = 30
       zmax_km  = 30.
       atmo     = AtmosphereHydrostatEquil()
       atmo.model_earthatmo_standard(zmax_km=zmax_km,nz=nz)
       T        = atmo.T
       P        = atmo.P/nc.bar
       mol      = 'H2O'
       opac     = CorrKappOpac(mol)
       opac.read_all_kappas()
       opac.interpolate_onto_temp_pres_values(T,P)

    This will create:

       opac.interpolated_kappas[ipt,inu,ig]

    where inu and ig are the same as above, and ipt is the index of
    the (T,P) point, in other words: the index of the layer in the
    atmosphere (the vertical grid point). In this example we have
    only 30 such vertical grid points, so ipt goes from 0 to 29.

    You can (optionally) write these interpolated opacities to file:

       dir      = 'earthatmo'
       opac.write_interpolated_kappa_metadata(dir=dir)
       opac.write_interpolated_kappa_to_file(mol,dir=dir)

    This will create a directory 'earthatmo/' and create two
    files:

       interpolated_correlkappa_metadata.txt  Text file containing metadata
       interpolated_correlkappa_H2O.npy       Numpy data file with kappas

    The interpolated_correlkappa_H2O.npy contains the full
    opac.interpolated_kappas[:,:,:] array, while the other file is
    a text file containing the metadata (e.g. the T and P grid,
    the frequency grid and the g grid).
    
    If you want to later read those interpolated data files back
    into python, please use the InterpolatedCorrKapp class.
    """
    def __init__(self,molecule_name,path=None):
        """
        ARGUMENTS:
           molecule_name                The name of the molecule (e.g. 'H2O')
           path                         If the data are elsewhere: path (otherwise None)
        """
        self.molecule_name = molecule_name
        self.dir           = molecule_name+'/'
        if path is not None:
            if path[-1]!='/': path=path+'/'
            self.dir = path+self.dir
        self.read_metadata()

    def read_metadata(self):
        filename = self.dir+'correlkappa_'+self.molecule_name+'_metadata.txt'
        with open(filename,'r') as f:
            l = f.readline()
            l = f.readline()
            self.ntemp = int(l)
            l = f.readline()
            self.temperature_list  = []
            self.temperature_array = np.zeros(self.ntemp)
            for i in range(self.ntemp):
                l = f.readline()
                self.temperature_list.append(l.strip())
                self.temperature_array[i] = float(l)
            l = f.readline()
            l = f.readline()
            self.npres = int(l)
            l = f.readline()
            self.pressure_list  = []
            self.pressure_array = np.zeros(self.npres)
            for i in range(self.npres):
                l = f.readline()
                self.pressure_list.append(l.strip())
                self.pressure_array[i] = float(l)
            l = f.readline()
            l = f.readline()
            self.ng = int(l)
            l = f.readline()
            self.g_values  = np.zeros(self.ng)
            self.g_weights = np.zeros(self.ng)
            for i in range(self.ng):
                l = f.readline()
                self.g_values[i] = float(l)
            l = f.readline()
            for i in range(self.ng):
                l = f.readline()
                self.g_weights[i] = float(l)
            l = f.readline()
            l = f.readline()
            self.nnu = int(l)
            l = f.readline()
            self.wl_micron = np.zeros(self.nnu)
            for i in range(self.nnu):
                l = f.readline()
                self.wl_micron[i] = float(l)
            cc   = 2.9979245800000e10      # Light speed             [cm/s]
            self.nu_hz = 1e4*cc/self.wl_micron
            self.nui   = np.sqrt(self.nu_hz[1:]*self.nu_hz[:-1])
            self.nui   = np.hstack((self.nu_hz[0],self.nui,self.nu_hz[-1]))
            self.dnu   = np.abs(self.nui[1:]-self.nui[:-1])

    def set_temperature_and_pressure(self):
        self.temp_string = self.temperature_list[self.it]
        self.pres_string = self.pressure_list[self.ip]
        self.temp_value  = float(self.temp_string)
        self.pres_value  = float(self.pres_string)
            
    def read_kappa(self,it,ip):
        self.it          = it
        self.ip          = ip
        self.set_temperature_and_pressure()
        filename = self.dir+'correlkappa_'+self.molecule_name+'_'+self.temp_string+'.K_'+self.pres_string+'bar.npy'
        self.kappa = np.load(filename)  # Kappa is in units of [cm^2/gram] where 'gram' is 'gram of this molecule'

    def read_all_kappas(self):
        """
        Read the correlated-kappa values for all available T and P values.

        RETURNS:
            self.all_kappas[it,ip,inu,ig]    Opacity in [cm^2/g]
        """
        self.all_kappas = np.zeros((self.ntemp,self.npres,self.nnu,self.ng))
        for it in range(self.ntemp):
            for ip in range(self.npres):
                self.read_kappa(it,ip)
                self.all_kappas[it,ip,:,:] = self.kappa.copy()

    def interpolate_onto_one_temp_press(self,temp,pres):
        kappa = np.zeros((self.nnu,self.ng))
        assert self.temperature_array[0]<self.temperature_array[-1], 'Error: interpolation oes not work if temperature grid is decreasing'
        assert self.pressure_array[0]<self.pressure_array[-1], 'Error: interpolation oes not work if pressure grid is decreasing'
        if temp<=self.temperature_array[0]:
            it   = 0
            epst = 0.0
        else:
            if temp>=self.temperature_array[-1]:
                it   = len(self.temperature_array)-2
                epst = 1.0
            else:
                it   = np.where(temp>self.temperature_array)[0][-1]
                epst = (temp-self.temperature_array[it])/(self.temperature_array[it+1]-self.temperature_array[it])
        if pres<=self.pressure_array[0]:
            ip   = 0
            epsp = 0.0
        else:
            if pres>=self.pressure_array[-1]:
                ip   = len(self.pressure_array)-2
                epsp = 1.0
            else:
                ip   = np.where(pres>self.pressure_array)[0][-1]
                lp0  = np.log(self.pressure_array[ip])
                lp1  = np.log(self.pressure_array[ip+1])
                epsp = (np.log(pres)-lp0)/(lp1-lp0)
        allk  = np.log(self.all_kappas)
        kappa = (1-epst)*((1-epsp)*allk[it,ip,:,:]+epsp*allk[it,ip+1,:,:]) +   \
                    epst*((1-epsp)*allk[it+1,ip,:,:]+epsp*allk[it+1,ip+1,:,:])
        kappa = np.exp(kappa)
        return kappa
        
    def interpolate_onto_temp_pres_values(self,T,P):
        self.interpolated_kappas_np = len(T)
        assert len(P)==self.interpolated_kappas_np, 'T and P have different size. Aborting.'
        self.interpolated_kappas_T = T.copy()
        self.interpolated_kappas_P = P.copy()
        self.interpolated_kappas   = np.zeros((self.interpolated_kappas_np,self.nnu,self.ng))
        for i in range(self.interpolated_kappas_np):
            temp  = T[i]
            pres  = P[i]
            self.interpolated_kappas[i,:,:] = self.interpolate_onto_one_temp_press(temp,pres)
            
    def write_interpolated_kappa_metadata(self,dir=None):
        if dir is not None:
            if dir[-1]!='/': dir=dir+'/'
            if not os.path.isdir(dir):
                os.makedirs(dir)
        else:
            dir = ''
        with open(dir+'interpolated_correlkappa_metadata.txt','w') as f:
            f.write('# Nr of T,P value pairs:\n')
            f.write('{}\n'.format(self.interpolated_kappas_np))
            f.write('# List of T,P values in Kelvin, bar:\n')
            for i in range(self.interpolated_kappas_np):
                str = '{0:13.6e} {1:13.6e}\n'.format(self.interpolated_kappas_T[i],self.interpolated_kappas_P[i])
                f.write(str)
            f.write('# Nr of g grid points:\n')
            f.write('{}\n'.format(len(self.g_values)))
            f.write('# List of g values:\n')
            for i in range(len(self.g_values)):
                f.write('{0:12.6e}'.format(self.g_values[i])+'\n')
            f.write('# List of g weights:\n')
            for i in range(len(self.g_weights)):
                f.write('{0:12.6e}'.format(self.g_weights[i])+'\n')
            f.write('# Nr of wavelength grid points:\n')
            f.write('{}\n'.format(len(self.wl_micron)))
            f.write('# List of wavelengths in micrometer:\n')
            for i in range(len(self.wl_micron)):
                f.write('{0:12.6e}'.format(self.wl_micron[i])+'\n')

    def write_interpolated_kappa_to_file(self,name,dir=None):
        if dir is not None:
            if dir[-1]!='/': dir=dir+'/'
            if not os.path.isdir(dir):
                os.makedirs(dir)
        else:
            dir = ''
        np.save(dir+'interpolated_correlkappa_'+name,self.interpolated_kappas)


class InterpolatedCorrKapp(object):
    """
    Container for a 1D set of correlated-kappa opacities belonging to 
    a 1D set of Temperature-Pressure pairs (i.e. an atmospheric hydrostatic
    structure model).

    For creating these interpolated correlated-kappa files, please refer
    to the CorrKappOpac class. 

    Suppose you have a directory 'earthatmo/' containing the interpolated
    correlated-kappa files, and suppose one of the molecules is 'H2O', then
    here is how to read it:

       from read_ck import *
       opac = InterpolatedCorrKapp('H2O',dir='earthatmo')

    The result will be in opac.kappa
    You can also read in multiple species at once:

       from read_ck import *
       opac = InterpolatedCorrKapp(['H2O','CO2'],dir='earthatmo')

    In this case the result will be in opac.all_kappas.

    If you read in multiple species, you can mix them into a single
    opacity array, with the abundances of each of the species set
    by the keyword abun (abundance by mass!). The abundances do not
    need to add up to 1, because often most of the gas is opacity-free.
    Note that the mixing takes a bit of time, because the code has to
    perform Monte-Carlo sampling. Example:

       from read_ck import *
       opac = InterpolatedCorrKapp(['H2O','CO2'],dir='earthatmo',abun=[6e-3,4e-4])

    Note that you can also specify abun as a 2D array, with the rightmost
    index being the vertical structure. This way you can specify the abundance
    differently at different heights in the atmosphere.

    Once you made your mix, you can write it back to a file:

       opac.write_mixture('mymix')

    """
    def __init__(self,molecule_name=None,dir=None,abun=None,nmc=10000,ismixture=False,simplemix=False):
        """
        ARGUMENTS:
           molecule_name     The name of the molecule (e.g. 'H2O'). If it
                             is just one name, then just this single molecule
                             (or single mixture of molecules) is read. 
                             If it is a list of names, then it will
                             lead to a mixture of opacities. In that case
                             the abun keyword must be set to (at least) an
                             array or list of abundances (by mass) of equal
                             size.
           dir               Directory where the interpolated data are
           abun              Abundance (or list or array of them) of the
                             molecule(s).
           nmc               (for mixing of opacities) Number of Monte Carlo
                             samplings.
           ismixture         (only if the file you read is for a mixture of
                             molecular species): Read the mixmetadata file
                             so that the abundances of the constituents is
                             known (self.mix_names, self.mix_abun). In this
                             case self.gasmix_kappa will be set equal to 
                             self.kappa.
           simplemix         If True, then any mixing is done in the simplest
                             possible way (no Monte Carlo, but instead fast).

        RETURNS:
           self.kappa        The correlated-kappa data, for each PT-point.
                             Shape: self.kappa[npt,nnu,ng], where npt is the
                             number of PT-points, nnu the number of frequency
                             points, ng the number of g-value points. If you
                             read multiple species, then self.kappa=Null. See
                             self.all_kappas.
           self.all_kappas   The same as self.kappa, but in case multiple 
                             species are read (nr of species is: self.nspec). 
                             Shape: self.all_kappas[nspec,npt,nnu,ng].
           self.gasmix_kappa The same as self.kappa, but for the mixture
                             (if you created one).
        """
        if dir is not None:
            if dir[-1]!='/': dir=dir+'/'
            self.dir = dir
        else:
            self.dir = ''
        self.read_metadata()
        if molecule_name is not None:
            self.molecule_name = molecule_name
            if type(self.molecule_name)==list or type(self.molecule_name)==np.ndarray:
                #
                # Read the opacities for all these molecules
                #
                self.nspec = len(molecule_name)
                self.all_kappas = np.zeros((self.nspec,self.npt,self.nnu,self.ng))
                for ispec in range(self.nspec):
                    self.read_kappas(self.molecule_name[ispec])
                    self.all_kappas[ispec,:,:,:] = self.kappa[:,:,:].copy()
                self.kappa = None
                #
                # Mix opacities, if abun is set
                # The mixing is done with a Monte Carlo method
                #
                if abun is not None:
                    self.mix_opacities(abun,nmc=nmc,simplemix=simplemix)
            else:
                #
                # Read one opacity (one molecular species) only OR an already
                # computed mixture.
                #
                self.read_kappas(self.molecule_name)
                if ismixture:
                    self.gasmix_kappa = self.kappa
                    self.read_mixture_abun(self.molecule_name)
                else:
                    if abun is not None:
                        self.gasmix_kappa = abun * self.kappa

    def mix_opacities(self,abun,nmc=10000,simplemix=False):
        """
        If multiple species are read in (i.e. if self.all_kappas is set), then
        this method computes a Correlated-Kappa array for the mixture of these
        molecular species, where their abundances are given by abun. Note that
        the sum of the abundances does not need to be (and generally is not) 
        unity, because the abundance is relative to the overall gas. 

        ARGUMENT:
           abun        Array of abundances by mass (!) of the species. It
                       has to be an array of [self.nspec] elements, i.e. the
                       abundance of each species. But it CAN also be a 2D
                       array [self.nspec,self.npt], so that you can specify
                       the abundance also as a function of height in the
                       atmosphere.
           nmc         (for mixing of opacities) Number of Monte Carlo
                       samplings.
           simplemix   If True, then do not do the Monte Carlo, but simply
                       add the opacities (this is not correct, but for not
                       too crazy opacities it might sort-of work nonetheless).
        """
        assert type(abun)==list or type(abun)==np.ndarray
        abun = np.array(abun)
        assert self.nspec==abun.shape[0]
        if len(abun.shape)==1:
            abuns = np.zeros((self.nspec,self.npt))
            for ipt in range(self.npt):
                abuns[:,ipt] = abun[:]
        else:
            assert abun.shape[1]==self.npt, 'Error: if you specify abun as 2D array: must be [nspec,npt]'
            abuns = abun
        self.abun = abun.copy()
        self.gasmix_kappa = np.zeros((self.npt,self.nnu,self.ng))
        for ipt in range(self.npt):
            if not simplemix:
                # Correct mixing
                print('Mixing Monte Carlo at PT point {}'.format(ipt))
                for inu in range(self.nnu):
                    rn    = np.random.random(nmc*self.nspec).reshape((self.nspec,nmc))
                    kappa = np.zeros_like(rn)
                    for ispec in range(self.nspec):
                        kappa[ispec,:] = abuns[ispec,ipt]*np.exp(np.interp(rn[ispec,:], \
                                         self.g_values,np.log(self.all_kappas[ispec,ipt,inu,:])))
                    kaptot = kappa.sum(axis=0)
                    kaptot.sort()
                    index_g = (self.g_values * nmc).astype(int)
                    assert index_g.max()<nmc
                    self.gasmix_kappa[ipt,inu,:] = kaptot[index_g]
            else:
                # Simplistic mixing
                self.gasmix_kappa[ipt,:,:] = 0
                for ispec in range(self.nspec):
                    self.gasmix_kappa[ipt,:,:] += abuns[ispec,ipt]*self.all_kappas[ispec,ipt,:,:]

    def read_metadata(self):
        filename = self.dir+'interpolated_correlkappa_metadata.txt'
        with open(filename,'r') as f:
            l = f.readline()
            l = f.readline()
            self.npt = int(l)
            l = f.readline()
            self.T  = np.zeros(self.npt)
            self.P  = np.zeros(self.npt)
            for i in range(self.npt):
                l = f.readline().split()
                self.T[i] = float(l[-2])
                self.P[i] = float(l[-1])
            l = f.readline()
            l = f.readline()
            self.ng = int(l)
            l = f.readline()
            self.g_values  = np.zeros(self.ng)
            self.g_weights = np.zeros(self.ng)
            for i in range(self.ng):
                l = f.readline()
                self.g_values[i] = float(l)
            l = f.readline()
            for i in range(self.ng):
                l = f.readline()
                self.g_weights[i] = float(l)
            l = f.readline()
            l = f.readline()
            self.nnu = int(l)
            l = f.readline()
            self.wl_micron = np.zeros(self.nnu)
            for i in range(self.nnu):
                l = f.readline()
                self.wl_micron[i] = float(l)
            cc   = 2.9979245800000e10      # Light speed             [cm/s]
            self.nu_hz = 1e4*cc/self.wl_micron
            self.nui   = np.sqrt(self.nu_hz[1:]*self.nu_hz[:-1])
            self.nui   = np.hstack((self.nu_hz[0],self.nui,self.nu_hz[-1]))
            self.dnu   = np.abs(self.nui[1:]-self.nui[:-1])
    
    def read_kappas(self,molecule_name):
        filename = self.dir+'interpolated_correlkappa_'+molecule_name+'.npy'
        self.kappa = np.load(filename)  # Kappa is in units of [cm^2/gram] where 'gram' is 'gram of this molecule'

    def write_mixture(self,name):
        """
        If you created a mixture of opacities (i.e. self.gasmix_kappa), then 
        this method will allow you to write it out into the same format as
        the original unmixed (but interpolated) opacities.

        ARGUMENTS:
          name         The name of the mixture
        """
        
        np.save(self.dir+'interpolated_correlkappa_'+name,self.gasmix_kappa)
        with open(self.dir+'interpolated_correlkappa_'+name+'_mixmetadata.txt','w') as f:
            f.write('# Nr of species mixed:\n')
            f.write('{}\n'.format(self.nspec))
            f.write('# List of species mixed:\n')
            for ispec in range(self.nspec):
                f.write(self.molecule_name[ispec]+'\n')
            f.write('# PT-dependent abundances (0=no, 1=yes)?\n')
            if(len(self.abun.shape)==1):
                f.write('0\n')
                f.write('# The abundances:\n')
                for a in self.abun:
                    f.write('{0:9.3e} '.format(a))
                f.write('\n')
            else:
                f.write('1\n')
                f.write('# The abundances:\n')
                for ipt in range(self.npt):
                    for a in self.abun[:,ipt]:
                        f.write('{0:9.3e} '.format(a))
                    f.write('\n')

    def read_mixture_abun(self,name):
        """
        If you want to know the mixing ratios (abundances) of the contents of a 
        mixed opacity, then there should be a file called 

           interpolated_correlkappa_<name>_mixmetadata.txt

        which this method reads.

        ARGUMENTS: 
           name                 The name of the mixture

        RETURNS:
           self.mix_names       List of names of the molecules in the mixture
           self.mix_abun        List of abundances of the molecules in the mixture
        """
        with open(self.dir+'interpolated_correlkappa_'+name+'_mixmetadata.txt','r') as f:
            l = f.readline()
            l = f.readline()
            mix_nspec = int(l)
            self.mix_names = []
            l = f.readline()
            for i in range(mix_nspec):
                l = f.readline()
                self.mix_names.append(l.strip())
            l = f.readline()
            l = f.readline()
            ptdep = int(l)
            l = f.readline()
            if ptdep==0:
                self.mix_abun = np.zeros(mix_nspec)
                l = f.readline().split()
                for i in range(mix_nspec):
                    self.mix_abun[i] = float(l[i])
            else:
                self.mix_abun = np.zeros((mix_nspec,self.npt))
                for ipt in range(self.npt):
                    l = f.readline().split()
                    for i in range(mix_nspec):
                        self.mix_abun[i,ipt] = float(l[i])
