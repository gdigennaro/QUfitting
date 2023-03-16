THIS FILE SUMMARISES THE STEPS REQUIRED FOR THE POLARISATION FITTING

The QUfitting_map.py software works on sets of FITS files only. 
Stokes I, Q and U files are located in folders (name must be stokes_i, stokes_q and stokes_u, respectively) 
containing FITS file of a single sub-band with a given Delta freq, smoother at the same resolution (*-image.smooth.regrid.fits). 
Running the software make_datacubes.py stacks the single Stokes folders into single data cubes 
and generates a text file 'lambda_sq.txt' with the squared wavelengths (column 0) and the corresponding frequencies (column 1);
then QUfitting_map.py perform QU fitting along the wavelength band.


PACKAGES NECESSARY (to run with python 3.X)
- numpy
- scipy
- pp
- astropy
- argparse
- corner
- emcee


FUNCTION USED BY QUfitting.py
- pol_func >> calculate the values and errors (MC simulations) of the polarization fraction and angle given the Stokes I,Q,U fluxes
- function_plots >> generate the polarisation plots (see Fig. 1 in Di Gennaro+2021, ApJ, 991, 3)
- function_polarization_MCMC >> functions for models and MCMC fitting


IMAGES REQUIRED TO RUN THE QU FIT
In order to run QUfitting_map.py, it is necessary to run pyrmsynth first (https://github.com/mrbell/pyrmsynth); it will produce the preliminary RM map and the averaged total intensity map (not corrected for the Ricean bias) 
the fit is parallelised on different CPUs (by default 10; if the number of CPUs needs to be changed, modify the option --ncpu)

- pyrmsynth outputs needed: 
  - RM map (i.e. *phi.fits)
  - Total average polarised map (i.e. *polint.fits)
- list of all the single-channel images in Stokes Q, U and I (name of the folders: stokes_q, stokes_u, stokes_i)


CODE USAGE
- run make_datacubes.py to create datacubes for Stokes Q, U and I;
  provide the cluster name and, if needed, the resolution of the observation

  python make_datacubes.py -i sourcename [--resol]

- run QUfitting_map.py
  - the option --docheck controls the pixels where to perform the QU fit (above a certain threshold, --maskthreshold, over the Ricean error in the *polint.fits image)
  - the option --doplots produces all the fitting plots (stored in the ./singlepixels folder; it also produces a compressed folder)
  - the option --ncpu controls the number of CPU used for the fit
  - the option --depol must be one of these: ExtDepol, IntDepol, None

  python QUfitting_map.py -i sourcename --depol DEPOL_MODEL [--resol]


CITATION:
If you make use of this code in its original form or portions of it, please cite:
Di Gennaro G. et al., Downstream Depolarization in the Sausage Relic: A 1-4 GHz Very Large Array Study, The Astrophysical Journal, Volume 911, 2021. https://doi:10.3847/1538-4357/abe620.

Also available on the arXiv: https://arxiv.org/abs/2102.06631 
