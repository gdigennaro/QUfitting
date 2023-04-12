# SCRIPT THAT FITS THE SOTKES I,Q,U DATACUBES TO GENERATE A MAP OF RM, POLARIZATION ANGLE AND FRACTION AND, EVENTUALLY, THE DEPOLARIZATION (SIGMA_RM) FOR THE FULL CLUSTER; THE FIT IS DONE IN PARALLEL WITH DIFFERENT CPUS
# LINKED WITH functions_polarization_MCMC
#
# to be run in the cluster polarisation folder
#
# G. Di Gennaro
# Feruary 2019

import numpy as np
import sys, os, glob
import datetime
import math,time
import pp
from scipy import optimize
from astropy.io import fits
from astropy.wcs import WCS
import argparse

# my function
import pol_func
import functions_polarization_MCMC
import function_plots


print ("Script is starting")
print (datetime.datetime.now())

def computerms(ms,masksup=1.e-7):
  m = ms[np.abs(ms)>masksup]
  rmsold = np.std(m)
  diff = 1.e-1
  cut = 3.
  med = np.median(m)
  for i in range(10):
    ind = np.where(np.abs(m-med)<rmsold*cut)[0]
    rms = np.std(m[ind])
    if np.abs((rms-rmsold)/rmsold)<diff: break
    rmsold = rms
  return rms


def QUfitting(source, polmap, mask, maskname, RMmap, stokesI, stokesQ, stokesU, noise_mapI, noise_mapQ, noise_mapU, res, beam_area, threshold, depol): 
  lambda_plot = np.loadtxt(DATADIR + "lambda_sq.txt")
  x_lambda = lambda_plot[:,0]
  freq = lambda_plot[:,1]

  bandwidth = np.shape(stokesI)[1]

  #### TODO not a constant noise but weighted to the beam response
  noise = np.sqrt(np.nanmean(polmap[:,:]**2))
  print (noise*1e6, "uJy/beam")

  I          = np.zeros(bandwidth)
  Q          = np.zeros(bandwidth)
  U          = np.zeros(bandwidth)
  noiseI     = np.zeros(bandwidth)
  noiseQ     = np.zeros(bandwidth)
  noiseU     = np.zeros(bandwidth)

  X          = np.zeros(bandwidth)
  P          = np.zeros(bandwidth)
  p          = np.zeros(bandwidth)
  noiseX     = np.zeros(bandwidth)
  noiseP     = np.zeros(bandwidth)
  noisep     = np.zeros(bandwidth)

  norm       = []
  a          = []
  curv       = []
  norm_neg   = []
  a_neg      = []
  curv_neg   = []
  norm_pos   = []
  a_pos      = []
  curv_pos   = []

  frac       = []
  angle      = []
  RM         = []
  sigma      = []
  frac_neg   = []
  angle_neg  = []
  RM_neg     = []
  sigma_neg  = []
  frac_pos   = []
  angle_pos  = []
  RM_pos     = []
  sigma_pos  = []

  chisqredI  = []
  chisqredQU = []

  y, x = np.where( (mask[0,0,:,:] == 0.) & (polmap[:,:] > (threshold*noise)))
  Npoints = len(x)

  RMguess = np.nanmean(RMmap[y,x])

  I[:] = sum(stokesI[0,:,y,x])
  Q[:] = sum(stokesQ[0,:,y,x])
  U[:] = sum(stokesU[0,:,y,x])

  noiseI[:] = np.sqrt( (Npoints/beam_area) *noise_mapI[:]**2 + (0.05*I[:])**2 )
  noiseQ[:] = np.sqrt( (Npoints/beam_area) *noise_mapQ[:]**2 + (0.05*Q[:])**2 )
  noiseU[:] = np.sqrt( (Npoints/beam_area) *noise_mapU[:]**2 + (0.05*U[:])**2 )
  
  
  X[:] = pol_func.angle(Q[:], U[:], noiseQ[:], noiseU[:])[0]
  P[:] = pol_func.fraction(Q[:], U[:], I[:], noiseQ[:], noiseU[:], noiseI[:])[0]

  noiseX[:] = pol_func.angle(Q[:], U[:], noiseQ[:], noiseU[:])[1]
  noiseP[:] = pol_func.fraction(Q[:], U[:], I[:], noiseQ[:], noiseU[:], noiseI[:])[1]

  if not os.path.exists(DATADIR + "/integrated"):
    os.mkdir(DATADIR + "/integrated")
  imagename = DATADIR + "/integrated/"+source+maskname
  fit = functions_polarization_MCMC.polarization_fitting(depol, x_lambda, I[:], noiseI[:], Q[:], noiseQ[:], U[:], noiseU[:], RMguess, None, None, imagename, docornerplot=True)

  fit_I  = fit[0]
  
  norm = fit_I[0][1]
  norm_neg = fit_I[0][2]
  norm_pos = fit_I[0][3]

  a = fit_I[1][1]
  a_neg = fit_I[1][2]
  a_pos = fit_I[1][3]
  
  curv = fit_I[2][1]
  curv_neg = fit_I[2][2]
  curv_pos = fit_I[2][3]  
  
  
  fit_QU = fit[1]
  
  frac = fit_QU[0][1]
  frac_neg = fit_QU[0][2]
  frac_pos = fit_QU[0][3]

  angle = fit_QU[1][1]
  angle_neg = fit_QU[1][2]
  angle_pos = fit_QU[1][3]
  
  RM = fit_QU[2][1]
  RM_neg = fit_QU[2][2]
  RM_pos = fit_QU[2][3]  

  sigma = np.sqrt(fit_QU[3][1])
  sigma_neg = 0.5*(fit_QU[3][2])*(fit_QU[3][1]**(-(1./2.)))
  sigma_pos =  0.5*(fit_QU[3][3])*(fit_QU[3][1]**(-(1./2.)))

  chisqredI = function_plots.models(depol, x_lambda, norm, a, curv, frac, angle, RM, sigma, \
    I[:], noiseI[:], Q[:], noiseQ[:], U[:], noiseU[:])[3]

  chisqredQU = function_plots.models(depol, x_lambda, norm, a, curv, frac, angle, RM, sigma, \
      I[:], noiseI[:], Q[:], noiseQ[:], U[:], noiseU[:])[4]

  print ("I FIT RESULTS")
  print ("I0       :", norm, norm_neg, norm_pos)
  print ("a        :", a, a_neg, a_pos)
  print ("b        :", curv, curv_neg, curv_pos)
  print ("alpha    :", 2*curv*np.log10(0.650) + a)

  print ("red chisq:", chisqredI)

  print ("\n")
  print ("QU FIT RESULTS")
  print ("p0       :", frac, frac_neg, frac_pos)
  print ("chi0     :", angle, angle_neg, angle_pos)
  print ("RM       :", RM, RM_neg, RM_pos)
  print ("sigmaRM  :", sigma, sigma_neg, sigma_pos)
  
  print ("red chisq:", chisqredQU)
  print ("\n")

  f1 = open(DATADIR+"/integrated/"+source+maskname+"_fitresults_stokesI_"+depol+".txt", "w+")
  f1.write("normaliz:"+str(norm)+"  "+str(norm_neg)+"  "+str(norm_pos)+"\n"+
					"a        :"+str(a)+"  "+str(a_neg)+"  "+str(a_pos)+"\n"+
					"curvature:"+str(curv)+" "+str(curv_neg)+" "+str(curv_pos))
  f1.close()

  f2 = open(DATADIR+"/integrated/"+source+maskname+"_fitresults_stokesQU_"+depol+".txt", "w+")
  f2.write("intr pol frac:"+str(frac)+"  "+str(frac_neg)+"  "+str(frac_pos)+"\n"+
					"intr pol angl:"+str(angle)+"  "+str(angle_neg)+"  "+str(angle_pos)+"\n"+
					"Rotat Measure:"+str(RM)+" "+str(RM_neg)+" "+str(RM_pos)+"\n"+
					"Depolariz    :"+str(sigma)+" "+str(sigma_neg)+" "+str(sigma_pos))
  f2.close()

  function_plots.plots(depol, imagename, x_lambda, norm, a, curv, frac, angle, RM, sigma, \
                      I[:], noiseI[:], Q[:], noiseQ[:], U[:], noiseU[:], X[:], noiseX[:], P[:], noiseP[:], \
                      xlim=[round(x_lambda.min(),3)-0.001, round(x_lambda.max(),2)+0.01], imgformat="pdf", saveplot=True)





## MAIN SCRIPT
parser = argparse.ArgumentParser(description='Make QU fit pixel per pixel using MCMC for the uncertainties on the total intensity polarization paramenters (I0, a, b; p0, chi0, RM, sigmaRM)')

parser.add_argument('-i','--sourcename', help='Name of the region to fit to produce the polarization parameter maps', type=str)
parser.add_argument('--RA', help='RA of cluster in deg', required=False, type=float)
parser.add_argument('--DEC', help='DEC of cluster in deg', required=False, type=float)
parser.add_argument('--dx', help='RA size of the optical image in degree', required=False, type=float)
parser.add_argument('--dy', help='DEC size of the optical image in degree', required=False, type=float)
parser.add_argument('--depol', help='Depolarization model to use (None, ExtDepol, IntDepol)', required=True, type=str)
parser.add_argument('--resol', help='Taper of the observations (e.g. TAPER50kpc)', required=False, type=str)
parser.add_argument('--thrsh', help='threshold for the QU fit, default=3.0', default=3.0, type=float)
parser.add_argument('--maskname', help='if provided, select only a given part of the cluster', required=True, type=str)

args = vars(parser.parse_args())
source    = args['sourcename']
depol     = args['depol']
if args['resol']:
  res       = args['resol']
else:
  res = ''
threshold = args['thrsh']
maskname  = args['maskname']


if res == '':
  DATADIR = '/export/data/group-brueggen/digennaro/HighRedshiftClusters/PlanckSZ/JVLA/polarization/'+source+'/None/'
else:
  DATADIR = '/export/data/group-brueggen/digennaro/HighRedshiftClusters/PlanckSZ/JVLA/polarization/'+source+'/Low/'


print ("resolution   :", res)
polmap, hdr = fits.getdata(glob.glob(DATADIR + "*polint.fits")[0], header=True)
mask = fits.getdata(glob.glob(DATADIR + maskname+".mask.fits")[0])
RMmap       = fits.getdata(glob.glob(DATADIR + "*phi.fits")[0])
stokesI = fits.getdata(DATADIR + "Idatacube"+res+".fits")
stokesQ = fits.getdata(DATADIR + "Qdatacube"+res+".fits")
stokesU = fits.getdata(DATADIR + "Udatacube"+res+".fits")

print ("maskname:", DATADIR + maskname+".mask.fits")

"""
QU fitting for a given region centred on [RA,DEC] with size [dx, dy]
"""
if args['RA']:
  ra        = args['RA']
  dec       = args['DEC']
  dx        = args['dx']
  dy        = args['dy']
  
  w = WCS(hdr, naxis=2)
  xmin, ymin = w.wcs_world2pix(ra+(dx/2)/np.cos(dec*np.pi/180.), dec-dy/2, 0)
  xmax, ymax = w.wcs_world2pix(ra-(dx/2)/np.cos(dec*np.pi/180.), dec+dy/2, 0)   
  
  xmin, ymin = int(xmin), int(ymin)
  xmax, ymax = int(xmax), int(ymax)
else:
  xmin, xmax = 0, hdr['NAXIS1']
  ymin, ymax = 0, hdr['NAXIS2']

bmaj = hdr['BMAJ'] * 3600.
bmin = hdr['BMIN'] * 3600.
cellsize = hdr['CDELT2']*3600.
beam_area = (np.pi/(4*np.log(2))) * (bmaj*bmin)/cellsize**2

bandwidth = np.shape(stokesI)[1]
# define the noise in the Q,U,I datacubes
noise_mapI = np.zeros(bandwidth)
noise_mapQ = np.zeros(bandwidth)
noise_mapU = np.zeros(bandwidth)

noise_mapI[:] = computerms(stokesI[0,:,ymin:ymax,xmin:xmax])*1e3
noise_mapQ[:] = computerms(stokesQ[0,:,ymin:ymax,xmin:xmax])*1e3
noise_mapU[:] = computerms(stokesU[0,:,ymin:ymax,xmin:xmax])*1e3

gauss2d = (2*np.pi)/(8*np.log(2))
stokesI = stokesI * ( cellsize**2 / (gauss2d * bmaj * bmin) )*1e3
stokesQ = stokesQ * ( cellsize**2 / (gauss2d * bmaj * bmin) )*1e3
stokesU = stokesU * ( cellsize**2 / (gauss2d * bmaj * bmin) )*1e3



QUfitting(source, polmap, mask, maskname, RMmap, stokesI, stokesQ, stokesU, noise_mapI, noise_mapQ, noise_mapU, res, beam_area, threshold, depol)

sys.exit()

