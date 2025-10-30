# SCRIPT THAT GENERATES THE DATACUBES FOR SOTKES I,Q,U GIVEN THE SINGLE-CHANNEL IMAGES PRODUCED VIA WSCLEAN
# STEP 1 IN THE POLARIZATION FITTING PROCEDURE
#
# G. Di Gennaro
# October 2018

import numpy as np
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.fits import getheader
from astropy.io.fits import Header
import argparse
import sys

def makedatacubes(imagelistI, imagelistQ, imagelistU, xmin, xmax, ymin, ymax, res):
  # WRITE CUBES
  bandwidth = len(imagelistQ)
  print ("number of channels: ", bandwidth)

  lambda_sq = (np.arange(bandwidth))*0.
  freq = (np.arange(bandwidth))*0.

  # define header
  hdu = fits.open(str(imagelistI[0]))
  xaxis = int(xmax)-int(xmin) #hdu[0].header['NAXIS1']
  yaxis = int(ymax)-int(ymin) #hdu[0].header['NAXIS2']

  print (int(xmax), int(xmin))
  print (int(ymax), int(ymin))
  print (xaxis, yaxis)

  del hdu[0].header['CTYPE4']
  del hdu[0].header['CRVAL4']
  del hdu[0].header['CDELT4']
  del hdu[0].header['CRPIX4']
  del hdu[0].header['CUNIT4']
  del hdu[0].header['CUNIT3']
  del hdu[0].header['PC*']

  cubeQ = np.zeros((1,bandwidth,yaxis,xaxis))
  cubeU = np.zeros((1,bandwidth,yaxis,xaxis))
  cubeI = np.zeros((1,bandwidth,yaxis,xaxis))


  c = 3.e8 #light speed m/s

  print ("Doing Q datacube...")
  i=0
  for image in imagelistQ:
    Q = fits.getdata(image)  
    cubeQ[:,i,:,:] = Q[:,:,int(ymin):int(ymax),int(xmin):int(xmax)]

    i=i+1
  fits.writeto(DATADIR + 'Qdatacube'+res+'.fits', cubeQ, hdu[0].header, overwrite=True)

  print ("Doing U datacube...")
  i=0
  for image in imagelistU:
    U = fits.getdata(image)
    cubeU[:,i,:,:] = U[:,:,int(ymin):int(ymax),int(xmin):int(xmax)]

    i=i+1
  fits.writeto(DATADIR + 'Udatacube'+res+'.fits', cubeU, hdu[0].header, overwrite=True)

  print ("Doing I datacube...")
  i=0
  for image in imagelistI:
    I = fits.getdata(image)
    cubeI[:,i,:,:] = I[:,:,int(ymin):int(ymax),int(xmin):int(xmax)]
    
    hdulist_lambda = fits.open(image)
    hdu_lambda = hdulist_lambda[0]
  
    lambda_sq[i] = (c/hdu_lambda.header['CRVAL3'])**2  
    freq[i] = hdu_lambda.header['CRVAL3']/1.e9
    
    i=i+1
  fits.writeto(DATADIR + 'Idatacube'+res+'.fits', cubeI, hdu[0].header, overwrite=True)
  print (lambda_sq, "m^2")
  np.savetxt(DATADIR + 'lambda_sq.txt', np.transpose([lambda_sq, freq]))




## MAIN SCRIPT
parser = argparse.ArgumentParser(description='Make IQU FITS cubes')

parser.add_argument('-i','--sourcename', help='Name of the region to fit to produce the polarization parameter maps', type=str)
parser.add_argument('--resol', help='Taper of the observations (e.g. TAPER50kpc)', required=False, type=str)

args = vars(parser.parse_args())
clustername = args['sourcename']
res         = args['resol']

if not args['resol']:
  res = ''
  DATADIR = '/export/data2/AG_Brueggen/digennaro/HighRedshiftClusters/PlanckSZ/JVLA/polarization/'+clustername+'/None/'
else:  
  DATADIR = '/export/data2/AG_Brueggen/digennaro/HighRedshiftClusters/PlanckSZ/JVLA/polarization/'+clustername+'/Low/'
path_stokesQ = DATADIR + 'stokes_q/'
path_stokesU = DATADIR + 'stokes_u/'
path_stokesI = DATADIR + 'stokes_i/'

imagelistQ = sorted(glob.glob(path_stokesQ + '*image.smooth.regrid.fits'))
imagelistU = sorted(glob.glob(path_stokesU + '*image.smooth.regrid.fits'))
imagelistI = sorted(glob.glob(path_stokesI + '*image.smooth.regrid.fits'))

#print (imagelistI)
print (len(imagelistQ), len(imagelistU), len(imagelistI))

if not len(imagelistQ) == len(imagelistI):
  print ('check number of channels')
  sys.exit()

else:
  polmap, hdr = fits.getdata(glob.glob(DATADIR+"*_polint.fits")[0], header=True)
  ra, dec = hdr['CRVAL1'], hdr['CRVAL2']
  dx, dy = hdr['CDELT2']*hdr['NAXIS1'], hdr['CDELT2']*hdr['NAXIS2']

  print (clustername)
  print ('Resolution:', res)
  print ("Right Ascension :",360+ra)
  print ("Declination     :", dec)
  print ("Size R.A. [deg] :", dx)
  print ("Size Dec [deg]  :", dy)

  stokes, hdr = fits.getdata(glob.glob(path_stokesQ + "*image.smooth.regrid.fits")[0], header=True)
  coords = [ra, dec]
  width  = [dx, dy]
  w = WCS(hdr, naxis=2)
  xmin, ymin = w.wcs_world2pix(coords[0]+(width[0]/2)/np.cos(coords[1]*np.pi/180.), coords[1]-width[1]/2, 1)
  xmax, ymax = w.wcs_world2pix(coords[0]-(width[0]/2)/np.cos(coords[1]*np.pi/180.), coords[1]+width[1]/2, 1)  

  makedatacubes(imagelistI, imagelistQ, imagelistU, xmin, xmax, ymin, ymax, res)
