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

def checkpixeltofit(polmap, threshold):
  """
  Check the pixels in the full *polint.fits image above a given threshold
  """
  noise = np.sqrt(np.nanmean(polmap[:,:]**2))  #computerms(polmap[0,0,ymin:ymax,xmin:xmax])
  print (noise*1e6, "uJy/beam")
  
  y, x = np.where(polmap[:,:] > (threshold*noise))
  pos = np.copy(polmap)*np.nan
  pos[y,x] = 1
  
  fits.writeto(DATADIR + "goodpixels_"+source+"_thrsh"+str(threshold)+".fits", pos, hdr, overwrite=True)
  return

def QUfitting(source, polmap, RMmap, stokesI, stokesQ, stokesU, res, threshold, depol, ncpus, xmin, xmax, ymin, ymax, doplots): 
  lambda_plot = np.loadtxt(DATADIR + "lambda_sq.txt")
  x_lambda = lambda_plot[:,0]
  freq = lambda_plot[:,1]

  bandwidth = np.shape(stokesI)[1]


  #### TODO not a constant noise but weighted to the beam response
  ##noise = np.sqrt(np.nanmean(polmap[0,0,ymin:ymax,xmin:xmax]**2))  #computerms(polmap[0,0,ymin:ymax,xmin:xmax])
  noise = np.sqrt(np.nanmean(polmap[:,:]**2))
  print (noise*1e6, "uJy/beam")
  print ("Producing single-pixel plot(s):", str(doplots))

  noise_mapQ = np.zeros(bandwidth)
  noise_mapU = np.zeros(bandwidth)
  noise_mapI = np.zeros(bandwidth)

  I          = np.copy(stokesI)*np.nan
  Q          = np.copy(stokesI)*np.nan
  U          = np.copy(stokesI)*np.nan
  noiseI     = np.copy(stokesI)*np.nan
  noiseQ     = np.copy(stokesI)*np.nan
  noiseU     = np.copy(stokesI)*np.nan
  X          = np.copy(stokesI)*np.nan
  P          = np.copy(stokesI)*np.nan
  p          = np.copy(stokesI)*np.nan
  noiseX     = np.copy(stokesI)*np.nan
  noiseP     = np.copy(stokesI)*np.nan
  noisep     = np.copy(stokesI)*np.nan

  norm       = np.copy(polmap)*np.nan
  a          = np.copy(polmap)*np.nan
  curv       = np.copy(polmap)*np.nan
  norm_neg   = np.copy(polmap)*np.nan
  a_neg      = np.copy(polmap)*np.nan
  curv_neg   = np.copy(polmap)*np.nan
  norm_pos   = np.copy(polmap)*np.nan
  a_pos      = np.copy(polmap)*np.nan
  curv_pos   = np.copy(polmap)*np.nan

  frac       = np.copy(polmap)*np.nan
  angle      = np.copy(polmap)*np.nan
  RM         = np.copy(polmap)*np.nan
  sigma      = np.copy(polmap)*np.nan
  frac_neg   = np.copy(polmap)*np.nan
  angle_neg  = np.copy(polmap)*np.nan
  RM_neg     = np.copy(polmap)*np.nan
  sigma_neg  = np.copy(polmap)*np.nan
  frac_pos   = np.copy(polmap)*np.nan
  angle_pos  = np.copy(polmap)*np.nan
  RM_pos     = np.copy(polmap)*np.nan
  sigma_pos  = np.copy(polmap)*np.nan

  chisqredI  = np.copy(polmap)*np.nan
  chisqredQU = np.copy(polmap)*np.nan

  #noise_mapI[:] = np.array([computerms(stokesI[0,z,int(ymin):int(ymax),int(xmin):int(xmax)])*1.e3 for z in range(bandwidth)])
  #noise_mapQ[:] = np.array([computerms(stokesQ[0,z,int(ymin):int(ymax),int(xmin):int(xmax)])*1.e3 for z in range(bandwidth)])
  #noise_mapU[:] = np.array([computerms(stokesU[0,z,int(ymin):int(ymax),int(xmin):int(xmax)])*1.e3 for z in range(bandwidth)])

  noise_mapI[:] = np.array([computerms(stokesI[0,z,:,:])*1.e3 for z in range(bandwidth)])
  noise_mapQ[:] = np.array([computerms(stokesQ[0,z,:,:])*1.e3 for z in range(bandwidth)])
  noise_mapU[:] = np.array([computerms(stokesU[0,z,:,:])*1.e3 for z in range(bandwidth)])



  #### SETS THE PARALLEL INPUTS
  print ("Number of CPU used:", ncpus)
  
  # tuple of all parallel python servers to connect with
  ppservers = ()
  mysecret = depol
          
  # Creates jobserver with ncpus workers
  job_server = pp.Server(ncpus, ppservers=ppservers, secret=mysecret)
  #print ("Starting pp with", job_server.get_ncpus(), "workers")

  jobs = []
  k = 0
  ####

  #idx = np.where(polmap[ymin:ymax,xmin:xmax] > (threshold*noise))
  idx = np.where(polmap[:,:] > (threshold*noise))
  numb_good_pix = len(idx[0])

  parts = int(numb_good_pix / 1000)
  step = 0
  print ("Total number of pixels where to perform the fit:", numb_good_pix)
  print ("Cycles of parallelisation:", parts+1)

  for x in range(xmin, xmax): # range(0, xmax-xmin): #
    for y in range(ymin, ymax): # range(0, ymax-ymin): #
      if polmap[y,x] > (threshold*noise):
        RMguess = RMmap[y,x]
        #print ("RMguess", RMguess)

        I[0,:,y,x] = stokesI[0,:,y,x]*1.e3
        Q[0,:,y,x] = stokesQ[0,:,y,x]*1.e3
        U[0,:,y,x] = stokesU[0,:,y,x]*1.e3

        noiseI[0,:,y,x] = np.sqrt( noise_mapI[:]**2 + (0.05*I[0,:,y,x])**2 )
        noiseQ[0,:,y,x] = np.sqrt( noise_mapQ[:]**2 + (0.05*Q[0,:,y,x])**2 )
        noiseU[0,:,y,x] = np.sqrt( noise_mapU[:]**2 + (0.05*U[0,:,y,x])**2 )
        
        if doplots:
          X[0,:,y,x] = pol_func.angle(Q[0,:,y,x], U[0,:,y,x], noiseQ[0,:,y,x], noiseU[0,:,y,x])[0]
          P[0,:,y,x] = pol_func.fraction(Q[0,:,y,x], U[0,:,y,x], I[0,:,y,x], noiseQ[0,:,y,x], noiseU[0,:,y,x], noiseI[0,:,y,x])[0]
        
          noiseX[0,:,y,x] = pol_func.angle(Q[0,:,y,x], U[0,:,y,x], noiseQ[0,:,y,x], noiseU[0,:,y,x])[1]
          noiseP[0,:,y,x] = pol_func.fraction(Q[0,:,y,x], U[0,:,y,x], I[0,:,y,x], noiseQ[0,:,y,x], noiseU[0,:,y,x], noiseI[0,:,y,x])[1]


        if not os.path.exists(DATADIR + "/singlepixel"):
          os.mkdir(DATADIR + "/singlepixel")
        imagename = DATADIR + "/singlepixel/pixel_"+str(x)+"-"+str(y)
          
        # Execute the same task with different amount of active workers
        jobs.append(job_server.submit(functions_polarization_MCMC.polarization_fitting, (depol, x_lambda, I[0,:,y,x], noiseI[0,:,y,x], Q[0,:,y,x], noiseQ[0,:,y,x], U[0,:,y,x], noiseU[0,:,y,x], RMguess, x, y, imagename, True,),(),("numpy","emcee","scipy.optimize","corner","matplotlib.pyplot","matplotlib",)))
        
        k+=1

        if step != parts:
          kmax = 1000
          text = "start again to collect the jobs; step #"
        else:
          kmax = numb_good_pix - (1000*parts)
          text = "all the jobs are collected; step #"
          
        if k == kmax:
          print ("collect the jobs")
          print ("step:", step)

            
          for job in jobs:
            fitI, fit, xx, yy = job()          
            norm[yy,xx] = fitI[0][1]
            a[yy,xx] = fitI[1][1]
            curv[yy,xx] = fitI[2][1]
            
            norm_neg[yy,xx] = fitI[0][2]
            a_neg[yy,xx] = fitI[1][2]
            curv_neg[yy,xx] = fitI[2][2]

            norm_pos[yy,xx] = fitI[0][3]
            a_pos[yy,xx] = fitI[1][3]
            curv_pos[yy,xx] = fitI[2][3]
                   
                   
            frac[yy,xx] = fit[0][1]
            angle[yy,xx] = fit[1][1] #*180/np.pi
            RM[yy,xx] = fit[2][1]
            if depol != None:
              sigma[yy,xx] = np.sqrt(fit[3][1])
                
            frac_neg[yy,xx] = fit[0][2]
            angle_neg[yy,xx] = fit[1][2]
            RM_neg[yy,xx] = fit[2][2]
            if depol != None:
              sigma_neg[yy,xx] = 0.5*(fit[3][2])*(fit[3][1]**(-(1./2.)))
                
            frac_pos[yy,xx] = fit[0][3]
            angle_pos[yy,xx] = fit[1][3] 
            RM_pos[yy,xx] = fit[2][3]
            if depol != None:
              sigma_pos[yy,xx] = 0.5*(fit[3][3])*(fit[3][1]**(-(1./2.)))

            chisqredI[yy,xx] = function_plots.models(depol, x_lambda, norm[yy,xx], a[yy,xx], curv[yy,xx], \
              frac[yy,xx], angle[yy,xx], RM[yy,xx], sigma[yy,xx], \
                I[0,:,yy,xx], noiseI[0,:,yy,xx], Q[0,:,yy,xx], noiseQ[0,:,yy,xx], U[0,:,yy,xx], noiseU[0,:,yy,xx])[3]

            
            chisqredQU[yy,xx] = function_plots.models(depol, x_lambda, norm[yy,xx], a[yy,xx], curv[yy,xx], \
              frac[yy,xx], angle[yy,xx], RM[yy,xx], sigma[yy,xx], \
                I[0,:,yy,xx], noiseI[0,:,yy,xx], Q[0,:,yy,xx], noiseQ[0,:,yy,xx], U[0,:,yy,xx], noiseU[0,:,yy,xx])[4]

            #print ("\n")
            
            print ("I FIT RESULTS")
            print ("I0       :", norm[yy,xx], norm_neg[yy,xx], norm_pos[yy,xx])
            print ("a        :", a[yy,xx], a_neg[yy,xx], a_pos[yy,xx])
            print ("b        :", curv[yy,xx], curv_neg[yy,xx], curv_pos[yy,xx])
            
            print ("red chisq:", chisqredI[yy,xx])

            print ("\n")
            print ("QU FIT RESULTS")
            print ("p0       :", frac[yy,xx], frac_neg[yy,xx], frac_pos[yy,xx])
            print ("chi0     :", angle[yy,xx], angle_neg[yy,xx], angle_pos[yy,xx])
            print ("RM       :", RM[yy,xx], RM_neg[yy,xx], RM_pos[yy,xx])
            print ("sigmaRM  :", sigma[yy,xx], sigma_neg[yy,xx], sigma_pos[yy,xx])
            
            print ("red chisq:", chisqredQU[yy,xx])
            print ("\n")
            
            if doplots:           #xlim=[0.005, 0.06]
              imagename = DATADIR + "/singlepixel/pixel_"+str(xx)+"-"+str(yy)
              function_plots.plots(depol, imagename, x_lambda, norm[yy,xx], a[yy,xx], curv[yy,xx], \
                frac[yy,xx], angle[yy,xx], RM[yy,xx], sigma[yy,xx], \
                  I[0,:,yy,xx], noiseI[0,:,yy,xx], Q[0,:,yy,xx], noiseQ[0,:,yy,xx], U[0,:,yy,xx], noiseU[0,:,yy,xx],\
                    X[0,:,yy,xx], noiseX[0,:,yy,xx], P[0,:,yy,xx], noiseP[0,:,yy,xx], xlim=[round(x_lambda.min(),3)-0.001, round(x_lambda.max(),2)+0.01], \
                      imgformat="pdf", saveplot=True)
                        

              f1 = open(imagename+"_fitresults_stokesI.txt", "w+")
              f1.write("normaliz:"+str(norm[yy,xx])+"  "+str(norm_neg[yy,xx])+"  "+str(norm_pos[yy,xx])+"\n"+
                      "a        :"+str(a[yy,xx])+"  "+str(a_neg[yy,xx])+"  "+str(a_pos[yy,xx])+"\n"+
                      "curvature:"+str(curv[yy,xx])+" "+str(curv_neg[yy,xx])+" "+str(curv_pos[yy,xx]))
              f1.close()

              f2 = open(imagename+"_fitresults_stokesQU_"+depol+".txt", "w+")
              f2.write("intr pol frac :"+str(frac[yy,xx])+"  "+str(frac_neg[yy,xx])+"  "+str(frac_pos[yy,xx])+"\n"+
                      "intr pol angl :"+str(angle[yy,xx])+"  "+str(angle_neg[yy,xx])+"  "+str(angle_pos[yy,xx])+"\n"+
                      "Rotat Measure :"+str(RM[yy,xx])+" "+str(RM_neg[yy,xx])+" "+str(RM_pos[yy,xx])+"\n"+
                      "Depolarization:"+str(sigma[yy,xx])+" "+str(sigma_neg[yy,xx])+" "+str(sigma_pos[yy,xx]))
              f2.close()

          step+=1
            
          k = 0
          jobs = []
          print (text, step)

  print ("")
  print ("Script has finished")
  print (datetime.datetime.now())
  if doplots:
    print ("Doing single pixel plots")
    os.system("tar -zcvf "+ DATADIR + "singlepixel.tar.gz "+ DATADIR + "singlepixel")
    #sys.exit()

  hdu = fits.open(glob.glob(DATADIR + "*_polint.fits")[0])
  del hdu[0].header['CUNIT3']
  del hdu[0].header['CTYPE4']
  del hdu[0].header['CRVAL4']
  del hdu[0].header['CDELT4']
  del hdu[0].header['CRPIX4']
  del hdu[0].header['CUNIT4']

  del hdu[0].header['PV2_1']
  del hdu[0].header['PV2_2']
  
  #### I
  name = ["I0", "a", "b"]
  pol_parms = [norm, a, curv]
  for i in range(len(pol_parms)):
    fits.writeto(DATADIR + source + "_QUfittingMap_"+name[i]+res+".fits", pol_parms[i], hdu[0].header, overwrite=True)

  #### I negative error
  name = ["I0_ErrNeg", "a_ErrNeg", "b_ErrNeg"]
  pol_parms = [norm_neg, a_neg, curv_neg]
  for i in range(len(pol_parms)):
    fits.writeto(DATADIR + source + "_QUfittingMap_"+name[i]+res+".fits", pol_parms[i], hdu[0].header, overwrite=True)

  #### I positive error
  name = ["I0_ErrPos", "a_ErrPos", "b_ErrPos"]
  pol_parms = [norm_pos, a_pos, curv_pos]
  for i in range(len(pol_parms)):
    fits.writeto(DATADIR + source + "_QUfittingMap_"+name[i]+res+".fits", pol_parms[i], hdu[0].header, overwrite=True)

  #### I educed chisquare
  fits.writeto(DATADIR + source + "_ChiSqRedMapI_"+depol+res+".fits", chisqredI, hdu[0].header, overwrite=True)


  #### QU
  if depol == "ExtDepol" or depol == "IntDepol":
    pol_parms = [frac, angle, RM, sigma]
    name = ["PolFrac", "IntrAngl", "RM", "sigmaRM"]
  else:
    pol_parms = [frac, angle, RM]
    name = ["PolFrac", "IntrAngl", "RM"]
  for i in range(len(pol_parms)):
    fits.writeto(DATADIR + source + "_QUfittingMap_"+name[i]+res+".fits", pol_parms[i], hdu[0].header, overwrite=True)

  #### QU negative error
  if depol == "ExtDepol" or depol == "IntDepol":
    pol_parms = [frac_neg, angle_neg, RM_neg, sigma_neg]
    name = ["PolFrac_ErrNeg", "IntrAngl_ErrNeg", "RM_ErrNeg", "sigmaRM_ErrNeg"]
  else:
    pol_parms = [frac_neg, angle_neg, RM_neg]
    name = ["PolFrac_ErrNeg", "IntrAngl_ErrNeg", "RM_ErrNeg"]
  for i in range(len(pol_parms)):
    fits.writeto(DATADIR + source + "_QUfittingMap_"+name[i]+res+".fits", pol_parms[i], hdu[0].header, overwrite=True)
    
  #### QU positive error
  if depol == "ExtDepol" or depol == "IntDepol":
    pol_parms = [frac_pos, angle_pos, RM_pos, sigma_pos]
    name = ["PolFrac_ErrPos", "IntrAngl_ErrPos", "RM_ErrPos", "sigmaRM_ErrPos"]
  else:
    pol_parms = [frac_pos, angle_pos, RM_pos]
    name = ["PolFrac_ErrPos", "IntrAngl_ErrPos", "RM_ErrPos"]
  for i in range(len(pol_parms)):
    fits.writeto(DATADIR + source + "_QUfittingMap_"+name[i]+res+".fits", pol_parms[i], hdu[0].header, overwrite=True)

  #### QU reduced chisquare
  fits.writeto(DATADIR + source + "_ChiSqRedMapQU_"+depol+res+".fits", chisqredQU, hdu[0].header, overwrite=True)





## MAIN SCRIPT
parser = argparse.ArgumentParser(description='Make QU fit pixel per pixel using MCMC for the uncertainties on the total intensity polarization paramenters (I0, a, b; p0, chi0, RM, sigmaRM)')

parser.add_argument('-i','--sourcename', help='Name of the region to fit to produce the polarization parameter maps', type=str)
parser.add_argument('--RA', help='RA of cluster in deg', required=False, type=float)
parser.add_argument('--DEC', help='DEC of cluster in deg', required=False, type=float)
parser.add_argument('--dx', help='RA size of the optical image in degree', required=False, type=float)
parser.add_argument('--dy', help='DEC size of the optical image in degree', required=False, type=float)
parser.add_argument('--depol', help='Depolarization model to use (None, ExtDepol, IntDepol)', required=True, type=str)
parser.add_argument('--ncpu', help='Number of CPU to use for the fit; by default it runs in parallel with 20 CPU', default=20, type=int)
parser.add_argument('--resol', help='Taper of the observations (e.g. TAPER50kpc)', required=False, type=str)
parser.add_argument('--maskthreshold', help='threshold for the QU fit, default=3.0', default=3.0, type=float)
parser.add_argument('--docheck', help='chek which are the pixels in the map above the noise threshold', action='store_true')
parser.add_argument('--doplots', help='set it to True if you want to have a plot of the fit for a single pixel; default=False', default=False, action='store_true')

args = vars(parser.parse_args())
source    = args['sourcename']
depol     = args['depol']
if args['resol']:
  res       = args['resol']
else:
  res = ''
ncpu      = args['ncpu']
#weight    = args['weight']
threshold = args['maskthreshold']


if res == '':
  DATADIR = './polarization/'+source+'/None/'
else:
  DATADIR = './polarization/'+source+'/'+res+'/'


print ("resolution   :", res)
polmap, hdr = fits.getdata(glob.glob(DATADIR + "*polint.fits")[0], header=True)
RMmap       = fits.getdata(glob.glob(DATADIR + "*phi.fits")[0])
stokesI = fits.getdata(DATADIR + "Idatacube"+res+".fits")
stokesQ = fits.getdata(DATADIR + "Qdatacube"+res+".fits")
stokesU = fits.getdata(DATADIR + "Udatacube"+res+".fits")

"""
QU fitting for a given region centred on [RA,DEC] with size [dx, dy]; if not provided, it uses all the fits image
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



if args['docheck']:
 #checkpixeltofit(polmap, threshold, coords=[ra, dec], width=[dx,dy])
 checkpixeltofit(polmap, threshold)
 cmd = "ds9 " + DATADIR + "goodpixels_"+source+"_thrsh"+str(threshold)+".fits &"
 print (cmd)
 os.system(cmd)

else:
  QUfitting(source, polmap, RMmap, stokesI, stokesQ, stokesU, res, threshold, depol, ncpu, xmin, xmax, ymin, ymax, doplots = args['doplots'])
  #pixelfitting(source, polmap, RMmap, stokesI, stokesQ, stokesU, res, threshold, depol, ncpu, doplots = args['doplots'])

sys.exit()




















