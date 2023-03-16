import numpy
import corner
import emcee
import matplotlib.pyplot
import matplotlib #as mpl
#import matplotlib.ticker as mtick
#import sys
#from tabulate import tabulate
import scipy.optimize
import sys, glob



def polarization_fitting(depol, wave, fluxI, err_fluxI, fluxQ, err_fluxQ, fluxU, err_fluxU, rm, x, y, imagename, docornerplot):

  matplotlib.rcParams['xtick.direction']='in'
  matplotlib.rcParams['ytick.direction']='in'
  matplotlib.rcParams['xtick.labelsize']= 14
  matplotlib.rcParams['ytick.labelsize']= 14

  #print (depol)
  ######################################################################################################
  # MCMC SPIX
  #def lnprior_spix(theta):
    #norm, a, curv = theta
  
    #return 0.0
    
  def lnprior_spix(theta):
    norm, a, curv = theta
    if norm>=0. and -numpy.inf<a<numpy.inf and -numpy.inf<curv<numpy.inf:
      return 0.

    return -numpy.inf

  def lnlike_spix(theta,x,y,err): 
    norm, a, curv = theta  
    model = norm * x**(curv*numpy.log10(x) + a)  #see Massaro+04, A&A, 413, 489
    
  
    return -0.5*numpy.sum((y - model)**2/err**2)

  def lnprob_spix(theta,x,y,err):
    lp = lnprior_spix(theta)
    if not numpy.isfinite(lp):
      return -numpy.inf
  
    return lp + lnlike_spix(theta, x, y, err)



  # MCMC POLARIZATION 
  def lnprior_pol(theta):
    if depol == "ExtDepol" or depol == "IntDepol":
      p0, chi0, rm, sigma_rm = theta
      if 0.<=p0<=1. and -400.<=rm<=400. and sigma_rm>=0.: #and 0.<=chi0<numpy.pi 
        return 0.	
    else:
      p0, chi0, rm = theta
      if 0.<=p0<=1. and -400.<=rm<=400.: #and 0.<=chi0<numpy.pi:
        return 0.

    return -numpy.inf


  def lnlike_pol(theta,x,y1,err1,y2,err2,norm,a,curv): 
    if depol == "ExtDepol" or depol == "IntDepol":
      p0, chi0, rm, sigma_rm = theta
    
      freq = (3.e8/numpy.sqrt(x))*1.e-9
      stokes_i = norm * freq**(curv*numpy.log10(freq) + a)
    
      if depol == "ExtDepol":
        model1 = stokes_i * p0 * numpy.exp(-2*(sigma_rm)*(x**2)) * numpy.cos(2*(chi0 + rm*x))
        model2 = stokes_i * p0 * numpy.exp(-2*(sigma_rm)*(x**2)) * numpy.sin(2*(chi0 + rm*x))
      elif depol == "IntDepol":
        model1 = stokes_i * p0 * ((1.-numpy.exp(-2*(sigma_rm)*(x**2)))/(2*(sigma_rm)*(x**2))) * numpy.cos(2*(chi0 + rm*x))
        model2 = stokes_i * p0 * ((1.-numpy.exp(-2*(sigma_rm)*(x**2)))/(2*(sigma_rm)*(x**2))) * numpy.sin(2*(chi0 + rm*x))
      
    else:
      p0, chi0, rm  = theta
    
      freq = (3.e8/numpy.sqrt(x))*1.e-9
      stokes_i = norm * freq**(curv*numpy.log10(freq) + a)
      
      model1 = stokes_i * p0 * numpy.cos(2*(chi0 + rm*x))
      model2 = stokes_i * p0 * numpy.sin(2*(chi0 + rm*x))    

    return -0.5 * numpy.sum( ((y1 - model1)/err1)**2 + ((y2 - model2)/err2)**2  )


  def lnprob_pol(theta,x,y1,err1,y2,err2,norm,a,curv):
    lp = lnprior_pol(theta)
    if not numpy.isfinite(lp):
      return -numpy.inf

    return lp + lnlike_pol(theta, x, y1, err1,y2,err2,norm,a,curv)



  def cornerplot(depol, samples, labelnames, stokes, imagename):
    labelist = [labelnames[i] for i in range(len(labelnames))]
  
    corner.corner(samples, labels=labelist, bins=[20]*ndim, label_kwargs={"fontsize": 20}, \
      show_titles=False, title_kwargs={"fontsize": 8}, title_fmt='.3f', quantiles=[0.16, 0.5, 0.84])
  
    matplotlib.pyplot.savefig(imagename+"_cornerplot"+stokes+depol+".pdf")
    #matplotlib.pyplot.show()
    matplotlib.pyplot.close()
    '''
    fig, axes = matplotlib.pyplot.subplots(ncols=1, nrows=4)
    fig.set_size_inches(12,6)
    axes[0].plot(sampler.chain[:, :, 0].transpose(), color='black', alpha=0.05)
    axes[0].set_ylabel(labelname[0])
    axes[0].axvline(burn, ls='dashed', color='red', label = 'End of burn-in')
    axes[1].plot(sampler.chain[:, :, 1].transpose(), color='black', alpha=0.05)
    axes[1].set_ylabel(labelname[1])
    axes[1].axvline(burn, ls='dashed', color='red')
    axes[2].plot(sampler.chain[:, :, 2].transpose(), color='black', alpha=0.05)
    axes[2].set_ylabel(labelname[2])
    axes[2].axvline(burn, ls='dashed', color='red')
    axes[3].plot(sampler.chain[:, :, 3].transpose(), color='black', alpha=0.05)
    axes[3].set_ylabel(labelname[3])
    axes[3].axvline(burn, ls='dashed', color='red')
    axes[3].set_xlabel('Step number')
    fig.tight_layout()
    fig.suptitle('MCMC tracks trough parameter space')
    matplotlib.pyplot.savefig(imagename+"_MCMCburns"+stokes+depol+".pdf")
    matplotlib.pyplot.close()
    '''
    
######################################################################################################    
    

  # STOKES I FIT
  freq = (3.e8/numpy.sqrt(wave))*1.e-9
  
  func = lambda p, i: p[0] * i**(p[1]+p[2]*numpy.log10(i))
  guess= [1.0,-1.0,0] #norm, a, curv
  
  errfunc = lambda p, i, j, err: (j - func(p, i)) / err
  out = scipy.optimize.leastsq(errfunc, guess, args=(freq, fluxI, err_fluxI), full_output=True)
           
  coeff = out[0]
  norm = coeff[0]
  a = coeff[1]
  curv = coeff[2]
  
  #alpha = a + 2.*curv*numpy.log10(2)
  
  #print (x, y)
  
  parms = [norm,a,curv]
  
  ndim, nwalkers = len(parms), 200
  pos = [parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
  
  runs = 1000 #500 ##GDG: fixed by Wout
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_spix, args=(freq, fluxI, err_fluxI))
  
  sampler.run_mcmc(pos, runs)
  burn = 200 #100 ##GDG: fixed by Wout
  samples = sampler.chain[:, burn:, :].reshape((-1, ndim))


  #### FIT SUMMARY
  lim_low = 50. - (68.27/2.)
  lim_upp = 50. + (68.27/2.)
  val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                   for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
  labelnames = ["$I_0$ [mJy]","$a$","$b$"]
  fit_dataI = [[labelnames[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]  
  
  norm = fit_dataI[0][1]
  a    = fit_dataI[1][1]
  curv = fit_dataI[2][1]
  
  if docornerplot:
    cornerplot(depol, samples, labelnames, "I", imagename)
    
    
  ####
  
  # STOKES QU FIT
  stokes_i = norm * (freq)**(a + curv*numpy.log10(freq))

    
  func = lambda p, x: 2.*(p[1] + p[2]*x) 


  
  if depol == "":
    guess = [1,1,rm]
    
    errfunc = lambda p, x1, y1, err1, x2, y2, err2,stokes_i: \
      abs( (y1 - (stokes_i * p[0]*numpy.cos(func(p,x1))))/err1 ) + \
      abs( (y2 - (stokes_i * p[0]*numpy.sin(func(p,x2))))/err2 )

    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i), full_output=True)
    coeff = out[0]
    covar = out[1]
    
    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]
    elif coeff[0] > 1:
      coeff[0] = 1

    '''
    #GDG: fixed by Wout
    # boundaries on chi0    
    if coeff[1] >= numpy.pi:
      coeff[1] = coeff[1] -numpy.pi  
    if coeff[1] < 0.:
      coeff[1] = coeff[1] + numpy.pi
    '''

    # boundaries on chi0    
    if coeff[1] >= numpy.pi or coeff[1] < 0:
      coeff[1] = coeff[1] % numpy.pi

    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = numpy.nan
    
    parms = [p0,chi0,rm_fit]
 
    ndim, nwalkers = len(parms), 200
    pos = [parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
  
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol, args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    
    runs = 1000 #500
    sampler.run_mcmc(pos, runs)
    burn = 200 #100
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    labelnames = ["$p_0$",r"$\chi_0$ [rad]","RM [rad m$^{-2}$]"]
    fit_data = [[labelnames[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]    

    if docornerplot:
      cornerplot(depol, samples, labelnames, "QU", imagename)


    
  elif depol == "ExtDepol":
    guess = [0.6,1,rm,1]
    errfunc = lambda p, x1, y1, err1, x2, y2, err2, stokes_i: \
      abs( (y1 - (stokes_i * (p[0]*numpy.exp(-2*(p[3])*(x1**2))) * numpy.cos(func(p,x1))) )/err1 ) + \
      abs( (y2 - (stokes_i * (p[0]*numpy.exp(-2*(p[3])*(x1**2))) * numpy.sin(func(p,x2))) )/err2 )

    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i))        
    coeff = out[0]

    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]

    '''
    # boundaries on chi0    
    if coeff[1] >= numpy.pi:
      coeff[1] = coeff[1] -numpy.pi  
    if coeff[1] < 0.:
      coeff[1] = coeff[1] + numpy.pi
    '''

   # boundaries on chi0
    if coeff[1] >= numpy.pi or coeff[1] < 0:
        coeff[1] = coeff[1] % numpy.pi

    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = abs(coeff[3])
    
    parms = [p0,chi0,rm_fit,sigma_rm]
    

    ndim, nwalkers = len(parms), 200
    pos = [parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
  
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol, args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    
    runs = 1000 #500
    sampler.run_mcmc(pos, runs)
    burn = 300 #200
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    
    samples[:,1] = samples[:,1] % numpy.pi # Range of chi0 was set to -pi to pi to avoid boundary problem -> need to mod samples    
    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    labelnames = ["$p_0$",r"$\chi_0$ [rad]","RM [rad m$^{-2}$]",  r"$\sigma_{\rm RM}^2$ [rad$^{-2}$ m$^{-4}$]"]
    fit_data = [[labelnames[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]    

    if docornerplot:
      cornerplot(depol, samples, labelnames, "QU", imagename)


  elif depol == "IntDepol":
    guess = [1,1,rm,1]
    
    errfunc = lambda p, x1, y1, err1, x2, y2, err2, stokes_i: \
      abs( (y1 - (stokes_i * p[0] * ((1. - numpy.exp(-2*(p[3])*(x1**2))) / (2*(p[3])*(x1**2))) * numpy.cos(func(p,x1))) )/err1 ) + \
      abs( (y2 - (stokes_i * p[0] * ((1. - numpy.exp(-2*(p[3])*(x1**2))) / (2*(p[3])*(x1**2))) * numpy.sin(func(p,x2))) )/err2 )
  
    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i), full_output=True)
    coeff = out[0]

    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]
    
    '''
    # boundaries on chi0    
    if coeff[1] >= numpy.pi:
      coeff[1] = coeff[1] -numpy.pi
    if coeff[1] < 0.:
      coeff[1] = coeff[1] + numpy.pi
    '''

    # boundaries on chi0    
    if coeff[1] >= numpy.pi or coeff[1] < 0:
      coeff[1] = coeff[1] % numpy.pi

    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = abs(coeff[3])

    parms = [p0,chi0,rm_fit,sigma_rm]

    ndim, nwalkers = len(parms), 200
    pos = [parms + 1e-4*numpy.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol, args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    

    runs = 1000 #500
    sampler.run_mcmc(pos, runs)
    burn = 200
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))    

    #samples[:,:,2] = samples[:,:,2]%np.pi #### FOLDING THE ANGLE TO BE BETWEEN 0 AND PI

    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    
    labelnames = ["$p_0$",r"$\chi_0$ [rad]","RM [rad m$^{-2}$]", r"$\varsigma_{\rm RM}^2$ [rad m$^{-2}$]"]
    fit_data = [[labelnames[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]    

    if docornerplot:
      cornerplot(depol, samples, labelnames, "QU", imagename)
    
    
  return(fit_dataI, fit_data, x, y)
  #return(fit_data, x, y)
  #return(fit_dataI,fit_data)
