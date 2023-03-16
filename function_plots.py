# FUNCTION FOR STOKES I Q U AND POLARIZATION ANGLE AND FRACTION PLOTS
import numpy as np
import sys, os

#plots
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['patch.force_edgecolor'] = True

def models(depol, x, norm, a, curv, p0, X0, RM, sigma, obs_i, err_i, obs_q, err_q, obs_u, err_u):
  freq = (3.e8/np.sqrt(x))*1.e-9
  mod_i = norm * freq**(a + curv*np.log10(freq))
  
  if len(x) == len(obs_i):
    dof = len(x)-2
    chisqred_i = (sum( ((obs_i - mod_i)/err_i)**2 ))/dof
  else:
    chisqred_i = None
    
  if np.isfinite(sigma):
    if depol == "ExtDepol":
      f = np.exp(-2*(sigma**2)*(x**2))
    elif depol == "IntDepol":
      f = (1.-np.exp(-2*(sigma**2)*(x**2)))/(2*(sigma**2)*(x**2))    
    dof = 2*len(x)-4
  else:
    f = 1
    dof = 2*len(x)-3
    
  mod_q = p0*f*mod_i *np.cos(2.* (X0 + RM*x))
  mod_u = p0*f*mod_i *np.sin(2.* (X0 + RM*x))
    
  if len(x) == len(obs_q):
    chisqred_qu = (sum( ((obs_q - mod_q)/err_q)**2 + ((obs_u - mod_u)/err_u)**2 ))/dof  
  else:
    chisqred_qu = None
    
  mod_X = 0.5*np.arctan2(mod_u,mod_q)
  mod_p = np.sqrt(mod_q**2 + mod_u**2) / mod_i
  
  return(mod_i, mod_q, mod_u, chisqred_i, chisqred_qu, mod_X, mod_p)



def plots(depol, imagename, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u, y_X, err_X, y_p, err_p, xlim=[0.005, 0.06], imgformat="pdf",saveplot=True):
  
  def freq_to_lambda(freq):
    wave_sq = (3.e8/freq)**2
    return(wave_sq)

  #freq = (3.e8/np.sqrt(x_lambda))/1e9
  #freqmin = (3.e8/np.sqrt(x_lambda))/1e9
  #freqmax = round(freq.max()+0.01,1)
  #dfreq = round((freqmax-freqmin)/6,1)
  #freqs = np.arange(freqmin, freqmax+dfreq, dfreq)*1e9
  freqs = np.array([1.25,1.5,2.,2.5,3.,3.5])*1.e9
  freqticks = np.array([freq_to_lambda(freq) for freq in freqs])
  
  fig = plt.figure(figsize=(10,14))
  gs = gridspec.GridSpec(6,1,height_ratios = [4,1,4,1,4,1], hspace=0.)
  ax1 = fig.add_subplot(gs[0])
  ax2 = fig.add_subplot(gs[2],sharex = ax1)  
  ax3 = fig.add_subplot(gs[4],sharex = ax1)
  ax4 = ax1.twiny() 
  plt.tight_layout(pad=6)

  ax1.grid(color='lightgrey', linestyle=':', linewidth=0.7)
  ax2.grid(color='lightgrey', linestyle=':', linewidth=0.7)
  ax3.grid(color='lightgrey', linestyle=':', linewidth=0.7)

  ax1.set_ylabel('$I$ [mJy]', fontsize=28)
  ax2.set_ylabel('$Q$ [mJy]', fontsize=28)
  ax3.set_ylabel('$U$ [mJy]', fontsize=28)
  ax4.set_xlabel(r'$\nu$ [GHz]', fontsize=28)
  
  ax1.tick_params(axis='y',labelsize=20)
  ax2.tick_params(axis='y',labelsize=20)
  ax3.tick_params(axis='y',labelsize=20)
  ax4.tick_params(axis='x',labelsize=20)
  
  ax2.xaxis.set_ticks_position('both')
  ax3.xaxis.set_ticks_position('both')
  ax1.yaxis.set_ticks_position('both')
  ax2.yaxis.set_ticks_position('both')
  ax3.yaxis.set_ticks_position('both')
  #ax2.yaxis.set_ticks([-0.2,-0.1,0,0.1,0.2])
  
  ax1.set_xlim([xlim[0], xlim[1]])
  ax4.set_xlim([xlim[0], xlim[1]])
  
  ax4.set_xticks(freqticks)
  ax4.set_xticklabels('{:.1f}'.format(freq) for freq in freqs/1.e9)
  
  xticklabels1 = ax1.get_xticklabels()
  xticklabels2 = ax2.get_xticklabels()
  xticklabels3 = ax3.get_xticklabels()
  plt.setp(xticklabels1, visible=False)
  plt.setp(xticklabels2, visible=False)
  plt.setp(xticklabels3, visible=False)
  #plt.suptitle(sourcetitles[row], fontsize=18)
  
  
  ax1.errorbar(x_lambda, y_i, yerr=err_i, fmt='o', color='k', mfc='none', markersize=6,elinewidth=1.)
  ax2.errorbar(x_lambda, y_q, yerr=err_q, fmt='o', color='k', mfc='none',markersize=6,elinewidth=1.)
  ax3.errorbar(x_lambda, y_u, yerr=err_u, fmt='o', color='k', mfc='none',markersize=6,elinewidth=1.)
  

  # print overlay models
  x = np.linspace(xlim[0], xlim[1], 1000)
  fit_i = models(depol, x, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[0]
  fit_q = models(depol, x, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[1]
  fit_u = models(depol, x, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[2]

  chisqred_i  = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[3]
  chisqred_qu = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[4]

  leg1, = ax1.plot(x, fit_i,'-',color='royalblue',linewidth=5.,zorder=len(y_i)+1, label=r"$\chi_{\rm red, I}^2=$"+str(round(chisqred_i,2)))
  leg2, = ax2.plot(x, fit_q,'-',color='royalblue',linewidth=5.,zorder=len(y_q)+1, label=r"$\chi_{\rm red, QU}^2=$"+str(round(chisqred_qu,2)))
  ax3.plot(x, fit_u,'-',color='royalblue',linewidth=5.,zorder=len(y_u)+1)


  ## print res
  ax1res = fig.add_subplot(gs[1],sharex = ax1)
  ax2res = fig.add_subplot(gs[3],sharex = ax1)
  ax3res = fig.add_subplot(gs[5],sharex = ax1)
  
  ax1res.set_xlim([xlim[0], xlim[1]]) 
  ax2res.set_xlim([xlim[0], xlim[1]]) 
  ax3res.set_xlim([xlim[0], xlim[1]]) 
  #axres.set_ylim([-0.3, 0.3])
  #ax2res.set_ylim([-0.3, 0.3])
  #ax3res.set_ylim([-0.15, 0.15])


  ax1res.grid(color='lightgrey', linestyle=':', linewidth=0.7)
  ax2res.grid(color='lightgrey', linestyle=':', linewidth=0.7)
  ax3res.grid(color='lightgrey', linestyle=':', linewidth=0.7)

  ax1res.set_ylabel('res', fontsize=24)
  ax2res.set_ylabel('res', fontsize=24)
  ax3res.set_ylabel('res', fontsize=24)
  ax3res.set_xlabel(r'$\lambda^2$ [m$^2$]', fontsize=28)
  
  ax1res.tick_params(axis='y',labelsize=16)
  ax2res.tick_params(axis='y',labelsize=16)
  ax3res.tick_params(axis='y',labelsize=16)
  ax3res.tick_params(axis='x',labelsize=20)
  
  ax1res.xaxis.set_ticks_position('both')
  ax2res.xaxis.set_ticks_position('both')
  ax3res.xaxis.set_ticks_position('both')
  ax1res.yaxis.set_ticks_position('both')
  ax2res.yaxis.set_ticks_position('both')
  ax3res.yaxis.set_ticks_position('both')  
  
  xticklabels1 = ax1res.get_xticklabels()
  xticklabels2 = ax2res.get_xticklabels()
  xticklabels3 = ax3res.get_xticklabels()
  plt.setp(xticklabels1, visible=False)
  plt.setp(xticklabels2, visible=False)
  
  model_i = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[0]
  model_q = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[1]
  model_u = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[2]
    
  
  ax1res.errorbar(x_lambda, y_i-model_i, yerr=err_i,fmt='.',markersize=4,color='k',alpha=0.75)
  ax1res.plot(x,[0]*1000,'--',color='royalblue',linewidth=2.5, zorder=len(y_i)+1)
  ax2res.errorbar(x_lambda, y_q-model_q, yerr=err_q,fmt='.',markersize=4,color='k',alpha=0.75)
  ax2res.plot(x,[0]*1000,'--',color='royalblue',linewidth=2.5, zorder=len(y_q)+1)
  ax3res.errorbar(x_lambda, y_u-model_u, yerr=err_u,fmt='.',markersize=4,color='k',alpha=0.75)
  ax3res.plot(x,[0]*1000,'--',color='royalblue',linewidth=2.5, zorder=len(y_u)+1)

  #leg1= r"$\chi_{\rm red, I}^2=$"+str(round(chisqred_i,2))
  #leg2= r"$\chi_{\rm red, QU}^2=$"+str(round(chisqred_qu,2))
  #ax1.text(0.02, 0.9, leg1,  fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes,  bbox=dict(facecolor='white', edgecolor='lightgrey', alpha=0.7,  boxstyle='round'))  
  #ax2.text(0.02, 0.9, leg2,  fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes,  bbox=dict(facecolor='white', edgecolor='lightgrey', alpha=0.7,  boxstyle='round'))  
  
  #ax1.legend(handles=[leg1], loc="upper left", prop={'size':14})
  #ax2.legend(handles=[leg2], loc="upper left", prop={'size':14})
  
  plt.tight_layout()
  
  if saveplot:
    plt.savefig(imagename+"_stokesIQUvslambdasq"+depol+"."+imgformat)
    #plt.show()
  
  
  #PLOT CHI and POL FRACTION
  fig = plt.figure(figsize=(10,14))
  gs = gridspec.GridSpec(4,1,height_ratios = [4,1,4,1], hspace=0)
  #plt.tight_layout()
  ax1 = fig.add_subplot(gs[0])
  ax2 = fig.add_subplot(gs[2], sharex = ax1)
  ax3 = ax1.twiny()

  ax1.grid(color='lightgrey', linestyle=':', linewidth=0.7)
  ax2.grid(color='lightgrey', linestyle=':', linewidth=0.7)

  ax1.set_ylabel(r'$\chi = \frac{1}{2} \arctan \left ( \frac{U}{Q} \right )$ [rad]', fontsize=28)
  ax2.set_ylabel(r'$p = \frac{\sqrt{Q^2 + U^2}}{I}$', fontsize=28) 
  ax3.set_xlabel(r'$\nu$ [GHz]', fontsize=28)
  
  ax1.tick_params(axis='y',labelsize=20)
  ax2.tick_params(axis='y',labelsize=20)
  ax3.tick_params(axis='x',labelsize=20) 
  
  ax1.yaxis.set_ticks_position('both')
  ax2.yaxis.set_ticks_position('both')
  ax2.xaxis.set_ticks_position('both')

  ax1.set_xlim([xlim[0], xlim[1]])
  ax3.set_xlim([xlim[0], xlim[1]])
  #ax2.set_ylim([-0.05,0.65])
  #ax2.set_yticks([0.,0.1,0.2,0.3,0.4,0.5,0.6])
  
  ax3.set_xticks(freqticks)
  ax3.set_xticklabels('{:.1f}'.format(freq) for freq in freqs/1.e9)
  
  xticklabels1 = ax1.get_xticklabels()
  xticklabels2 = ax2.get_xticklabels()
  plt.setp(xticklabels1, visible=False)
  plt.setp(xticklabels2, visible=False)
  #plt.suptitle(sourcetitles[row], fontsize=18)


  idx = np.where(y_p+err_p > 1.5)
  err_p[idx] = None
  y_p[idx] = None
 
  ax1.errorbar(x_lambda, y_X, yerr=err_X, fmt='o', color='k', mfc='none', markersize=6, elinewidth=1.)
  ax2.errorbar(x_lambda, y_p, yerr=err_p, fmt='o', color='k', mfc='none', markersize=6, elinewidth=1.)

  # plot overlay models
  fit_X = models(depol, x, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[5]
  fit_p = models(depol, x, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[6]
  
  ax1.plot(x, fit_X,'-',color='royalblue',linewidth=5., zorder=len(y_X)+1,label=r"$\rm RM=$"+str("%.2f" % RM+"rad m$^{-2}$"))
  ax2.plot(x, fit_p,'-',color='royalblue',linewidth=5.,zorder=len(y_p)+1)
  
  # print res
  ax1res = fig.add_subplot(gs[1],sharex = ax1)
  ax2res = fig.add_subplot(gs[3],sharex = ax1)
  
  ax1res.set_xlim([xlim[0], xlim[1]])
  ax2res.set_xlim([xlim[0], xlim[1]])
  #ax1res.set_ylim([-1, 1])
  #ax2res.set_ylim([-1.5, 1.5])

  ax1res.grid(color='lightgrey', linestyle=':', linewidth=0.7)
  ax2res.grid(color='lightgrey', linestyle=':', linewidth=0.7)

  ax1res.set_ylabel('res', fontsize=24)
  ax2res.set_ylabel('res', fontsize=24)
  ax2res.set_xlabel(r'$\lambda^2$ [m$^2$]', fontsize=28)
  
  ax1res.tick_params(axis='y',labelsize=16)
  ax2res.tick_params(axis='y',labelsize=16)
  ax2res.tick_params(axis='x',labelsize=20)
  
  ax1res.xaxis.set_ticks_position('both')
  ax2res.xaxis.set_ticks_position('both')
  ax1res.yaxis.set_ticks_position('both')
  ax2res.yaxis.set_ticks_position('both')
  
  xticklabels1 = ax1res.get_xticklabels()
  xticklabels2 = ax2res.get_xticklabels()
  plt.setp(xticklabels1, visible=False)


  model_X = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[5]
  model_p = models(depol, x_lambda, norm, a, curv, p0, X0, RM, sigma, y_i, err_i, y_q, err_q, y_u, err_u)[6]

  y_X[np.where(y_X >= model_X + np.pi/2.)] -= np.pi
  y_X[np.where(y_X <= model_X - np.pi/2.)] += np.pi
  
  
  ax1res.errorbar(x_lambda, y_X - model_X, yerr=err_X,fmt='.',markersize=4,color='k',alpha=0.75)
  ax1res.plot(x,[0]*1000,'--',color='royalblue',linewidth=2.5, zorder=len(y_X)+1)
  ax2res.errorbar(x_lambda, y_p - model_p,yerr=err_p,fmt='.',markersize=4,color='k',alpha=0.75)
  ax2res.plot(x,[0]*1000,'--',color='royalblue',linewidth=2.5, zorder=len(y_p)+1)
  
  plt.tight_layout()
  
  if saveplot:
    plt.savefig(imagename+"_pXvslambdasq"+depol+"."+imgformat)
    #plt.show()
    
  return
