# FUNCTIONS FOR THE CALCULATION OF THE VALUE OF POLARIZATION ANGLE AND FRACTION WITH MONTECARLO ERROR, AND SECOND ORDER FIT FOR STOKES I,Q,U AND POLARIZATION ANGLE AND FRACTION ASSUMING A GIVEN POLARIZATION MODEL (no depolarization, external depolarization and internal depolarization)
# G. Di Gennaro
# November 2018

import numpy as np
from numpy import r_
from scipy import optimize
import sys
import cmath
from scipy.stats import circstd


def angle(fluxQ, fluxU, fluxQ_err, fluxU_err):
  chi = 0.5 * np.arctan2(fluxU,fluxQ)

  chi_err = np.array([])
  for j in range(len(fluxQ)):
    chi_mc = np.array([])
    for i in range(150):
      fluxQ_mc = np.random.normal(fluxQ[j], scale=fluxQ_err[j])  
      fluxU_mc = np.random.normal(fluxU[j], scale=fluxU_err[j])
    
      func_chi = 0.5 * np.arctan2(fluxU_mc,fluxQ_mc)
      chi_mc = np.append(chi_mc, func_chi)
    
      chi_mc[np.where(chi[j] > chi_mc[i] + np.pi/2.)] += np.pi
      chi_mc[np.where(chi[j] < chi_mc[i] - np.pi/2.)] -= np.pi


    chi_err = np.append(chi_err, circstd(chi_mc, np.pi))
  return (chi, chi_err)


def fraction(fluxQ, fluxU, fluxI, fluxQ_err, fluxU_err, fluxI_err):
  #p = np.sqrt(fluxQ**2 + fluxU**2)
  p = np.sqrt((fluxQ**2 + fluxU**2) / fluxI**2)

  p_err = np.array([])
  for j in range(len(fluxQ)):
    p_mc = np.array([])
    for i in range(150):
      fluxQ_mc = np.random.normal(fluxQ[j], scale=fluxQ_err[j])  
      fluxU_mc = np.random.normal(fluxU[j], scale=fluxU_err[j])
      fluxI_mc = np.random.normal(fluxI[j], scale=fluxI_err[j])
      
      #func_p = np.sqrt(fluxQ_mc**2 + fluxU_mc**2)
      #p_mc = np.append(p_mc, func_p)
      func_p = np.sqrt((fluxQ_mc**2 + fluxU_mc**2) / fluxI_mc**2)
      p_mc = np.append(p_mc, func_p)
    
    p_err = np.append(p_err,np.std(p_mc))
  return (p, p_err)

