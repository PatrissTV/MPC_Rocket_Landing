import shutil, sys, os.path, math, time, copy

import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from numpy import eye, array, asarray, exp
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float': '{: 6.4f}'.format})
from math import sqrt

#%matplotlib inline
import matplotlib
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
rcParams["font.serif"] = "cmr14"
rcParams['savefig.dpi'] = 300
rcParams["figure.dpi"] = 100
rcParams.update({'font.size': 18})
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
params = {'legend.fontsize': 12,
          'legend.handlelength': 2}
plt.rcParams.update(params)

import scipy
from scipy import optimize 
from scipy.optimize import fsolve, line_search, minimize, LinearConstraint, NonlinearConstraint, Bounds

import scipy.linalg as la
from scipy.linalg import expm, solve_continuous_lyapunov, sqrtm

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

from platform import python_version
print("Running Python:",python_version())

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def nicegrid(ax=plt):
    ax.grid(True, which='major', color='#666666', linestyle=':')
    ax.grid(True, which='minor', color='#999999', linestyle=':', alpha=0.2)
    ax.minorticks_on()

def sat(x , ll = -1, uu = 1):
    if x > uu:
        return uu
    elif x < ll:
        return ll
    return x

def lqr(A,B,Rxx,Ruu):
    Pss = la.solve_continuous_are(A, B, Rxx, Ruu)
    Kss = np.linalg.inv(np.atleast_2d(Ruu))@B.T@Pss
    return Kss, Pss
