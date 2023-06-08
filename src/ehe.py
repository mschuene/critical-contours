from utils import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wurlitzer import sys_pipes
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
U = 1 # upper limit
N = 10000
epsilon = 1 # reset parameter
deltaU = 0.022
alpha = 1 - 1/np.sqrt(N)

def ehe_critical_weights(N,U=1):
    return np.ones([N,N])*(1 - 1/np.sqrt(N)) * U / N

def simulate_model(units=np.random.random(N)*U,numAvalanches=1000,
                   W=None,deltaU = deltaU):
    if W is None:
        W = ehe_critical_weights(len(units))
    avalanche_sizes = []
    avalanche_durations = []
    sumavd = 0
    while sumavd < numAvalanches:
        r = np.random.randint(len(units))
        units[r] += deltaU
        if units[r] >= U:
            avs,avd = handle_avalanche(r,units,W,epsilon=1,U=U)
            avalanche_sizes.append(avs)
            avalanche_durations.append(avd)
            sumavd += avd
    return avalanche_sizes,avalanche_durations

def handle_avalanche(start_unit,units,W,epsilon=1,U=U):
    avalanche_size = 0
    avalanche_duration = 0
    # handle starting single unit avalanche
    A = np.zeros_like(units)
    A[start_unit] = 1
    units[start_unit] = epsilon * (units[start_unit] - U)
    s = 1
    while s > 0:
        avalanche_size += s
        avalanche_duration += 1
        units += W * A #interior input to all units
        A = units >= U
        units[A] = epsilon * (units[A] - U) # resetting threshold crossed units
        s = np.sum(A)
    return avalanche_size,avalanche_duration

from functools import reduce
import itertools

def spiking_configurations(iterable):
    "powerset([1,2,3]) --> reverse of (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s),0,-1))

def intersection(hyperr1,hyperr2):
    """compute intersection of two hyperrectangles given by list of  end values along each axes.
       So the rectangles always have their lower values at 0"""
    if len(hyperr1) == 0 or len(hyperr2) == 0:
        return []
    intersection  = []
    for (u1l,u1u),(u2l,u2u) in zip(hyperr1,hyperr2):
        il,iu = (max(u1l,u2l),min(u1u,u2u))
        if iu <= il:
            return []
        intersection.append((il,iu))
    return intersection

def volume(hyperr):
    if len(hyperr) == 0:
        return 0
    return prod(hu - hl for hu,hl in hyperr)

def volume_union(hyperrectangles):
    # uses approach from https://stackoverflow.com/questions/28146260
    # TODO use best known algorithm for Klee's measure problem
    # https://pdfs.semanticscholar.org/9528/ef345d013577fd6c9d7fde8a54a1b86d3bbe.pdf
    dim = len(hyperrectangles[0])
    gridlines = [np.unique([hyperr[i][0] for hyperr in hyperrectangles] +
                           [hyperr[i][1] for hyperr in hyperrectangles]) for i in range(dim)]
    grid = np.zeros([len(gl)-1 for gl in gridlines],dtype=bool) #TODO Better to create grid on the fly to save space?
    #fill the grid
    for hyperr in hyperrectangles:
        grid_slices = tuple(slice(np.searchsorted(gridlines[i],hyperr[i][0]),np.searchsorted(gridlines[i],hyperr[i][1]))
                            for i in range(dim))
        grid[grid_slices] = True #indicate grid cells covered by hyperr
    # calculate total volume from grid and gridlines
    it = np.nditer(grid, flags=['multi_index'])
    volume = 0
    while not it.finished:
        if(it[0]): # for each filled cell compute it's volume and add it
            volume += prod(grid[i+1] - grid[i] for grid,i in zip(gridlines,it.multi_index))
        it.iternext()
    return volume

def R_sp(W,spiking_configuration,num_dims):
    """returns excluded hyper rectangle induced by the spiking_configuration"""
    ind_sp = np.zeros(num_dims,dtype=int)
    for s in spiking_configuration:
        ind_sp[s] = 1
    limits = (W @ ind_sp) # Don't rescale here
    sp_comp = np.ones_like(ind_sp) - ind_sp # 1 at nodes that are not in sp and 0 otherwise
    return [(0,x) for x in np.maximum(limits,sp_comp)]

def noninhabited_region(units,W,N=None):
    if N is None:
      N = len(units)
    return volume_union([R_sp(W,sp,N) for sp in spiking_configurations(units)])

from itertools import chain, combinations

def prod(gen):
    res = 1;
    for i in gen:
        res *= i
    return res

def nonempty_subsets(iterable):
    "nonempty_subsets([1,2,3]) -->  (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)))

class NoninhabitedRegion():
    def __init__(self,W):
        self.results = {}
        self.W = W;

    def noninhabited_region(self,N=None,U=None):
        W = self.W
        if N is None:
            N = range(W.shape[0])
        elif type(N) == int:
            N = range(N)
        N = tuple(N)
        if U is None:
            U = np.ones(len(N))
        U = np.array(U)
        res = self.results.get((N,tuple(U)))
        if res is not None:
            res
        row_sums = np.array([sum(W[i,j] for j in N) for i in N]) # works with sympy
        res = prod(row_sums)#Gamma with I=N
        for I in nonempty_subsets(N):
            I_inds = np.in1d(N,I)
            res += self.noninhabited_region(I,U=row_sums[I_inds]) * prod(U[~I_inds]-row_sums[~I_inds])
        self.results[(N,tuple(U))] = res
        return res

import sympy as sym
def symbolic_matrix(N):
     W = np.empty((N,N),dtype=object);
     for i in range(N):
         for j in range(N):
             W[i,j] = sym.symbols('w'+str(i)+str(j))
     return W

from sympy.combinatorics import Permutation
def symbolic_det(W,I):
    ret = 0
    for perm_idx in itertools.permutations(range(len(I))):
        ret += Permutation(perm_idx).signature() * prod(W[i,I[j_idx]] for i,j_idx in zip(I,perm_idx))
    return ret

def hypothesis_volume(W,N):
    if type(N) == int:
        N = range(N)
    return sum((-1 if len(I) % 2 == 0 else 1) * symbolic_det(W,I) for I in spiking_configurations(N))

def test_conjecture(N=5):
    W = symbolic_matrix(N)
    NR = NoninhabitedRegion(W)
    Lambda = NR.noninhabited_region()
    print('Lambda',Lambda,flush=True)
    Lambda = sym.simplify(Lambda)
    Lambda_bar = hypothesis_volume(W,N)
    print('Lambda_bar',Lambda_bar,flush=True)
    Lambda_bar = sym.simplify(Lambda_bar)
    print(sym.simplify(Lambda - Lambda_bar))

from sympy import Symbol, Rational, binomial, expand_func
from sympy.utilities.iterables import partitions

def gen_partitions(N):
    for p in partitions(N,m=N):
        yield reduce(lambda a,b:a+b,([k]*v for k,v in p.items()))


def coef_i(p,i,N):
    return sum((-1)**(len(J) - 1)*binomial(N - sum(J),i - sum(J)) for J in (list(nonempty_subsets(p)) + [p]))

def coef(p,N):
    return sum((-1)**(N-i)*coef_i(p,i,N) for i in range(1,N+1))


def sign(p,N):
    if sum(p) != N:
        return 0;
    return (-1)**(sum(lz-1 for lz in p))

def expected_sign(p,N): 
    return (-1)**(N-1)*sign(p,N)

def points2pdf(points,lower_limit=1,upper_limit=None,return_unique=True):
    if upper_limit is None:
        upper_limit = max(points)
    unique,counts = np.unique(points,return_counts=True)
    pdf = np.zeros(upper_limit - lower_limit + 1)
    for u,c in zip(unique,counts):
        pdf[u - lower_limit] = c
    if return_unique:
        return (pdf/np.sum(pdf),np.array(list(range(lower_limit,upper_limit+1))))
    return pdf/np.sum(pdf)

import matplotlib.pyplot as plt

def loglogplot(pdf,ax = None, figsize=(6,4),from_pdf=False,label=None):
    if from_pdf:
        norm_counts,unique = pdf
    else:
        points = pdf
        unique,counts = np.unique(points,return_counts=True)
        norm_counts = counts/np.sum(counts)
    if ax is None:
        f,ax = plt.subplots(figsize=figsize)
    ax.loglog(unique,norm_counts,label=label)
    #ax.set_xscale('log',nonposx='mask');
    #ax.set_yscale('log',nonposy='mask');
    #ax.scatter(unique,norm_counts,facecolors='none',edgecolors='b')
    return ax

def storekey(res,key):
   def inner(value):
       res[key] = value
   return inner
