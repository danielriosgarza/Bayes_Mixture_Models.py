from __future__ import division
import numpy as np
import math
import random

from Cholesky import Cholesky
from Gauss_Wishart_probability_model import Gauss_Wishart_model as GWM
from Gaussian_component import Gaussian_component as GC
from Gaussian_variable import Gaussian_variable as GV
from sklearn.cluster import KMeans as km
from scipy.misc import logsumexp as lse

class Gaussian_clusters:
    def __init__(self, k=1, Xm=None, alpha=None, nk=None):
        self.d = self.__d(Xm)
        
        self.n = self.__n(Xm)
        
        self.k=k
        
        self.Xm=Xm
        
        self.alpha=alpha
        
        self.nk = self.__nk()#assign nth data-point to the kth cluster
        
        self.pi = self.__pi()
        
        self.GaussComp = self.__GaussComp()
        
        self.pnk = None # assignment probabilities of the nth datapoint to the kth cluster
        
    
    
    def __d(self, Xm=None):
        if Xm is None:
            self.d=1
        elif len(Xm.shape)==1:
            self.d=1
        else:
            self.d = len(Xm[0])
        return self.d
        
        
    def __n(self, Xm=None):
        if Xm is None:
            self.n =0
        elif self.d==1:
            self.n = len(np.array(Xm).flatten())
        else:
            self.n=len(Xm.flatten())/self.d
        return int(self.n)
        
    def __nk(self, nk=None):
        if self.Xm is None:
            self.nk = None
        elif nk is None:
            #self.nk = np.random.randint(0,self.k, self.n) 
            self.nk = km(n_clusters=self.k).fit_predict(self.Xm)
        else:
            try:
                self.nk = nk
                self.nk.shape=(self.n,)
            except ValueError:
                print "nk must be of the same lenght as Xm"
        return self.nk
    
    def __pi(self):
        if self.k==1:
            self.pi=1
        else:
            self.pi = np.log(np.random.dirichlet([self.alpha+len(self.nk[self.nk==cluster])/self.alpha for cluster in xrange(self.k)]))
        return self.pi
        
    def __GaussComp(self):
        
        self.GaussComp={cluster: GC(d=self.d, X=self.Xm[self.nk==cluster]) for cluster in xrange(self.k)}
        
        return self.GaussComp
    
    def __pnk(self):
        self.pnk = np.zeros((self.n, self.k))
        gvs = {cluster: GV(d =self.GaussComp[cluster].d, S = self.GaussComp[cluster].chol_prec_rvs(), \
        mu = self.GaussComp[cluster].mu_rvs(), method='chol_prec') for cluster in xrange(self.k)}
        for datapoint in xrange(self.n):
            a =self.pnk[datapoint] = np.array([self.pi[cluster]+gvs[cluster].logp(self.Xm[datapoint])\
            for cluster in xrange(self.k)])
            self.pnk[datapoint] = np.exp(a-lse(a))
            
        return self.pnk
        
    def __update_nk(self):
        if self.pnk is None:
            self.pnk = self.__pnk()
        self.nk = np.array([np.random.choice(np.arange(self.k), p = self.pnk[datapoint]) for datapoint in xrange(self.n)])
        return self.nk
    
    def __Gibbs_update(self, t=1):
        if t==1:
            self.__GaussComp()
            self.__pnk()
            self.__update_nk()
            self.__pi()
        
        elif int(t)>1:
            for i in xrange(t):
                self.__GaussComp()
                self.__pnk()
                self.__update_nk()
                self.__pi()
    
    def __coll_Gibbs_update(self, t=1):
        
        arr = np.arange(self.n)
        np.random.shuffle(arr)
        
        for datapoint in arr:
            self.nk[datapoint] = self.k+1
            ppd = np.array([math.log(self.alpha/self.k+len(self.nk[self.nk==cluster])) +\
            GWM(GC(d=self.d, X=self.Xm[self.nk==cluster])).post_pred_lp_(self.Xm[datapoint]) for cluster in xrange(self.k)])
            ppd = np.exp(ppd-lse(ppd))
            self.nk[datapoint] = np.random.choice(np.arange(self.k), p = ppd)
    
            
            
