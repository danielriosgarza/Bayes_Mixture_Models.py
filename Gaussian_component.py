from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
from Gaussian_variable import Gaussian_variable
import random
trm = linalg.get_blas_funcs('trmm')
from copy import copy


class Gaussian_component(object, Gaussian_variable):
    
    '''Deals with multiple measurements of a Gaussian variable random variable.'''
    
    def __init__(self, d, kappa_0 = 0, v_0=0, mu_0=None, S_0=None, X=None):
        
        
        
        
        self.d = self._Gaussian_variable__d(d)

        self.n = self.__n(X)
        
        self.kappa_0 = kappa_0
        
        self.v_0 = v_0
        
        self.mu_0 = self._Gaussian_variable__mu(mu_0)
        
        
        self.S_0 = self._Gaussian_variable__S(S_0)
        
        
        self.X = self.__X(X)
        
        self.sX = None
        
        self.mu= self.__mu()
        
        self.emp_mu = self.__emp_mu()
        
        self.scale = None
        
        self.chol_scale=None

        self.inv_scale = None
        
        self.chol_inv_scale = None
        
        self.XX_T=None
        
        self.cov = None
        
        self.chol_cov=None
        
        self.prec= None
        
        self.chol_prec=None

    def __n(self, X=None):
        if X is None:
            self.n =0
        elif self.d==1:
            self.n = len(np.array(X).flatten())
        else:
            self.n=len(X.flatten())/self.d
        return self.n
    
    def __X(self, X=None):
        if X is None:
            self.X = np.array(self._Gaussian_variable__Xi())
            self.X.shape=(1,self.d)
        
        elif self.d==1:
            if np.array(X).shape!=(self.d,0) and self.n>1:
                print '\x1b[5;31;46m'+'Warning: data was flattened'+ '\x1b[0m'
            self.X = np.array(X)
            self.X.shape = (len(self.X.flatten()),1)
            
            
        else:
            self.X = X
            self.X.shape = (self.n, self.d)
        
        return self.X
    
    def __sX(self, X=None):
        if self.n == 0:
            if self.d==1:
                self.sX=0
            else:
                self.sX = np.zeros(self.d)
        elif self.n == 1:
            if self.d==1:
                self.sX = float(self.X)
            else:
                self.sX = self.X.flatten()
        else:
            if self.d==1:
                self.sX = float(np.einsum('ij->j', self.X))
            else:
                self.sX = np.einsum('ij->j', self.X) 
        return self.sX
    
    def __mu(self):
        if self.n == 0:
            self.mu = self.mu_0 
            
        else:
            self.mu = (self.kappa_0*self.mu_0 + self.__sX(self.X))/(self.kappa_0+self.n)
        
        return self.mu
    
    def __emp_mu(self):
        if self.n is 0:
            self.emp_mu = self.mu_0 
            return self.emp_mu
        else:
            self.emp_mu = (self.__sX(self.X))/(self.n)
            return self.emp_mu


    def __XX_T(self):
        if self.d==1:
            #returns the sum of Xi*Xi
            self.XX_T = float(np.einsum('ij, ij->j', self.X,self.X))
        else:
            self.XX_T = np.einsum('ij, iz->jz', self.X, self.X)
        return self.XX_T
        
    def __scale(self):
        if self.n == 0:
            self.scale = self.S_0
                
        else:
            
            if self.d==1:
                self.scale = self.S_0 + self.__XX_T() + self.kappa_0*(self.mu_0**2)-(self.kappa_0+self.n)*(self.mu**2)
            
            else:
                self.scale= self.S_0 + self.__XX_T() + self.kappa_0*np.einsum('i,j->ji', self.mu_0, self.mu_0)-(self.kappa_0+self.n)*np.einsum('i,j->ji', self.mu, self.mu)
        
        return self.scale
    
    def __chol_scale(self):
        if self.scale is None:
            self.scale = self.__scale()
        if self.d==1:
            self.chol_scale=self.scale
        else:
            self.chol_scale = Cholesky(self.scale).lower
        
        return self.chol_scale
    
    def __inv_scale(self):
        if self.scale is None:
            self.scale = self.__scale()
        if self.d==1:
            self.inv_scale=1./self.scale
        else:
            self.inv_scale = Cholesky(self.scale)._Cholesky__inv()
        
        return self.inv_scale
    
    def __chol_inv_scale(self):
        if self.inv_scale is None:
            self.inv_scale = self.__inv_scale()
        if self.d==1:
            self.chol_inv_scale=self.inv_scale
        else:
            self.chol_inv_scale = Cholesky(self.inv_scale).lower
        
        return self.chol_inv_scale
    
    def __cov(self):
        '''Empirical covariance matrix'''

        if self.n == 0:
            if self.d==1:
                self.cov = 1.
                
            else:
                self.cov= np.eye(self.d) + self.kappa_0*np.einsum('i,j->ji', self.mu_0, self.mu_0)
                
        else:
            
            if self.d==1:
                self.cov = (1./self.n)*float(self.__XX_T() - 2*self.emp_mu*self.sX + self.n*(self.emp_mu**2))
            
            else:
                self.cov= (1./self.n)*(self.__XX_T() - 2*np.einsum('i,j->ji', self.sX, self.emp_mu)+self.n*np.einsum('i,j->ji', self.emp_mu, self.emp_mu))
        
        return self.cov
            
    def __chol_cov(self):
        cov = self.__cov()
        if np.sum(cov!=0)==0:
            self.chol_cov=cov
        else:
            self.chol_cov = Gaussian_variable(d=self.d, S=cov)._Gaussian_variable__chol_cov()
        return self.chol_cov
        
    def __prec(self):
        cov = self.__cov()
        if np.sum(cov!=0)==0:
            self.prec=cov
        else:
            self.prec = Gaussian_variable(d=self.d, S=cov)._Gaussian_variable__prec()
        return self.prec
    
    def __chol_prec(self):
        cov = self.__cov()
        if np.sum(cov!=0)==0:
            self.chol_prec=cov
        else:
            self.chol_prec = Gaussian_variable(d=self.d, S=cov)._Gaussian_variable__chol_prec()
        return self.chol_prec

    def chol_prec_rvs(self):
        if self.v_0+self.n<self.d+1:
            df = self.d+1
        else:
            df = self.v_0+self.n
        
        if self.d==1:
            
            if self.scale is None:
                s = self.__scale()
            else:
                s=self.scale
            
            return random.gammavariate(0.5*df, 2./s)
        
        else:
                    
            if self.chol_inv_scale is None:
                a = self.__chol_inv_scale()
            else:
                a= self.chol_inv_scale
            
            ind = np.tril_indices(self.d,-1) #lower triangular non-diagonal elements
            
            B = np.zeros((self.d,self.d)) 
            
            norm = np.random.standard_normal(len(ind[0])) #normal samples for the lower triangular non-diagonal elements
    
            B[ind] = norm 
    
            chisq = [math.sqrt(random.gammavariate(0.5*(df-(i+1)+1), 2.0)) for i in xrange(self.d)]
    
            B = B+np.diag(chisq)
    
            ch_d =trm(alpha=1, a=a, b=B, lower=1)
    
            dg= np.diag(ch_d)#Assuring the result is the Cholesky decomposition that contains positive diagonals
    
            adj = np.tile(dg/abs(dg), (self.d,1))
    
            return ch_d*adj

    def prec_rvs(self):
        if self.d==1:
            return self.chol_prec_rvs()
        else:
            return Cholesky(self.chol_prec_rvs(), method='lower').matrix

    def cov_rvs(self):
        
        if self.d==1:
            return 1./self.chol_prec_rvs()
        else:
            return Cholesky(self.chol_prec_rvs(), method='lower')._Cholesky__inv()
    
    def chol_cov_rvs(self):
        
        if self.d==1:
            return 1./self.chol_prec_rvs()
        else:
            return Cholesky(self.chol_prec_rvs(), method='lower')._Cholesky__chol_of_the_inv()

    def mu_rvs(self, chol_prec=None):
        
        if chol_prec is None:
            chol_prec = self.chol_prec_rvs()
        
        if self.n==0:
            return Gaussian_variable(d= self.d, mu=self.mu, S = chol_prec, method='chol_prec').rvs()
        else:
            
            return Gaussian_variable(d= self.d, mu=self.mu, S = math.sqrt(self.kappa_0+self.n)*chol_prec, method='chol_prec').rvs()

    
    def rvs(self):
        chol_prec = self.chol_prec_rvs()
        mu = self.mu_rvs(chol_prec)
        return Gaussian_variable(self.d, mu=mu, S = chol_prec, method='chol_prec').rvs()

    def rvs_coll(self):
        v =self.v_0+self.n
        
        if v>self.d:
            df = v + 1
        else:
            df = self.d+1

        if self.d>1:
            df = df-self.d
            
            x = np.sqrt(df/np.random.chisquare(df))
            
            scale = np.sqrt(((self.kappa_0 +self.n + 1)/((self.kappa_0+self.n)*(df-self.d+1))))*self.__chol_scale()
            
            return self.mu+(scale.dot(np.random.standard_normal(self.d))*x)
        
        else:
             df = v-1
             scale = ((self.kappa_0+self.n+1.)/((self.kappa_0+self.n)*(self.kappa_0+self.n)))*self.__chol_scale()
             
             return self.mu+np.random.standard_t(df)*math.sqrt(scale)

        
    def down_date(self, ind_X):
        
        if self.n ==0:
            print 'component is already empty'
            return None
        
        try:
            dwnX = self.X[ind_X]
        except IndexError:
            print 'Failled to downdate: ind_X not in X'
            return None
        
        if self.n-1==0:
            print 'Component is now empty. Setting parameters to priors'
            
            self.n=0
            self.X = self.__X()
            self.sX = self.__sX()
            self.emp_mu = self.__emp_mu()
            self.mu = self.__mu()
            if self.scale is not None:
                self.scale=self.__scale()
            if self.inv_scale is not None:
                self.inv_scale=self.__inv_scale()
            if self.cov is not None:
                self.cov=self.__cov()
            if self.prec is not None:
                self.prec=self.__prec()
            if self.XX_T is not None:
                self.XX_T = self.__XX_T()
            if self.chol_cov is not None:
                self.chol_cov = self.__chol_cov()
            if self.chol_prec is not None:
                self.chol_prec = self.__chol_prec()
        
        else:
            ind = np.ones(self.n, dtype=np.bool)
            ind[ind_X]=0
            
            self.n-=1
            self.X = self.X[ind]
            self.sX = self.__sX()
            self.emp_mu = self.__emp_mu()
            self.mu = self.__mu()
            
            if self.scale is not None:
                self.scale=self.__scale()
            if self.inv_scale is not None:
                self.inv_scale=self.__inv_scale()
            if self.cov is not None:
                self.cov=self.__cov()
            if self.prec is not None:
                self.prec=self.__prec()
            if self.XX_T is not None:
                self.XX_T = self.__XX_T()
            if self.chol_cov is not None:
                self.chol_cov = self.__chol_cov()
            if self.chol_prec is not None:
                self.chol_prec = self.__chol_prec()
            
        
            
    
    
    def up_date(self, Xi):
        
        if self.n==0:
            self.n = self.__n(Xi)
            self.X = self.__X(Xi)
            self.sX = self.__sX(Xi)
            self.emp_mu = self.__emp_mu()
            self.mu = self.__mu()
            if self.scale is not None:
                self.scale=self.__scale()
            if self.inv_scale is not None:
                self.inv_scale=self.__inv_scale()
            if self.cov is not None:
                self.cov=self.__cov()
            if self.prec is not None:
                self.prec=self.__prec()
            if self.XX_T is not None:
                self.XX_T = self.__XX_T()
            if self.chol_cov is not None:
                self.chol_cov = self.__chol_cov()
            if self.chol_prec is not None:
                self.chol_prec = self.__chol_prec()
        
        else:
            
            try:
                Xi.shape=(1, self.d)
            except ValueError:
                print '\n\nError: to add a data-point to a Gaussian Component, dimensions should be (1,d)\n\n'
            
            self.X = np.concatenate([self.X, Xi])
            self.n = self.__n(self.X)
            self.sX = self.__sX(self.X)
            self.emp_mu = self.__emp_mu()
            self.mu = self.__mu()
            if self.scale is not None:
                self.scale=self.__scale()
            if self.inv_scale is not None:
                self.inv_scale=self.__inv_scale()
            if self.cov is not None:
                self.cov=self.__cov()
            if self.prec is not None:
                self.prec=self.__prec()
            if self.XX_T is not None:
                self.XX_T = self.__XX_T()
            if self.chol_cov is not None:
                self.chol_cov = self.__chol_cov()
            if self.chol_prec is not None:
                self.chol_prec = self.__chol_prec()   

        
      
    def __update_params(self, scale=False, chol_scale=False, inv_scale=False, chol_inv_scale=False, \
    XX_T=False, cov=False, chol_cov=False, prec=False, chol_prec=False):
        if scale:
            self.__scale()
        if chol_scale:
            self.__chol_scale()
        if inv_scale:
            self.__inv_scale()
        
        if chol_inv_scale:
            self.__chol_inv_scale()
        
        if XX_T:
            self.__XX_T
        
        if cov:
            self.__cov()
        
        if chol_cov:
            self.__chol_cov()
        
        if prec:
            self.__prec()
        
        if chol_prec:
            self.__chol_prec()
            
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    def copy(self):
        return self.__copy__()
