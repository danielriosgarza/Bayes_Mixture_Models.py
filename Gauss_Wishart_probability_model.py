from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
import random
trm = linalg.get_blas_funcs('trmm')
from scipy.special import gammaln as gamlog
from Gaussian_component import Gaussian_component



class Gauss_Wishart_model:
    '''Probability distributions for a Bayesian Normal Wishart probability model.'''
    
    def __init__(self, Gaussian_component):
        self.GaussComp = Gaussian_component
        self.GaussComp._Gaussian_component__update_all() 
        self.d = self.GaussComp.d
        self.n = self.GaussComp.n
        self.prec_mu_norm_Z = None #normalizing constant of the prior probability
        self.prec_norm_Z = None #normalizing constant of the prior probability of the precision matrix
        self.mu_norm_Z = None #normalizing constant of the prior probability of the mean
        
        self.prior_lp = None #log prior probability of the mean and precision matrix
        self.prior_lp_prec = None #log prior probability of the precision matrix 
        self.prior_lp_mu = None #log prior probability of the mean
        
        self.data_lp = None #log probability of the data 
        
        self.joint_lp = None #Joint log probability (data, mean, precision)
        
        self.post_lp = None #posterior log probability
        
        self.post_pred_lp=None #posterior predictive log probability
        
        self.marginal_lp = None #marginal log probability (precision and mean integrated out)
        
        self.marginal_mu_lp = None #marginal posterior of the mean
        self.marginal_prec_lp = None #marginal posterior of the precision

    def __chol_S(self,S=None):
        if S is None:
            S = self.GaussComp.S_0
            
        if self.d == 1:
            return S
        else:
            return Cholesky(S).log_determinant()

    def __df(self, v=None, d=None):
        '''sets the degreed of freedom in the model to be used in several functions'''
        if v is None:
            v= self.GaussComp.v_0
        if d is None:
            d= self.d
        if v<d:
            return d+1
        else:
            return v+1
            
    def __prec_mu_norm_Z(self,S=None, kappa=None,v = None, d=None):
        
        '''Log of the normalizing constant of the precision and mean. This normalizing function is 
        used throughout different equations.eq(1), (2), (3)'''
        if S is None:
            S = self.GaussComp.S_0
        if kappa is None:
            kappa = self.GaussComp.kappa_0
        if v is None:
            v = self.GaussComp.v_0
        if d is None:
            d = self.d
        
        df = self.__df(v,d)
        
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
       
        if d==1:
            self.prec_mu_norm_Z = -kt +0.5*math.log(math.pi) - 0.5*(df-1)*math.log(S)+0.5*df*math.log(2) + gamlog(0.5*(df-1))
        else:
            self.prec_mu_norm_Z = (0.5*df*d*math.log(2))+ (0.25*(d*(d+1))*math.log(math.pi)) - kt \
            -((0.5*(df-1))*self.__chol_S(S))+sum(gamlog(0.5*(df-np.arange(1,d+1))))
        
        return self.prec_mu_norm_Z

    def __mu_norm_Z(self, kappa=None,d=None):
        '''Computes the normalizing constant of the distribution of the mean. eq(2), (3)'''
        
        if kappa is None:
            kappa = self.GaussComp.kappa_0
        
        if d is None:
            d = self.d
        
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
        
        self.mu_norm_Z = (-0.5*d*(math.log(2*math.pi)))+kt
        
        return self.mu_norm_Z

    
        
                
    def __prec_norm_Z(self, S=None,v=None,d=None):
        
        '''normalizing constant for the probability that the precision matrix is Lamda, given the prior.
        This is modelled as Wi(v, S). The normalizing constant is 1/Z and 
        Z = [2**(0.5*v*d)][det(S)**(-0.5*v)][prod(gamma_func(v+1-i)) for i in xrange(d)].'''
        
        if S is None:
            S = self.GaussComp.S_0
        if v is None:
            v = self.GaussComp.v_0
        if d is None:
            d = self.d
        
        df = self.__df(v,d)
        if d==1:
            self.prec_norm_Z = (-0.5*(df-1)*(math.log(S)-math.log(2))) +gamlog(0.5*(df-1))
        else:
            self.prec_norm_Z = (0.5*d*(df-1)*math.log(2))-((0.5*(df-1))*self.__chol_S(S)) + ((0.25*d*(d-1))*math.log(math.pi))\
            +sum(gamlog(0.5*(df-np.arange(1, d+1))))
        
        return self.prec_norm_Z

        
    def prior_lp_prec_(self, prec=None,S=None, v=None,d=None):
        '''Computes the log prior probability of the precision matrix.eq (9), (10), (11), (12)'''
        
        if prec is None:
            prec = self.GaussComp.prec
        if S is None:
            S = self.GaussComp.S_0
        if v is None:
            v = self.GaussComp.v_0
        if d is None:
            d = self.d
        
        df = self.__df(v,d)
        
        
        if d == 1:
            self.prior_lp_prec = (-self.__prec_norm_Z(S, df-1, d)+(0.5*(df-3)*math.log(prec))-(0.5*prec*S))
               
        else:
            self.prior_lp_prec = -self.__prec_norm_Z(S, df-1, d)+(0.5*(df-d-2)*log_chol_prec)-(0.5*np.einsum('ij, ij', prec, S))
        return self.prior_lp_prec

    def prior_lp_mu_(self, prec=None, emp_mu=None, mu_0=None, kappa=None,d=None):
        '''Computes the log prior probability of the mean. Eqs (13), (14),(15), (16).'''
        if prec is None:
            prec = self.GaussComp.prec
        if kappa is None:
            kappa = self.GaussComp.kappa_0
        if emp_mu is None:
            emp_mu = self.GaussComp.emp_mu
        if mu_0 is None:
            mu_0 = self.GaussComp.mu_0
        if d is None:
            d = self.d
        
        if d==1:
            self.prior_lp_mu = self.__mu_norm_Z(kappa,1)+(0.5*math.log(prec))- (0.5*(kappa*prec)*((emp_mu-mu_0)**2))
        else:
            self.prior_lp_mu = self.__mu_norm_Z(kappa, d)+(0.5*Cholesky(prec).log_determinant())-(0.5*np.einsum('ij, ij', kappa*prec, np.einsum('i,j->ij', (emp_mu)-mu_0, (emp_mu)-mu_0)))
        
        return self.prior_lp_mu

    def prior_lp_(self, prec=None, emp_mu=None, mu_0=None, S=None, kappa=None,v=None, d=None):
        if prec is None:
            prec = self.GaussComp.prec
        if kappa is None:
            kappa = self.GaussComp.kappa_0
        if emp_mu is None:
            emp_mu = self.GaussComp.emp_mu
        if mu_0 is None:
            mu_0 = self.GaussComp.mu_0
        if v is None:
            v=self.GaussComp.v_0
        if S is None:
            S=self.GaussComp.S_0
        if d is None:
            d = self.d
        
        df = self.__df(v,d)
        
        if d==1:
            self.prior_lp = -self.__prec_mu_norm_Z(S, kappa, df-1, d)+0.5*(df-2)*math.log(prec) - 0.5*prec*(S+kappa*(emp_mu-mu_0)**2)
        else:
            self.prior_lp = -self.__prec_mu_norm_Z(S, kappa, df-1,d)+((0.5*(df-1-d))*Cholesky(prec).log_determinant())\
            -(0.5*np.einsum('ij, ij', prec, (kappa*np.einsum('i,j->ij', emp_mu-mu_0, emp_mu-mu_0))+S))
        
        return self.prior_lp
    
    def data_lp_(self, prec=None, XX_T=None, sX=None, emp_mu=None, n=None, d=None):
        
        if prec is None:
            prec = self.GaussComp.prec
        if XX_T is None:
            XX_T = self.GaussComp.XX_T
        if sX is None:
            sX = self.GaussComp.sX
        if emp_mu is None:
            emp_mu = self.GaussComp.emp_mu
        if n is None:
            n = self.n
        if d is None:
            d=self.d
               
        if n==0:
            n=1
        if d ==1:
            self.data_lp = (-0.5*(n*d)*math.log(2*math.pi)) + ((0.5*n)*math.log(prec)) - (0.5*prec*((XX_T) -( 2*emp_mu*sX)+(n*emp_mu*emp_mu)))
        else:
            self.data_lp= (-0.5*(n*d)*math.log(2*math.pi)) + ((0.5*n)* Cholesky(prec).log_determinant()) - (0.5*np.einsum('ij, ij', prec, (n*np.einsum('i,j->ij', emp_mu, emp_mu)-2*np.einsum('i,j->ij', sX, emp_mu) + XX_T)))
        return self.data_lp
    
    def joint_lp_(self, prec, XX_T, S, mu_0, sX, emp_mu, kappa, v, n,d):
        df = self.__df(v,d)
        if n==0:
            n=1

        self.joint_lp =  (-0.5*(n*d)*math.log(2*math.pi)) -self.__prec_mu_norm_Z(S, kappa, df-1,d) + ((0.5*(df-1+n-d))*Cholesky(prec).log_determinant())\
        -(0.5*np.einsum('ij, ij', prec, ((kappa+n)*np.einsum('i,j->ij', emp_mu, emp_mu) - 2*np.einsum('i,j->ij', emp_mu, ((kappa*mu_0)+sX)) +  kappa*np.einsum('i,j->ij', mu_0, mu_0)+XX_T+S)))
        return self.joint_lp 
    
    def marginal_lp_(self, cov, S, kappa, v, n,d):
        df = self.__df(v,d)
        if n==0:
            n=1

        self.marginal_lp = (-0.5*(n*d)*math.log(2*math.pi))+ self.__prec_mu_norm_Z(cov, n+kappa, n+df-1,d) - self.__prec_mu_norm_Z(S, kappa, df-1,d)
        return self.marginal_lp

    def marginal_mu_lp_(self, mu_i, prec, mu, kappa, v, n,d):
        df= v+n+1-d
        if n==0:
            n=1
        M = (df*(kappa+n))*prec
        
        self.marginal_mu_lp = gamlog(0.5*(df+d)) - gamlog(0.5*df) + 0.5*Cholesky(M).log_determinant()-0.5*d*math.log(df*math.pi)\
        - 0.5*(df+d)*(math.log(1+(1/df)*np.einsum('i,ij,j->', mu_i-mu, M, mu_i-mu)))
        
        return self.marginal_mu_lp
    
    
    def marginal_prec_lp_(self, prec_i, prec, v, n,d):
        df = self.__df(v,d)
        if n==0:
            n=1
            
        self.marginal_prec_lp = -self.__prec_norm_Z(prec,df-1+n,d)+(0.5*(df-1+n-d-1))*Cholesky(prec_i).log_determinant()\
        -0.5*np.einsum('ij,ij', prec, prec_i)
        return self.marginal_prec_lp
        
        
        
    def post_lp_(self, prec, cov, mu,emp_mu, sX, kappa, v, n, d):
    
        df = self.__df(v,d)
        if n ==0:
            n=1
        
        self.post_lp = -self.__prec_mu_norm_Z(cov, kappa+n, df-1+n,d) + (0.5*(df-1+n-d))*Cholesky(prec).log_determinant()\
        -(0.5*np.einsum('ij, ij', prec, ((kappa+n)*np.einsum('i,j->ij', mu-emp_mu, mu-emp_mu) + cov)))
        return self.post_lp
          
    def post_pred_lp_(self, Xi, prec, mu, kappa,v, n,d):
        df = self.__df(v,d)
        df= df+n-d
        if n==0:
            n=1
        M = (((kappa+n)*df)/(kappa+n+1)) * prec
        
        self.post_pred_lp = gamlog(0.5*(df+d)) - gamlog(0.5*df) + 0.5*Cholesky(M).log_determinant()-0.5*d*math.log(df*math.pi)\
        - 0.5*(df+d)*(math.log(1+(1/df)*np.einsum('i,ij,j->', Xi-mu, M, Xi-mu)))
        
        return self.post_pred_lp
