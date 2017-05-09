from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
import random
trm = linalg.get_blas_funcs('trmm')
from scipy.special import gammaln as gamlog




class Gauss_Wishart_probability_model:
    '''Probability distributions for a Bayesian Normal Wishart probability model.'''
    
    def __init__(self):
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

    def __df(self, v, d):
        '''sets the degrees of freedom for the model and deals with the cases where v is smaller than the number of dimensions.
        parameters
        --------
        v:scalar quantity. int or float.
        Represents the degrees of freedom of the precision
        d: int
        Dimensions of the model
        
         Returns
         --------
         df: int or float
         Adjusted degrees of freedom.'''
          
        if v<d:
            return d+1
        else:
            return v+1
            
    def __prec_mu_norm_Z(self,S, kappa,v, d):
        
        df = self.__df(v,d)
        
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
        
        self.prec_mu_norm_Z = (0.5*df*d*math.log(2))+ (0.25*(d*(d+1))*math.log(math.pi)) - kt \
        -((0.5*(df-1))*Cholesky(S).log_determinant())+sum(gamlog(0.5*(df-np.arange(1,d+1))))
        
        return self.prec_mu_norm_Z
    
    def __prec_norm_Z(self, S,v,d):
        '''normalizing constant for the probability that the precision matrix is Lamda, given the prior.
        This is modelled as Wi(v, S). The normalizing constant is 1/Z and 
        Z = [2**(0.5*v*d)][det(S)**(-0.5*v)][prod(gamma_func(v+1-i)) for i in xrange(d)].'''
        
        df = self.__df(v,d)
              
        self.prec_norm_Z = (0.5*(df-1)*d*math.log(2))-((0.5*(df-1))*Cholesky(S).log_determinant()) + ((0.25*d*(d-1))*math.log(math.pi))\
        +sum(gamlog(0.5*(df-np.arange(1, d+1))))
         
        return self.prec_norm_Z

    def __mu_norm_Z(self, kappa,d):
        
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
        
        self.mu_norm_Z = (-0.5*d*(math.log(2*math.pi)))+kt
        
        return self.mu_norm_Z
        
    def prior_lp_prec_(self, prec,S, v,d):

        df = self.__df(v,d)

        self.prior_lp_prec = -self.__prec_norm_Z(S, df-1, d)+(0.5*(df-d-2)*Cholesky(prec).log_determinant())-(0.5*np.einsum('ij, ij', prec, S))
        
        return self.prior_lp_prec

    def prior_lp_mu_(self, prec, emp_mu, mu_0, kappa,d):
        
        self.prior_lp_mu = self.__mu_norm_Z(kappa, d)+(0.5*Cholesky(prec).log_determinant())-(0.5*np.einsum('ij, ij', kappa*prec, np.einsum('i,j->ij', (emp_mu)-mu_0, (emp_mu)-mu_0)))
        
        return self.prior_lp_mu

    def prior_lp_(self, prec, emp_mu, mu_0, S, kappa,v, d):
        
        df = self.__df(v,d)
        
        self.prior_lp = -self.__prec_mu_norm_Z(S, kappa, df-1,d)+((0.5*(df-1-d))*Cholesky(prec).log_determinant())\
        -(0.5*np.einsum('ij, ij', prec, (kappa*np.einsum('i,j->ij', emp_mu-mu_0, emp_mu-mu_0))+S))
        
        return self.prior_lp
    
    def data_lp_(self, prec, XX_T, sX, emp_mu, n, d):
        
        if n==0:
            n=1

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
