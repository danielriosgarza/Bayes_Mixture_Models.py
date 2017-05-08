from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
trm = linalg.get_blas_funcs('trmm')

class Gaussian_variable:
    '''Methods for a multivariate Gaussian model object. 
    Can be called by minimally assigning a number of dimensions. In this case, a standard normal variable is created.
    One can also specify a mean vector (mu), a sample vector (Xi), and a matrix (S). The supported methods for S are:
        'cov' - assumes that S is a covariance matrix
        'chol_cov' -assumes that S is the Cholesky dec. of the covariance matrix
        'prec' - assumes that S is a precision matrix (inverse of the covariance)
        'chol_prec' - assumes that S is the Cholesky dec. of the precision matrix.
        
    For example, if S is the lower Cholesky decomposition of the precision matrix, then:
    >Gaussian_variable(mu, Xi, S, method='chol_prec').logp()
    returns the same result as:
    >scipy.stats.multivariate_normal(mean=mu, cov= np.linalg.inv(S.dot(S.T)).logpdf(X)''' 
    
    def __init__(self, d, Xi=None, S=None,  mu=None, method = 'cov'):
        if Xi is None:
            self.Xi = np.zeros(d)
        else:
            self.Xi = Xi
        self.d = d
        self.method = method
        self.mu = self.__mu(mu)
        self.S = self.__S(S)
    
    def __mu(self,mu=None):
        if mu is None:
            return np.zeros(self.d)
        else:
            return mu
            
    def __S(self, S=None):
        if S is None:
            return np.eye(self.d)
        else:
            return S
    
    def __cov(self):
        '''returns the covariance matrix'''
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return self.S
        elif self.method=='chol_cov':
            return Cholesky(self.S).mat(self.S)
        elif self.method=='prec':
            return Cholesky(self.S).inv()
        elif self.method=='chol_prec':
            return Cholesky(self.S).inv(self.S)    
    
    def __chol_cov(self):
        '''returns the Cholesky dec. of the covariance matrix'''
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return Cholesky(self.S).lower()
        elif self.method=='chol_cov':
            return self.S
        elif self.method=='prec':
            return Cholesky(self.S).chol_of_the_inv()
        elif self.method=='chol_prec':
            return Cholesky(self.S).chol_of_the_inv(self.S)    
    
    def __prec(self):
        '''returns the precision matrix'''
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return Cholesky(self.S).inv()
        elif self.method=='chol_cov':
            return Cholesky(self.S).chol_of_the_inv(self.S)
        elif self.method=='prec':
            return self.S
        elif self.method=='chol_prec':
            return Cholesky(self.S).mat(self.S)
    
    def __chol_prec(self):
        '''returns the Cholesky dec. of the precision matrix'''
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return Cholesky(self.S).chol_of_the_inv()
        elif self.method=='chol_cov':
            return Cholesky(self.S).chol_of_the_inv(self.S)
        elif self.method=='prec':
            return Cholesky(self.S).lower()
        elif self.method=='chol_prec':
            return self.S
    
    def delta(self):
        'returns (Xi-mu)'''
        return self.Xi-self.mu
    def xi_xit(self):
        '''returns the matrix XiXi' '''
        return np.einsum('i,j->ij', self.Xi, self.Xi)
        
    def rvs(self, n=1):
        '''returns a random multivariate Gaussian variable 
        to avoid matrix iversions, provide the precision matrix. Method has a slightly different 
        output than numpy or scipy's multivariate normal rvs, but has similar statistical properties.
        The algorithm is mentioned in the book 'Handbook of Monte Carlo Methods'
        from Kroese et al.(2011) (algorithm 5.2, pag. 155)'''
               
        m = self.__chol_prec().T
        Z = np.random.standard_normal(size=(self.d,n))
        return self.mu + linalg.solve_triangular(m, Z, lower=0, overwrite_b=1, check_finite=0).T

    def logp(self):
        
        '''Multivariate normal probability function.    
        The equation is (Wikipedia or Kevin P. Murphy, 2007):
            (2pi)**(-0.5*d) |prec_m|**(0.5) exp[0.5(X-mu_v)' prec_m (x-mu_v)]
        Should give the same result as scipy.stats.multivariate_normal(mu_v, inv(prec_m)).pdf(X)
        
        Outputs
        --------
        logpdf estimate of Xi'''
        
        pm = self.__prec()
        cpm = self.__chol_prec()
        det = Cholesky(cpm).log_determinant(cpm)
        delta = self.delta()
        in_exp = delta.T.dot(pm).dot(delta)
        
        
        return (-0.5*self.d)*math.log(2*math.pi) + 0.5*det -0.5*in_exp
    
    def p(self):
        '''returns the pdf estimate of Xi'''
        return math.exp(self.logp())
    
        
        

   
