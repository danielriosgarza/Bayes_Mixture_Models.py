from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
trm = linalg.get_blas_funcs('trmm')

class Gaussian_variable:
    
    
    def __init__(self, d=None, Xi=None, S=None,  mu=None, method = 'cov'):
        
        self.d = self.__d(d)
        
        self.Xi = self.__Xi(Xi)
                
        self.method = self.__method(method)
        
        self.mu = self.__mu(mu)
        
        self.S = self.__S(S)

        self.cov = None
        
        self.chol_cov = None
        
        self.prec = None
        
        self.chol_prec=None
        

        self.logp_=None
        self.p_ = None

    def __d(self, d=None):
        if d is None:
            self.d=0
        else:
            try:
                self.d = int(d)
                assert int(d)>0, '\n\n\nd must be greater than 0\n\n\n' 
            except ValueError:
                print '\n\n\nError: d must be an integer\n\n\n'
            
        
        return self.d
        
    
    def __Xi(self, Xi=None):
        if Xi is None:
            if self.d==1:
                self.Xi=0
            else:
                self.Xi = np.zeros(self.d)
        else:
            if self.d==1:
                assert isinstance(Xi, int) or isinstance(Xi, float),\
                '\n\n\nError: if d=1, Xi must be an int or a float\n\n\n'
                self.Xi=Xi
            else:
                assert len(np.array(Xi).flatten())==self.d,\
                '\n\n\nError: Xi must have the same number of dimensions as d\n\n\n'
                self.Xi = np.array(Xi)
        return self.Xi

    def __method(self, method='cov'):
        assert method=='cov' or method== 'chol_cov' or method=='prec' or method== 'chol_prec',\
        "\n\n\nError: method must be one of 'cov','chol_cov','prec','chol_prec\n\n\n"
        self.method=method
        return self.method
    
    def __mu(self,mu=None):
        if mu is None:
            if self.d==1:
                self.mu=0
            else:
                self.mu = np.zeros(d)
        else:
            if self.d==1:
                assert isinstance(mu, int) or isinstance(mu, float),\
                '\n\n\nError: if d=1, mu must be an int or a float\n\n\n'
                self.mu=mu
            else:
                assert len(np.array(mu).flatten())==self.d,\
                '\n\n\nError: mu must have the same number of dimensions as d\n\n\n'
                self.mu = np.array(mu)
        return self.mu
            
    def __S(self, S=None):
        if S is None:
            if self.d==1:
                self.S=1
            else:
                self.S=np.eye(self.d)
        else:
            if self.d==1:
                assert isinstance(S, int) or isinstance(S, float),\
                '\n\n\n if d=1, S must be an int or float'
                self.S=S
            else:
                if self.method=='chol_cov' or self.method=='chol_prec':
                    assert sum(np.array(S)[np.triu_indices(self.d,k=1)])==0,\
                    '\n\n\nError: S must be a lower triangular matrix\n\n\n'
                    self.S=np.array(S)
                else:
                    assert np.array(S).shape==(self.d, self.d),\
                    '\n\n\nS must be a d-dimensional square matrix\n\n\n'
                    self.S=np.array(S)
        return self.S
    
    def __cov(self):
        '''returns the covariance matrix'''
        if self.d==1:
            if self.method=='cov' or self.method=='chol_cov':
                self.cov=self.S
            elif self.method=='prec' or self.method=='chol_prec':
                self.cov = 1./self.S
        else:
            if self.method=='cov':
                self.cov = self.S
            elif self.method=='chol_cov':
                self.cov = Cholesky(self.S).mat(self.S)
            elif self.method=='prec':
                self.cov = Cholesky(self.S).inv()
            elif self.method=='chol_prec':
                self.cov= Cholesky(self.S).inv(self.S)    
        return self.cov
    
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
        if self.d==0:
            print "Can't generate rv from a 0 dimensional distribution"
            return None       
        else:
            m = self.__chol_prec().T
            Z = np.random.standard_normal(size=(self.d,n))
            return self.mu + linalg.solve_triangular(m, Z, lower=0, overwrite_b=1, check_finite=0).T

    def logp(self):
        
        '''Normal probability function.    
        The equation is (Wikipedia or Kevin P. Murphy, 2007):
            (2pi)**(-0.5*d) |prec_m|**(0.5) exp[0.5(X-mu_v)' prec_m (x-mu_v)]
        Should give the same result as scipy.stats.multivariate_normal(mu_v, inv(prec_m)).pdf(X)
        
        Outputs
        --------
        logpdf estimate of Xi'''
        if self.d==0:
            self.logp_ = 0
            return self.logp_
        elif self.d==1:
            self.logp_ =-0.5*math.log(2*math.pi*self.S)-0.5*(((self.Xi-self.mu)**2)/(self.S))
            return self.logp_
        else:
            
            pm = self.__prec()
            cpm = self.__chol_prec()
            det = Cholesky(cpm).log_determinant(cpm)
            delta = self.delta()
            in_exp = delta.T.dot(pm).dot(delta)
        
            self.logp_ = (-0.5*self.d)*math.log(2*math.pi) + 0.5*det -0.5*in_exp
            return self.logp_
    
    def p(self):
        '''returns the pdf estimate of Xi'''
        if self.d==0:
            self.p_=0
            return self.p_
        else:
            
            self.p_= math.exp(self.logp())
            return self.p_
    
        
        

   
