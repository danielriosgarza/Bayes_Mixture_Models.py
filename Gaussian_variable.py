from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky


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

        self.delta = None
        
        self.Xi_Xi_T = None
        

        self.logp_=None
        self.p_ = None

    def __d(self, d=None):
        if d is None:
            self.d=1
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
                self.cov = Cholesky(self.S, method = 'lower').matrix
            elif self.method=='prec':
                self.cov = Cholesky(self.S)._Cholesky__inv()
            elif self.method=='chol_prec':
                self.cov= Cholesky(self.S, method='lower')._Cholesky__inv()    
        return self.cov
    
    def __chol_cov(self):
        '''returns the Cholesky dec. of the covariance matrix'''
        if self.d==1:
            if self.method=='cov' or self.method=='chol_cov':
                self.chol_cov=self.S
            elif self.method=='prec' or self.method=='chol_prec':
                self.chol_cov = 1./self.S
        else:
            if self.method=='cov':
                self.chol_cov = Cholesky(self.S).lower
            elif self.method=='chol_cov':
                self.chol_cov = self.S
            elif self.method=='prec':
                self.chol_cov = Cholesky(self.S).chol_of_the_inv()
            elif self.method=='chol_prec':
                self.chol_cov= Cholesky(self.S, method='lower').chol_of_the_inv()    
        return self.chol_cov  
    
    def __prec(self):
        '''returns the precision matrix'''
        if self.d==1:
            if self.method=='cov' or self.method=='chol_cov':
                self.prec=1./self.S
            elif self.method=='prec' or self.method=='chol_prec':
                self.prec = self.S
        else:
            if self.method=='cov':
                self.prec = Cholesky(self.S)._Cholesky__inv()
            elif self.method=='chol_cov':
                self.prec = Cholesky(self.S, method='lower')._Cholesky__inv()  
            elif self.method=='prec':
                self.prec = self.S
            elif self.method=='chol_prec':
                self.prec= Cholesky(self.S, method='lower').matrix    
        return self.prec
    
    def __chol_prec(self):
        '''returns the Cholesky dec. of the precision matrix'''
        if self.d==1:
            if self.method=='cov' or self.method=='chol_cov':
                self.chol_prec=1./self.S
            elif self.method=='prec' or self.method=='chol_prec':
                self.chol_prec = self.S
        else:
            if self.method=='cov':
                self.chol_prec = Cholesky(self.S).chol_of_the_inv()
            elif self.method=='chol_cov':
                self.chol_prec = Cholesky(self.S, method='lower').chol_of_the_inv()  
            elif self.method=='prec':
                self.chol_prec = Cholesky(self.S).lower
            elif self.method=='chol_prec':
                self.chol_prec= self.S
        return self.chol_prec  
    
    def __delta(self):
        '''returns (Xi-mu)'''
        self.delta =  self.Xi-self.mu
        return self.delta

    def __Xi_Xi_T(self):
        '''returns the matrix XiXi' '''
        if d==1:
            self.Xi_Xi_T = self.Xi**2
        else:
            self.Xi_Xi_T = np.einsum('i,j->ij', self.Xi, self.Xi)
        return self.Xi_Xi_T

    def logp(self):
        
        '''Gaussian probability function.    
        The equation is (Wikipedia or Kevin P. Murphy, 2007):
        
        --------
        logpdf estimate of Xi'''
                
        if self.d==1:
            #parametrized by the mean and prec.
            pr = self.__prec()
            self.logp_ =-0.5*math.log(2*math.pi) +0.5*math.log(pr)-0.5*(((self.__delta())**2)*(pr))
            return self.logp_
        
        else:
                    
            cpm = self.__chol_prec()
            
            det = Cholesky(cpm, method='lower').log_determinant()
            delta = self.__delta()
            in_exp = delta.T.dot(Cholesky(cpm, method='lower').matrix).dot(delta)
        
            self.logp_ = (-0.5*self.d)*math.log(2*math.pi) + 0.5*det -0.5*in_exp
            
            return self.logp_
    
    def p(self):
        '''returns the pdf estimate of Xi'''
        self.p_= math.exp(self.logp())
        return self.p_

    def __r_mat(self, a_in_rad, vec):
        '''take a 2 component vector and rotate by a given angle''' 
        t_mat = np.array([[np.cos(a_in_rad), -1*np.sin(a_in_rad)], [np.sin(a_in_rad), np.cos(a_in_rad)]])
        
        return np.dot(t_mat, vec)
    
    def standard_normal_Gaussian(self, n):
        '''generative implementation of a standard normal distribution'''
        #define a radius using the square root of the exponential distribution
        #where theta equals two.
        rad = np.array([[0, math.sqrt(np.random.exponential(2))] for i in xrange(n)])
        #rotate the radius by random uniform angle in the interval [0, 2pi],
        #the coordinates will be samples from two independent standard normal dist.
        norm_sample = np.array([self.__r_mat(np.random.uniform(0, 2*math.pi), i) for i in rad])
        
        return norm_sample
    
    def Box_Muller(self):
        '''Box_Muller (1958) for a single draw from two independent standard normal distributions.'''
        u_1 = np.random.uniform()
        u_2 = np.random.uniform()
        return np.array([math.sqrt(-2*math.log(u_1))*math.sin(2*math.pi*u_2), math.sqrt(-2*math.log(u_1))*math.cos(2*math.pi*u_2)])

    def rvs(self, n=1):
        '''returns a random multivariate Gaussian variable 
        to avoid matrix iversions, provide the precision matrix. Method has a slightly different 
        output than numpy or scipy's multivariate normal rvs, but has similar statistical properties.
        The algorithm is mentioned in the book 'Handbook of Monte Carlo Methods'
        from Kroese et al.(2011) (algorithm 5.2, pag. 155)'''
        pr = self.__chol_prec()
        
        if self.d==1:
            return np.random.standard_normal(n)*(math.sqrt(1./pr))+self.mu

        else:
            Z = np.random.standard_normal(size=(self.d,n))
            return self.mu + linalg.solve_triangular(pr.T, Z, lower=0, overwrite_b=1, check_finite=0).T

    
    
        
        

   
