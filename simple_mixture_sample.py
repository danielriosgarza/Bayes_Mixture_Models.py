import numpy as np

alpha = np.array([500,100,50])

A1 =  np.random.randint(0, 10)*np.random.rand(2,2)
A1 = A1.dot(A1.T)

A2 =  np.random.randint(0, 10)*np.random.rand(2,2)
A2 = A2.dot(A2.T)

A3 =  np.random.randint(0, 10)*np.random.rand(2,2)
A3 = A3.dot(A3.T)

mu1 = np.random.randint(-10, 10, 2)
mu2 = np.random.randint(-10, 10, 2)
mu3 = np.random.randint(-10, 10, 2)

X=[]

covs=np.array([A1, A2, A3])
mus = np.array([mu1, mu2, mu3])


for i in xrange(1000):
    ind = np.random.choice(np.arange(3), p=np.random.dirichlet(alpha))
    X.append(np.random.multivariate_normal(mean=mus[ind], cov=covs[ind]))

X= np.array(X)
