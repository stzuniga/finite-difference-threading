from numpy import *
from poisson import poisson

# Validation of the threaded matrix-vector result and comparing
# it to a serial matrix-vector multiply using the global Poisson matrix
# found in poisson.py.

#Computing the relative difference with testing harness.
def check_matvec(fpp_numeric, n, fcn):
    
    fpp_numeric = fpp_numeric.reshape(-1,)

    Acheck = poisson((n, n), format='csr')
    h = 1./(n-1)
    Acheck *= 1/h**2

    # Create a grid
    pts = linspace(0,1,n)
    X,Y = meshgrid(pts, pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,) 
    f_vals = fcn(X,Y)

    # Standard matrix-vector result
    fpp_check = Acheck*f_vals

    # Relative error norm
    diff = fpp_check - fpp_numeric
    diff_norm = sqrt(dot(diff, diff)) / sqrt(dot(fpp_check, fpp_check))

    # Terminal output--> Desired output: 0
    print("Matvec check (should be zero): " + str( diff_norm )) 

