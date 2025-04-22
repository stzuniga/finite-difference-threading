from numpy import *
from scipy import sparse
import time
from threading import Thread
from matplotlib import pyplot
from poisson import poisson
from check_matvec import check_matvec
import matplotlib.pyplot as plt 


def L2norm(e, h):
    '''
    Take L2-norm of e
    '''
    # ensure e has a compatible shape for taking a dot-product
    e = e.reshape(-1,)

    
    return  h*sqrt(sum(e**2)) # The square root of the integral of e^2
                              # Assuming a uniform grid in x and y and applying
                              # Applying the midpoint formula.
                              # Assuming each grid point represents the midpoint of a 
                              # equally sized regions
                            


##
#Threaded Computations of Finite Difference Approximations.
##
def compute_fd(n, nt, k, f, fpp_num):
    """
Compute the numeric second derivative of a function `f` using finite difference
and domain decomposition across multiple threads.

Each thread:
- Constructs its local grid (including halo rows),
- Applies a 5-point Poisson stencil,(can be found in poisson.py)
- Writes results to its section of the global output array.

Parameters
----------
n : int
    Total number of n grid points in each direction on a n x n grid.
nt : int
    Number of threads
k : int
    The thread index (n=0 initial)
f : function
    Function f(x, y)
fpp_num : np.ndarray
    Global array storing computed second derivative

Notes
-----
We divide the domain by rows: thread `k` owns rows from
`k*(n/nt)` to `(k+1)*(n/nt)`.

Example: If n=4 and nt=2,
- Thread 0 owns rows 0 and 1
- Thread 1 owns rows 2 and 3

Each thread also includes halo rows to correctly apply the stencil near boundaries.
"""


    # Determine the row range this thread is responsible for.
    # For example, with 4 rows and 2 threads:
    # thread 0 → start=0, end=2 → handles rows 0 and 1
    # thread 1 → start=2, end=4 → handles rows 2 and 3
    start = int(k*(n/nt))
    end = int((k+1)*(n/nt))


    # Expand this thread's row range to include neighboring rows (halo region)
    # needed for computing derivatives using a stencil.
    # First thread skips the top halo; last thread skips the bottom.
    # Example: thread 0 might cover rows 0–1, but needs halo row 2 → end_halo = 3
    start_halo = start - 1 if k != 0 else start
    end_halo = end + 1 if k != (nt - 1) else end

    # Build the local 5-point stencil matrix, CSR matrix (including halo rows),
    # Scaled by 1/h^2 to match the discretization of the Laplacian.
    A = poisson((end_halo - start_halo, n), format='csr')
    h = 1./(n-1)
    A *= 1/h**2

    # Create the full grid of (x, y) coordinates for this thread's halo region.
    # Each thread uses all x-points, and a slice of y-points including halos.
    x_pts = linspace(0, 1, n)
    y_pts = linspace(0, 1, n)
    X, Y = meshgrid(x_pts, y_pts)
    X = X.reshape(-1,)  # Flatten to 1D arrays to vectors
    Y = Y.reshape(-1,)


   # Evaluate the function f(x, y) over the full grid,
    # then extract this thread's portion (including halo rows).
    f_vals = f(X, Y)
    local_f_vals = f_vals[start_halo * n : end_halo * n]


    # Apply the finite difference stencil to local values.
    # Trim off halo rows from the output based on thread position.
    output = A * local_f_vals
    if k == 0:
        output = output[:(end - start) * n]        # No top halo to trim
    elif k == nt - 1:
        output = output[1 * n :]                   # No bottom halo to trim
    else:
        output = output[1 * n : -1 * n]            # Trim both halos

    # Trimmed output into the global result array
    fpp_num[start * n : end * n] = output




#Test Function Definition
def fcn(x, y):
    return cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) + sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))
    #debug function
    #return x**2 + y**2

#Exact second derivative for error analysis
def fcnpp(x, y):
    fcnppx = (-2. / 9.) * (x + 1) ** (-5. / 3.) * (cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))) + (1. / 9.) * (x + 1) ** (-4. / 3.) * (-cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)))
    fcnppy = (-2. / 9.) * (y + 1) ** (-5. / 3.) * (cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))) + (1. / 9.) * (y + 1) ** (-4. / 3.) * (-cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)))
    return fcnppx + fcnppy
    #debug function
    #return 4 * ones_like(x)

#Performance configurations
option = 2 #MODIFY this to selection configuration

if option == 1:
    NN = array([840*6]) # Large grid for HPC Cluster
    num_threads = [1,2,3,4,5,6,7,8]

elif option == 2: # Consider starting with num_threads=[1], debug in serial where num_threads=[1]
    NN = 210*arange(1,6) # Moderate Grid for convergence study on on local machine.
    num_threads = [1] # Number of Threads

elif option == 3: # Consider starting with num_threads=[1], debug in serial where num_threads=[1]
    NN = array([6]) # Small Grid for debugging on local.
    num_threads = [2] # Number of threads

else:
    print("Incorrect Option!")





##############################
# Begin main computation loop #
##############################

# Set up result tables to track error and timing
# Rows = thread counts, Columns = grid sizes
error = zeros((len(num_threads), len(NN)))
timings = zeros((len(num_threads), len(NN)))



# Loop over various numbers of threads count and grid size

for i,nt in enumerate(num_threads):
    # Loop over all combinations of j and
    for j,n in enumerate(NN):
        # Where
        # i: thread index
        # j: grid size index

        # Allocate space for the computed second derivative over the full grid
        fpp_numeric = zeros((n * n,))
        # Number of timings to do for each run
        ntimings = 5 
        # Carry out timing experiment
        min_time = 10000 


        # Time the threaded computation multiple times
        for m in range(ntimings): 

            # This loop creates a thread for each subdomain in order to compute fpp numerically
            # in the interior threads.
            # t_list = [ Thread_object_1, Thread_object_2, ... ,  Thread_object_nt  ]
            # each thread will compute one thread's contributions to fpp_numeric
            t_list = []


            for k in range(nt):
                t_list.append(Thread(target=compute_fd, args=(n, nt, k, fcn, fpp_numeric)))
            
            # Start timing
            start = time.perf_counter()

            
            # Looping over each thread object to launch them, then looping over each
            # thread object to join threads.
            for t in t_list:
                t.start()

            for t in t_list:
                t.join()
            end = time.perf_counter()
            min_time = min([end - start, min_time])
        
        # End loop over timings
        print(" ")


        ##
        # Use testing-harness to make sure your threaded matvec works
        # This call should print zero (or a numerically zero value)
        # Not to be run with HPC 
        if option == 2 or option == 3:
            check_matvec(fpp_numeric, n, fcn)

        # Grid of evenly spaced points for a reference evaluation of
        # the double derivative
        h = 1./(n-1)
        pts = linspace(0,1,n)
        X,Y = meshgrid(pts, pts)
        X = X.reshape(-1,)
        Y = Y.reshape(-1,)
        fpp = fcnpp(X,Y)

        # Account for domain boundaries.
        #
        # Correcting boundary grid points using one-sided approximations.
        # The boundary_points array is a Boolean array, mask on an array.
        # Simulating ghost values just outside the domain using the original function f.
        # For example if boundary_points is True at 10 points and False at 90 points, 
        # then x[boundary_points] will be a length 10 array at those 10 True locations
  
        ###Boundaries

        # Bottom boundary 
        boundary_points = (Y == 0)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points], Y[boundary_points]-h)

        # Top Boundary
        boundary_points = (Y == 1)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points], Y[boundary_points] + h)

        # Left Boundary
        boundary_points = (X == 0)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points] - h, Y[boundary_points])

        #Right Boundary
        boundary_points = (X == 1)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points] + h, Y[boundary_points])

        
        # Computing error and timing
        e = fpp_numeric - fpp
        error[i, j] = L2norm(e, h)

        # Record the best runtime for this configuration
        timings[i, j] = min_time

        # Output
        print("fpp numeric: ", fpp_numeric)
        print("fpp: ", fpp)
        print(min_time)
    print(" ") #Spacer

    ### End Loop over various grid-sizes

    ####################### NOTE ####################
    # If using option 1 Large grid for HPC Cluster   #
    # Comment out plotting functions.                #
    #################################################

    # Saving plot and output timings for Option 2 and Option 3
    # Plot convergence curve
    pyplot.loglog(NN, error[i, :], label='Error')  # Plot error values
    pyplot.loglog(NN.astype(float), (NN.astype(float) ** -2), label='Ref Quadratic', linestyle='--')

    # Formatting the plot
    pyplot.xlabel('Grid Size (N)', fontsize=14)
    pyplot.ylabel('Error (L2 Norm)', fontsize=14)
    pyplot.title(f'Convergence for {nt} threads', fontsize=16)
    pyplot.legend(fontsize=12)
    pyplot.savefig(f'error{i}.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0)

    # Save timings for Option 1 and Option 2
    savetxt('timings.txt', timings)
    savetxt('timings1thread.txt', timings)
    savetxt('timings2thread.txt', timings)
    savetxt('timings3thread.txt', timings)

    ####################### NOTE ####################
    #       If using option 2 or Option 3            #
    #       Comment out plotting functions.          #
    #################################################

    ## Save timings for Option 1
    #savetxt('timingsCARC.txt', timings)





    # Post Analysis: Examine the stencil structure at different grid locations

    # Uncomment this block to inspect how the Poisson matrix (A) varies
    # between corner, edge, and interior points. Useful for verifying stencil correctness.

    import numpy as np
    from scipy.sparse import csr_matrix

    n = 5  
    A = poisson((n, n), format='csr')

    corner_row = 0          # Top-left corner
    edge_row = 1            # Next to corner
    interior_row = n + 1    # Fully interior point

    print("Matrix row for corner point (0, 0):")
    print("Data:", A[corner_row, :].data)
    print("Indices:", A[corner_row, :].indices)

    print("\\nMatrix row for edge point (0, 1):")
    print("Data:", A[edge_row, :].data)
    print("Indices:", A[edge_row, :].indices)

    print("\\nMatrix row for interior point (1, 1):")
    print("Data:", A[interior_row, :].data)
    print("Indices:", A[interior_row, :].indices)


