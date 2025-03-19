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

    # Task:
    # Return the L2-norm, i.e., the square roof of the integral of e^2
    # Assume a uniform grid in x and y, and apply the midpoint rule.
    # Assume that each grid point represents the midpoint of an equally sized region
    return  h*sqrt(sum(e**2))


def compute_fd(n, nt, k, f, fpp_num):
    '''
    Compute the numeric second derivative of the function 'f' with a
    threaded matrix-vector multiply.

    Input
    -----
    n   <int>       :   Number of grid points in x and y for global problem
    nt  <int>       :   Number of threads
    k   <int>       :   My thread number
    f   <func>      :   Function to take second derivative of
    fpp_num <array> :   Global array of size n**2


    Output
    ------
    fpp_num will have this thread's local portion of the second derivative
    written into it


    Notes
    -----
    We do a 1D domain decomposition.  Each thread 'owns' the k*(n/nt) : (k+1)*(n/nt) rows
    of the domain.

    For example,
    Let the global points in the x-dimension be [0, 0.33, 0.66, 1.0]
    Let the global points in the y-dimension be [0, 0.33, 0.66, 1.0]
    Let the number of threads be two (nt=2)

    Then for the k=0 case (for the 0th thread), the domain rows  'owned' are
    y = 0,    and x = [0, 0.33, 0.66, 1.0]
    y = 0.33, and x = [0, 0.33, 0.66, 1.0]

    Then for the k = 1, case, the domain rows 'owned' are
    y = 0.66, and x = [0, 0.33, 0.66, 1.0]
    y = 1.0,  and x = [0, 0.33, 0.66, 1.0]

    We assume that n/nt divides evenly.

    '''

    # Task:
    # Compute start, end
    #
    # These indices allow you to index into arrays and grab only this thread's
    # portion.  For example, using the y = [0, 0.33, 0.66, 1.0] example above,
    # and considering thread 0, will yield start = 0 and end = 2, so that
    # y[start:end] --> [0, 0.33]
    start = int(k*(n/nt))
    end = int((k+1)*(n/nt))

    # Task:
    # Compute start_halo, and end_halo
    #
    # These values are the same as start and end, only they are expanded to
    # include the halo region.
    #
    # Halo regions essentially expand a thread's local domain to include enough
    # information from neighboring threads to carry out the needed computation.
    # For the above example, that means
    #   - Including the y=0.66 row of points for the k=0 case
    #     so that y[start_halo : end_halo] --> [0, 0.33, 0.66]
    #   - Including the y=0.33 row of points for the k=1 case
    #     so that y[start_halo : end_halo] --> [0.33, 0.66, 1.0]
    #   - Note that for larger numbers of threads, some threads
    #     will have halo regions including domain rows above and below.
    start_halo = start - 1 if k != 0 else start
    end_halo = end + 1 if k != (nt - 1) else end


    # Construct local CSR matrix.  Here, you're given that function in poisson.py
    # This matrix will contain the extra halo domain rows
    A = poisson((end_halo - start_halo, n), format='csr')
    h = 1./(n-1)
    A *= 1/h**2

    # Task:
    # Inspect a row or two of A, and verify that it's the correct 5 point stencil
    # You can print a few rows of A, with print(A[k,:])
    #print(f"Row {start} of A:")
    #print(A[start, :])

    #print(f"Row {start + 1} of A:")
    #print(A[start + 1, :])
    # Task:
    # Construct a grid of evenly spaced points over this thread's halo region
    #
    # x_pts contains all of the points in the x-direction in this thread's halo region
    x_pts = linspace(0,1,n)
    #
    # y_pts contains all of the points in the y-direction for this thread's halo region
    # For the above example and thread 1 (k=1), this is y_pts = [0.33, 0.66, 1.0]
    y_pts = linspace(0, 1, n)

    # Task:
    # There is no coding to do here, but examime how meshgrid works and
    # understand how it gives you the correct uniform grid.
    X,Y = meshgrid(x_pts, y_pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,)

    # Task:
    # Compute local portion of f by using X and Y
    f_vals = f(X, Y)
    #print("f_vals", + f_vals)
    #print("x,y", + X, + Y)
    local_f_vals = f_vals[start_halo * n: end_halo * n]

    # Task:
    # Compute the correct range of output values for this thread
    #print(f"Thread {k} - Shape of A: {A.shape}, Shape of f_vals: {f_vals.shape}")
    output = A*local_f_vals
    if k == 0:
        output = output[:(end - start) * n]
    elif k == nt - 1:
        output = output[(1 * n):]
    else:
        output = output[(1 * n):-(1 * n)]

        # Task:
    # Set the output array
    fpp_num[start * n: end * n] = output


def fcn(x, y):
    return cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) + sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))
    #return x**2 + y**2

def fcnpp(x, y):
    fcnppx = (-2. / 9.) * (x + 1) ** (-5. / 3.) * (cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))) + (1. / 9.) * (x + 1) ** (-4. / 3.) * (-cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)))
    fcnppy = (-2. / 9.) * (y + 1) ** (-5. / 3.) * (cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.))) + (1. / 9.) * (y + 1) ** (-4. / 3.) * (-cos((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)) - sin((x + 1) ** (1. / 3.) + (y + 1) ** (1. / 3.)))
    return fcnppx + fcnppy
    #return 4 * ones_like(x)




##
# Here are three problem size options for running.  The instructor has chosen these
# for you.
option = 2
if option == 1:
    # Choose this if doing a final run on CARC for your strong scaling study
    NN = array([840*6]) # array of grid sizes "n" to loop over
    num_threads = [1,2,3,4,5,6,7,8]
elif option == 2:
    # Choose this for printing convergence plots on your laptop/lab machine,
    # and for initial runs on CARC.
    # You may want to start with just num_threads=[1] and debug the serial case first.
    NN = 210*arange(1,6) # array of grid sizes "n" to loop over
    num_threads = [1] #eventually include 2, 3
elif option == 3:
    # Choose this for code development and debugging on your laptop/lab machine
    # You may want to start with just num_threads=[1] and debug the serial case first.
    NN = array([6]) # array of grid sizes "n" to loop over
    num_threads = [2] #eventually include 2,3
else:
    print("Incorrect Option!")

##
# Begin main computation loop
##

# Task:
# Initialize your data arrays
error = zeros((len(num_threads), len(NN)))
timings = zeros((len(num_threads), len(NN)))


# Loop over various numbers of threads
for i,nt in enumerate(num_threads):
    # Loop over various problem sizes
    for j,n in enumerate(NN):

        # Task:
        # Initialize output array
        fpp_numeric = zeros((n * n,))

        # Task:
        # Choose the number of timings to do for each run
        ntimings = 5

        # Carry out timing experiment
        min_time = 10000
        for m in range(ntimings):

            # This loop will set up each Thread object to compute fpp numerically in the
            # interior of each thread's domain.  That is, after this loop
            # t_list = [ Thread_object_1, Thread_object_2, ...]
            # where each Thread_object will be ready to compute one thread's contribution
            # to fpp_numeric.  The threads are launched below.
            t_list = []
            for k in range(nt):
                # Task:
                # Finish this call to Thread(), passing in the correct target and arguments
                t_list.append(Thread(target=compute_fd, args=(n, nt, k, fcn, fpp_numeric)))

            start = time.perf_counter()
            # Task:
            # Loop over each thread object to launch them.  Then separately loop over each
            # thread object to join the threads.
            for t in t_list:
                t.start()

            for t in t_list:
                t.join()
            end = time.perf_counter()
            min_time = min([end - start, min_time])
        ##
        # End loop over timings
        print(" ")

        ##
        # Use testing-harness to make sure your threaded matvec works
        # This call should print zero (or a numerically zero value)
        if option == 2 or option == 3:
            check_matvec(fpp_numeric, n, fcn)

        # Construct grid of evenly spaced points for a reference evaluation of
        # the double derivative
        h = 1./(n-1)
        pts = linspace(0,1,n)
        X,Y = meshgrid(pts, pts)
        X = X.reshape(-1,)
        Y = Y.reshape(-1,)
        fpp = fcnpp(X,Y)

        # Account for domain boundaries.
        #
        # The boundary_points array is a Boolean array, that acts like a
        # mask on an array.  For example if boundary_points is True at 10
        # points and False at 90 points, then x[boundary_points] will be a
        # length 10 array at those 10 True locations
        boundary_points = (Y == 0)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points], Y[boundary_points]-h)

        # Task:
        # Account for the domain boundaries at Y == 1, X == 0, X == 1
        boundary_points = (Y == 1)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points], Y[boundary_points] + h)

        boundary_points = (X == 0)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points] - h, Y[boundary_points])

        boundary_points = (X == 1)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points] + h, Y[boundary_points])

        # Task:
        # Compute error
        print("fpp numeric: ", + fpp_numeric)
        print("fpp:", + fpp)
        e = fpp_numeric - fpp
        error[i, j] = L2norm(e, h)
        timings[i, j] = min_time
        print(min_time)
    ##
    # End Loop over various grid-sizes
    print(" ")

    # Task:
    # Generate and save plot showing convergence for this thread number
    # --> Comment out plotting before running on CARC
    pyplot.loglog(NN, error[i, :], label='Error')  # Plot error values
    pyplot.loglog(NN.astype(float), (NN.astype(float) ** -2), label='Ref Quadratic', linestyle='--')

    # Formatting the plot
    pyplot.xlabel('Grid Size (N)', fontsize=14)
    pyplot.ylabel('Error (L2 Norm)', fontsize=14)
    pyplot.title(f'Convergence for {nt} threads', fontsize=16)
    pyplot.legend(fontsize=12)
    pyplot.savefig(f'error{i}.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0)

# Save timings for future use LOCAL
#savetxt('timings.txt', timings)
#savetxt('timings1thread.txt', timings)
#savetxt('timings2thread.txt', timings)
#savetxt('timings3thread.txt', timings)


# Save timings for future use CARC
#savetxt('timingsCARC.txt', timings)


"""
Comment this out during CARC.
"""
""""""

## Post Analysis
#Domain Walls

# import numpy as np
# from scipy.sparse import csr_matrix

# n = 5  

# A = poisson((n, n), format='csr')

# corner_row = 0
# edge_row = 1
# interior_row = n + 1  

# #(0,0)
# print("Matrix row for corner point (0, 0):")
# print("Data:", A[corner_row, :].data)
# print("Indices:", A[corner_row, :].indices)

# #(0,1)
# print("\nMatrix row for edge point (0, 1):")
# print("Data:", A[edge_row, :].data)
# print("Indices:", A[edge_row, :].indices)

# #(1,1)
# print("\nMatrix row for interior point (1, 1):")
# print("Data:", A[interior_row, :].data)
# print("Indices:", A[interior_row, :].indices)

