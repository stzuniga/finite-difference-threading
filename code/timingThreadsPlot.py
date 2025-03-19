"""
This Python script will generate the plots for the TIMINGS of the threads
as requested in Task 2 of Homework number 4. This file takes in the
.txt files that are generated for each individual timing generated
in the hw4_skeleton.py file and will output a plot and save 
it as timings_plot.png
"""



import matplotlib.pyplot as plt
import numpy as np

def load_timings(filename):
    with open(filename, 'r') as file:
        return [float(value) for value in file.read().split()]

timings_1_thread = load_timings('/Users/stevenzuniga/Desktop/cs471/2024-fall-471-zunigasteven/hw4/code/timings1thread.txt')
timings_2_threads = load_timings('/Users/stevenzuniga/Desktop/cs471/2024-fall-471-zunigasteven/hw4/code/timings2thread.txt')
timings_3_threads = load_timings('/Users/stevenzuniga/Desktop/cs471/2024-fall-471-zunigasteven/hw4/code/timings3thread.txt')

x = np.arange(1, len(timings_1_thread) + 1)
plt.figure(figsize=(10, 6))
plt.plot(x, timings_1_thread, linestyle='-', label='1 Thread')
plt.plot(x, timings_2_threads, linestyle='-', label='2 Thread')
plt.plot(x, timings_3_threads, linestyle='-', label='3 Thread')
plt.xlabel('Index')
plt.ylabel('Time (seconds)')
plt.title('Timings for Different Numbers of Threads')
plt.legend()

plt.savefig('timings_plot.png')
plt.show()


