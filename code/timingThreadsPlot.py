import matplotlib.pyplot as plt
import numpy as np

def load_timings(filename):
    with open(filename, 'r') as file:
        return [float(value) for value in file.read().split()]

# Outputs Option 2 or Option 3 
timings_1_thread = load_timings('output/1threadSerialTimings.txt')
timings_2_thread = load_timings('output/2threadSerialTimings.txt')
timings_3_thread = load_timings('output/3threadSerialTimings.txt')

#Outputs for Option 1
parallelThreads = load_timings('outputHPC/parallelTimings.txt')

#Serial Plot
x = np.arange(1, len(timings_1_thread) + 1)
plt.figure(figsize=(10, 6))
plt.plot(x, timings_1_thread, linestyle='-', label='1 Thread')
plt.plot(x, timings_2_thread, linestyle='-', label='2 Thread')
plt.plot(x, timings_3_thread, linestyle='-', label='3 Thread')
plt.xlabel('Thread')
plt.ylabel('Time (seconds)')
plt.title('Timings for Different Numbers of Threads')
plt.legend()
plt.savefig('output/serialTimingsPlot.png')
plt.show()

#Strong Scaling
plt.plot(np.arange(1, len(parallelThreads) + 1), parallelThreads, 
         linestyle='--', marker='o', label='Parallel Option 1')
plt.xlabel('Thread')
plt.ylabel('Time (seconds)')
plt.title('Weak Scaling')
plt.legend()
plt.savefig('output/weakScaling.png')
plt.show()