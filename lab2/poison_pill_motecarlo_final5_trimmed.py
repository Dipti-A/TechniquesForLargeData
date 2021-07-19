import multiprocessing
import time
import random
from ipython_genutils.py3compat import xrange
from math import pi
import json
import os
class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        """
        Handles task from queue.
        """
        proc_name = self.name # from Process
        while True:
            next_task = self.task_queue.get() # multiprocessing.JoinableQueue() init from Consumer, get task obj
            if next_task is None: # poision pill
                #print(f'Exiting {proc_name}')

                self.task_queue.task_done()
                break
            #print(proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)

class sample_pi():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        time.sleep(0.1)  # pretend to take some time to do the work
        #print("Hello from a worker", os.getpid())
        s = 0
        #random.seed(self.b) #increments randome seed.

        random.seed()
        for i in range(self.a): #args ?
            x = random.random()
            y = random.random()
            #print(f'x {x}, y {y}')
            if x ** 2 + y ** 2 <= 1.0:
                s += 1
        print(f'S = {s}')
        return s


def run_jobs(num_consumers, accuracy, num_of_predictions):
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue() # like a list

    # Start consumers


    # multiprocessing.cpu_count()

    #print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)] # create a list if Consumer instances.

    list(worker.start() for worker in consumers)
    # start() Start the process’s activity.
    # This must be called at most once per process object.
    # It arranges for the object’s run() method to be invoked in a separate process.


    # Enqueue jobs
    pi_est = 0
    #accuracy = 0.01 # format ok 99% = 0.01 99.9 0.001
    # 0.001
    n = 0

    #print(f' abs val {abs(pi - pi_est)}')
    total_result = 0
    count_tasks = 0
    inc_rnd = 0
    while abs(pi - pi_est) > accuracy:
        inc_rnd += 1
        #Problem, we only put a new task wjen we evaulate the the error.

        print(f'num of consumers{num_consumers}')
        for i in range(num_consumers):
            tasks.put(sample_pi(a=num_of_predictions, b=inc_rnd)) # we put task for the consumers 12 + 12 + 12 +12 8 +8+ 8+ 8)
        tasks.join()
        count_tasks += num_consumers
        for i in range(1, num_consumers+1):
            print(f'printing i {i}')
            result = results.get()
            print('Results queue:', result)
            total_result += result
        print('Result Total:', total_result)
        n += num_of_predictions
        pi_est = (4.0 * total_result) / (n*num_consumers)


        #print(f'pi estimate = {pi_est}')
        #print(f'steps {n}')
        print(" Steps\tSuccess\tPi est.\tError")
        print("%6d\t%7d\t%1.5f\t%1.5f" % (n, total_result, pi_est, pi - pi_est))

        #n = 1000 # comp per task/

    for i in range(num_consumers):
        tasks.put(None)  # Terminate
        tasks.join()

    #print(f'number of tasks {n/count_tasks}')
    tasks.join()
    return n

if __name__ == '__main__':
    accuracy = 0.00001

    num_of_predictions = 10000


    timer = {}
    max_range = multiprocessing.cpu_count()

    for i in [1,2,4,8, 16, 24, 32]:
        start_time = time.time()
        tasks_taken =  run_jobs(num_consumers=i, accuracy=accuracy, num_of_predictions=num_of_predictions)
        time_taken = (time.time() - start_time)
        print('Time', time_taken)
        timer[str(i)] = (tasks_taken, time_taken, (tasks_taken/time_taken)/10000)

    print(f'time {timer}')
    # with open('pi_acc'+str(accuracy)+'.json', 'w') as outfile:
    f_name = 'new2pi_acc_pred'+str(num_of_predictions)+'_accs'+str(accuracy)
    with open(f_name+'.json', 'w') as outfile:
        json.dump(timer, outfile)
    exit()


#he speedup is then calculated as how many samples per second the parallel version produces
# divided by the number of samples per second the serial version produces. See [#/s] below.