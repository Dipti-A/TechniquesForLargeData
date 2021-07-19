import multiprocessing
import time
import random
from ipython_genutils.py3compat import xrange
from math import pi

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print(f'Exiting {proc_name}')

                self.task_queue.task_done()
                break
            print(proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)

class Task(object):
    def __init__(self, a):
        self.a = a

    def __call__(self):
        time.sleep(0.1)  # pretend to take some time to do the work
        random.seed()
        print("Hello from a worker")
        s = 0
        for i in range(self.a):
            x = random.random()
            y = random.random()
            if x ** 2 + y ** 2 <= 1.0:
                s += 1
        return s



if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start consumers
    num_consumers = 12 # multiprocessing.cpu_count() * 2

    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]

    list(worker.start() for worker in consumers)

    # Enqueue jobs
    num_jobs = 100 # when should n be the limit
    for i in range(num_jobs):
        tasks.put(Task(i))

    # Add a poison pill for each consumer
#



    # Start printing results
    while num_jobs:
        result = results.get()
        print('Result:', result)

        num_jobs -= 1
        pi_est = (4.0 * result) / num_jobs
        print(" Steps\tSuccess\tPi est.\tError")
        print("%6d\t%7d\t%1.5f\t%1.5f" % (num_jobs, result, pi_est, pi - pi_est))
        if pi - pi_est < 0.8:
            for i in range(num_consumers):
                tasks.put(None)  # 99%
                # Wait for all of the tasks to finish
        tasks.join()
