from multiprocessing import Process, Value, Array, Queue
import  random
import argparse # See https://docs.python.org/3/library/argparse.html
import time
from math import pi

def f(q):
    s = 0
    # random.seed(self.b) #increments randome seed.
    n = 100000
    random.seed()
    for i in range(n):  # args ?
        x = random.random()
        y = random.random()
        # print(f'x {x}, y {y}')
        if x ** 2 + y ** 2 <= 1.0:
            s += 1
    return q.put(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers', '-w',
                        default='1',
                        type=int,
                        help='Number of parallel processes')
    parser.add_argument('--accuracy', '-a',
                        default='0.0001',
                        type=float,
                        help='Number of steps in the Monte Carlo simulation')
    args = parser.parse_args()

    timer = {}
    start_time = time.time()
    q = Queue()
    total_result = 0
    n = 0
    pi_est = 0
    #while total_result <1000000:
    while abs(pi - pi_est) > args.accuracy:

        consumers = [Process(target=f, args=(q,))
                     for i in range(args.workers)]  # create a list if Consumer instances.

        x = list(worker.start() for worker in consumers)

        for worker in consumers:
            #print(f'worker pid {worker.pid}')
            total_result+=q.get()

        n += 100000
        pi_est = (4.0 * total_result) / (n * args.workers)
        #print(f'pi est {pi_est}')

        list(worker.join() for worker in consumers)

    time_taken = (time.time() - start_time)
    print('Time', time_taken)
    #print(f'total result {total_result}')
    print(f'workers = {args.workers}')
    print(f'n {n*args.workers}')
    print(f'calcu per sec {((n*args.workers)/time_taken)/10000}')