import multiprocessing
import logging


def worker(num):
    """worker function"""
    print('Worker', num)
    return


if __name__ == '__main__':
    jobs = []
    #multiprocessing.log_to_stderr(logging.DEBUG)

    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)

        p.start()
