import multiprocessing
import logging


class MyFancyClass(object):

    def __init__(self, name):
        self.name = name

    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print('Doing something fancy in %s for %s!' % (proc_name, self.name))

def worker(q):
    """worker function"""
    task = q.get()
    task.do_something()


if __name__ == '__main__':
    queue = multiprocessing.Queue()
    #    multiprocessing.log_to_stderr(logging.DEBUG)
    for i in range(10):
        p = multiprocessing.Process(target=worker, args=(queue,))
        p.start()

        queue.put(MyFancyClass('Fancy pants'))
    queue.close()
    queue.join_thread()
    p.join()


