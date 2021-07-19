
from Problem1a import CalculateStats
import sys

import time

if __name__ == '__main__':
    begin_time = time.time()
    # Remove ['-r', 'local'] when running locally
    #mr_job = CalculateStats(args=['--cat-output', 'mist-5000.dat'])
    # Uncomment below stmnt for testing in Bayes
    mr_job = CalculateStats(args=['-r', 'local','--cat-output', 'assignment3.dat'])
    #mr_job = CalculateStats(args=['--cat-output'])

    with mr_job.make_runner() as runner:
        runner.run()
        for key, value in mr_job.parse_output(runner.cat_output()):
            print(key, value)
    print("Execution time in seconds ", time.time() - begin_time)