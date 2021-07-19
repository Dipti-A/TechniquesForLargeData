from pyspark import SparkContext 
import argparse
import time

start_time = time.time()

def Statistics(args):
    sc = SparkContext(master = "local[%s]" % args.cores)
    datafile = sc.textFile(args.file)

    data = datafile.map(lambda line: line.split()).map(lambda line: float(line[2]))
    count = data.count() 
    
    mean = data.reduce(lambda a, b: a + b) / count
    sd = (data.map(lambda line: (line - mean)**2).reduce(lambda a, b: a + b) / count)**0.5
    mini = data.reduce(lambda a, b: min(a, b))
    maxi = data.reduce(lambda a, b: max(a, b)) 

    print("Mean : %1.4f" % mean)
    print("St.Dev: %1.4f" % sd)
    print("Min : %1.4f" % mini)
    print("Max : %1.4f" % maxi)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file', '-f',
                        type = str,
                        help='datafile')
    parser.add_argument('--cores', '-c',
                        type = int,
                        default = 1,
                        help='Number of cores')
    args = parser.parse_args()
    Statistics(args)
    print("Execution time: %s seconds" % (time.time() - start_time))
