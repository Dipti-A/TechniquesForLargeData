from pyspark import SparkContext 
import argparse
import time

start_time = time.time()

def Statistics(args):
    
    # Initate pyspark. Sparkcontext is the entry to all spark functionalities
    sc = SparkContext(master = 'local[%s]' % args.cores) # number of cores is an input argument
    distfile = sc.textFile(args.file) # datafile is an input argument
    
    
    data = distfile.map(lambda t: float(t.split()[2])) # make column 3 (values) into a float
    count = data.count() # the length of the data matrix
    
    # Calculate the basic statistics: mean, standard deviation, minimum and maximum 
    mean = data.reduce(lambda a, b: a + b) / count 
    sd = (data.map(lambda t: (t - mean)**2).reduce(lambda a, b: a + b) / count)**0.5
    mini = data.reduce(lambda a, b: min(a, b))
    maxi = data.reduce(lambda a, b: max(a, b))
    
    # Print basic statistics
    print('Mean : %1.4f' % mean)
    print('St.Dev: %1.4f' % sd)
    print('Min : %1.4f' % mini)
    print('Max : %1.4f' % maxi)
    
    # For loop that yields the bin borders and the bin count
    width = (maxi - mini) / 10 # width of bins
    bins = []
    bin_count = []
    for i in range(0, 10):
        upper_i = mini + width*(i + 1)
        lower_i = mini + width*i
        bin_i = data.filter(lambda t: t < (mini + width*(i + 1)) and t >= (mini + width*i)).count()
        if i == 9: # since the last bin will include observations that fall on the left and right bin border
            bin_i = data.filter(lambda t: t <= (mini + width*(i + 1)) and t >= (mini + width*i)).count()
        bins.append([lower_i, upper_i])
        bin_count.append(bin_i)
        
    # For loop that printse each bin and associated bin count on a new row
    for i,j in zip(bins, bin_count): 
        print(i, j)

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
    print("Execution time in seconds: %s" % (time.time() - start_time))
