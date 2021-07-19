#!/usr/bin/env python
#
# File: kmeans_new.py
# Author: Alexander Schliep (alexander@schlieplab.org)
#
#
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
from multiprocessing import Pool
import functools

def generateData(n, c):
    begin_time = time.time()
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers = c, cluster_std=1.7, shuffle=False,
                      random_state = 2122)
    end_time = time.time()
    print("Execution time of generateData in seconds ", end_time - begin_time)
    return X

#calculate the new cluster, the cluster size and variance for all points in the chunk received
def nearestCentroid(k, centroids, data):

    N = len(data)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)
    variation = np.zeros(k)
    cluster_sizes = np.zeros(k, dtype=int)

    for i in range(len(data)):
        # norm(a-b) is Euclidean distance, matrix - vector computes difference for all rows of matrix
        datum=data[i]
        dist = np.linalg.norm(centroids - datum, axis=1)
        cluster = np.argmin(dist)
        dist = np.min(dist)
        # Assign data points to nearest centroid
        c[i] = cluster
        cluster_sizes[cluster] += 1
        variation[cluster] += dist ** 2

    return c, cluster_sizes, variation


def kmeans(workers, k, data, nr_iter):

    begin_time = time.time()
    N = len(data)

    # Total time to calculate nearest centroid
    nc_total_time=0
    # Total time to calculate recompute centroids
    rc_total_time=0
    # Total time to calculate total variations
    vc_total_time=0

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    logging.debug("Initial centroids\n", centroids)

    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0

    # For each iteration
    for j in range(nr_iter):
        logging.debug("=== Iteration %d ===" % (j+1))

        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)

        # Create a dictionary to store the index of data point and its cluster,dist tuple information
        #dict_cluster_dist = {}
        start_time = time.time()

        # Pre-filling function with initial centroids, k and initial centroids
        assignmentFunction = functools.partial(nearestCentroid, k, centroids)

        # Split data into subLists, each subList would then be sent to workers
        splitList = np.array_split(data, workers)
        subList = []
        for i in splitList:
            subList.append(i)

        # Invoking parallelizable function by passing split data list
        # Pool.map takes care of distributing these chunks to appropriate worker
        with Pool(processes = workers) as pool:
            result = pool.map(assignmentFunction, [subList[i] for i in range(len(subList))])

        # Retrieve the cluster index, cluster size and variation for each chunk received from worker
        cluster_index_list = []
        for tup in result:
            c=tup[0]
            cluster_sizes=tup[1]
            variation=tup[2]
            cluster_index_list.append(c)
            #print("c, cluster size, variation",c, cluster_sizes, variation )

        # Read all the lists of cluster index from all workers and convert them into a single flat list
        flat_list = []
        for elem in cluster_index_list:
            flat_list.extend(elem)

        stop_time = time.time()
        nc_time = stop_time - start_time
        nc_total_time +=nc_time

        #### Calculate total variation #####
        begin_time1 = time.time()

        delta_variation = -total_variation
        total_variation = sum(variation) 
        delta_variation += total_variation
        vc_time = time.time() - begin_time1
        vc_total_time += vc_time
        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids
        start_time2 = time.time()
        centroids = np.zeros((k,2)) # This fixes the dimension to 2
        for i in range(N):
            centroids[flat_list[i]] += data[i]

        centroids = centroids / cluster_sizes.reshape(-1,1)
        rc_time=time.time() - start_time2

        rc_total_time += rc_time

    #print("Total Execution time to compute nearest centroids in seconds ", nc_total_time)
    #print("Total time for calculating variation in seconds ", vc_total_time)
    #print("Total Execution time to recompute centroids in seconds ", rc_total_time)

    return total_variation, c

def computeClustering(args):
    begin_time = time.time()
    if args.verbose:
        logging.basicConfig(format='# %(message)s',level=logging.INFO)
    if args.debug: 
        logging.basicConfig(format='# %(message)s',level=logging.DEBUG)
    begin_time1 = time.time()

    X = generateData(args.samples, args.classes)

    start_time = time.time()
    #
    # Modify kmeans code to use args.worker parallel threads
    total_variation, assignment = kmeans(args.workers, args.k_clusters, X, nr_iter = args.iterations)
    #
    #
    end_time = time.time()
    logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
    print("Clustering complete in %3.2f [s]" % (end_time - start_time))
    #print(f"Total variation {total_variation}")

    if args.plot: # Assuming 2D data
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
        plt.title("k-means result")
        #plt.show()        
        fig.savefig(args.plot)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: kmeans_new.py -v -k 4 --samples 50 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default='2',
                        type = int,
                        help='Number of parallel processes to use')
    parser.add_argument('--k_clusters', '-k',
                        default='3',
                        type = int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        default='100',
                        #default='2',
                        type = int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        default='1000',
                        #default='10',
                        type = int,
                        help='Number of samples to generate as input')
    parser.add_argument('--classes', '-c',
                        default='3',
                        type = int,
                        help='Number of classes to generate samples from')   
    parser.add_argument('--plot', '-p',
                        type = str,
                        help='Filename to plot the final result')   
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print verbose diagnostic output')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Print debugging output')
    args = parser.parse_args()
    computeClustering(args)

