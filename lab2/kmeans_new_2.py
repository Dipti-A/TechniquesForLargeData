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
import multiprocessing as mp
import functools

def generateData(n, c):
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers = c, cluster_std=1.7, shuffle=False,
                      random_state = 2122)
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    #print("In nearestCentroid - datum and centroid", datum)
    #print("Centroids ", centroids)
    dist = np.linalg.norm(centroids - datum, axis=1)
    #print("cluster and dist", np.argmin(dist), np.min(dist))
    return np.argmin(dist), np.min(dist)

def kmeans(workers, k, data, nr_iter = 100):
#def kmeans(k, data, nr_iter = 100):

    begin_time = time.time()
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    logging.debug("Initial centroids\n", centroids)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0

    # For each iteration
    for j in range(nr_iter):
        logging.debug("=== Iteration %d ===" % (j+1))
        ###print("Iteration# :", j+1)

        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)

        ########## Added code for parallelization ###########

        # Split data to number of processors/workers
        split_data = np.array_split(data, workers)
        #print("num of clusters, split data ============>",k, split_data)

        # Assign data points to nearest centroid ---> OLD CODE
        #for i in range(N):
            #cluster, dist = nearestCentroid(data[i],centroids)

        #Create a list to aggregate all cluster-dist tuple information
        list_cluster_dist = []

        start_time = time.time()
        # Iterate through the list of split data
        for i in range(len(split_data)):
            assignmentFunction = functools.partial(nearestCentroid, centroids)
            pool = mp.Pool(workers)
            result = pool.map(assignmentFunction, split_data[i])
            ###print("Result:", result)
            # Retrieve the cluster and distance from the result and consolidate data into list
            for tup in result:
                list_cluster_dist.append(tup)

        #print("********** length and list_cluster_dist ===>", len(list_cluster_dist), list_cluster_dist)
        stop_time = time.time()
        print("Execution time of nearestCentroid in seconds ", stop_time - start_time)

        ########## End code for parallelization ###########

        for i in range(N):
            for cluster,dist in list_cluster_dist:  # newly added
                c[i] = cluster
                cluster_sizes[cluster] += 1
                variation[cluster] += dist**2
                #print("cluster cluster_size variation ==>", c[i], cluster_sizes[cluster], variation[cluster])

        delta_variation = -total_variation
        total_variation = sum(variation) 
        delta_variation += total_variation

        print("iteration# total_variation delta_variation", j, total_variation, delta_variation)
        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids
        start_time1 = time.time()
        centroids = np.zeros((k,2)) # This fixes the dimension to 2
        for i in range(N):
            centroids[c[i]] += data[i]        
        centroids = centroids / cluster_sizes.reshape(-1,1)
        stop_time1 = time.time()
        print("Execution time for recomputing centroids in seconds ", stop_time1 - start_time1)

        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)
        print("cluster cluster_size centroids =====>", c, cluster_sizes, centroids)

    return total_variation, c


def computeClustering(args):
    if args.verbose:
        logging.basicConfig(format='# %(message)s',level=logging.INFO)
    if args.debug: 
        logging.basicConfig(format='# %(message)s',level=logging.DEBUG)

    X = generateData(args.samples, args.classes)

    start_time = time.time()
    #
    # Modify kmeans code to use args.worker parallel threads
    total_variation, assignment = kmeans(args.workers, args.k_clusters, X, nr_iter = args.iterations)
    #
    #
    end_time = time.time()
    logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
    print(f"Total variation {total_variation}")

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
                        help='Number of parallel processes to use (NOT IMPLEMENTED)')
    parser.add_argument('--k_clusters', '-k',
                        default='3',
                        type = int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        ##default='100',
                        default='100',
                        type = int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        #default='10000',
                        default='10',
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

