from mrjob.job import MRJob
import math

class CalculateStats(MRJob):

    def mapper_init(self):
        self.group = int(self.options.group)

    def mapper(self, _, line):
        words = line.split('\t')
        grp = int(words[1])  # column <group> from the data file
        if grp == self.group:
            yield "None", float(words[2])  # Yield <value> column from data field

    def configure_args(self):
        super(CalculateStats, self).configure_args()
        self.add_passthru_arg('--group', default=1,
                    help="python Problem1d.py --group 13 assignment3.dat")

    # Function to calculate frequency of values in given range
    def count_records_in_range(self, values, minrange, maxrange):
        counter = 0
        # Loop thru all values
        for val in values:
            if val >= minrange and val <= maxrange:
                counter += 1
        return counter

    def combiner(self, _, value):
        count = 0
        values = 0
        sumofsquares = 0
        listofvalues = []

        for idx, v in enumerate(value):
            count += 1
            values += v
            sumofsquares += v**2
            listofvalues.append(v)

        # Yield (sum of values, count, sum of squares, min, max, list of values) tuple for each mapper
        yield "None", (values, count, sumofsquares, min(listofvalues), max(listofvalues), listofvalues)

    def reducer(self, _, tuple):
        counts = 0
        values = 0
        sumofsquares = 0
        listofminvalues = []
        listofmaxvalues = []
        listofvalues = []

        for idx,tup in enumerate(tuple): # Iterating through Iterator object
            values += tup[0]  # Calculate sum of values
            counts += tup[1]  # Calculate total count
            sumofsquares += tup[2] # Aggregate total sum of squares of all values
            listofminvalues.append(tup[3])  # Group all minimum values from each combiner
            listofmaxvalues.append(tup[4])  # Group all maximum values from each combiner
            listofvalues.append(tup[5])  # Group all the values from each combiner

        # Retrieve min and max values from list retrieved from combiner
        minval = min(listofminvalues)
        maxval = max(listofmaxvalues)

        # Calculate Mean
        mean = values / counts

        # Calculate Std. deviation
        stddev = math.sqrt(sumofsquares / counts - mean ** 2)

        # Yield Mean, Std. deviation, Min, Max values
        yield "Mean", mean
        yield "Stddev", stddev
        yield "MinValue", minval
        yield "MaxValue", maxval

        ### Calculate data for histogram i.e. Bin interval and the number of records in that range ###
        numofbins = 10
        listofallvalues = []

        # Read all the lists of values from combiner and convert them into a single flat list
        for elem in listofvalues:
            listofallvalues.extend(elem)

        # Calculate bin width - << Rounding off to 5 decimals >>
        bin_width = (maxval - minval)/numofbins
        bin_width = round(bin_width,5)

        # Create list of bin intervals with minimum value
        bin_intervals = [minval]
        newval = minval

        # Calculate bin-intervals
        for i in range(numofbins):
            # bin interval = previous value + bin width
            newval += bin_width
            if newval < (maxval + bin_width):
                bin_intervals.append((newval))

        # Calculate bin-intervals and frequency of values in that range and yield results
        # <<Not rounding off the bin-intervals as they are long float values>>
        for i in range(len(bin_intervals) - 1):
            yield (bin_intervals[i], bin_intervals[i + 1]), self.count_records_in_range(listofallvalues, bin_intervals[i],bin_intervals[i + 1])


if __name__ == '__main__':

    CalculateStats.run()