from mrjob.job import MRJob, MRStep
import statistics as stat


class MRWordFrequencyCount(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_data,
                   combiner=self.combiner),
            MRStep(
                   reducer=self.reducer)]

    def mapper_get_data(self, _, line): # 2 c -> 2 mappers
        yield "None", float(line.split()[2])

    def combiner(self, _, value):
        count = 0
        #values = 0
        values = []
        for _, v in enumerate(value):
            count += 1
            values.append(v)
        yield "Sum", (sum(values),min(values), count)
        #yield "Min", (min(values))# Yield (sum of values, count) tuple for each mapper

    def combiner_min(self, _, value):
        values = []
        for _, v in enumerate(value):
            values.append(v)
        yield "min", (min(values))  # Yield (sum of values, count) tuple for each mapper


    def reducer(self, key, tuple):
        counts = 0
        values = 0
        min_val = []

        for _,tup in enumerate(tuple): # Iterating thru Iterator object
            values += tup[0]
            min_val.append(tup[1]) # Retrieve sum of values)
            counts += tup[2]  # Retrieve total count
        mean = values/counts
        yield "mean", mean
        yield "min", (min(min_val))


    def reducer_min(self, _, value):
        values = []
        for _, v in enumerate(value):
            values.append(v)
        yield "min", (min(values))  # Yield (sum of values, count) tuple for each mapper


if __name__ == '__main__':
    MRWordFrequencyCount.run()