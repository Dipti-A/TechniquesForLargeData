# import BF library
from bloom_filter import BloomFilter
from collections import defaultdict

d = defaultdict(int)
Inputs = ["1", "2", "3", "2", "2", "2", "2", "3", "3", "3"]

myBF=BloomFilter(max_elements=10, error_rate=0.001)
myBF2=BloomFilter(max_elements=10, error_rate=0.01)
myBF3=BloomFilter(max_elements=10, error_rate=0.01)


for x in Inputs:
    print(f'x {x}')
    if x not in myBF: #90%
        print(f'adding bf1 {x}')
        myBF.add(x)  # 90% low high error
    elif x not in myBF2:
        print(f'adding bf2 {x}')
        myBF2.add(x)  # 90% low high error
    elif x not in myBF3:
        print(f'adding bf3 {x}')
        myBF3.add(x)  # 90% low high error
    else:
        d[x] += 1

print(d.keys())
