# import BF library
from bloom_filter import BloomFilter

# Create a BF and elements to it
def Create_and_fillin_BF(inputList,maxElement, Error):
    myBF = BloomFilter(max_elements=maxElement, error_rate=Error)
    for element in inputList:
        if element not in myBF:
            myBF.add(element)
        else:
            print("Element {} is already in the BF".format(element))
    print('*********** Bloom Filter is created and filled-in ***************')        
    return myBF

def Query(BF, QueryInput, error):

    for query in QueryInput:
        if query in BF:
            print("Element {} is in the BF with {} error rate".format(query, error))
        else:
            print("Element {} is certainly NOT in the BF".format(query))

ErrorRate=0.001
MaxBFsize=10
#Inputs for filling-in the BF
Inputs = ["a", "b", "f", "g", "c","d","a"]

#create and fill-in the BF
myBF=Create_and_fillin_BF(Inputs, MaxBFsize, ErrorRate)

# Queries BF all the inputs that were added + "l" & "d"
Query(myBF, Inputs + ["l","d","a"],ErrorRate) 
