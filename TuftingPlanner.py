# This library is made for using with tufting robot project 
# by manipulating data and organize to help getting a cleaner data
import PathFindingAlgorithm as pfa


def PathFindOptimized(datapoints):
    # Now assuming the case where there is no duplicated data
    totalDistance, listOfData = pfa.PathFind(datapoints)
    return pfa.OptimizePathByOpt2(totalDistance, listOfData)
     