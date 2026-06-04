import math

def EuclideanDistanceBetweenTwoPoints(datapoint1, datapoint2):
    x1, y1 = datapoint1
    x2, y2 = datapoint2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


# Let listOfCoordiantes be the list that each elements contain two elements inside which is x and y value
def PathFind(listOfCoordinates):

    copiedList = listOfCoordinates.copy()
    sizeOfList = len(listOfCoordinates)

    if (len(listOfCoordinates) == 0):
        raise ValueError("The list of coordinates is empty")
    elif (sizeOfList == 1):
        return 0, listOfCoordinates[0], 
    
    # so from here we expect the list to have at least 2 elements
    visitedDatapoint = []

    # We can build funciton to find the ideal starting point later
    # This assume that the first value is the ideal one
    visitedDatapoint.append(listOfCoordinates[0])
    copiedList.remove(listOfCoordinates[0])

    totalDistance = 0

    while len(copiedList) != 0:
        # The last time this loop will run is when the copiedList has 1 element

        minimalDistanceFromBaseValue = math.inf
        minimalDistanceFromBaseDatapoint = None
        # visitedDatapoint will never be empty since we set the origin point to be visited
        lastVisitedDatapoint = visitedDatapoint[-1]

        for candidatePoint in copiedList:

            # Considering the case that there is duplicated value
            # Can later deal with it before passing into this funciton
            if lastVisitedDatapoint == candidatePoint:
                continue
            
            distance = EuclideanDistanceBetweenTwoPoints(lastVisitedDatapoint, candidatePoint)
            
            if(distance < minimalDistanceFromBaseValue):
                minimalDistanceFromBaseValue = distance
                minimalDistanceFromBaseDatapoint = candidatePoint
        
        
        # After the loop is finished, the datapoint and distance value will not be empty
        # The datapoint we got should be the closest or last element in the list
        visitedDatapoint.append(minimalDistanceFromBaseDatapoint)
        copiedList.remove(minimalDistanceFromBaseDatapoint)
        totalDistance += minimalDistanceFromBaseValue
    
    # This has no optimization yet
    return totalDistance, visitedDatapoint


def OptimizePathByOpt2(totalDistance, visitedDatapoint):
    optimizingPath = visitedDatapoint.copy()
    #total distance will just use for comparing the reduced cost
    # reduce cost = totalDistance - newCost
    # If reduce cost is positive then it is improved 

    if(len(optimizingPath) < 4):
        return totalDistance, optimizingPath
    
    # If the number of edges in the visitedDatapoint is n,
    # there is n(n-3)/2 possibly combination way to try and improve
    # So the cost of each optimization will be around O(n^2)

    # same as the value of the number of edges
    numberOfNodes = len(optimizingPath)

    totalNumberOfIteration = 0
    maxNumberOfIteration = numberOfNodes * (numberOfNodes - 3) / 2

    updatedCost = totalDistance

    while True:
        totalNumberOfIteration += 1
        changed = False
        for i in range(0, numberOfNodes - 3):
            for j in range(i + 2, numberOfNodes - 1):
                s1 = i
                e1 = i + 1
                s2 = j
                e2 = j + 1

                # the swap A-B and C-D edges is swap into A-C and B-D
                # and then travel in the opposite direction of the initial path in the period of B to C becoming C to B

                # before swapping we can just calculate the cost if distance from A-C + B-D is lower than A-B and C-D, if so then it is improved

                localInitialCost = (EuclideanDistanceBetweenTwoPoints(optimizingPath[s1], optimizingPath[e1]) 
                                                + EuclideanDistanceBetweenTwoPoints(optimizingPath[s2], optimizingPath[e2]))
                localCandidateCost = (EuclideanDistanceBetweenTwoPoints(optimizingPath[s1], optimizingPath[s2]) 
                                                + EuclideanDistanceBetweenTwoPoints(optimizingPath[e1], optimizingPath[e2]))
                reducedCost = localInitialCost - localCandidateCost
                if(reducedCost > 0):
                    #swap the paths
                    optimizingPath[e1:s2+1] = optimizingPath[e1:s2+1][::-1]
                    changed = True
                    updatedCost -= reducedCost
                    break
            if(changed is True):
                break
                
        # If there is change still then it might not be optimal yet
        # But if change is still false or all the candidate that we try to swap does not produce any lower cost. It is optimal for 2opt.
        # It has finite iteration because there exist a point where no improvement can be made no matter how many swapping combination you try. 
        # There exist minimal distance combination so there must be finite iteration
        if(changed is False):
            break

        #case where iteration never stop
        # debugging purposes
        if(totalNumberOfIteration > maxNumberOfIteration):
            print("Reached the point where number of totalNumberOfIteration exceed its max")
            break
    return updatedCost, optimizingPath