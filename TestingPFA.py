import PathFindingAlgorithm as pfa

points_far = [
    # (0, 0),
    (0, 10),
    (10, 10),
    (10, 0),
    (50, 5),
    (0, 0),
]

points_grid_3x3 = [
    (0, 0),  (10, 0),  (20, 0),
    (0, 10), (10, 10), (20, 10),
    (0, 20), (10, 20), (20, 20),
]

totalDistance, listOfData = pfa.PathFind(points_far)
print("Initial distance:", totalDistance)
optDistance, optPath = pfa.OptimizePathByOpt2(totalDistance, listOfData)
print("Optimized distance:", optDistance)
for p in optPath:
    print(p)