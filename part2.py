import numpy as np
import os
from part1 import parseProfiles


def extractCB(ratingsFilePath, k = 20, t = 10, epsilon = 0.01, usersClusteringPath = "U.csv", itemsClusteringPath = "V.csv",  codebookPath = "B.csv"):
    l = k

    ratings = np.genfromtxt(ratingsFilePath, delimiter=',', dtype=[int, int, float, long])[1:]  # ignore table's title
    userProfiles, itemProfiles = parseProfiles(ratings)
    userCount = len(userProfiles)
    itemCount = len(itemProfiles)
    vArray = np.array([itemProfiles.keys(), np.random.randint(0, l-1, len(itemProfiles), int)])
    uArray = np.array([userProfiles.keys(), np.random.randint(0, k - 1, len(userProfiles), int)])
    bArray = np.zeros([k, l])
    for i in range(0, k):
        for j in range(0, l):
            usersIDs = extractcluster(uArray, i)
            itemsIDs = extractcluster(vArray, j)
            bArray[i, j] = sumCluster(userProfiles, usersIDs, itemsIDs)
    print ""
    return

def extractcluster(clusterMap, clusterID):
    clusterObjects = []
    for i in range(0, len(clusterMap[0])):
        if clusterMap[1, i] == clusterID:
            clusterObjects.append(clusterMap[0, i])

    return clusterObjects


def sumCluster(userProfiles, clusterUserIDs, clusterItemIDs):
    ratingsCounter = 0
    ratingsSum = 0
    users = [userProfiles[userID] for userID in clusterUserIDs]
    for user in users:
        ratings = np.array([user[1], user[2]])
        ratingsCounter += len(ratings)
        np.sum(rating)
        print ""

    


extractCB("ratings.csv")