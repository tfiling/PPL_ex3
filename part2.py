import numpy as np
import pandas as pd
import os
from part1 import parseProfiles
from time import gmtime, strftime
from multiprocessing import Pool, Pipe

import threading


def extractCB(ratingsFilePath, k = 4, t = 10, epsilon = 0.01, usersClusteringPath = "U.csv", itemsClusteringPath = "V.csv",  codebookPath = "B.csv"):
    global dfUserProfiles, vArray, uArray, bArray, globalSystemRatingsAverage
    l = k
    np.random.seed(10)


    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
    ratings = np.genfromtxt(ratingsFilePath, delimiter=',', dtype=[int, int, float, long])[1:]  # ignore table's title

    # ratings is a list of tuples where the 3rd(f2) element is the rating - extract a series of the ratings and get the mean(average) of that series
    globalSystemRatingsAverage = pd.DataFrame.from_records(ratings, exclude=['f0', 'f1', 'f3'])['f2'].mean()

    userProfiles, itemProfiles = parseProfiles(ratings)
    dfUserProfiles = pd.DataFrame.from_dict({i: userProfiles[i] for i in userProfiles.keys() }, orient='index', columns=['userID', 'items', 'ratings'])
    dfItemProfiles = pd.DataFrame.from_dict({i: itemProfiles[i] for i in itemProfiles.keys() }, orient='index', columns=['itemID', 'users', 'ratings'])
    userCount = len(userProfiles)
    itemCount = len(itemProfiles)
    vArray = pd.DataFrame({"itemID" : itemProfiles.keys(), "cluster" : np.random.randint(0, l-1, len(itemProfiles), int)})
    uArray = pd.DataFrame({"userID" : userProfiles.keys(), "cluster" : np.random.randint(0, k - 1, len(userProfiles), int)})
    bArray = pd.DataFrame(np.zeros([k, l]))

    print strftime("%Y-%m-%d %H:%M:%S", gmtime())

    p = Pool(20)
    indexes = [(i, j) for i in range(0, k) for j in range(0,l)]
    results = p.map(calculateB, indexes)
    for (i, j, result) in results:
        bArray.at[i, j] = result
    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
    t = 1


    print ""
    return

def calculateB((i, j)):
    clusterUsers = (uArray[uArray.cluster == i]).loc[:,["userID"]]
    itemsIDs = (vArray[vArray.cluster == j]).loc[:,["itemID"]]
    clusterUsers = pd.merge(clusterUsers, dfUserProfiles, how='inner')
    sum = 0
    ratingsCount = 0
    for index, row in clusterUsers.iterrows():# TODO can be improved
        ratings = pd.DataFrame({"itemID" : row["items"], "ratings" : row["ratings"]})
        ratings = pd.merge(itemsIDs, ratings, how='inner', on='itemID')
        sum += ratings["ratings"].sum()
        ratingsCount += len(ratings)
    if ratingsCount > 0:
        result = sum / ratingsCount
    else:
        result = globalSystemRatingsAverage
    print i,j
    return (i, j, result)


def extractcluster(clusterMap, clusterID):
    clusterObjects = []
    for i in range(0, len(clusterMap[0])):
        if clusterMap[1, i] == clusterID:
            clusterObjects.append(clusterMap[0, i])

    return clusterObjects


extractCB("ratings.csv")