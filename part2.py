import numpy as np
import pandas as pd
import os
from part1 import parseProfiles
from time import gmtime, strftime


def extractCB(ratingsFilePath, k = 20, t = 10, epsilon = 0.01, usersClusteringPath = "U.csv", itemsClusteringPath = "V.csv",  codebookPath = "B.csv"):
    l = k
    np.random.seed(10)

    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
    ratings = np.genfromtxt(ratingsFilePath, delimiter=',', dtype=[int, int, float, long])[1:]  # ignore table's title
    userProfiles, itemProfiles = parseProfiles(ratings)
    dfUserProfiles = pd.DataFrame.from_dict({i: userProfiles[i] for i in userProfiles.keys() }, orient='index', columns=['userID', 'items', 'ratings'])
    dfItemProfiles = pd.DataFra4me.from_dict({i: itemProfiles[i] for i in itemProfiles.keys() }, orient='index', columns=['itemID', 'users', 'ratings'])
    userCount = len(userProfiles)
    itemCount = len(itemProfiles)
    vArray = pd.DataFrame({"itemID" : itemProfiles.keys(), "cluster" : np.random.randint(0, l-1, len(itemProfiles), int)})
    uArray = pd.DataFrame({"userID" : userProfiles.keys(), "cluster" : np.random.randint(0, k - 1, len(userProfiles), int)})
    bArray = pd.DataFrame(np.zeros([k, l]))
    for i in range(0, k):
        for j in range(0, l):
            print i, j
            print strftime("%Y-%m-%d %H:%M:%S", gmtime())
            clusterUsers = (uArray[uArray.cluster == i]).loc[:,["userID"]]
            itemsIDs = (vArray[vArray.cluster == j]).loc[:,["itemID"]]
            clusterUsers = pd.merge(clusterUsers, dfUserProfiles, how='inner')
            sum = 0
            print strftime("%Y-%m-%d %H:%M:%S", gmtime())
            for index, row in clusterUsers.iterrows():
                ratings = pd.DataFrame({"itemID" : row["items"], "ratings" : row["ratings"]})
                ratings = pd.merge(itemsIDs, ratings, how='inner', on='itemID')
                sum += ratings["ratings"].sum()
            print strftime("%Y-%m-%d %H:%M:%S", gmtime())
            bArray.at[i, j] = sum
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