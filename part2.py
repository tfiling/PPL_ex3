import numpy as np
import pandas as pd
from part1 import parseProfiles
from time import gmtime, strftime
from multiprocessing import Pool
import math

MAX_THREAD_POOL = 6


def extractCB(ratingsFilePath, k = 4, maxSteps = 10, epsilon = 0.01, usersClusteringPath = "U.csv", itemsClusteringPath = "V.csv",  codebookPath = "B.csv"):
    global dfUserProfiles, dfItemProfiles, vArray, uArray, bArray, globalSystemRatingsAverage, vArrayDict, uArrayDict, userProfiles, itemProfiles, globalRatingsCount
    l = k
    np.random.seed(10)#TODO remove seed


    print strftime("start - %Y-%m-%d %H:%M:%S", gmtime())
    ratings = np.genfromtxt(ratingsFilePath, delimiter=',', dtype=[int, int, float, long])[1:]  # ignore table's title

    # ratings is a list of tuples where the 3rd(f2) element is the rating - extract a series of the ratings and get the mean(average) of that series
    globalSystemRatingsAverage = pd.DataFrame.from_records(ratings, exclude=['f0', 'f1', 'f3'])['f2'].mean()

    userProfiles, itemProfiles = parseProfiles(ratings)
    dfUserProfiles = pd.DataFrame.from_dict({i: userProfiles[i] for i in userProfiles.keys() }, orient='index', columns=['userID', 'items', 'ratings'])
    dfItemProfiles = pd.DataFrame.from_dict({i: itemProfiles[i] for i in itemProfiles.keys() }, orient='index', columns=['itemID', 'users', 'ratings'])
    globalRatingsCount = sum(map(len, dfItemProfiles["users"]))
    userCount = len(userProfiles)#TODO remove
    itemCount = len(itemProfiles)
    vArray = pd.DataFrame({"itemID" : itemProfiles.keys(), "cluster" : np.random.randint(0, l-1, len(itemProfiles), int)})
    vArrayDict = vArray.set_index("itemID")["cluster"].to_dict()
    uArray = pd.DataFrame({"userID" : userProfiles.keys(), "cluster" : np.random.randint(0, k - 1, len(userProfiles), int)})
    uArrayDict = uArray.set_index("userID")["cluster"].to_dict()
    bArray = pd.DataFrame(np.zeros([k, l]))

    print strftime("data loaded - %Y-%m-%d %H:%M:%S", gmtime())

    # calculate codebook
    p = Pool(MAX_THREAD_POOL)
    indexes = [(i, j) for i in range(0, k) for j in range(0,l)]
    results = p.map(calculateB, indexes)
    p.close()
    for (i, j, result) in results:
        bArray.at[i, j] = result
    print strftime("B calculated %Y-%m-%d %H:%M:%S", gmtime())
    t = 0
    currentRMSEDiff = 5  # rate is in range [0,5] this is the maximum theoretical value
    currentRMSE = 5
    while t < maxSteps and currentRMSEDiff > epsilon:
        # improve user clusters - steps 8-9
        p = Pool(MAX_THREAD_POOL)
        userIDs = userProfiles.keys()
        newClusters = p.map(getBestUserCluster, userIDs)
        p.close()
        print strftime("user cluster improved - %Y-%m-%d %H:%M:%S", gmtime())
        uArrayDict = dict(newClusters)
        uArray = pd.DataFrame({"userID" : uArrayDict.keys(), "cluster" : uArrayDict.values()})

        # calculate codebook - step 10
        p = Pool(MAX_THREAD_POOL)
        indexes = [(i, j) for i in range(0, k) for j in range(0, l)]
        results = p.map(calculateB, indexes)
        p.close()
        print strftime("B1 calculated - %Y-%m-%d %H:%M:%S", gmtime())
        for (i, j, result) in results:
            bArray.at[i, j] = result

        # update item clusters - steps 11-12
        p = Pool(MAX_THREAD_POOL)
        itemIDs = itemProfiles.keys()
        newClusters = p.map(getBestItemCluster, itemIDs)
        p.close()
        print strftime("item clusters improved - %Y-%m-%d %H:%M:%S", gmtime())
        vArrayDict = dict(newClusters)
        vArray = pd.DataFrame({"itemID" : uArrayDict.keys(), "cluster" : uArrayDict.values()})

        # calculate codebook - step 13
        p = Pool(MAX_THREAD_POOL)
        indexes = [(i, j) for i in range(0, k) for j in range(0, l)]
        results = p.map(calculateB, indexes)
        p.close()
        print strftime("B1 calculated - %Y-%m-%d %H:%M:%S", gmtime())
        for (i, j, result) in results:
            bArray.at[i, j] = result

        newRSME = calculateRSME()
        currentRMSEDiff = math.fabs(newRSME - currentRMSE)
        print strftime("new RSME calculated - %Y-%m-%d %H:%M:%S", gmtime())
        print "old RSME - {} new RSME - {} current diff - {}".format(currentRMSE, newRSME, currentRMSEDiff)
        currentRMSE = newRSME

    print strftime("trainig completed - %Y-%m-%d %H:%M:%S", gmtime())
    return

###################################################################
## calcualteB
###################################################################


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
    return (i, j, result)



###################################################################
## getBestUserCluster
###################################################################

def getBestUserCluster(userID):
    k = 4  # TODO get from upper scope
    items = userProfiles[userID][1]
    itemRankings = userProfiles[userID][2]
    clusterErrors = np.zeros(k)
    for i in range(0, len(items)):
        itemID = items[i]
        rating = itemRankings[i]
        itemCluster = vArrayDict[itemID]
        for cluster in range(0, k):
            bRating = bArray.at[cluster, itemCluster]
            result = (rating - bRating) ** 2
            clusterErrors[cluster] += result

    bestCluster = clusterErrors.argmin()
    return (userID, bestCluster)



###################################################################
## getBestItemCluster
###################################################################

def getBestItemCluster(itemID):
    l = 4 #TODO get from upper scope

    users = itemProfiles[itemID][1]
    userRankings = itemProfiles[itemID][2]
    clusterErrors = np.zeros(l)
    for i in range(0, len(users)):
        userID = users[i]
        rating = userRankings[i]
        userCluster = uArrayDict[userID]
        for cluster in range(0, l):
            bRating = bArray.at[userCluster, cluster]
            result = (rating - bRating) ** 2
            clusterErrors[cluster] += result

    bestCluster = clusterErrors.argmin()
    return (itemID, bestCluster)


###################################################################
## calculateRSME
###################################################################

def calculateRSME():
    k = l = 4 #TODO get from global scope
    p = Pool(MAX_THREAD_POOL)
    itemIDs = itemProfiles.keys()
    errorSum = sum(p.map(getErrorSum, itemIDs))
    p.close()
    result = math.sqrt(errorSum / globalRatingsCount)
    return result

def getErrorSum(itemID):
    l = 4 #TODO get from upper scope

    users = itemProfiles[itemID][1]
    userRankings = itemProfiles[itemID][2]
    itemCluster = vArrayDict[itemID]
    errorSum = 0
    for i in range(0, len(users)):
        userID = users[i]
        rating = userRankings[i]
        userCluster = uArrayDict[userID]
        bRating = bArray.at[userCluster, itemCluster]
        result = (rating - bRating) ** 2
        errorSum += result

    return errorSum



extractCB("ratings.csv")