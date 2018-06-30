import sys
import numpy as np
import pandas as pd
import os
from part1 import parseProfiles
from time import gmtime, strftime
from multiprocessing import Pool
import math
import ast

from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

MAX_THREAD_POOL = 6
V_CACHE_FILE_PATH = "v.csv"
U_CACHE_FILE_PATH = "u.csv"
B_CACHE_FILE_PATH = "b.csv"
USERS_CACHE_FILE_PATH = "USERS_CACHE_FILE.csv"


def extractCB(ratingsFilePath, _k = 20, maxSteps = 10, epsilon = 0.01, usersClusteringPath = U_CACHE_FILE_PATH, itemsClusteringPath = V_CACHE_FILE_PATH,  codebookPath = B_CACHE_FILE_PATH):
    # IMPORTANT NOTE
    # To be able parallel comutation in my implementation I had to use multiprocessing.Pool
    # this library creates additional processes that can use additional CPU cores.
    # using thread will limit me to one process, therefor I had to use multiple python process as Pool does
    # due to a bug in python 2.7 you can run only global functions in child process
    # which forced me to implement to define the parallel functions in global scope and therefor make the below variables to be global
    # (did not find a proper way implementing parallel calculation inside a class since it was suffering the same issue)
    global dfUserProfiles, dfItemProfiles, vArray, uArray, bArray, \
        globalSystemRatingsAverage, vArrayDict, uArrayDict, \
        userProfiles, itemProfiles, globalRatingsCount, k, l, \
        testItemProfiles, testGlobalRatingCount
    k = _k
    l = k

    print strftime("start - %Y-%m-%d %H:%M:%S", gmtime())
    trainRatings, testRatings, allRatings = splitDataset(ratingsFilePath)
    testItemProfiles, testGlobalRatingCount = extractRequiredTestDataStructures(testRatings)

    # ratings is a list of tuples where the 3rd(f2) element is the rating - extract a series of the ratings and get the mean(average) of that series
    globalSystemRatingsAverage = allRatings['rating'].mean()

    userProfiles, itemProfiles = parseProfiles(trainRatings.values)
    dfUserProfiles = pd.DataFrame.from_dict({i: userProfiles[i] for i in userProfiles.keys() }, orient='index', columns=['userID', 'items', 'ratings'])
    dfItemProfiles = pd.DataFrame.from_dict({i: itemProfiles[i] for i in itemProfiles.keys() }, orient='index', columns=['itemID', 'users', 'ratings'])
    globalRatingsCount = sum(map(len, dfItemProfiles["users"]))
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

    vArray.to_csv(V_CACHE_FILE_PATH)
    uArray.to_csv(U_CACHE_FILE_PATH)
    bArray.to_csv(B_CACHE_FILE_PATH)
    vArray.to_csv(itemsClusteringPath)
    uArray.to_csv(usersClusteringPath)
    bArray.to_csv(codebookPath)
    print strftime("trainig completed - %Y-%m-%d %H:%M:%S", gmtime())
    return



def extractRequiredTestDataStructures(testRatings):
    testUserProfiles, testItemProfiles = parseProfiles(testRatings.values)
    dfItemProfiles = pd.DataFrame.from_dict({i: testItemProfiles[i] for i in testItemProfiles.keys() }, orient='index', columns=['itemID', 'users', 'ratings'])
    testGlobalRatingCount = sum(map(len, dfItemProfiles["users"]))
    return testItemProfiles, testGlobalRatingCount

def splitDataset(ratingsFilePath):
    dataset = pd.read_csv(ratingsFilePath)
    # randomly select 80% of the dataset to be train dataset and the rest to be test dataset
    df = pd.DataFrame(np.random.randn(len(dataset), 2))
    mask = np.random.rand(len(df)) < 0.8
    train = dataset[mask]
    test = dataset[~mask]
    return train, test, dataset

###################################################################
## calcualteB
###################################################################


def calculateB((i, j)):
    clusterUsers = (uArray[uArray.cluster == i]).loc[:,["userID"]]
    itemsIDs = (vArray[vArray.cluster == j]).loc[:,["itemID"]]
    clusterUsers = pd.merge(clusterUsers, dfUserProfiles, how='inner')
    sum = 0
    ratingsCount = 0
    for index, row in clusterUsers.iterrows(): # gal can be improved - seems like a bottleneck
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
    p = Pool(MAX_THREAD_POOL)
    itemIDs = testItemProfiles.keys()
    errorSum = sum(p.map(getErrorSum, itemIDs))
    p.close()
    result = math.sqrt(errorSum / testGlobalRatingCount)
    return result

def getErrorSum(itemID):
    errorSum = 0
    if testItemProfiles.has_key(itemID) and vArrayDict.has_key(itemID):
        users = testItemProfiles[itemID][1]
        userRankings = testItemProfiles[itemID][2]
        itemCluster = vArrayDict[itemID]
        for i in range(0, len(users)):
            userID = users[i]
            rating = userRankings[i]
            userCluster = uArrayDict[userID]
            bRating = bArray.at[userCluster, itemCluster]
            result = (rating - bRating) ** 2
            errorSum += result

    return errorSum

###################################################################
## cacheLookup
###################################################################

def cacheLookup():
    found = os.path.isfile(U_CACHE_FILE_PATH) and os.path.isfile(V_CACHE_FILE_PATH) and os.path.isfile(B_CACHE_FILE_PATH)
    if found:
        global uArray, vArray, bArray, usersCache
        uArray = pd.read_csv(U_CACHE_FILE_PATH)
        vArray = pd.read_csv(V_CACHE_FILE_PATH)
        bArray = pd.read_csv(B_CACHE_FILE_PATH)
        usersCache = pd.read_csv(USERS_CACHE_FILE_PATH)
    return found


###################################################################
## parseArguments
###################################################################

def parseArguments(args):
    if len(args) < 6:
        raise Exception('Expecting at least 5 arguments: '
                        'ExtractCB '
                        '[the rating input file] '
                        '[U output directory as csv file] '
                        '[V output directory as csv file] '
                        '[B output directory as csv file] that dont have default value')

    if len(args) < 9:
        print "found less arguments than full argument signature:\n" \
              "Part2.py " \
              "ExtractCB " \
              "[the rating input file] " \
              "[K size] " \
              "[T size] "\
              "[epsilon size] " \
              "[U output directory as csv file] " \
              "[V output directory as csv file] " \
              "[B output directory as csv file]\n"\
              "according to a faculty answer in the forum, " \
              "we expect the following signature where missing arguments take the default values:\n" \
              "Part2.py " \
              "ExtractCB " \
              "rating_file=[the rating input file] " \
              "k=[K size] " \
              "t=[T size] " \
              "epsilon=[epsilon size] " \
              "u_out=[U output directory as csv file] " \
              "v_out=[V output directory as csv file] " \
              "b_out=[B output directory as csv file]\n\n"

        print "required arguments with no default value:\n"\
                "ExtractCB "\
                "[the rating input file] "\
                "[U output directory as csv file] "\
                "[V output directory as csv file] "\
                "[B output directory as csv file] that dont have default value"

        rating_file = None
        u_out = None
        v_out = None
        b_out = None
        k = 20
        l = 20
        t = 10
        epsilon = 0.01
        splitted = map(lambda arg: arg.split('='), args)
        splitted = filter(lambda x: len(x) > 1, splitted)
        for arg in splitted:
            if arg[0].lower() == "rating_file":
                rating_file = arg[1]
            if arg[0].lower() == "u_out":
                u_out = arg[1]
            if arg[0].lower() == "v_out":
                v_out = arg[1]
            if arg[0].lower() == "b_out":
                b_out = arg[1]
            if arg[0].lower() == "k":
                k = int(arg[1])
                l = int(arg[1])
            elif arg[0].lower() == "t":
                t = int(arg[1])
            elif arg[0].lower() == "epsilon":
                epsilon = float(arg[1])

        if rating_file is None:
            raise Exception("rating_file=<value> argument is missing, no default value was assigned for the argument!")
        if u_out is None:
            raise Exception("u_out=<value> argument is missing, no default value was assigned for the argument!")
        if v_out is None:
            raise Exception("v_out=<value> argument is missing, no default value was assigned for the argument!")
        if b_out is None:
            raise Exception("b_out=<value> argument is missing, no default value was assigned for the argument!")

    # No args are missing and will appear in order
    else:
        rating_file = args[2]
        k = l = int(args[3])
        t = int(args[4])
        epsilon = float(args[5])
        u_file = args[6]
        v_file = args[7]
        b_file = args[8]
    return rating_file, k, t, epsilon, u_file, v_file, b_file


###################################################################
## handle post request
###################################################################

@app.route('/', methods=['POST'])
def recommend():
    # parse params:
    userID = request.values.get('userid')
    if userID is None:
        print("userID was not provided in the request")
        return jsonify([])
    N = request.values.get('n')
    if N is None:
        print("userID was not provided in the request")
        return jsonify([])
    userID = int(userID)
    N = int(N)
    recommendationJson = jsonify(getRecommendation(userID, N))
    print "response contents: ", recommendationJson
    return recommendationJson


def getRecommendation(userID, N):
    recommendationList = []
    cacheLookup()
    if len(usersCache[usersCache["userID"] == userID]) == 0:
        print("requested userID could not be found.\n "
              "all ratings by this user were selected to be in the test dataset and therefore there is not data regarding this user!\n"
              "plese note that each row in the input files had ~80% to be in the train dataset (see splitDataset function)")
    else:
        user = usersCache[usersCache["userID"] == 1][["userID", "items", "ratings"]].iloc[0].to_dict()
        userCluster = uArray[uArray["userID"] == userID]["cluster"].iloc[0]
        itemsClusterRatings = bArray.loc[userCluster][1:].sort_values(ascending=False)
        alreadyRecommended = user["items"] # prevent recommendation of items the user laready rated
        if type(alreadyRecommended) == str:
            alreadyRecommended = ast.literal_eval(alreadyRecommended) # parse the string representation of the the list into an actual list

        for clusterID in itemsClusterRatings.index:
            clusterItems = vArray[vArray["cluster"] == int(clusterID)]["itemID"].values
            items = list(set(clusterItems) - set(alreadyRecommended))
            alreadyRecommended.extend(items)
            recommendationList.extend(items)
            if len(recommendationList) > N:
                break

        recommendationList = recommendationList[:N] # trim the resulted list
    return recommendationList


###################################################################
## main
###################################################################

if __name__ == '__main__':
    print "IMPORTANT! - please note I expect you to:\n" \
          "1 - run part2 with ExtractCB first, the clusters will be calculated and saved\n" \
          "2 - run part2 without arguments which will load the clusters and provide recommendation\n\n" \
          "according to the assumptions provided in the faculty answers in the forum\n"

    args = sys.argv
    if len(args) > 1 and args[1] == "ExtractCB":
        rating_file, k, t, epsilon, u_file, v_file, b_file = parseArguments(args)
        extractCB(rating_file, k, t, epsilon, u_file, v_file, b_file)
    else:
        app.run(debug=True)
        print "running regular webserver (without training), the clusters will be loaded from previous runs!\n" \
              "please be sure you applied ExtractCB first"
        found = cacheLookup()
        if not found:
            print "no cached data found! Plaese run ExtractCB method and then run part 2 again\n" \
                  "I assumed you will run ExtractCB before running the server handling the post requests"

            exit(1)



