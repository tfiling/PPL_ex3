import numpy as np
import sys


def extractProfiles(inputFile, userProfileFilePath, itemProfileFilePath):
    ratings = np.genfromtxt(inputFile, delimiter=',', dtype=[int, int, float, long])[1:]# ignore table's title
    userProfiles, itemProfiles = parseProfiles(ratings)
    saveProfiles(userProfiles, itemProfiles, userProfileFilePath, itemProfileFilePath)


def parseProfiles(ratings):
    userProfiles = {}
    itemProfiles = {}
    for row in ratings:
        (userID, itemID, itemRating, ts) = row
        if not userProfiles.has_key(userID):
            userProfiles[userID] = [userID, [], []]

        userProfiles[userID][1].append(itemID)
        userProfiles[userID][2].append(itemRating)

        if not itemProfiles.has_key(itemID):
            itemProfiles[itemID] = [itemID, [], []]

        itemProfiles[itemID][1].append(userID)
        itemProfiles[itemID][2].append(itemRating)

    return userProfiles, itemProfiles


def saveProfiles(userProfiles, itemProfiles, userProfileFilePath, itemProfileFilePath):
    saveDictionary(userProfileFilePath, userProfiles)
    saveDictionary(itemProfileFilePath, itemProfiles)


def saveDictionary(filePath, dict):
    file = open(filePath, "wb")
    rows = []
    for id in dict:
        rows.append(str(id) + ',' + str(dict[id][1]) + ',' + str(dict[id][2]) + '\n')
    stringContent = ''.join(rows)
    file.write(stringContent)
    file.close()


argv = sys.argv
if len(argv) >= 5 and argv[1] == 'ExtractProfiles':
    inputFile = argv[2]
    userProfileFilePath = argv[3]
    itemProfileFilePath = argv[4]
    extractProfiles(inputFile, userProfileFilePath, itemProfileFilePath)
else:
    print 'Invalid arguments, should be in the form: ' \
          'ExtractProfiles [the rating input file] [user profile csv output directory] [item profile csv output directory].'
