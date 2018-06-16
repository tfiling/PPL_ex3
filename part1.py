import numpy as np
import sys


def extractProfiles(inputFile, userProfileFilePath, itemProfileFilePath):
    ratings = np.genfromtxt(inputFile, delimiter=',', dtype=[int, int, float, long])[1:]
    userProfiles = parseUserProfiles(ratings)
    itemProfiles = parseItemProfiles(ratings)
    saveProfiles(userProfiles, itemProfiles, userProfileFilePath, itemProfileFilePath)




argv = sys.argv
if len(argv) >= 5 and argv[1] == 'ExtractProfiles':
    inputFile = argv[2]
    userProfileFilePath = argv[3]
    itemProfileFilePath = argv[4]
    extractProfiles(inputFile, userProfileFilePath, itemProfileFilePath)
else:
    print 'Invalid arguments, should be in the form: ' \
          'ExtractProfiles [the rating input file] [user profile csv output directory] [item profile csv output directory].'
