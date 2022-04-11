from unicodedata import name
from KNN import *
from MissingData import *
from testAlgorithm import *
import threading
from os.path import exists


def getTrainingData(nameOfTrainingDataFile, results, index):
    # Read the file
    trainDataLines = []
    with open("Data/" + nameOfTrainingDataFile, "r") as trainDataFile:
        trainDataLines = trainDataFile.readlines()

    data = []
    for x in range(len(trainDataLines)):
        temp = trainDataLines[x].split()
        data.append(temp)

    results[index] = data
    return data


def FillTestData1(data, kValue, nameOfFile, results, index):
    print("Getting missing data k = " + str(kValue))
    missingDataFilled = FillMissingData(data, kValue)
    # store this in a file
    fileName = "Solutions/" + nameOfFile + "_K" + str(kValue) + ".txt"
    f = open(fileName, "w")
    for missingingDataRow in missingDataFilled:
        for item in missingingDataRow:
            f.write(item + "\t")
        f.write("\n")

    f.close()

    results[index] = data
    return missingDataFilled


dataResults = [None] * 3
filledDataResults = [None] * 3

fileNames = ["MissingData1.txt", "MissingData2.txt", "MissingData3.txt"]


for x in range(0, len(fileNames)):
    # get training data
    t2 = threading.Thread(
        target=getTrainingData,
        args=(
            fileNames[x],
            dataResults,
            x,
        ),
    )
    t2.start()

    t2.join()

    # fill missing parts of the training data
    t2_2 = threading.Thread(
        target=FillTestData1,
        args=(
            dataResults[x],
            5,
            fileNames[x][:-4] + "Filled",
            filledDataResults,
            x,
        ),
    )
    t2_2.start()
    t2_2.join()
