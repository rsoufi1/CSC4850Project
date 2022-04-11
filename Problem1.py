from unicodedata import name
from KNN import *
from MissingData import *
from testAlgorithm import *
import threading
from os.path import exists
from NaiveBayes import *


def FillTestData1(data, kValue, nameOfFile, results, index):
    print("Getting missing data k = " + str(kValue))
    missingDataFilled = FillMissingData(data, kValue)
    # store this in a file
    fileName = nameOfFile + "_K" + str(kValue) + ".txt"
    f = open(fileName, "w")
    for missingingDataRow in missingDataFilled:
        for item in missingingDataRow:
            f.write(item + "\t")
        f.write("\n")

    f.close()

    results[index] = data
    return missingDataFilled


def getTestData(fileName, results, index):
    # get test data
    testData = []
    with open("Data/" + fileName, "r") as testDataFile:
        testData = testDataFile.readlines()

    testDataArray = []
    for line in testData:
        temp = []
        if "," in line:
            temp = line.split(",")
        else:
            temp = line.split()

        for x in range(0, len(temp)):
            if "\n" in temp[x]:
                temp[x] = temp[x][:-1]

        testDataArray.append(temp)

    if index != -1:
        results[index] = testDataArray
    return testDataArray


def getTrainingData(nameOfTrainingDataFile, nameOfTrainingLabelFile, results, index):
    # Read the file
    trainDataLines = []
    with open("Data/" + nameOfTrainingDataFile, "r") as trainDataFile:
        trainDataLines = trainDataFile.readlines()

    labelLines = []
    with open("Data/" + nameOfTrainingLabelFile, "r") as labelsFile:
        labelLines = labelsFile.readlines()

    data = []
    for x in range(len(trainDataLines)):
        temp = trainDataLines[x].split()
        temp.append(labelLines[x][:-1])
        data.append(temp)

    results[index] = data
    return data


def getFilledTrainingData(nameOfFile, results, index):
    data = []
    with open(nameOfFile, "r") as testDataFile:
        data = testDataFile.readlines()

    dataArray = []
    for line in data:
        lineToAdd = line.split("\t")
        if lineToAdd[len(lineToAdd) - 1] == "\n":
            lineToAdd.remove("\n")
        dataArray.append(lineToAdd)

    if index != -1:
        results[index] = dataArray
    return dataArray


def testingData(results, i, kValue, fileName):
    # TESTING - see which k value is the best for these
    testResults = testThis(results[i])

    # Save the test results to a file
    testingFileName = "TestResults_K" + str(kValue) + ".txt"
    if exists(testingFileName):
        f = open(testingFileName, "a")
        f.write(
            "Test results for complete data file "
            + fileName
            + "Filled_K"
            + str(kValue)
            + ".txt\n"
        )
        for result in testResults:
            f.write(str(result) + "\n")

        f.write("\n\n\n\n\n")
        f.close()
    else:
        f = open(testingFileName, "w")
        f.write(
            "Test results for complete data file "
            + fileName
            + "Filled_K"
            + str(kValue)
            + ".txt\n"
        )
        for result in testResults:
            f.write(str(result) + "\n")

        f.write("\n\n\n\n\n")
        f.close()


def testingNBayes(results, i, kValue, fileName):
    # TESTING - see which k value is the best for these
    testResults = testNaiveBayes(results[i])

    # Save the test results to a file
    testingFileName = "TestResults_NaiveBayes" + str(kValue) + ".txt"
    if exists(testingFileName):
        f = open(testingFileName, "a")
        f.write(
            "Test results for complete data file "
            + fileName
            + "Filled_K"
            + str(kValue)
            + ".txt\n"
        )
        for result in testResults:
            f.write(str(result) + "\n")

        f.write("\n\n\n\n\n")
        f.close()
    else:
        f = open(testingFileName, "w")
        f.write(
            "Test results for complete data file "
            + fileName
            + "Filled_K"
            + str(kValue)
            + ".txt\n"
        )
        for result in testResults:
            f.write(str(result) + "\n")

        f.write("\n\n\n\n\n")
        f.close()


def writeSolutions(fileName, labels):
    f = open("Solutions/" + fileName, "w")
    for label in labels:
        f.write(str(label) + "\n")

    f.close()


def NB(input, test, results, index):
    result = NaiveBayes(input, test)
    results[index] = result
    return result


def get1():
    trainignData1 = getFilledTrainingData("TrainData1Filled_K5.txt", [], -1)
    testData1 = getTestData(testingDataFileName[0], [], -1)
    predictionLabels1 = KNN2(trainignData1, testData1, 5)
    writeSolutions("SoufiClassification1.txt", predictionLabels1)


def get2():
    trainignData2 = getFilledTrainingData("TrainData2Filled_K5.txt", [], -1)
    testData2 = getTestData(testingDataFileName[1], [], -1)
    predictionLabels2 = KNN2(trainignData2, testData2, 5)
    writeSolutions("SoufiClassification2.txt", predictionLabels2)


def get3():
    trainignData3 = getFilledTrainingData("TrainData3Filled_K5.txt", [], -1)
    testData3 = getTestData(testingDataFileName[2], [], -1)
    predictionLabels3 = NaiveBayes(trainignData3, testData3)
    writeSolutions("SoufiClassification3.txt", predictionLabels3)


def get4():
    trainignData4 = getFilledTrainingData("TrainData4Filled_K5.txt", [], -1)
    testData4 = getTestData(testingDataFileName[3], [], -1)
    predictionLabels4 = KNN2(trainignData4, testData4, 5)
    writeSolutions("SoufiClassification4.txt", predictionLabels4)


def get5():
    trainignData5 = getFilledTrainingData("TrainData5Filled_K7.txt", [], -1)
    testData5 = getTestData(testingDataFileName[4], [], -1)
    predictionLabels5 = KNN2(trainignData5, testData5, 7)
    writeSolutions("SoufiClassification5.txt", predictionLabels5)


# get the training data
trainingDataFileNames = [
    "TrainData1.txt",
    "TrainData2.txt",
    "TrainData3.txt",
    "TrainData4.txt",
    "TrainData5.txt",
]
testingDataFileName = [
    "TestData1.txt",
    "TestData2.txt",
    "TestData3.txt",
    "TestData4.txt",
    "TestData5.txt",
]
trainingLabelFileNames = [
    "TrainLabel1.txt",
    "TrainLabel2.txt",
    "TrainLabel3.txt",
    "TrainLabel4.txt",
    "TrainLabel5.txt",
]

trainingDataResults = [None] * 6
trainignDataFilledresults = [None] * 6

# get the training data and fill in any missing spots
for kValue in range(3, 9, 2):
    for index in range(0, len(trainingDataFileNames)):
        # check if the data is already there
        finishedFileName = (
            trainingDataFileNames[index][:-4] + "Filled" + "_K" + str(kValue) + ".txt"
        )
        if not exists(finishedFileName):
            # get training data
            t2 = threading.Thread(
                target=getTrainingData,
                args=(
                    trainingDataFileNames[index],
                    trainingLabelFileNames[index],
                    trainingDataResults,
                    index,
                ),
            )
            t2.start()

            t2.join()

            # fill missing parts of the training data
            t2_2 = threading.Thread(
                target=FillTestData1,
                args=(
                    trainingDataResults[index],
                    kValue,
                    trainingDataFileNames[index][:-4] + "Filled",
                    trainignDataFilledresults,
                    index,
                ),
            )
            t2_2.start()
            t2_2.join()
        # ONLY INCLUDE AS PART OF TESTING
        # else:
        # get the data from the file
        #    t2 = threading.Thread(
        #        target=getFilledTrainingData,
        #        args=(
        #            trainingDataFileNames[index][:-4]
        #            + "Filled_K"
        #            + str(kValue)
        #            + ".txt",
        #            trainignDataFilledresults,
        #            index,
        #        ),
        #    )
        #    t2.start()
        #    t2.join()

        # testing - K nearest neighbor
        # t3 = threading.Thread(
        #    target=testingData,
        #    args=(
        #        trainignDataFilledresults,
        #        index,
        #        kValue,
        #        trainingDataFileNames[index][:-4],
        #    ),
        # )
        # t3.start()
        # t3.join()

        # testing - Naive Bayes
        # t3 = threading.Thread(
        #    target=testingNBayes,
        #    args=(
        #        trainignDataFilledresults,
        #        index,
        #        kValue,
        #        trainingDataFileNames[index][:-4],
        #    ),
        # )
        # t3.start()
        # t3.join()


solutionsArray = [None] * 5
testingDataArray = [None] * 5
finalSolutions = [None] * 5

# solve problem 1
# DATA 1 - use filled data k = 5, then use k= 5 to solve
t = threading.Thread(target=get1, args=())
t.start()
t.join()

# DATA 2 - use filled data k = 5, then used k = 5 to solve
t = threading.Thread(target=get2, args=())
t.start()
t.join()

# DATA 3  - use filled data k = 5, use NB to solve
t = threading.Thread(target=get3, args=())
t.start()
t.join()

# DATA 4 - use filled data k = 5, then used k - 5 to solve
t = threading.Thread(target=get4, args=())
t.start()
t.join()

# DATA 5 - use filled data k = 7, then use k = 7 to solve
t = threading.Thread(target=get5, args=())
t.start()
t.join()

print("Done!!!!")
