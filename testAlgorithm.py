from math import *
from KNN import *
from NaiveBayes import *


def testThis(data1):
    # So testing the accuracy of these with different k values
    dataLength = len(data1)
    partition = round(dataLength * 9 / 10)

    testingAccuracyData = data1[0:partition]
    testingAccuracyTesting = data1[partition + 1 :]

    testResults = []

    # remove label from testingAccuracyTesting
    actualLabels = []
    for lineNumber in range(len(testingAccuracyTesting)):
        actualLabels.append(
            testingAccuracyTesting[lineNumber][
                len(testingAccuracyTesting[lineNumber]) - 1
            ]
        )
        testingAccuracyTesting[lineNumber] = testingAccuracyTesting[lineNumber][:-1]

    for k in range(1, 9, 2):
        results = KNN2(testingAccuracyData, testingAccuracyTesting, k)
        # compare
        numberCorrect = 0
        for x in range(len(actualLabels)):
            if actualLabels[x] == results[x]:
                numberCorrect += 1

        testResults.append(
            {
                "Accuracy": numberCorrect / len(actualLabels),
                "K Value": k,
                "Actual labels": actualLabels,
                "Predicted labels": results,
            }
        )

    return testResults


def testNaiveBayes(data1):
    # So testing the accuracy of these with different k values
    dataLength = len(data1)
    partition = round(dataLength * 9 / 10)

    testingAccuracyData = data1[0:partition]
    testingAccuracyTesting = data1[partition + 1 :]

    testResults = []

    # remove label from testingAccuracyTesting
    actualLabels = []
    for lineNumber in range(len(testingAccuracyTesting)):
        actualLabels.append(
            testingAccuracyTesting[lineNumber][
                len(testingAccuracyTesting[lineNumber]) - 1
            ]
        )
        testingAccuracyTesting[lineNumber] = testingAccuracyTesting[lineNumber][:-1]

    results = NaiveBayes(testingAccuracyData, testingAccuracyTesting)
    # compare
    numberCorrect = 0
    for x in range(len(actualLabels)):
        if actualLabels[x] == results[x]:
            numberCorrect += 1

    testResults.append(
        {
            "Accuracy": numberCorrect / len(actualLabels),
            "Actual labels": actualLabels,
            "Predicted labels": results,
        }
    )

    return testResults
