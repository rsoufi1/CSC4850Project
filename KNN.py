from math import *


def KNN(trainingData, testData, kValue):
    # print("k nearist neighbor")
    predictionLabels = []

    # print("number of samples: ")
    # print(len(testData))
    for x in range(len(testData)):
        rowsWithShortestDistance = []
        for k in range(kValue):
            rowsWithShortestDistance.append([-1, float("inf")])
        for y in range(len(trainingData)):
            # calculate distance between this row and the training data
            distance = calculateDistance(trainingData[y], testData[x])
            # see if this distance is less than the other rows
            # figure out which row is the least
            minValue = float("inf")
            minValueIndex = -1
            infIsThere = False
            for t in range(len(rowsWithShortestDistance)):
                if rowsWithShortestDistance[t][1] == float("inf"):
                    minValueIndex = t
                    minValue = rowsWithShortestDistance[t][1]
                    infIsThere = True
                else:
                    if rowsWithShortestDistance[t][1] < minValue and not infIsThere:
                        minValueIndex = t
                        minValue = rowsWithShortestDistance[t][1]

            # see if distance is less than the min
            if distance < minValue:
                rowsWithShortestDistance[minValueIndex] = [
                    y,
                    distance,
                    trainingData[y][len(trainingData[y]) - 1],
                ]

        # print(rowsWithShortestDistance)
        # make predicition
        vote = []
        for row in rowsWithShortestDistance:
            # see if the label is already in vote
            isInVote = False
            isInVoteIndex = -1
            for v in range(len(vote)):
                if vote[v][0] == row[2]:
                    isInVoteIndex = v
                    isInVote = True

            if isInVote:
                vote[isInVoteIndex][1] += 1
            else:
                vote.append([row[2], 1])

        # see which label has the most votes
        mostVotedLabel = 0
        numVotes = -1

        for v in vote:
            if v[1] > numVotes:
                mostVotedLabel = v[0]
                numVotes = v[1]

        # print prediction
        predictionLabels.append(mostVotedLabel)
    return predictionLabels


def KNN2(trainingData, testData, kValue):
    # print("k nearist neighbor")
    predictionLabels = []

    for x in range(len(testData)):
        distances = []
        for y in range(len(trainingData)):
            # calculate distance between this row and the training data
            distance = calculateDistance(trainingData[y], testData[x])
            # see if this distance is less than the other rows
            distances.append([y, distance, trainingData[y][len(trainingData[y]) - 1]])

        distances.sort(key=lambda x: x[1])
        # get the top k
        rowsWithShortestDistance = []
        for k in range(kValue):
            rowsWithShortestDistance.append(
                [distances[k][0], distances[k][1], distances[k][2]]
            )
        # print(rowsWithShortestDistance)
        # make predicition
        vote = []
        for row in rowsWithShortestDistance:
            # see if the label is already in vote
            isInVote = False
            isInVoteIndex = -1
            for v in range(len(vote)):
                if vote[v][0] == row[2]:
                    isInVoteIndex = v
                    isInVote = True

            if isInVote:
                vote[isInVoteIndex][1] += 1
            else:
                vote.append([row[2], 1])

        # see which label has the most votes
        mostVotedLabel = 0
        numVotes = -1

        for v in vote:
            if v[1] > numVotes:
                mostVotedLabel = v[0]
                numVotes = v[1]

        # print prediction
        predictionLabels.append(mostVotedLabel)
    return predictionLabels


def calculateDistance(row1, row2):
    distance = 0.0
    for x in range(len(row2)):
        distance += (float(row1[x]) - float(row2[x])) ** 2
    return sqrt(distance)


def voteLabel(closestNeighbors):
    possibleLabels = []
    for n in closestNeighbors:
        xInPossible = False
        for x in possibleLabels:
            if possibleLabels[x][0] == n[2]:
                xInPossible = True
                possibleLabels[x][1] += 1
        if not xInPossible:
            possibleLabels.append([n[2], 1])

    possibleLabels.sort()

    return possibleLabels[len(possibleLabels) - 1]


def KNN3(trainingData, testData, kValue, labelColumn):
    # print("k nearist neighbor")
    predictionLabels = []

    # print("number of samples: ")
    # print(len(testData))
    for x in range(len(testData)):
        rowsWithShortestDistance = []
        for k in range(kValue):
            rowsWithShortestDistance.append([-1, float("inf"), 0])

        for y in range(len(trainingData)):
            # calculate distance between this row and the training data
            distance = calculateDistanceWithColumnToIgnore(
                trainingData[y], testData[x], labelColumn
            )
            # see if this distance is less than the other rows
            # figure out which row is the least
            minValue = float("inf")
            minValueIndex = -1
            infIsThere = False
            for t in range(len(rowsWithShortestDistance)):
                if rowsWithShortestDistance[t][1] == float("inf"):
                    minValueIndex = t
                    minValue = rowsWithShortestDistance[t][1]
                    infIsThere = True
                else:
                    if rowsWithShortestDistance[t][1] < minValue and not infIsThere:
                        minValueIndex = t
                        minValue = rowsWithShortestDistance[t][1]

            # see if distance is less than the min
            if distance < minValue:
                rowsWithShortestDistance[minValueIndex] = [
                    y,
                    distance,
                    trainingData[y][labelColumn],
                ]

        # print(rowsWithShortestDistance)
        # make predicition
        vote = []
        for row in rowsWithShortestDistance:
            # see if the label is already in vote
            isInVote = False
            isInVoteIndex = -1
            for v in range(len(vote)):
                if vote[v][0] == row[2]:
                    isInVoteIndex = v
                    isInVote = True

            if isInVote:
                vote[isInVoteIndex][1] += 1
            else:
                vote.append([row[2], 1])

        # see which label has the most votes
        mostVotedLabel = 0
        numVotes = -1

        for v in vote:
            if v[1] > numVotes:
                mostVotedLabel = v[0]
                numVotes = v[1]

        # print prediction
        predictionLabels.append(mostVotedLabel)
    return predictionLabels


def KNN3(trainingData, testData, kValue, labelColumn):
    # print("k nearist neighbor")
    predictionLabels = []

    # print("number of samples: ")
    # print(len(testData))
    for x in range(len(testData)):
        rowsWithShortestDistance = []
        for k in range(kValue):
            rowsWithShortestDistance.append([-1, float("inf"), 0])

        for y in range(len(trainingData)):
            # calculate distance between this row and the training data
            distance = calculateDistanceWithColumnToIgnore(
                trainingData[y], testData[x], labelColumn
            )
            # see if this distance is less than the other rows
            # figure out which row is the least
            minValue = float("inf")
            minValueIndex = -1
            infIsThere = False
            for t in range(len(rowsWithShortestDistance)):
                if rowsWithShortestDistance[t][1] == float("inf"):
                    minValueIndex = t
                    minValue = rowsWithShortestDistance[t][1]
                    infIsThere = True
                else:
                    if rowsWithShortestDistance[t][1] < minValue and not infIsThere:
                        minValueIndex = t
                        minValue = rowsWithShortestDistance[t][1]

            # see if distance is less than the min
            if distance < minValue:
                rowsWithShortestDistance[minValueIndex] = [
                    y,
                    distance,
                    trainingData[y][labelColumn],
                ]

        # print(rowsWithShortestDistance)
        # make predicition
        vote = []
        for row in rowsWithShortestDistance:
            # see if the label is already in vote
            isInVote = False
            isInVoteIndex = -1
            for v in range(len(vote)):
                if vote[v][0] == row[2]:
                    isInVoteIndex = v
                    isInVote = True

            if isInVote:
                vote[isInVoteIndex][1] += 1
            else:
                vote.append([row[2], 1])

        # see which label has the most votes
        mostVotedLabel = 0
        numVotes = -1

        for v in vote:
            if v[1] > numVotes:
                mostVotedLabel = v[0]
                numVotes = v[1]

        # print prediction
        predictionLabels.append(mostVotedLabel)
    return predictionLabels


def calculateDistanceWithColumnToIgnore(trainingDataRow, testingDataRow, column):
    distance = 0.0
    for x in range(len(testingDataRow)):
        if x != column:
            distance += (float(trainingDataRow[x]) - float(testingDataRow[x])) ** 2
    return sqrt(distance)
