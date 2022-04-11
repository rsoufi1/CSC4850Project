# So we need to be able to find the probabilites of each attribute in the data set


def NaiveBayes(data, testingData):
    # get number of rows
    numRows = len(data)
    numColumns = 0
    if numRows != 0:
        numColumns = len(data[0])
    else:
        return

    # dictionary to store how often each attribute appears
    # format:
    # {
    #  c1: {
    #    attribute1Name:
    #      {
    #      class: count
    #      },
    #    attribute2Name:
    #      {
    #      class: count
    #      }
    #    },
    #  c2: {},
    #  c3: {}
    # }

    attributeCount = {}
    classCount = {}
    for rowNumber in range(0, numRows):
        classType = data[rowNumber][numColumns - 1]
        # first, increase the class count
        if classType in classCount:
            classCount[classType] = classCount[classType] + 1
        else:
            classCount[classType] = 1

        # then, increase the attribute count
        for columnNumber in range(0, numColumns - 1):
            columnName = "C" + str(columnNumber)
            attribute = data[rowNumber][columnNumber]
            # first, check if the column is already in the dictionary
            if columnName in attributeCount:
                # then, check if that specific attribute is in the dictionary
                if attribute in attributeCount[columnName]:
                    # then, check if that specific class is part of that attribute
                    if classType in attributeCount[columnName][attribute]:
                        # if it does, incremenet the count by 1
                        attributeCount[columnName][attribute][classType] = (
                            attributeCount[columnName][attribute][classType] + 1
                        )
                    else:  # otherwise, add the class
                        attributeCount[columnName][attribute][classType] = 1
                else:  # if not, add it
                    attributeCount[columnName][attribute] = {classType: 1}
            else:  # if it is not, add it
                attributeCount[columnName] = {attribute: {classType: 1}}

    # predict for each row in testing data
    predictionLabels = []
    for row in testingData:
        maxValue = -1
        predictionClass = 0
        for c in classCount:
            count = classCount[c]
            pC = count / numRows

            # go through all the attributes and determine the probability for this class
            for columnNumber in range(0, len(row)):
                columnName = "C" + str(columnNumber)
                value = 0
                # if this attribute existed in the training data, update the value
                if (
                    row[columnNumber] in attributeCount[columnName]
                    and c in attributeCount[columnName][row[columnNumber]]
                ):
                    value = attributeCount[columnName][row[columnNumber]][c]
                pC = pC * (value / count)

            if pC > maxValue:
                maxValue = pC
                predictionClass = c

        predictionLabels.append(predictionClass)

    return predictionLabels
