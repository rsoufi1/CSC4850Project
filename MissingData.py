from KNN import KNN3


def FillMissingData(data, K):
    # Fill missing data
    # The data will need to be reformated so we can put it in KNN algorithm
    tempData = data
    numRows = len(data)
    if numRows == 0:
        return
    numColumns = len(data[numRows - 1])

    # This is a progress tracker - delete later
    numMissing = 0
    for column in range(numColumns):
        for row in range(numRows):
            if tempData[row][column] == "1.00000000000000e+99":
                numMissing += 1

    numFinished = 0
    for column in range(numColumns):
        missingDataRow = []
        notMisisngDataRow = []
        # get all the rows where column column has a missing value
        for row in range(numRows):
            if tempData[row][column] == "1.00000000000000e+99":
                missingDataRow.append(tempData[row])
                numFinished += 1
                print("Progress: " + str(numFinished / numMissing))
            else:
                notMisisngDataRow.append(tempData[row])

        # for all the rows missing data, do KNN and fill the gaps
        for missingDataIndex in range(len(missingDataRow)):
            predictionForMissingData = KNN3(
                notMisisngDataRow, [missingDataRow[missingDataIndex]], K, column
            )
            # print(predictionForMissingData)
            missingDataRow[missingDataIndex][column] = predictionForMissingData[0]

        # put both rows back together
        tempData.clear()
        tempData.extend(missingDataRow)
        tempData.extend(notMisisngDataRow)

    print("Done")
    return tempData
