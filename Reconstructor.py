import math
import numpy as np
import statistics as stat

def dataCleaner(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList):
    #Run data through the z-score filter
    zScoreCleanedList = (outlierFinder(scmList, 0.8))[0]
    #Remove leftover rows
    barRemovedList = leftoverBarRemover(zScoreCleanedList)
    #remove data outside fce and flh boundaries
    insideBoundsList = purgeDataOutsideBounds(barRemovedList, parameterMatrix, scmList, efiList)

    cleanedData = insideBoundsList

    return cleanedData

def outlierFinder(data,threshold):
    scmValues = np.array([row[1:] for row in data]) #cut off unneeded frequency values from each row
    zScores = []
    whistlerCoords = []
    outliers = np.full_like(scmValues, np.nan)            #make another 2d array filled with NaN values
    numRows, numCols = scmValues.shape                    #find number of rows and columns
    for i in range(len(scmValues)):                       #iterate through each frequency row by row
        '''
        For better statistics, each row will find a localized mean and standard deviation
        by taking the data from a group of 3 rows: The given row, the row below it, and the row above it.
        For the first row (row 0) we take the first 3 rows 0, 1, and 2.
        Likewise, for the last row we take the last 3 rows 28, 29, and 30
        '''

        start = max(0, i-1)
        end = min(numRows, i+2)
        localRows = scmValues[start:end, :] #automatically finds local rows
        mean = np.nanmean(localRows)
        std = np.nanstd(localRows)          #find the stats for the local groupings
        
        z = (scmValues[i] - mean) / std     #calculate z-scores
        zScores.append(z)

        outlierMask = z >= threshold        #create a list of bool values to mask whichever values are outliers

        outliers[i, outlierMask] = scmValues[i, outlierMask] #add outlier values to array
        outliers[i, ~outlierMask] = np.nan                   #non-outliers are replaced with NaNs
        whistlerCoords.extend([(i, j) for j, v in enumerate(outlierMask) if v]) #make a list of indices where there were outliers

    return outliers.tolist(), whistlerCoords, zScores

def purgeDataOutsideBounds(barRemovedList, parameterMatrix, scmList, efiList):
    freqs = np.array([row[0] for row in scmList])
    scmValues = np.array([row[1:] for row in scmList])
    barRemovedArray = np.array(barRemovedList)

    fce_values, flh_values, fpe_values = frequencyInterpolation(parameterMatrix, len(scmValues[0])) #Interpolate Frequencies

    mask = (flh_values <= freqs[:, None]) & (fce_values >= freqs[:, None])
    insideBoundsArray = np.where(mask, barRemovedArray, np.nan)

    return insideBoundsArray.tolist()

def leftoverBarRemover(zScoreCleanedList):
    arr = np.array(zScoreCleanedList)
    percentReal = np.sum(~np.isnan(arr), axis = 1) / arr.shape[1] * 100 #array where each element is the percent of non-NaN values in a row of "arr"
    mask = percentReal < 10
    arr[~mask] = np.nan         #set all to NaN if not <10%
    return arr.tolist()

def frequencyInterpolation(parameterMatrix,length):
    fTime = [row[0] for row in parameterMatrix]
    baseline = fTime[0]
    fce = [row[1] for row in parameterMatrix]
    flh = [row[2] for row in parameterMatrix]
    fpe = [row[3] for row in parameterMatrix]
    
    j_values = np.linspace(baseline, baseline + length - 1, length)
    
    fce_values = np.interp(j_values, fTime, fce)
    flh_values = np.interp(j_values, fTime, flh)
    fpe_values = np.interp(j_values, fTime, fpe)

    return fce_values, flh_values, fpe_values

def bCalc(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList):  
    recalculatedB = []
    N, whistlerData, channels = nCalc(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList)
    for i in range(len(whistlerData)):  #iterate through each row
        row = []
        for j in range(len(whistlerData[i])):   #iterate through each data point
            foo = (whistlerData[i][j])*(pow((N[i][j]),2))*(100/9)               ### !!! RECONSTRUCTOR FUNCTION !!! ###
            row.append(foo)
        recalculatedB.append(row)
    return recalculatedB

def nCalc(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList):
    N = []
    channels = [row[0] for row in scmList]
    values = [row[1:] for row in scmList]

    whistlerData = efiFilter(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList)
    
    fce_values, flh_values, fpe_values = frequencyInterpolation(parameterMatrix, len(values[0]))
    
    for f in channels:
        n_values = (fpe_values / f) * np.power(np.abs((fce_values / f) - 1), -0.5)
        N.append(n_values)
    
    return N, whistlerData, channels

def efiFilter(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList):
    efiValues = np.array([row[1:] for row in efiList])
    scmValues = np.array(dataCleaner(parameterMatrix, scmTimeList, scmList, efiTimeList, efiList))

    efiFiltered = np.where(~np.isnan(scmValues), efiValues, np.nan)

    return efiFiltered.tolist()