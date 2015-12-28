#from pylab import plot,show
from numpy import array, random
from scipy.cluster.vq import kmeans,vq
from scipy.stats import pearsonr
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import copy as cp
import math as math

InputDirectoryPath = "C:\Users\David B\Documents\Data Analytics\Projects\Recommender Systems\\"
OutputDirectoryPath = "C:\Users\David B\Documents\Data Analytics\Projects\Recommender Systems\\"

IfileList = np.array([InputDirectoryPath+"VPS Dataset1.csv",InputDirectoryPath+"VPS Dataset2.csv",InputDirectoryPath+"VPS Dataset3.csv",InputDirectoryPath+"VPS Dataset4.csv"])
EfileList = np.array([OutputDirectoryPath+"VPS Results1.csv",OutputDirectoryPath+"VPS Results2.csv",OutputDirectoryPath+"VPS Results3.csv",OutputDirectoryPath+"VPS Results4.csv"])


# Loads a CSV files into an array.
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
# fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    columnHeaders = strlineToTuple(lines[0])
    
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    
    return columnHeaders, array(dataset,dtype=np.float32)

# Exports data in CSV format
def exportCSV(fileName, columnHeaders, table):
    fileHandler = open(fileName, "wt")
    fileHandler.write(strtupleToLine(columnHeaders))    
    for row in range(len(table)):
        fileHandler.write(tupleToLine(table[row]))    
    fileHandler.close()
    
# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    cleanLine = line.strip()                  # remove leading/trailing whitespace and newlines
    cleanLine = cleanLine.replace('"', '')# get rid of quotes
    lineList = cleanLine.split(",")          # separate the fields
    stringsToNumbers(lineList)            # convert strings into numbers
    lineTuple = array(lineList)
    return lineTuple

def tupleToLine(row):
    line = ','.join(map(str, row))
    line = line.replace('nan', '')
    line = line + '\n'   
    return line    

def strtupleToLine(row):
    line = ','.join(row)
    line = line + '\n'
    return line

# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def strlineToTuple(line):
    cleanLine = line.strip()                  # remove leading/trailing whitespace and newlines
    cleanLine = cleanLine.replace('"', '')# get rid of quotes
    lineList = cleanLine.split(",")          # separate the fields
    lineTuple = array(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])
        else:
            myList[i] = np.nan

# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if len(s) > 1 and s[0] == "-":
    s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True

# Loads a csv worksheet into an array
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
# fileName: name of the CSV file to be read
# Returns: a list of tuples
#def loadPartPrices(fileName, worksheet):
    #import the vehicle part excel file defined as a 2-Dimensional matrix rows as part numbers and columns as countrys prices
#    xlsx = pd.read_excel(fileName, sheetname=worksheet)

    #import the vehicle part excel file into a 2-Dimesional matrix array
#    return xlsx.as_matrix();

#Need to determine parts with common set of recommendations to help provide data for missing country prices
#Convert all nans to -1; And all non zeros to 1;
def commonizePartPrices(pricedata):
    commonRecommendations = cp.deepcopy(pricedata)
    where_are_NaNs = np.isnan(commonRecommendations)
    commonRecommendations[where_are_NaNs] = -1
    where_are_nonZeros = np.where(commonRecommendations > 0)
    commonRecommendations[where_are_nonZeros] = 1
    commonRecommendations[:,0] = 0       #ignore the part number column
    commonRecommendations[:,1] = 1       #ignore the part group column
    commonRecommendations[:,2] = 1       #ignore the part weight column
    commonRecommendations[:,3] = 1       #ignore the part ft3 column
    return commonRecommendations;

def findMatchingClusterSets(clstColumn,clstGroup):
    matchingClusterSets = np.zeros(len(centroids))
            
    for clstGroupMatch in range(len(centroids)):
        if(clstGroup <> clstGroupMatch):
           
            if((centroids[clstGroup][clstColumn] + centroids[clstGroupMatch][clstColumn]) == 0):
                #this other cluster has a pricing data in needed column...  check that there are twos in other columns
                if(checkfortwos(centroidsSet1 = centroids[clstGroup], centroidsSet2 = centroids[clstGroupMatch]) == True):
                    #save this pair for similiarity ratings
                    matchingClusterSets[clstGroupMatch] = 1
    
    return matchingClusterSets;

def checkfortwos(centroidsSet1, centroidsSet2):
   centroidSum = centroidsSet1+centroidsSet2
   if(sum(centroidSum==2) > 0):
        return True;
   return False;

def calculateSimiliarity(a,b):
    pcorr, p_value = pearsonr(a, b)
    return pcorr

def calculatePrediction(clstGroupPartrowSimPrices,clstMatchSetPartrowSimPrices, partpriceSimiliarity, clstMatchSetPartrowPrediction, startofpriceindex):
    #print "clstGroupPartrowSimPrices: " , clstGroupPartrowSimPrices[startofpriceindex:]
    #print "clstMatchSetPartrowSimPrices: " , clstMatchSetPartrowSimPrices[startofpriceindex:]
    #print "partpriceSimiliarity: " , partpriceSimiliarity
    #print "clstMatchSetPartrowPrediction: " , clstMatchSetPartrowPrediction

    if(np.isnan(partpriceSimiliarity) == False):
        clstGroupPartrowSimPricesmean = np.average(clstGroupPartrowSimPrices[startofpriceindex:])
        clstMatchSetPartrowSimPricesmean = np.average(np.append(clstMatchSetPartrowSimPrices[startofpriceindex:],[clstMatchSetPartrowPrediction]))
        predictionClstGroupPartRow = clstGroupPartrowSimPricesmean + (clstMatchSetPartrowPrediction - clstMatchSetPartrowSimPricesmean)
        return predictionClstGroupPartRow
    return float('nan')
    

def bestPrediction(partSimPred):
    #print "partSimPred:"
    #print partSimPred
    
    maxSimIndex = np.nanargmax(partSimPred, axis=0)
    #print "maxSimIndex: ", maxSimIndex
    bestPrice = partSimPred[maxSimIndex[2]][3]
    #print "bestPrice: ", bestPrice
    return bestPrice

#generate a prediction based on the most similiar row
def findBestmatchAndMakePricePrediciton(clstColumn, clstGroup, matchingClusterSets):
    #iterate through all of the parts in cluster group (clstGroup) for the missing part price in column (clstColumn)
    for clstGroupPartrow in range(len(groups[clstGroup])):
        #print "clstGroup: ", clstGroup, " clstGroupPartrow: ", clstGroupPartrow, " clstColumn: ", clstColumn
    
        partSimPred = []
        #print "clstGroupPartrow: ", clstGroupPartrow
        for mClsSets in range(len(matchingClusterSets)):
            if(matchingClusterSets[mClsSets] == 1):
                #find common columns that can be used to test pearson correlation
                #print "mClsSets: ", mClsSets
                commonColPearsonCorr = centroids[clstGroup] + centroids[mClsSets]
                clstGroupPartrowSimPrices = groups[clstGroup][clstGroupPartrow][commonColPearsonCorr==2]
                for clstMatchSetPartrow in range(len(groups[mClsSets])):
                    clstMatchSetPartrowSimPrices = groups[mClsSets][clstMatchSetPartrow][commonColPearsonCorr==2]
                    clstMatchSetPartrowPrediction = groups[mClsSets][clstMatchSetPartrow][clstColumn]
                    #print "clstGroupPartrowSimPrices: ", clstGroupPartrowSimPrices, " clstMatchSetPartrowSimPrices: ", clstMatchSetPartrowSimPrices
                    #calculate similiarity and prediction for this part
                    partpriceSimiliarity = calculateSimiliarity(clstGroupPartrowSimPrices,clstMatchSetPartrowSimPrices)
                    partpricePrediction = calculatePrediction(clstGroupPartrowSimPrices,clstMatchSetPartrowSimPrices, partpriceSimiliarity, clstMatchSetPartrowPrediction,3)
                    
                    if(np.isnan(partpriceSimiliarity) == False and np.isnan(partpricePrediction) == False):
                        partSimPred.append([groups[clstGroup][clstGroupPartrow][0], groups[mClsSets][clstMatchSetPartrow][0], partpriceSimiliarity, partpricePrediction])
        
        #select the matching part price with the highest simiarity
        if(len(partSimPred) > 0):
            groups[clstGroup][clstGroupPartrow][clstColumn] = bestPrediction(partSimPred)
            partpricedata[groups[clstGroup][clstGroupPartrow][0]][clstColumn] = groups[clstGroup][clstGroupPartrow][clstColumn]
    return;

def checkCentroids():
    #quick check that the centroids or either 0,1, or -1
    #print centroids
    for clst in range(len(centroids)):
        one=sum(1 for item in centroids[clst] if item==(1))
        negone=sum(1 for item in centroids[clst] if item==(-1))
        zero=sum(1 for item in centroids[clst] if item==(0))
        if((one+negone+zero) != len(centroids[clst])):
            return False
    return True

for fileItr in range(len(IfileList)):
    print "**************************************************************************************************************************"         

    # data generation. Set file location here
    #partpricedata = loadPartPrices("C:\Users\David B\Documents\Data Analytics\Projects\Recommender Systems\Vehicle Part Regional Prices.xlsm", "Matrix Subset Ax2")
    #partpricedata = loadPartPrices("C:\Users\David B\Documents\Data Analytics\Projects\Recommender Systems\Vehicle Part Regional Prices.xlsm", "Matrix Subset Ax3")
    print "Importing CSV File ", IfileList[fileItr]
    columnHeaders, partpricedata = loadCSV(IfileList[fileItr])
            
    #convert regional part prices into 0s and 1s (0 - representing no part price available and 1 - representing part price is available in that cooresponding region
    print "Commonizing Parts for Kmeans"
    commonRecommendations = commonizePartPrices(partpricedata)
    
    # computing K-Means with K = 2 raised to the power of number of columns in partpricingdata minus the header columns
    K = math.pow(2,len(partpricedata[0]) - 4)
    
    random.seed(234)

    condition = True
    while condition:
        print "Running Kmeans with K = ", K
        centroids,_ = kmeans(commonRecommendations,K)
        condition = not checkCentroids()
        K = K + 1

    print "Completed Kmeans with K = ", len(centroids)
    #print centroids
    
    # assign each sample to a cluster
    idx,_ = vq(commonRecommendations,centroids)
    #print idx

    #group parts with indexes
    groups = []
    for grpi in range(len(centroids)):
        groups.append(partpricedata[idx==grpi,])
    #    print grpi
    #    print partpricedata[idx==grpi,]
    
    #loop through each cluster center  
    for clstGroup in range(len(centroids)):
        print "clstGroup: ", clstGroup, " of ", len(centroids) - 1
        #loop through each missing column within cluster center
        for clstColumn in range(len(centroids[clstGroup])):
            #if this column is missing a price (i.e. -1) then go look for other clusters that have a price (i.e. +1)
            if(centroids[clstGroup][clstColumn] == -1):
                #look for other clusters that have a price (i.e. +1)
                matchingClusterSets = findMatchingClusterSets(clstColumn=clstColumn,clstGroup=clstGroup)
                #if there is a matching set then find row with best similiarity and make prediction
                if(sum(matchingClusterSets) > 0):
                    findBestmatchAndMakePricePrediciton(clstColumn=clstColumn, clstGroup=clstGroup, matchingClusterSets=matchingClusterSets)
    
    print "Export Results to CSV ", EfileList[fileItr]   
    exportCSV(EfileList[fileItr],columnHeaders, partpricedata)