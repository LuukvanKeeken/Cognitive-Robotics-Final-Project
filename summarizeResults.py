import math
import os
import pandas
import numpy as np

directory = "./results"
tableData = []

def normalApproximation(sample):
    sampleSize = len(sample)
    p = sample.mean()
    q = 1-p

    if not (sampleSize * p > 5 and sampleSize * q > 5):
        return -1,-1
    mu = p * sampleSize
    std = math.sqrt(mu*q)
  
    return mu, std

def normalApproximationFromMean(sampleSize, p):
    q = 1-p

    if not (sampleSize * p > 5 and sampleSize * q > 5):
        return -1,-1
    mu = p * sampleSize
    std = math.sqrt(mu*q)
  
    return mu, std


for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        experiment = filename.replace(',', '')
        experiment = experiment.replace('.csv', '')
        words = experiment.split()
        for index, word in enumerate(words):
            if word == 'scenario': scenario = words[index+1]
            if word == 'segmenter': segmenter = words[index+1]
            if word == 'matching': matchingNetwork = words[index+1]
            if word == 'network': graspingNetwork = words[index+1]
        

        df = pandas.read_csv("./results/" + filename)

        runs = len(df.index)

        runsSuccess = df.correctObjectManipulations.mean()
        meanRunsSuccess, STDrunsSuccess = normalApproximation(df.correctObjectManipulations)
        #runsSuccessSTD = df.correctObjectManipulations.std()
        
        predictionSuccesDividedRun = df.correctObjectMatch / df.totalPredictions
        meanPredictionSuccessPerRun = predictionSuccesDividedRun.mean()
        STDPredictionSuccessPerRun = predictionSuccesDividedRun.std()
        predictionSuccesPerRun = df.correctObjectMatch.sum() / df.totalPredictions.sum()
        
        runsWithoutPileManipulation = len(df[df.pileManipulations == 0])/runs
        meanRunsWithoutPileManipulation,STDRunsWithoutPileManipulation = normalApproximationFromMean(runs, runsWithoutPileManipulation)

        meanPileManipulations = df.pileManipulations.mean()
        STDPileManipulations= df.pileManipulations.std()

        runsWithCorrectPredictionPerRun = len(df[df.correctObjectMatch>0])/runs
        meanRunsWithCorrectPredictionPerRun,STDRunsWithCorrectPredictionPerRun = normalApproximationFromMean(runs, runsWithCorrectPredictionPerRun)
        
        tableRow = [scenario, segmenter, matchingNetwork, graspingNetwork, meanRunsSuccess, STDrunsSuccess, predictionSuccesPerRun, meanPredictionSuccessPerRun, STDPredictionSuccessPerRun, meanRunsWithoutPileManipulation,STDRunsWithoutPileManipulation, meanPileManipulations,STDPileManipulations, meanRunsWithCorrectPredictionPerRun,STDRunsWithCorrectPredictionPerRun]
        tableData.append(tableRow)
   

        #numberOfFaultPredictedSegments
        #pileManipulations
        #wrongWorstPrediction
        #pileManipulationGraspFaults
        #correctObjectMatch
        #correctManipulations
        #correctGrasps     
        #totalPredictions

    else:
        continue
columns = ['scenario', 'segmenter', 'matchingNetwork', 'graspingNetwork', 'meanRunSuccess', 'STDRunSuccess', 'predictionSuccesPerRun', 'meanPredictionSuccessPerRun', 'STDPredictionSuccessPerRun ', 'meanRunsWithoutPileManipulation', 'STDRunsWithoutPileManipulation', 'meanPileManipulations', 'STDPileManipulations', 'meanRunsWithCorrectPredictionPerRun', 'STDRunsWithCorrectPredictionPerRun']


table = pandas.DataFrame(np.array(tableData), columns = columns)
print(table.to_markdown())
table.to_csv("summary.csv")