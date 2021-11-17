import os
import pandas
import numpy as np
import matplotlib.pyplot as plt


directory = "./results"
tableData = []
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

        runsSuccess = df.correctObjectManipulations.mean()/runs
        predictionSuccess = df.correctObjectMatch.sum() / df.totalPredictions.sum()
        
        runsWithoutPileManipulation = len(df[df.pileManipulations == 0])/runs
        averagePileManipulationsPerRun = df.pileManipulations.mean()

        atleastOneCorrectPredictionPerRun = len(df[df.correctObjectMatch>0])/runs
        
        tableRow = [scenario, segmenter, matchingNetwork, graspingNetwork, runsSuccess, predictionSuccess, runsWithoutPileManipulation, averagePileManipulationsPerRun, atleastOneCorrectPredictionPerRun]
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

table = pandas.DataFrame(np.array(tableData), columns = ['scenario', 'segmenter', 'matchingNetwork', 'graspingNetwork', 'runsSuccess', 'predictionSuccess', 'runsWithoutPileManipulation','averagePileManipulationsPerRun', 'atleastOneCorrectPredictionPerRun'])
print(table.to_markdown())