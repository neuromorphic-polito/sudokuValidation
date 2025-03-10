import csv
import numpy as np
import pandas as pd

difficulties = ['easy', 'med', 'hard']
puzzles = [1, 2, 3]
bugFix = [0, 1]
enhanced = [0, 1]
labels = [
    'trial',
    'difficulty', 'puzzle', 'bugFix', 'enhanced',
    'flagSol', 'flagBug', 'flagCheck', 'flagIf', 'flagMem',
    'timeSol', 'timeBug', 'timeCheck', 'timeIf', 'timeMem',
    'binSol', 'binBug', 'binCheck', 'binIf', 'binMem',
    'spikeSol', 'spikeCheck', 'spikeIf', 'spikeMem',
]
i, results = 0, []
for d in difficulties:
    for p in puzzles:
        for b in bugFix:
            for e in enhanced:
                name = f'GENN{d}{p}_b{b}_e{e}'
                file = open(f'../results/GENN/{name}.csv', 'r')
                for line in file.readlines():
                    data = [i]+line.rstrip().split(',')[1:]
                    if e == 1:
                        data[5] = data[9]
                        # data[20] = data[23]
                    results.append(data)
                    i += 1
                file.close()
file = open('../results/resultsGENN.csv', 'w', newline='')
write = csv.writer(file)
write.writerow(labels)
write.writerows(results)
file.close()

resultsGENN = pd.read_csv('../results/resultsGENN.csv')

#########################
# ##### Bug Count ##### #
#########################
bugCount = 0
for row in resultsGENN.values.tolist():
    _, _, _, _, enhanced, _, flagBug, _, _, _, _, _, _, _, _, _, binBug, binCheck, _, _, _, _, _, _ = row
    if flagBug == 1 and ((enhanced == 1 and binBug < binCheck) or enhanced == 0):
        bugCount += 1
print(f'The bug is present: {bugCount} times\n')

############################
# ##### Success rate ##### #
############################
successCount = pd.pivot_table(
    resultsGENN,
    index='puzzle', columns=['difficulty', 'bugFix', 'enhanced'],
    values='flagSol',
    aggfunc='sum'
)
print(f'SUCCESS RATE\n{successCount}\n')

###########################
# ##### Spike count ##### #
###########################
resultsGENN['spikeCount'] = resultsGENN['spikeSol']+resultsGENN['spikeCheck']+resultsGENN['spikeIf']+resultsGENN['spikeMem']
spikeCount = pd.pivot_table(
    resultsGENN.loc[(resultsGENN['bugFix'] == 0)],
    columns=['difficulty', 'enhanced'],
    values='spikeCount',
    aggfunc=np.median
)
print(f'SPIKE COUNT\n{spikeCount}\n')

#############################
# ##### Solution time ##### #
#############################
timeSolution = pd.pivot_table(
    resultsGENN.loc[(resultsGENN['bugFix'] == 1) & (resultsGENN['enhanced'] == 1) & (resultsGENN['flagSol'] == 1)],
    index='puzzle', columns=['difficulty', 'bugFix', 'enhanced'],
    values='timeSol',
    aggfunc=np.min
)
print(f'TIME TO SOLUTION MIN\n{timeSolution}\n')

timeSolution = pd.pivot_table(
    resultsGENN.loc[(resultsGENN['bugFix'] == 1) & (resultsGENN['enhanced'] == 1) & (resultsGENN['flagSol'] == 1)],
    index='puzzle', columns=['difficulty', 'bugFix', 'enhanced'],
    values='timeSol',
    aggfunc=np.max
)
print(f'TIME TO SOLUTION MAX\n{timeSolution}\n')
