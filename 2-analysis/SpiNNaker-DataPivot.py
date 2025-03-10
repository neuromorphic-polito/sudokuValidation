import csv
import pandas as pd
import numpy as np

difficulties = ['easy', 'med', 'hard']
puzzles = [1, 2, 3]
strategy = [0, 1]
labels = ['trial', 'difficulty', 'puzzle', 'bugFix', 'enhanced', 'flagSol', 'spikeCount', 'timeSimulation', 'timeExtraction']
i, results = 0, []
for d in difficulties:
    for p in puzzles:
        for s in strategy:
            name = f'SpiNNaker{d}{p}_b{s}_e{s}'
            file = open(f'../results/SpiNNaker/{name}.csv', 'r')
            for line in file.readlines():
                results.append([i]+line.rstrip().split(',')[1:])
                i += 1
            file.close()
file = open('../results/resultsSpiNNaker.csv', 'w', newline='')
write = csv.writer(file)
write.writerow(labels)
write.writerows(results)
file.close()

resultsSpiNNaker = pd.read_csv('../results/resultsSpiNNaker.csv')

############################
# ##### Success rate ##### #
############################
successCount = pd.pivot_table(
    resultsSpiNNaker,
    index='puzzle', columns=['difficulty', 'bugFix'],
    values='flagSol',
    aggfunc='sum'
)
print(f'SUCCESS RATE\n{successCount}\n')

#############################
# ##### Solution time ##### #
#############################
timeSolution = pd.pivot_table(
    resultsSpiNNaker.loc[(resultsSpiNNaker['bugFix'] == 1) & (resultsSpiNNaker['enhanced'] == 1) & (resultsSpiNNaker['flagSol'] == 1)],
    index='puzzle', columns=['difficulty', 'bugFix', 'enhanced'],
    values='timeExtraction',
    aggfunc=np.min
)
print(f'TIME TO SOLUTION MIN\n{timeSolution}\n')

timeSolution = pd.pivot_table(
    resultsSpiNNaker.loc[(resultsSpiNNaker['bugFix'] == 1) & (resultsSpiNNaker['enhanced'] == 1) & (resultsSpiNNaker['flagSol'] == 1)],
    index='puzzle', columns=['difficulty', 'bugFix', 'enhanced'],
    values='timeExtraction',
    aggfunc=np.max
)
print(f'TIME TO SOLUTION MAX\n{timeSolution}\n')

