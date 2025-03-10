import nni
import argparse
from sudokuDataset import *
import numpy as np
import spynnaker8 as model
import time
import csv
import os


def main(argument):

    name = f'SpiNNaker{argument.difficulty}{argument.puzzle}_b{argument.bugFix}_e{argument.enhanced}'

    start = 0
    try:
        file = open(f'../results/{name}.csv', 'r')
        start = len(file.readlines())
        file.close()
    except:
        pass

    if start >= argument.trials:
        return 0

    ##################################
    # ##### Problem definition ##### #
    ##################################
    puzzle = None
    if argument.difficulty == 'easy':
        puzzle = puzzleEasy
    elif argument.difficulty == 'med':
        puzzle = puzzleMed
    elif argument.difficulty == 'hard':
        puzzle = puzzleHard
    elif argument.difficulty == 'imp':
        puzzle = puzzleImp

    ##### General variables #####
    variables = 9
    (sudokuPop, sudoku, sudokuSol) = puzzle[argument.puzzle]
    sudoku = np.array(sudoku)
    timeSimulation = 60  # s
    timeSteps = int(timeSimulation*1e3)

    ################################
    # ##### Model definition ##### #
    ################################
    model.setup(timestep=1.0, min_delay=2, max_delay=144)

    ##### Neuron parameters #####
    paramLif = {
        'cm': 0.25,  # nF
        'tau_m': 20.0,  # ms
        'i_offset': 0.3,  # nA
        'v_rest': -65.0,  # mV
        'v_thresh': -50.0,  # mV
        'v_reset': -70.0,  # mV
        'tau_refrac': 2.0,  # ms
        'tau_syn_E': 5.0,  # ms
        'tau_syn_I': 5.0,  # ms
    }

    # ##### Populations neurons ##### #
    solverPop = sudokuPop

    popStim = [[0 for _ in range(variables)] for _ in range(variables)]
    for row in range(variables):
        for col in range(variables):
            popStim[row][col] = model.Population(
                label=f'stim{row}{col}',
                size=variables*solverPop,
                cellclass=model.SpikeSourcePoisson(rate=20, duration=timeSteps)
            )

    popVar = [[0 for _ in range(variables)] for _ in range(variables)]
    for row in range(variables):
        for col in range(variables):
            popVar[row][col] = model.Population(
                label=f'var{row}{col}',
                size=variables*solverPop,
                cellclass=model.IF_curr_exp, cellparams=paramLif
            )
            if argument.enhanced == 0: popVar[row][col].record('spikes')

    # ##### Synaptic connections ##### #
    ##### Stim-Var excitation #####
    for row in range(variables):
        for col in range(variables):
            clue = sudoku[row, col]-1
            weight = np.random.uniform(low=1.4, high=1.6, size=variables*solverPop)
            synapsesExcit = np.diag(weight)
            if clue >= 0:
                mask = np.zeros(variables*solverPop)
                mask[clue*solverPop:(clue+1)*solverPop] = 1
                synapsesExcit = synapsesExcit*np.diag(mask)
            synapsesExcitList = []
            for so in range(variables*solverPop):
                for to in range(variables*solverPop):
                    synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
            model.Projection(
                label=f'stimExcit{row}{col}',
                presynaptic_population=popStim[row][col], postsynaptic_population=popVar[row][col],
                receptor_type='excitatory',
                connector=model.FromListConnector(synapsesExcitList)
            )

    ##### Var-Var internal inhibition #####
    for row in range(variables):
        for col in range(variables):
            synapsesInhib = np.random.uniform(low=-0.08, high=0.0, size=(variables*solverPop, variables*solverPop))
            for i in range(variables):
                synapsesInhib[i*solverPop:(i+1)*solverPop, i*solverPop:(i+1)*solverPop] = 0
            synapsesInhibList = []
            for so in range(variables*solverPop):
                for to in range(variables*solverPop):
                    synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
            model.Projection(
                label=f'internalInhib{row}{col}',
                presynaptic_population=popVar[row][col], postsynaptic_population=popVar[row][col],
                receptor_type='inhibitory',
                connector=model.FromListConnector(synapsesInhibList)
            )

    ##### Var-Var lateral inhibition #####
    indexes = [(row, col) for row in range(variables) for col in range(variables)]
    constraints = []
    for i, cell1 in enumerate(indexes):
        for j, cell2 in enumerate(indexes):
            if (cell1[0] == cell2[0] or cell1[1] == cell2[1]) and i != j:
                constraints.append((cell1, cell2))
            if (cell2[0]//3 == cell1[0]//3 and cell2[1]//3 == cell1[1]//3) and (cell2[0] != cell1[0] and cell2[1] != cell1[1]) and i != j:
                constraints.append((cell1, cell2))
    for (rowSo, colSo), (rowTo, colTo) in constraints:
        if sudoku[rowTo][colTo] == 0 or argument.bugFix == 0:
            synapsesInhib = np.zeros(shape=(variables*solverPop, variables*solverPop))
            for i in range(variables):
                synapsesInhib[i*solverPop:(i+1)*solverPop, i*solverPop:(i+1)*solverPop] = np.random.uniform(low=-0.08, high=0.0, size=(solverPop, solverPop))
            synapsesInhibList = []
            for so in range(variables*solverPop):
                for to in range(variables*solverPop):
                    synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
            model.Projection(
                label=f'lateralInhib{rowSo}{colSo}{rowTo}{colTo}',
                presynaptic_population=popVar[rowSo][colSo], postsynaptic_population=popVar[rowTo][colTo],
                receptor_type='inhibitory',
                connector=model.FromListConnector(synapsesInhibList)
            )

    if argument.enhanced == 1:
        # ##### Validation network ##### #
        ##### Neuron parameters #####
        paramLif['i_offset'] = 0.1
        validationPop = 10

        # ##### Neuron populations ##### #
        popPolisher = [[0 for _ in range(variables)] for _ in range(variables)]
        for row in range(variables):
            for col in range(variables):
                popPolisher[row][col] = model.Population(
                    label=f'polisher{row}{col}',
                    size=variables*validationPop,
                    cellclass=model.IF_curr_exp,
                    cellparams=paramLif
                )

        popNetChecker = [[0 for _ in range(variables)] for _ in range(3)]
        for c in range(3):
            for v in range(variables):
                popNetChecker[c][v] = model.Population(
                    label=f'netChecker{c}{v}',
                    size=(variables+1)*validationPop,
                    cellclass=model.IF_curr_exp,
                    cellparams=paramLif
                )

        ##### Var-Polisher excitation #####
        for row in range(variables):
            for col in range(variables):
                synapsesExcit = np.zeros(shape=(variables*solverPop, variables*validationPop))
                for i in range(variables):
                    synapsesExcit[i*solverPop:(i+1)*solverPop, i*validationPop:(i+1)*validationPop] = 1.0
                synapsesExcitList = []
                for so in range(variables*solverPop):
                    for to in range(variables*validationPop):
                        synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
                model.Projection(
                    label=f'polisherExcit{row}{col}',
                    presynaptic_population=popVar[row][col], postsynaptic_population=popPolisher[row][col],
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapsesExcitList)
                )

        ##### Polisher-Polisher internal inhibition #####
        for row in range(variables):
            for col in range(variables):
                synapsesInhib = -1.0*np.ones(shape=(variables*validationPop, variables*validationPop))
                for i in range(variables):
                    synapsesInhib[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 0.0
                synapsesInhibList = []
                for so in range(variables*validationPop):
                    for to in range(variables*validationPop):
                        synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
                model.Projection(
                    label=f'polisherInhib{row}{col}',
                    presynaptic_population=popPolisher[row][col], postsynaptic_population=popPolisher[row][col],
                    receptor_type='inhibitory',
                    connector=model.FromListConnector(synapsesInhibList)
                )

        ##### Row Polisher-NetChecker excitation #####
        for row in range(variables):
            for col in range(variables):
                synapsesExcit = np.zeros(shape=(variables*validationPop, (variables+1)*validationPop))
                for n in range(1, validationPop+1):
                    synapsesExcit[:, -n] = 0.15
                for i in range(variables):
                    synapsesExcit[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 1.0
                synapsesExcitList = []
                for so in range(variables*validationPop):
                    for to in range((variables+1)*validationPop):
                        synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
                model.Projection(
                    label=f'rowNetChecker{row}{col}',
                    presynaptic_population=popPolisher[row][col], postsynaptic_population=popNetChecker[0][row],
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapsesExcitList)
                )

        ##### Col Polisher-NetChecker excitation #####
        for col in range(variables):
            for row in range(variables):
                synapsesExcit = np.zeros(shape=(variables*validationPop, (variables+1)*validationPop))
                for n in range(1, validationPop+1):
                    synapsesExcit[:, -n] = 0.15
                for i in range(variables):
                    synapsesExcit[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 1.0
                synapsesExcitList = []
                for so in range(variables*validationPop):
                    for to in range((variables+1)*validationPop):
                        synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
                model.Projection(
                    label=f'colNetChecker{row}{col}',
                    presynaptic_population=popPolisher[row][col], postsynaptic_population=popNetChecker[1][col],
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapsesExcitList)
                )

        ##### Cel Polisher-NetChecker excitation #####
        mapping = [[] for _ in range(variables)]
        i = 0
        for row in range(variables):
            for col in range(variables):
                if row % 3 == 1 and col % 3 == 1:
                    mapping[i].append((row-1, col-1))
                    mapping[i].append((row-1, col))
                    mapping[i].append((row-1, col+1))
                    mapping[i].append((row, col-1))
                    mapping[i].append((row, col))
                    mapping[i].append((row, col+1))
                    mapping[i].append((row+1, col-1))
                    mapping[i].append((row+1, col))
                    mapping[i].append((row+1, col+1))
                    i += 1
        for cel in range(variables):
            for v in range(variables):
                synapsesExcit = np.zeros(shape=(variables*validationPop, (variables+1)*validationPop))
                for n in range(1, validationPop+1):
                    synapsesExcit[:, -n] = 0.15
                for i in range(variables):
                    synapsesExcit[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 1.0
                synapsesExcitList = []
                for so in range(variables*validationPop):
                    for to in range((variables+1)*validationPop):
                        synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
                mapRow, mapCol = mapping[cel][v]
                model.Projection(
                    label=f'celNetChecker{mapRow}{mapCol}',
                    presynaptic_population=popPolisher[mapRow][mapCol], postsynaptic_population=popNetChecker[2][cel],
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapsesExcitList)
                )

        ##### NetChecker-NetChecker inhibition #####
        for c in range(3):
            for v in range(variables):
                synapsesInhib = -1.2*np.ones(shape=((variables+1)*validationPop, (variables+1)*validationPop))
                for i in range(variables+1):
                    synapsesInhib[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 0.0
                synapsesInhibList = []
                for so in range((variables+1)*validationPop):
                    for to in range((variables+1)*validationPop):
                        synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
                model.Projection(
                    label=f'netCheckerNetChecker{c}{v}',
                    presynaptic_population=popNetChecker[c][v], postsynaptic_population=popNetChecker[c][v],
                    receptor_type='inhibitory',
                    connector=model.FromListConnector(synapsesInhibList)
                )

        # ##### Neuron populations ##### #
        popIf = model.Population(
            label=f'if',
            size=2*validationPop,
            cellclass=model.IF_curr_exp, cellparams=paramLif
        )

        ##### NetChecker-If excitation #####
        for c in range(3):
            for v in range(variables):
                synapsesExcit = np.zeros(shape=((variables+1)*validationPop, 2*validationPop))
                synapsesExcit[0:9*validationPop, validationPop:] = 1.0
                synapsesExcit[9*validationPop:, 0:validationPop] = 0.11
                synapses_excitList = []
                for i1 in range((variables+1)*validationPop):
                    for i2 in range(2*validationPop):
                        synapses_excitList.append([i1, i2, synapsesExcit[i1, i2], 2.0])
                model.Projection(
                    label=f'ifExcit{c}{v}',
                    presynaptic_population=popNetChecker[c][v], postsynaptic_population=popIf,
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapses_excitList)
                )

        ##### If-If excitation #####
        synapsesExcit = np.zeros(shape=(2*validationPop, 2*validationPop))
        synapsesExcit[0:validationPop, 0:validationPop] = 1.1
        synapsesExcitList = []
        for so in range(2*validationPop):
            for to in range(2*validationPop):
                synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
        model.Projection(
            label=f'ifExcit',
            presynaptic_population=popIf, postsynaptic_population=popIf,
            receptor_type='excitatory',
            connector=model.FromListConnector(synapsesExcitList)
        )

        ##### If-If inhibition #####
        synapsesInhib = np.zeros(shape=(2*validationPop, 2*validationPop))
        synapsesInhib[0:validationPop, validationPop:] = -1.0
        synapsesInhib[validationPop:, 0:validationPop] = -1.0
        synapsesInhibList = []
        for so in range(2*validationPop):
            for to in range(2*validationPop):
                synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
        model.Projection(
            label=f'ifInhib',
            presynaptic_population=popIf, postsynaptic_population=popIf,
            receptor_type='inhibitory',
            connector=model.FromListConnector(synapsesInhibList)
        )

        ###########################################################################################
        popDelay = []
        popDelay.append(
            model.Population(
                label=f'ifDelay',
                size=2*validationPop,
                cellclass=model.IF_curr_exp,
                cellparams=paramLif
            )
        )

        ##### Sudoku network Inhibition #####
        synapsesExcit = np.zeros(shape=(2*validationPop, 2*validationPop))
        for i in range(validationPop):
            synapsesExcit[i, i] = 1
        synapsesExcit *= 2.5
        synapsesExcitList = []
        for so in range(2*validationPop):
            for to in range(2*validationPop):
                synapsesExcitList.append([so, to, synapsesExcit[so, to], 144.0])
        model.Projection(
            label=f'delyPop0',
            presynaptic_population=popIf, postsynaptic_population=popDelay[-1],
            receptor_type='excitatory',
            connector=model.FromListConnector(synapsesExcitList)
        )
        for d in range(1, 17):
            popDelay.append(
                model.Population(
                    label=f'ifDelay',
                    size=2*validationPop,
                    cellclass=model.IF_curr_exp,
                    cellparams=paramLif
                )
            )

            ##### Sudoku network Inhibition #####
            synapsesExcit = np.zeros(shape=(2*validationPop, 2*validationPop))
            for i in range(validationPop):
                synapsesExcit[i, i] = 1
            synapsesExcit *= 2.5
            synapsesExcitList = []
            for so in range(2*validationPop):
                for to in range(2*validationPop):
                    synapsesExcitList.append([so, to, synapsesExcit[so, to], 144.0])
            model.Projection(
                label=f'delyPop{d}',
                presynaptic_population=popDelay[-2], postsynaptic_population=popDelay[-1],
                receptor_type='excitatory',
                connector=model.FromListConnector(synapsesExcitList)
            )

        ##### If-Var inhibition #####
        for row in range(variables):
            for col in range(variables):
                synapsesInhib = -1.0*np.ones(shape=(2*validationPop, variables*solverPop))
                synapsesInhib[validationPop:, :] = 0
                synapsesInhibList = []
                for so in range(2*validationPop):
                    for to in range(variables*solverPop):
                        synapsesInhibList.append([so, to, synapsesInhib[so, to], 144.0])
                model.Projection(
                    label=f'ifSolverInhi{row}{col}',
                    presynaptic_population=popDelay[-1], postsynaptic_population=popVar[row][col],
                    receptor_type='inhibitory',
                    connector=model.FromListConnector(synapsesInhibList)
                )
        ########################################################################################

        # ##### Neuron populations ##### #
        memoryPop = 3
        popMem = [[0 for _ in range(variables)] for _ in range(variables)]
        for row in range(variables):
            for col in range(variables):
                popMem[row][col] = model.Population(
                    label=f'mem{row}{col}',
                    size=variables*memoryPop,
                    cellclass=model.IF_curr_exp,
                    cellparams=paramLif
                )
                popMem[row][col].record('spikes')

        ##### Var-Mem excitation #####
        for row in range(variables):
            for col in range(variables):
                synapsesExcit = np.zeros(shape=(variables*solverPop, variables*memoryPop))
                for i in range(variables):
                    synapsesExcit[i*solverPop:(i+1)*solverPop, i*memoryPop:(i+1)*memoryPop] = 1.0
                synapsesExcitList = []
                for so in range(variables*solverPop):
                    for to in range(variables*memoryPop):
                        synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
                model.Projection(
                    label=f'memExcit{row}{col}',
                    presynaptic_population=popVar[row][col], postsynaptic_population=popMem[row][col],
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapsesExcitList)
                )

        ##### Mem-mem excitation #####
        for row in range(variables):
            for col in range(variables):
                synapsesExcit = np.zeros(shape=(variables*memoryPop, variables*memoryPop))
                for i in range(variables):
                    synapsesExcit[i*memoryPop:(i+1)*memoryPop, i*memoryPop:(i+1)*memoryPop] = 0.8
                synapsesExcitList = []
                for so in range(variables*memoryPop):
                    for to in range(variables*memoryPop):
                        synapsesExcitList.append([so, to, synapsesExcit[so, to], 2.0])
                model.Projection(
                    label=f'memInteExcit{row}{col}',
                    presynaptic_population=popMem[row][col], postsynaptic_population=popMem[row][col],
                    receptor_type='excitatory',
                    connector=model.FromListConnector(synapsesExcitList)
                )

        ##### Mem-Mem lateral inhibition #####
        for row in range(variables):
            for col in range(variables):
                synapsesInhib = -0.3*np.ones(shape=(variables*memoryPop, variables*memoryPop))
                for i in range(variables):
                    synapsesInhib[i*memoryPop:(i+1)*memoryPop, i*memoryPop:(i+1)*memoryPop] = 0.0
                synapsesInhibList = []
                for so in range(variables*memoryPop):
                    for to in range(variables*memoryPop):
                        synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
                model.Projection(
                    label=f'memInteInhib{row}{col}',
                    presynaptic_population=popMem[row][col], postsynaptic_population=popMem[row][col],
                    receptor_type='inhibitory',
                    connector=model.FromListConnector(synapsesInhibList)
                )

        ##### If-Mem inhibition #####
        for row in range(variables):
            for col in range(variables):
                synapsesInhib = -0.6*np.ones(shape=(2*validationPop, variables*memoryPop))
                synapsesInhib[0:validationPop, :] = 0
                synapsesInhibList = []
                for so in range(2*validationPop):
                    for to in range(variables*memoryPop):
                        synapsesInhibList.append([so, to, synapsesInhib[so, to], 2.0])
                model.Projection(
                    label=f'ifInhiMem{row}{col}',
                    presynaptic_population=popIf, postsynaptic_population=popMem[row][col],
                    receptor_type='inhibitory',
                    connector=model.FromListConnector(synapsesInhibList)
                )

    ##########################
    # ##### Simulation ##### #
    ##########################
    timeSimulationStart = time.time()
    model.run(timeSteps)
    timeSimulationEnd = time.time()
    timeSimulation = timeSimulationEnd-timeSimulationStart

    ########################
    # ##### Analysis ##### #
    ########################
    binWidth = 100
    assert timeSteps % binWidth == 0
    binsTime = np.arange(0, timeSteps+binWidth, binWidth)

    flagSol, timeExtraction, spikeCount = 0, 0, 0
    if argument.enhanced == 0:
        timeBaseLineStart = time.time()
        popVarRec = [[0 for _ in range(variables)] for _ in range(variables)]
        for row in range(variables):
            for col in range(variables):
                spike = popVar[row][col].get_data('spikes')
                popVarRec[row][col] = [np.array(neuron) for neuron in spike.segments[0].spiketrains]
        timeBaseLineEnd = time.time()
        timeExtraction = timeBaseLineEnd-timeBaseLineStart

        ##### Solution validation #####
        solverBins = np.zeros(shape=(binsTime.size-1, variables, variables), dtype=int)
        for row in range(variables):
            for col in range(variables):
                value = np.zeros(shape=(binsTime.size-1, variables), dtype=int)
                for i in range(variables*solverPop):
                    digit = i//solverPop
                    interval = np.searchsorted(binsTime, popVarRec[row][col][i])-1
                    value[interval, digit] += 1
                for b in range(binsTime.size-1):
                    solverBins[b][row][col] = np.argmax(value[b, :])+1

        for b in range(binsTime.size-1):
            if np.array_equal(solverBins[b], sudokuSol) == True:
                flagSol = 1
                break

        ##### Spike Count #####
        for row in range(variables):
            for col in range(variables):
                for i in range(variables*solverPop):
                    spikeCount += len(popVarRec[row][col][i])

    elif argument.enhanced == 1:
        timeEnhancedStart = time.time()
        popMemRec = [[0 for _ in range(variables)] for _ in range(variables)]
        for row in range(variables):
            for col in range(variables):
                spike = popMem[row][col].get_data('spikes')
                popMemRec[row][col] = [np.array(neuron) for neuron in spike.segments[0].spiketrains]
        timeEnhancedEnd = time.time()
        timeExtraction = timeEnhancedEnd-timeEnhancedStart

        sudokuMem = np.zeros(shape=(variables, variables), dtype=int)
        for row in range(variables):
            for col in range(variables):
                value = np.zeros(shape=(variables), dtype=int)
                for i in range(variables*memoryPop):
                    digit = i//memoryPop
                    value[digit] += np.sum(np.array(popMemRec[row][col][i]))
                sudokuMem[row][col] = np.argmax(value)+1

        ##### Memory validation #####
        if np.array_equal(sudokuSol, sudokuMem):
            flagSol = 1

        for row in range(variables):
            for col in range(variables):
                for i in range(variables*memoryPop):
                    spikeCount += len(popMemRec[row][col][i])

    try:
        file = open(f'../results/{name}.csv', 'r')
        file.close()
    except:
        file = open(f'../results/{name}.csv', 'w', newline='')
        file.close()

    file = open(f'../results/{name}.csv', 'a', newline='')
    write = csv.writer(file)
    write.writerow([
        start,
        argument.difficulty, argument.puzzle, argument.bugFix, argument.enhanced,
        flagSol,
        timeSimulation, timeExtraction,
        spikeCount
    ])
    file.close()

    model.end()

    os.system('rm -r application_generated_data_files')
    os.system('rm -r reports')

    return 0


######################
# ##### Parser ##### #
######################
if __name__ == '__main__':
    paramsNNI = {
        'difficulty': 'easy',
        'conf': 0,
        'trials': 2,
    }

    paramsNNI.update(nni.get_next_parameter())

    parser = argparse.ArgumentParser(description='Sudoku solver SpiNNaker')

    parser.add_argument('-d', '--difficulty', help='Puzzle difficulty level', type=str, default='easy')  # paramsNNI['difficulty'])
    parser.add_argument('-p', '--puzzle', help='Puzzle type', type=int, default=1)
    parser.add_argument('-t', '--trials', help='Attempt of resolution', type=int, default=10)  # paramsNNI['trials'])
    parser.add_argument('-b', '--bugFix', help='Bug fix for changing state of original problem', type=int, default=1)  # paramsNNI['conf'])
    parser.add_argument('-e', '--enhanced', help='Use the enhanced pipeline', type=int, default=1)  # paramsNNI['conf'])
    parser.add_argument('-w', '--bin-width', help='Binning width', type=int, default=100)

    argument = parser.parse_args()
    main(argument)
