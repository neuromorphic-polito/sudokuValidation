import argparse
import os
from sudokuDataset import *
import numpy as np
from pygenn import GeNNModel, init_postsynaptic, init_weight_update
import csv


def main(argument):

    name = f'GENN{argument.difficulty}{argument.puzzle}_b{argument.bugFix}_e{argument.enhanced}'

    backend = None
    if argument.gpu == 1:
        backend = 'cuda'
        os.environ['CUDA_PATH'] = '/usr/local/cuda'
    else:
        backend = 'single_threaded_cpu'

    start = 0
    try:
        file = open(f'../results/GENN/{name}.csv', 'r')
        start = len(file.readlines())
        file.close()
    except:
        pass

    for trial in range(start, argument.trials):
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

        ################################
        # ##### Model definition ##### #
        ################################
        model = GeNNModel(precision='float', model_name=f'.{name}', backend=backend)
        model.dt = 1.0  # ms

        ##### Neuron parameters #####
        lifParam = {
            'C': 0.25,  # nF
            'TauM': 20.0,  # ms
            'Ioffset': 0.3,  # nA
            'Vrest': -65.0,  # mV
            'Vthresh': -50.0,  # mV
            'Vreset': -70.0,  # mV
            'TauRefrac': 2.0,  # ms
        }
        lifVar = {
            'V': lifParam['Vrest'],  # mV
            'RefracTime': 0.0,  # ms
        }

        # ##### Populations neurons ##### #
        solverPop = sudokuPop

        popStim = [[0 for _ in range(variables)] for _ in range(variables)]
        for row in range(variables):
            for col in range(variables):
                popStim[row][col] = model.add_neuron_population(
                    pop_name=f'stim{row}{col}',
                    num_neurons=variables*solverPop,
                    neuron='Poisson',
                    params={'rate': 20}, vars={'timeStepToSpike': 0}
                )

        popVar = [[0 for _ in range(variables)] for _ in range(variables)]
        for row in range(variables):
            for col in range(variables):
                popVar[row][col] = model.add_neuron_population(
                    pop_name=f'var{row}{col}',
                    num_neurons=variables*solverPop,
                    neuron='LIF',
                    params=lifParam, vars=lifVar
                )
                popVar[row][col].spike_recording_enabled = True

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
                model.add_synapse_population(
                    pop_name=f'stimExcit{row}{col}', matrix_type='DENSE',
                    source=popStim[row][col], target=popVar[row][col],
                    postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                    weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                )

        ##### Var-Var internal inhibition #####
        for row in range(variables):
            for col in range(variables):
                synapsesInhib = np.random.uniform(low=-0.08, high=0.0, size=(variables*solverPop, variables*solverPop))
                for i in range(variables):
                    synapsesInhib[i*solverPop:(i+1)*solverPop, i*solverPop:(i+1)*solverPop] = 0
                model.add_synapse_population(
                    pop_name=f'internalInhib{row}{col}', matrix_type='DENSE',
                    source=popVar[row][col], target=popVar[row][col],
                    postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                    weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
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
                model.add_synapse_population(
                    pop_name=f'lateralInhib{rowSo}{colSo}{rowTo}{colTo}', matrix_type='DENSE',
                    source=popVar[rowSo][colSo], target=popVar[rowTo][colTo],
                    postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                    weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
                )

        if argument.enhanced == 1:
            # ##### Validation network ##### #
            ##### Neuron parameters #####
            lifParam['Ioffset'] = 0.1
            validationPop = 10

            # ##### Neuron populations ##### #
            popPolisher = [[0 for _ in range(variables)] for _ in range(variables)]
            for row in range(variables):
                for col in range(variables):
                    popPolisher[row][col] = model.add_neuron_population(
                        pop_name=f'polisher{row}{col}',
                        num_neurons=variables*validationPop,
                        neuron='LIF',
                        params=lifParam, vars=lifVar
                    )

            popNetChecker = [[0 for _ in range(variables)] for _ in range(3)]
            for c in range(3):
                for v in range(variables):
                    popNetChecker[c][v] = model.add_neuron_population(
                        pop_name=f'netChecker{c}{v}',
                        num_neurons=(variables+1)*validationPop,
                        neuron='LIF',
                        params=lifParam, vars=lifVar
                    )
                    popNetChecker[c][v].spike_recording_enabled = True

            ##### Var-Polisher excitation #####
            for row in range(variables):
                for col in range(variables):
                    synapsesExcit = np.zeros(shape=(variables*solverPop, variables*validationPop))
                    for i in range(variables):
                        synapsesExcit[i*solverPop:(i+1)*solverPop, i*validationPop:(i+1)*validationPop] = 1.0
                    model.add_synapse_population(
                        pop_name=f'polisherExcit{row}{col}', matrix_type='DENSE',
                        source=popVar[row][col], target=popPolisher[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                    )

            ##### Polisher-Polisher internal inhibition #####
            for row in range(variables):
                for col in range(variables):
                    synapsesInhib = -1.0*np.ones(shape=(variables*validationPop, variables*validationPop))
                    for i in range(variables):
                        synapsesInhib[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 0.0
                    model.add_synapse_population(
                        pop_name=f'polisherInhib{row}{col}', matrix_type='DENSE',
                        source=popPolisher[row][col], target=popPolisher[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
                    )

            ##### Row Polisher-NetChecker excitation #####
            for row in range(variables):
                for col in range(variables):
                    synapsesExcit = np.zeros(shape=(variables*validationPop, (variables+1)*validationPop))
                    for n in range(1, validationPop+1):
                        synapsesExcit[:, -n] = 0.15
                    for i in range(variables):
                        synapsesExcit[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 1.0
                    model.add_synapse_population(
                        pop_name=f'rowNetChecker{row}{col}', matrix_type='DENSE',
                        source=popPolisher[row][col], target=popNetChecker[0][row],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                    )

            ##### Col Polisher-NetChecker excitation #####
            for col in range(variables):
                for row in range(variables):
                    synapsesExcit = np.zeros(shape=(variables*validationPop, (variables+1)*validationPop))
                    for n in range(1, validationPop+1):
                        synapsesExcit[:, -n] = 0.15
                    for i in range(variables):
                        synapsesExcit[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 1.0
                    model.add_synapse_population(
                        pop_name=f'colNetChecker{row}{col}', matrix_type='DENSE',
                        source=popPolisher[row][col], target=popNetChecker[1][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
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
                    mapRow, mapCol = mapping[cel][v]
                    model.add_synapse_population(
                        pop_name=f'celNetChecker{mapRow}{mapCol}', matrix_type='DENSE',
                        source=popPolisher[mapRow][mapCol], target=popNetChecker[2][cel],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                    )

            ##### NetChecker-NetChecker inhibition #####
            for c in range(3):
                for v in range(variables):
                    synapsesInhib = -1.2*np.ones(shape=((variables+1)*validationPop, (variables+1)*validationPop))
                    for i in range(variables+1):
                        synapsesInhib[i*validationPop:(i+1)*validationPop, i*validationPop:(i+1)*validationPop] = 0.0
                    model.add_synapse_population(
                        pop_name=f'netCheckerNetChecker{c}{v}', matrix_type='DENSE',
                        source=popNetChecker[c][v], target=popNetChecker[c][v],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
                    )

            # ##### Neuron populations ##### #
            popIf = model.add_neuron_population(
                pop_name=f'if',
                num_neurons=2*validationPop,
                neuron='LIF',
                params=lifParam, vars=lifVar
            )
            popIf.spike_recording_enabled = True

            ##### NetChecker-If excitation #####
            for c in range(3):
                for v in range(variables):
                    synapsesExcit = np.zeros(shape=((variables+1)*validationPop, 2*validationPop))
                    synapsesExcit[0:9*validationPop, validationPop:] = 11.0
                    synapsesExcit[9*validationPop:, 0:validationPop] = 0.02
                    model.add_synapse_population(
                        pop_name=f'ifExcit{c}{v}', matrix_type='DENSE',
                        source=popNetChecker[c][v], target=popIf,
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                    )

            ##### If-If excitation #####
            synapsesExcit = np.zeros(shape=(2*validationPop, 2*validationPop))
            synapsesExcit[0:validationPop, 0:validationPop] = 0.5
            model.add_synapse_population(
                pop_name=f'ifExcit', matrix_type='DENSE',
                source=popIf, target=popIf,
                postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
            )

            ##### If-If inhibition #####
            synapsesInhib = np.zeros(shape=(2*validationPop, 2*validationPop))
            synapsesInhib[0:validationPop, validationPop:] = -1.0
            synapsesInhib[validationPop:, 0:validationPop] = -1.0
            model.add_synapse_population(
                pop_name=f'ifInhib', matrix_type='DENSE',
                source=popIf, target=popIf,
                postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
            )

            ##### If-Var inhibition #####
            synapticDelay = []
            for row in range(variables):
                for col in range(variables):
                    synapsesInhib = -2.0*np.ones(shape=(2*validationPop, variables*solverPop))
                    synapsesInhib[validationPop:, :] = 0
                    synapticDelay.append(model.add_synapse_population(
                        pop_name=f'ifSolverInhi{row}{col}', matrix_type='DENSE',
                        source=popIf, target=popVar[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
                    ))
                    synapticDelay[-1].axonal_delay_steps = 400

            # ##### Neuron populations ##### #
            memoryPop = 3
            popMem = [[0 for _ in range(variables)] for _ in range(variables)]
            for row in range(variables):
                for col in range(variables):
                    popMem[row][col] = model.add_neuron_population(
                        pop_name=f'mem{row}{col}',
                        num_neurons=variables*memoryPop,
                        neuron='LIF',
                        params=lifParam, vars=lifVar
                    )
                    popMem[row][col].spike_recording_enabled = True

            ##### Var-Mem excitation #####
            for row in range(variables):
                for col in range(variables):
                    synapsesExcit = np.zeros(shape=(variables*solverPop, variables*memoryPop))
                    for i in range(variables):
                        synapsesExcit[i*solverPop:(i+1)*solverPop, i*memoryPop:(i+1)*memoryPop] = 1.0
                    model.add_synapse_population(
                        pop_name=f'memExcit{row}{col}', matrix_type='DENSE',
                        source=popVar[row][col], target=popMem[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                    )

            ##### Mem-mem excitation #####
            for row in range(variables):
                for col in range(variables):
                    synapsesExcit = np.zeros(shape=(variables*memoryPop, variables*memoryPop))
                    for i in range(variables):
                        synapsesExcit[i*memoryPop:(i+1)*memoryPop, i*memoryPop:(i+1)*memoryPop] = 0.6
                    model.add_synapse_population(
                        pop_name=f'memInteExcit{row}{col}', matrix_type='DENSE',
                        source=popMem[row][col], target=popMem[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
                    )

            ##### Mem-Mem lateral inhibition #####
            for row in range(variables):
                for col in range(variables):
                    synapsesInhib = -0.3*np.ones(shape=(variables*memoryPop, variables*memoryPop))
                    for i in range(variables):
                        synapsesInhib[i*memoryPop:(i+1)*memoryPop, i*memoryPop:(i+1)*memoryPop] = 0.0
                    model.add_synapse_population(
                        pop_name=f'memInteInhib{row}{col}', matrix_type='DENSE',
                        source=popMem[row][col], target=popMem[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
                    )

            ##### If-Mem inhibition #####
            for row in range(variables):
                for col in range(variables):
                    synapsesInhib = -0.6*np.ones(shape=(2*validationPop, variables*memoryPop))
                    synapsesInhib[0:validationPop, :] = 0
                    model.add_synapse_population(
                        pop_name=f'ifInhiMem{row}{col}', matrix_type='DENSE',
                        source=popIf, target=popMem[row][col],
                        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
                        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
                    )

        ##########################
        # ##### Simulation ##### #
        ##########################
        timeSimulation = 60  # s
        timeSteps = int(timeSimulation*1e3)
        model.build()
        model.load(num_recording_timesteps=timeSteps)

        while model.timestep < timeSteps:
            model.step_time()
        model.pull_recording_buffers_from_device()


        ########################
        # ##### Analysis ##### #
        ########################
        binWidth = 100
        assert timeSteps % binWidth == 0
        binsTime = np.arange(0, timeSteps+binWidth, binWidth)

        ##### Solution/Initial clue validation #####
        binsPop = np.arange(0, (variables+1)*solverPop, solverPop)
        solverBins = np.zeros(shape=(binsTime.size-1, variables, variables), dtype=np.uint8)
        spikeSol = 0
        for row in range(variables):
            for col in range(variables):
                times, index = popVar[row][col].spike_recording_data[0]
                wta = np.histogram2d(index, times, (binsPop, binsTime))[0]
                solverBins[:, row, col] = np.argmax(wta, axis=0)+1
                spikeSol += times.size

        flagSol, timeSol, binSol = 0, 0, 0
        for b in range(binsTime.size-1):
            if np.array_equal(solverBins[b], sudokuSol) is True:
                flagSol = 1
                timeSol = binWidth*(b+1)/1e3
                binSol = b
                break

        flagBug, timeBug, binBug = 0, 0, 0
        sudokuMask = np.where(sudoku == 0, 0, 1)
        for b in range(binsTime.size-1):
            if np.array_equal(solverBins[b]*sudokuMask, sudoku) is False:
                flagBug = 1
                timeBug = binWidth*(b+1)/1e3
                binBug = b
                break

        flagCheck, timeCheck, binCheck, spikeCheck = 0, 0, 0, 0
        flagIf, timeIf, binIf, spikeIf = 0, 0, 0, 0
        flagMem, timeMem, binMem, spikeMem = 0, 0, 0, 0
        if argument.enhanced == 1:
            ##### NetChecker validation #####
            binsPop = np.arange(0, (variables+2)*validationPop, validationPop)
            netCheckerBins = np.zeros(shape=(binsTime.size-1, 3, variables), dtype=int)
            spikeCheck = 0
            for row in range(3):
                for col in range(variables):
                    times, index = popNetChecker[row][col].spike_recording_data[0]
                    wta = np.histogram2d(index, times, (binsPop, binsTime))[0]
                    netCheckerBins[:, row, col] = np.argmax(wta, axis=0)+1
                    spikeCheck += times.size

            for b in range(binsTime.size-1):
                if np.sum(netCheckerBins[b]) == 270:
                    flagCheck = 1
                    timeCheck = binWidth*(b+1)/1e3
                    binCheck = b
                    break

            ##### If validation #####
            binsPop = np.arange(0, 3*validationPop, validationPop)
            ifBins = np.zeros(shape=binsTime.size-1, dtype=int)
            spikeIf = 0
            times, index = popIf.spike_recording_data[0]
            wta = np.histogram2d(index, times, (binsPop, binsTime))[0]
            ifBins[:] = np.argmax(wta, axis=0)+1
            spikeIf += times.size

            for b in range(binsTime.size-1):
                if ifBins[b] == 1:
                    flagIf = 1
                    timeIf = binWidth*(b+1)/1e3
                    binIf = b
                    break

            ##### Memory validation #####
            binsPop = np.arange(0, (variables+1)*memoryPop, memoryPop)
            memoryBins = np.zeros(shape=(binsTime.size-1, variables, variables), dtype=int)
            spikeMem = 0
            for row in range(variables):
                for col in range(variables):
                    times, index = popMem[row][col].spike_recording_data[0]
                    wta = np.histogram2d(index, times, (binsPop, binsTime))[0]
                    memoryBins[:, row, col] = np.argmax(wta, axis=0)+1
                    spikeMem += times.size

            for b in range(binsTime.size-1):
                if np.array_equal(memoryBins[b], sudokuSol) is True:
                    flagMem = 1
                    timeMem = binWidth*(b+1)/1e3
                    binMem = b
                    break

        try:
            file = open(f'../results/GENN/{name}.csv', 'r')
            file.close()
        except:
            file = open(f'../results/GENN/{name}.csv', 'w', newline='')
            file.close()

        file = open(f'../results/GENN/{name}.csv', 'a', newline='')
        write = csv.writer(file)
        write.writerow([
            trial,
            argument.difficulty, argument.puzzle, argument.bugFix, argument.enhanced,
            flagSol, flagBug, flagCheck, flagIf, flagMem,
            timeSol, timeBug, timeCheck, timeIf, timeMem,
            binSol, binBug, binCheck, binIf, binMem,
            spikeSol, spikeCheck, spikeIf, spikeMem,
        ])
        file.close()

        os.system('rm -r .*_CODE*')
    return 0


######################
# ##### Parser ##### #
######################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku solver GENN')

    parser.add_argument('-d', '--difficulty', help='Puzzle difficulty level', type=str, default='easy')
    parser.add_argument('-p', '--puzzle', help='Puzzle type', type=int, default=1)
    parser.add_argument('-t', '--trials', help='Attempt of resolution', type=int, default=301)
    parser.add_argument('-b', '--bugFix', help='Bug fix for changing state of original problem', type=int, default=0)
    parser.add_argument('-e', '--enhanced', help='Use the enhanced pipeline', type=int, default=0)
    parser.add_argument('-w', '--binWidth', help='Binning width', type=int, default=100)
    parser.add_argument('-g', '--gpu', help='GPU or CPU mode', type=int, default=1)

    argument = parser.parse_args()
    main(argument)
