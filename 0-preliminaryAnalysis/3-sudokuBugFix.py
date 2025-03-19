import os
import numpy as np
import matplotlib.pyplot as plt
from pygenn import GeNNModel, init_postsynaptic, init_weight_update


####################################
# ##### Environment variable ##### #
####################################
os.environ['CUDA_PATH'] = '/usr/local/cuda'; backend = 'cuda'
# backend = 'single_threaded_cpu'


def sudokuPlot():
    ##### Sudoku plot #####
    space = 1000
    axes = np.linspace(start=-space, stop=space, num=variables)
    offset = space/(variables-1)
    positions = [(x, y) for y in np.flip(axes) for x in axes]

    figsize, digitsize, maxwidth = 4.8, 200, 3
    figsize = np.array((figsize, figsize))*2
    plt.figure(figsize=figsize)

    plt.title('Sudoku', fontdict={'fontsize': 30})
    for i, (x, y) in enumerate(positions):
        digit = sudoku.flatten()[i]
        if digit != 0:
            plt.scatter(x=x, y=y, marker=f'${digit}$', color='#000000', s=digitsize)

    for i, axis in enumerate(axes):
        linewidth = 1
        if i%3 == 0:
            linewidth = maxwidth
        plt.vlines(x=axis-offset, ymin=-space-offset, ymax=space+offset, linewidth=linewidth, colors='#000000')
        plt.hlines(y=axis-offset, xmin=-space-offset, xmax=space+offset, linewidth=linewidth, colors='#000000')
    plt.vlines(x=axes[-1]+offset, ymin=-space-offset, ymax=space+offset, linewidth=maxwidth, colors='#000000')
    plt.hlines(y=axes[-1]+offset, xmin=-space-offset, xmax=space+offset, linewidth=maxwidth, colors='#000000')

    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    ##### Sudoku solution plot #####
    figsize, digitsize, maxwidth = 4.8, 200, 3
    figsize = np.array((figsize, figsize))*2
    plt.figure(figsize=figsize)

    plt.title('Sudoku Solution', fontdict={'fontsize': 30})
    for i, (x, y) in enumerate(positions):
        digit = sudokuSol.flatten()[i]
        if digit != 0:
            plt.scatter(x=x, y=y, marker=f'${digit}$', color='#005BB8', s=digitsize)
        digit = sudoku.flatten()[i]
        if digit != 0:
            plt.scatter(x=x, y=y, marker=f'${digit}$', color='#000000', s=digitsize)

    for i, axis in enumerate(axes):
        linewidth = 1
        if i%3 == 0:
            linewidth = maxwidth
        plt.vlines(x=axis-offset, ymin=-space-offset, ymax=space+offset, linewidth=linewidth, colors='#000000')
        plt.hlines(y=axis-offset, xmin=-space-offset, xmax=space+offset, linewidth=linewidth, colors='#000000')
    plt.vlines(x=axes[-1]+offset, ymin=-space-offset, ymax=space+offset, linewidth=maxwidth, colors='#000000')
    plt.hlines(y=axes[-1]+offset, xmin=-space-offset, xmax=space+offset, linewidth=maxwidth, colors='#000000')

    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    return 0


def sudokuSolver():
    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.sudokuBugFix', backend=backend)
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
    solverPop = sudokuEasyPop

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
            synapsesInhib = np.random.uniform(low=-0.2/2.5, high=0.0, size=(variables*solverPop, variables*solverPop))
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
        if sudoku[rowTo][colTo] == 0:
            synapsesInhib = np.zeros(shape=(variables*solverPop, variables*solverPop))
            for i in range(variables):
                synapsesInhib[i*solverPop:(i+1)*solverPop, i*solverPop:(i+1)*solverPop] = np.random.uniform(low=-0.2/2.5, high=0.0, size=(solverPop, solverPop))
            model.add_synapse_population(
                pop_name=f'lateralInhib{rowSo}{colSo}{rowTo}{colTo}', matrix_type='DENSE',
                source=popVar[rowSo][colSo], target=popVar[rowTo][colTo],
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

    flagSol, timeSol = False, None
    for b in range(binsTime.size-1):
        if np.array_equal(solverBins[b], sudokuSol) is True:
            flagSol = True
            timeSol = binWidth*(b+1)/1e3
            break

    flagBug, timeBug = False, None
    sudokuMask = np.where(sudoku == 0, 0, 1)
    for b in range(binsTime.size-1):
        if np.array_equal(solverBins[b]*sudokuMask, sudoku) is False:
            flagBug = True
            timeBug = binWidth*(b+1)/1e3
            break

    print('Evaluation metrics')
    print(f'+-------------------------------+')
    print(f'|\tTime simulation\t|\t{timeSimulation} s\t|')
    print(f'+-------------------------------+')
    print(f'|\tSolution found\t|\t{flagSol}\t|')
    print(f'+-------------------------------+')
    print(f'|\tSolution time\t|\t{timeSol} s\t|')
    print(f'+-------------------------------+')
    print(f'|\tSpike Counting\t|\t{spikeSol}\t|')
    print(f'+-------------------------------+')
    print(f'|\tClue change \t|\t{flagBug}\t|')
    print(f'+-------------------------------+')
    print(f'|\tClue time   \t|\t{timeBug} s\t|')
    print(f'+-------------------------------+')


    ####################
    # ##### Plot ##### #
    ####################
    popVarRec = [[0 for _ in range(variables)] for _ in range(variables)]
    for row in range(variables):
        for col in range(variables):
            times, index = popVar[row][col].spike_recording_data[0]
            records = [[] for _ in range(variables*solverPop)]
            for t, i in list(zip(times, index)):
                records[i].append(t)
            popVarRec[row][col] = records

    figsize = np.array((4.8, 4.8))*2
    fig, axs = plt.subplots(nrows=variables, ncols=variables, figsize=figsize)
    for row in range(variables):
        for col in range(variables):
            axs[row, col].eventplot(popVarRec[row][col])
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_xlim([0, timeSteps])
    fig.subplots_adjust(wspace=0, hspace=0)

    return 0


if __name__ == '__main__':
    ##############################
    # ##### Sudoku problem ##### #
    ##############################
    # http://lipas.uwasa.fi/~timan/sudoku/
    sudokuEasyPop = 27
    sudokuEasy = np.array([
        [0, 4, 0, 8, 0, 5, 2, 0, 0],
        [0, 2, 0, 0, 4, 0, 0, 5, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 9, 0, 0, 0, 3, 1, 2, 0],
        [1, 0, 6, 0, 7, 8, 0, 0, 3],
        [3, 7, 0, 9, 0, 4, 0, 8, 0],
        [0, 0, 0, 0, 0, 6, 7, 0, 0],
        [0, 0, 8, 3, 5, 9, 0, 1, 0],
        [0, 1, 9, 0, 0, 7, 6, 0, 0]
    ])
    sudokuEasySol = np.array([
        [9, 4, 7, 8, 3, 5, 2, 6, 1],
        [6, 2, 3, 7, 4, 1, 8, 5, 9],
        [5, 8, 1, 6, 9, 2, 3, 7, 4],
        [8, 9, 4, 5, 6, 3, 1, 2, 7],
        [1, 5, 6, 2, 7, 8, 9, 4, 3],
        [3, 7, 2, 9, 1, 4, 5, 8, 6],
        [4, 3, 5, 1, 2, 6, 7, 9, 8],
        [7, 6, 8, 3, 5, 9, 4, 1, 2],
        [2, 1, 9, 4, 8, 7, 6, 3, 5]
    ])

    sudokuMedPop = 27
    sudokuMed = np.array([
        [9, 0, 4, 0, 0, 0, 5, 0, 1],
        [6, 0, 0, 5, 0, 0, 0, 0, 7],
        [2, 5, 7, 8, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 6],
        [1, 0, 0, 0, 0, 0, 7, 0, 3],
        [0, 0, 0, 3, 9, 6, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 6, 0, 0, 8, 0],
        [0, 6, 0, 9, 2, 8, 0, 7, 4]
    ])
    sudokuMedSol = np.array([
        [9, 8, 4, 6, 7, 2, 5, 3, 1],
        [6, 1, 3, 5, 4, 9, 8, 2, 7],
        [2, 5, 7, 8, 3, 1, 6, 4, 9],
        [8, 3, 2, 1, 5, 7, 4, 9, 6],
        [1, 9, 6, 2, 8, 4, 7, 5, 3],
        [7, 4, 5, 3, 9, 6, 2, 1, 8],
        [3, 7, 8, 4, 1, 5, 9, 6, 2],
        [4, 2, 9, 7, 6, 3, 1, 8, 5],
        [5, 6, 1, 9, 2, 8, 3, 7, 4]
    ])

    sudokuAIEscargotPop = 30
    sudokuAIEscargot = np.array([
        [1, 0, 0, 0, 0, 7, 0, 9, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 8],
        [0, 0, 9, 6, 0, 0, 5, 0, 0],
        [0, 0, 5, 3, 0, 0, 9, 0, 0],
        [0, 1, 0, 0, 8, 0, 0, 0, 2],
        [6, 0, 0, 0, 0, 4, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 7, 0, 0, 0, 3, 0, 0]
    ])
    sudokuAIEscargotSol = np.array([
        [1, 6, 2, 8, 5, 7, 4, 9, 3],
        [5, 3, 4, 1, 2, 9, 6, 7, 8],
        [7, 8, 9, 6, 4, 3, 5, 2, 1],
        [4, 7, 5, 3, 1, 2, 9, 8, 6],
        [9, 1, 3, 5, 8, 6, 7, 4, 2],
        [6, 2, 8, 7, 9, 4, 1, 3, 5],
        [3, 5, 6, 4, 7, 8, 2, 1, 9],
        [2, 4, 1, 9, 3, 5, 8, 6, 7],
        [8, 9, 7, 2, 6, 1, 3, 5, 4]
    ])

    ##### Configurations #####
    variables = 9
    sudokuPop = sudokuEasyPop
    sudoku = sudokuEasy
    sudokuSol = sudokuEasySol

    sudokuPlot()
    sudokuSolver()

    plt.show()

    os.system("rm -r .*_CODE")
