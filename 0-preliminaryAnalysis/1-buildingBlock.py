import os
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
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


def singleNodeWithoutStimulation():
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='980px', width='100%', bgcolor='#222222', font_color='#FFFFFF')

    # ##### Populations neurons ##### #
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size
        )

    # ##### Synaptic connections ##### #
    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v{so}', to=f'v{to}',
                    width=1, color=colors['synInhi']
                )

    graph.toggle_physics(False)
    graph.show('singleVarPopWithoutStim.html')


    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.block1', backend=backend)
    model.dt = 1.0  # ms

    ##### Neuron parameters #####
    paramLif = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.3,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -70.0,  # mV
        'TauRefrac': 2.0,  # ms
    }
    varLif = {
        'V': paramLif['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }

    # ##### Populations neurons ##### #
    neuronsVar = 9
    neuronsPop = 25

    popVar = model.add_neuron_population(
        pop_name='popVar',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar.spike_recording_enabled = True

    # ##### Synaptic connections ##### #
    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib', matrix_type='DENSE',
        source=popVar, target=popVar,
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
    times, index = popVar.spike_recording_data[0]
    popVarRec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVarRec[i].append(t)


    ####################
    # ##### Plot ##### #
    ####################
    figsize = np.array((6.4, 4.8))*2
    plt.figure(figsize=figsize)
    plt.title('Winner-take-all')
    plt.eventplot(popVarRec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])

    return 0


def singleNodeWithOneStimulation():
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='980px', width='100%', bgcolor='#222222', font_color='#FFFFFF')

    # ##### Populations neurons ##### #
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size
        )

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    indexStim = 4
    for i in labels:
        if i == indexStim:
            graph.add_edge(
                source=f's{i}', to=f'v{i}',
                width=5, color=colors['synExci']
            )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v{so}', to=f'v{to}',
                    width=1, color=colors['synInhi']
                )

    graph.toggle_physics(False)
    graph.show('singleVarPopWithOneStim.html')


    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.block2', backend=backend)
    model.dt = 1.0  # ms

    ##### Neuron parameters #####
    paramLif = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.3,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -70.0,  # mV
        'TauRefrac': 2.0,  # ms
    }
    varLif = {
        'V': paramLif['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }

    # ##### Populations neurons ##### #
    neuronsVar = 9
    neuronsPop = 25

    popStim = model.add_neuron_population(
        pop_name='stim',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar = model.add_neuron_population(
        pop_name='popVar',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar.spike_recording_enabled = True

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        if i//neuronsPop == indexStim-1:
            weight = np.random.uniform(low=1.4, high=1.6)
            synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit', matrix_type='DENSE',
        source=popStim, target=popVar,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib', matrix_type='DENSE',
        source=popVar, target=popVar,
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
    times, index = popVar.spike_recording_data[0]
    popVarRec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVarRec[i].append(t)


    ####################
    # ##### Plot ##### #
    ####################
    figsize = np.array((6.4, 4.8))*2
    plt.figure(figsize=figsize)
    plt.title('Single Constraint')
    plt.eventplot(popVarRec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])

    return 0


def singleNodeWithAllStimulation():
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='980px', width='100%', bgcolor='#222222', font_color='#FFFFFF')

    # ##### Populations neurons ##### #
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size
        )

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    for i in labels:
        graph.add_edge(
            source=f's{i}', to=f'v{i}',
            width=5, color=colors['synExci']
        )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v{so}', to=f'v{to}',
                    width=1, color=colors['synInhi']
                )

    graph.toggle_physics(False)
    graph.show('singleVarPopWithAllStim.html')


    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.block3', backend=backend)
    model.dt = 1.0  # ms

    ##### Neuron parameters #####
    paramLif = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.3,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -70.0,  # mV
        'TauRefrac': 2.0,  # ms
    }
    varLif = {
        'V': paramLif['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }

    # ##### Populations neurons ##### #
    neuronsVar = 9
    neuronsPop = 25

    popStim = model.add_neuron_population(
        pop_name='stim',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar = model.add_neuron_population(
        pop_name='popVar',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar.spike_recording_enabled = True

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        weight = np.random.uniform(low=1.4, high=1.6)
        synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit', matrix_type='DENSE',
        source=popStim, target=popVar,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib', matrix_type='DENSE',
        source=popVar, target=popVar,
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
    times, index = popVar.spike_recording_data[0]
    popVarRec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVarRec[i].append(t)


    ####################
    # ##### Plot ##### #
    ####################
    figsize = np.array((6.4, 4.8))*2
    plt.figure(figsize=figsize)
    plt.title('Single Constraint')
    plt.eventplot(popVarRec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])

    return 0


def twoNodesWithOneStimulation():
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='980px', width='100%', bgcolor='#222222', font_color='#FFFFFF')

    # ##### Populations neurons ##### #
    ##### First node ####
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's1{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v1{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size
        )

    ##### Second node ####
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's2{labels[i]}', label=f'{labels[i]}',
            x=x[i]+400, y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v2{labels[i]}', label=f'{labels[i]}',
            x=x[i]+400, y=y[i],
            color=colors['neurVar'], size=size
        )

    # ##### Synaptic connections ##### #
    ##### Synapses first node #####
    ##### Stimulus excitation #####
    for i in labels:
        graph.add_edge(
            source=f's1{i}', to=f'v1{i}',
            width=5, color=colors['synExci']
        )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v1{so}', to=f'v1{to}',
                    width=1, color=colors['synInhi']
                )

    ##### Synapses second node #####
    ##### Stimulus excitation #####
    indexStim = 4
    for i in labels:
        if i == indexStim:
            graph.add_edge(
                source=f's2{i}', to=f'v2{i}',
                width=5, color=colors['synExci']
            )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v2{so}', to=f'v2{to}',
                    width=1, color=colors['synInhi']
                )

    ##### Lateral inhibition #####
    for i in labels:
        graph.add_edge(
            source=f'v1{i}', to=f'v2{i}',
            width=1, color=colors['synInhi']
        )

    graph.toggle_physics(False)
    graph.show('twoNodesWithOneStimulation.html')


    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.block4', backend=backend)
    model.dt = 1.0  # ms

    ##### Neuron parameters #####
    paramLif = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.3,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -70.0,  # mV
        'TauRefrac': 2.0,  # ms
    }
    varLif = {
        'V': paramLif['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }

    # ##### Populations neurons ##### #
    neuronsVar = 9
    neuronsPop = 25

    popStim1 = model.add_neuron_population(
        pop_name='stim1',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar1 = model.add_neuron_population(
        pop_name='popVar1',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar1.spike_recording_enabled = True

    popStim2 = model.add_neuron_population(
        pop_name='stim2',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar2 = model.add_neuron_population(
        pop_name='popVar2',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar2.spike_recording_enabled = True

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        weight = np.random.uniform(low=1.4, high=1.6)
        synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit1', matrix_type='DENSE',
        source=popStim1, target=popVar1,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib1', matrix_type='DENSE',
        source=popVar1, target=popVar1,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )

    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        if i//neuronsPop == indexStim-1:
            weight = np.random.uniform(low=1.4, high=1.6)
            synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit2', matrix_type='DENSE',
        source=popStim2, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib2', matrix_type='DENSE',
        source=popVar2, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )

    ##### Lateral inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop == to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib12', matrix_type='DENSE',
        source=popVar1, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )
    model.add_synapse_population(
        pop_name='synapsesInhib21', matrix_type='DENSE',
        source=popVar2, target=popVar1,
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
    times, index = popVar1.spike_recording_data[0]
    popVar1Rec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVar1Rec[i].append(t)

    times, index = popVar2.spike_recording_data[0]
    popVar2Rec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVar2Rec[i].append(t)


    ####################
    # ##### Plot ##### #
    ####################
    figsize = np.array((6.4, 2.4))*2
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.title('First node')
    plt.eventplot(popVar1Rec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])
    plt.subplot(1, 2, 2)
    plt.title('Second node')
    plt.eventplot(popVar2Rec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])

    return 0


def twoNodesWithTwoStimulation():
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='980px', width='100%', bgcolor='#222222', font_color='#FFFFFF')

    # ##### Populations neurons ##### #
    ##### First node ####
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's1{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v1{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size
        )

    ##### Second node ####
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's2{labels[i]}', label=f'{labels[i]}',
            x=x[i]+400, y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v2{labels[i]}', label=f'{labels[i]}',
            x=x[i]+400, y=y[i],
            color=colors['neurVar'], size=size
        )

    # ##### Synaptic connections ##### #
    ##### Synapses first node #####
    ##### Stimulus excitation #####
    indexStim1 = 4
    for i in labels:
        if i == indexStim1:
            graph.add_edge(
                source=f's1{i}', to=f'v1{i}',
                width=5, color=colors['synExci']
            )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v1{so}', to=f'v1{to}',
                    width=1, color=colors['synInhi']
                )

    ##### Synapses second node #####
    ##### Stimulus excitation #####
    indexStim2 = 7
    for i in labels:
        if i == indexStim2:
            graph.add_edge(
                source=f's2{i}', to=f'v2{i}',
                width=5, color=colors['synExci']
            )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v2{so}', to=f'v2{to}',
                    width=1, color=colors['synInhi']
                )

    ##### Lateral inhibition #####
    for i in labels:
        graph.add_edge(
            source=f'v1{i}', to=f'v2{i}',
            width=1, color=colors['synInhi']
        )

    graph.toggle_physics(False)
    graph.show('twoNodesWithOneStimulation.html')


    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.block5', backend=backend)
    model.dt = 1.0  # ms

    ##### Neuron parameters #####
    paramLif = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.3,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -70.0,  # mV
        'TauRefrac': 2.0,  # ms
    }
    varLif = {
        'V': paramLif['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }

    # ##### Populations neurons ##### #
    neuronsVar = 9
    neuronsPop = 25

    popStim1 = model.add_neuron_population(
        pop_name='stim1',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar1 = model.add_neuron_population(
        pop_name='popVar1',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar1.spike_recording_enabled = True

    popStim2 = model.add_neuron_population(
        pop_name='stim2',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar2 = model.add_neuron_population(
        pop_name='popVar2',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar2.spike_recording_enabled = True

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        if i//neuronsPop == indexStim1-1:
            weight = np.random.uniform(low=1.4, high=1.6)
            synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit1', matrix_type='DENSE',
        source=popStim1, target=popVar1,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib1', matrix_type='DENSE',
        source=popVar1, target=popVar1,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )

    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        if i//neuronsPop == indexStim2-1:
            weight = np.random.uniform(low=1.4, high=1.6)
            synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit2', matrix_type='DENSE',
        source=popStim2, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib2', matrix_type='DENSE',
        source=popVar2, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )

    ##### Lateral inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop == to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib12', matrix_type='DENSE',
        source=popVar1, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )
    model.add_synapse_population(
        pop_name='synapsesInhib21', matrix_type='DENSE',
        source=popVar2, target=popVar1,
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
    times, index = popVar1.spike_recording_data[0]
    popVar1Rec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVar1Rec[i].append(t)

    times, index = popVar2.spike_recording_data[0]
    popVar2Rec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVar2Rec[i].append(t)


    ####################
    # ##### Plot ##### #
    ####################
    figsize = np.array((6.4, 2.4))*2
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.title('First node')
    plt.eventplot(popVar1Rec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])
    plt.subplot(1, 2, 2)
    plt.title('Second node')
    plt.eventplot(popVar2Rec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])

    return 0


def twoNodesWithAllStimulation():
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='980px', width='100%', bgcolor='#222222', font_color='#FFFFFF')

    # ##### Populations neurons ##### #
    ##### First node ####
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's1{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v1{labels[i]}', label=f'{labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size
        )

    ##### Second node ####
    ##### Stimulus neurons #####
    radius, phi = 75, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's2{labels[i]}', label=f'{labels[i]}',
            x=x[i]+400, y=y[i],
            color=colors['neurStim'], size=size
        )

    ##### Variables neurons #####
    radius, phi = 150, -np.pi/2
    x = radius*np.cos(angles+phi)
    y = radius*np.sin(angles+phi)
    labels = [i+1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v2{labels[i]}', label=f'{labels[i]}',
            x=x[i]+400, y=y[i],
            color=colors['neurVar'], size=size
        )

    # ##### Synaptic connections ##### #
    ##### Synapses first node #####
    ##### Stimulus excitation #####
    for i in labels:
        graph.add_edge(
            source=f's1{i}', to=f'v1{i}',
            width=5, color=colors['synExci']
        )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v1{so}', to=f'v1{to}',
                    width=1, color=colors['synInhi']
                )

    ##### Synapses second node #####
    ##### Stimulus excitation #####
    for i in labels:
        graph.add_edge(
            source=f's2{i}', to=f'v2{i}',
            width=5, color=colors['synExci']
        )

    ##### Internal inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v2{so}', to=f'v2{to}',
                    width=1, color=colors['synInhi']
                )

    ##### Lateral inhibition #####
    for i in labels:
        graph.add_edge(
            source=f'v1{i}', to=f'v2{i}',
            width=1, color=colors['synInhi']
        )

    graph.toggle_physics(False)
    graph.show('twoNodesWithOneStimulation.html')


    ################################
    # ##### Model definition ##### #
    ################################
    model = GeNNModel(precision='float', model_name=f'.block6', backend=backend)
    model.dt = 1.0  # ms

    ##### Neuron parameters #####
    paramLif = {
        'C': 0.25,  # nF
        'TauM': 20.0,  # ms
        'Ioffset': 0.3,  # nA
        'Vrest': -65.0,  # mV
        'Vthresh': -50.0,  # mV
        'Vreset': -70.0,  # mV
        'TauRefrac': 2.0,  # ms
    }
    varLif = {
        'V': paramLif['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }


    # ##### Populations neurons ##### #
    neuronsVar = 9
    neuronsPop = 25

    popStim1 = model.add_neuron_population(
        pop_name='stim1',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar1 = model.add_neuron_population(
        pop_name='popVar1',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar1.spike_recording_enabled = True

    popStim2 = model.add_neuron_population(
        pop_name='stim2',
        num_neurons=neuronsVar*neuronsPop,
        neuron='Poisson',
        params={'rate': 20}, vars={'timeStepToSpike': 0}
    )

    popVar2 = model.add_neuron_population(
        pop_name='popVar2',
        num_neurons=neuronsVar*neuronsPop,
        neuron='LIF',
        params=paramLif, vars=varLif
    )
    popVar2.spike_recording_enabled = True

    # ##### Synaptic connections ##### #
    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        weight = np.random.uniform(low=1.4, high=1.6)
        synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit1', matrix_type='DENSE',
        source=popStim1, target=popVar1,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib1', matrix_type='DENSE',
        source=popVar1, target=popVar1,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )

    ##### Stimulus excitation #####
    synapsesExcit = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for i in range(neuronsVar*neuronsPop):
        weight = np.random.uniform(low=1.4, high=1.6)
        synapsesExcit[i, i] = weight
    model.add_synapse_population(
        pop_name='synapsesExcit2', matrix_type='DENSE',
        source=popStim2, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesExcit.flatten()}),
    )

    ##### Internal inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop != to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib2', matrix_type='DENSE',
        source=popVar2, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )

    ##### Lateral inhibition #####
    synapsesInhib = np.zeros(shape=(neuronsVar*neuronsPop, neuronsVar*neuronsPop))
    for so in range(neuronsVar*neuronsPop):
        for to in range(neuronsVar*neuronsPop):
            if so//neuronsPop == to//neuronsPop:
                weight = np.random.uniform(low=-0.2/2.5, high=0.0)
                synapsesInhib[so, to] = weight
    model.add_synapse_population(
        pop_name='synapsesInhib12', matrix_type='DENSE',
        source=popVar1, target=popVar2,
        postsynaptic_init=init_postsynaptic('ExpCurr', {"tau": 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': synapsesInhib.flatten()}),
    )
    model.add_synapse_population(
        pop_name='synapsesInhib21', matrix_type='DENSE',
        source=popVar2, target=popVar1,
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
    times, index = popVar1.spike_recording_data[0]
    popVar1Rec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVar1Rec[i].append(t)

    times, index = popVar2.spike_recording_data[0]
    popVar2Rec = [[] for _ in range(neuronsVar*neuronsPop)]
    for t, i in list(zip(times, index)):
        popVar2Rec[i].append(t)


    ####################
    # ##### Plot ##### #
    ####################
    figsize = np.array((6.4, 2.4))*2
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.title('First node')
    plt.eventplot(popVar1Rec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])
    plt.subplot(1, 2, 2)
    plt.title('Second node')
    plt.eventplot(popVar2Rec)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
    labels = [i+1 for i in range(neuronsVar)]
    plt.yticks(ticks=ticks, labels=labels)
    offset = 3000
    plt.xlim([0-offset, timeSteps+offset])

    return 0


if __name__ == '__main__':
    ##############################
    # ##### Sudoku Problem ##### #
    ##############################
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

    ##### General variables #####
    variables = 9
    sudoku = sudokuEasy
    sudokuSol = sudokuEasySol

    colors = {
        'neurStim': '#FF5733',
        'neurVar':  '#C9E3AC',
        'synInhi':  '#1192E8',
        'synExci':  '#FA4D56'
    }
    size = 20
    angles = np.linspace(start=0, stop=2*np.pi, num=variables, endpoint=False)

    sudokuPlot()
    singleNodeWithoutStimulation()
    singleNodeWithOneStimulation()
    singleNodeWithAllStimulation()
    twoNodesWithOneStimulation()
    twoNodesWithTwoStimulation()
    twoNodesWithAllStimulation()

    plt.show()

    os.system("rm -r .*_CODE")
    os.system("rm -r *.html")
    os.system("rm -r lib")
