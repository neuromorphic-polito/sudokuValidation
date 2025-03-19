import os
import numpy as np
from pygenn import GeNNModel, init_postsynaptic, init_weight_update
import matplotlib.pyplot as plt


####################################
# ##### Environment variable ##### #
####################################
os.environ['CUDA_PATH'] = '/usr/local/cuda'; backend = 'cuda'
# backend = 'single_threaded_cpu'


################################
# ##### Model definition ##### #
################################
model = GeNNModel(precision='float', model_name=f'.wta', backend=backend)
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
neuronsVar = 9
neuronsPop = 25

popVar = model.add_neuron_population(
    pop_name='popVar',
    num_neurons=neuronsVar*neuronsPop,
    neuron='LIF',
    params=lifParam, vars=lifVar
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
popVarRecords = [[] for _ in range(neuronsVar*neuronsPop)]
for t, i in list(zip(times, index)):
    popVarRecords[i].append(t)


####################
# ##### Plot ##### #
####################
figsize = np.array((6.4, 4.8))*2
plt.figure(figsize=figsize)
plt.title('Winner-take-all')
plt.eventplot(popVarRecords)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
ticks = [i for i in range(neuronsPop//2, neuronsVar*neuronsPop, neuronsPop)]
labels = [i for i in range(neuronsVar)]
plt.yticks(ticks=ticks, labels=labels)
offset = 3000
plt.xlim([0-offset, timeSteps+offset])
plt.show()

os.system("rm -r .*_CODE")
