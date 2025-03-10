import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
import matplotlib.image as mpimg


def sudokuPuzzlePlot(sudoku):
    ###########################
    # ##### Sudoku plot ##### #
    ###########################
    space = 1000
    axes = np.linspace(start=-space, stop=space, num=variables)
    offset = space / (variables - 1)
    positions = [(x, y) for y in np.flip(axes) for x in axes]

    figsize, digitsize, maxwidth = 4.8, 500, 3
    figsize = np.array((figsize, figsize)) * 2
    plt.figure(figsize=figsize)

    for i, (x, y) in enumerate(positions):
        digit = sudoku.flatten()[i]
        if digit != 0:
            plt.scatter(x=x, y=y, marker=f'${digit}$', color='#000000', s=digitsize)

    for i, axis in enumerate(axes):
        linewidth = 1
        if i % 3 == 0:
            linewidth = maxwidth
        plt.vlines(x=axis - offset, ymin=-space - offset, ymax=space + offset, linewidth=linewidth, colors='#000000')
        plt.hlines(y=axis - offset, xmin=-space - offset, xmax=space + offset, linewidth=linewidth, colors='#000000')
    plt.vlines(x=axes[-1] + offset, ymin=-space - offset, ymax=space + offset, linewidth=maxwidth, colors='#000000')
    plt.hlines(y=axes[-1] + offset, xmin=-space - offset, xmax=space + offset, linewidth=maxwidth, colors='#000000')

    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.tight_layout()

    plt.savefig('img/sudokuPlot.png')
    plt.show()

    return 0


def sudokuNetPlot(colors, sudoku):
    graph = Network(height='960px', width='50%', bgcolor='#F7F7F7', font_color='#000000')

    spacing, numCells, partition = 1300, 9, 3
    indexCells = [(y, x) for x in range(int(numCells / partition)) for y in range(int(numCells / partition))]
    axis = np.linspace(-spacing, spacing, partition)
    xOffset = [axis[x] for x, _ in indexCells]
    yOffset = [axis[y] for _, y in indexCells]

    spacingSub, numSubCells, partition = 400, 9, 3
    indexSubCells = [(y, x) for x in range(int(numSubCells / partition)) for y in range(int(numSubCells / partition))]
    axis = np.linspace(-spacingSub, spacingSub, partition)
    xSubOffset = [axis[x] for x, _ in indexSubCells]
    ySubOffset = [axis[y] for _, y in indexSubCells]

    positions = []
    for i1 in range(numCells):
        for i2 in range(numSubCells):
            positions.append((xOffset[i1] + xSubOffset[i2], yOffset[i1] + ySubOffset[i2]))
    positions = sorted(positions, key=lambda x: (x[1], x[0]))

    ##### Stimulation neurons #####
    variables, radius, phi, size = 9, 75, -np.pi / 2 + 0.17, 15
    angles = np.linspace(0, 2 * np.pi, variables, endpoint=False)
    x = radius * (np.cos(angles + phi) - np.cos(phi - 0.17))
    y = radius * (np.sin(angles + phi))
    labels = list(range(1, variables + 1))
    for row in range(variables):
        for col in range(variables):
            if sudoku[row, col] == 0:
                for stim in range(variables):
                    xPos, yPos = positions[row * variables + col]
                    graph.add_node(
                        n_id=f'row{row}_col{col}_stim{stim}', label=labels[stim],
                        x=xPos + x[stim], y=yPos + y[stim],
                        color=colors['neurStim'], size=size
                    )
            else:
                stim = sudoku[row, col] - 1
                xPos, yPos = positions[row * variables + col]
                graph.add_node(
                    n_id=f'row{row}_col{col}_stim{stim}', label=labels[stim],
                    x=xPos + x[stim], y=yPos + y[stim],
                    color=colors['neurStim'], size=size
                )

    ##### Variables neurons #####
    variables, radius, phi, size = 9, 150, -np.pi / 2 + 0.17, 15
    angles = np.linspace(0, 2 * np.pi, variables, endpoint=False)
    x = radius * (np.cos(angles + phi) - np.cos(phi - 0.17))
    y = radius * (np.sin(angles + phi))
    labels = list(range(1, variables + 1))
    for row in range(variables):
        for col in range(variables):
            for var in range(variables):
                xPos, yPos = positions[row * variables + col]
                graph.add_node(
                    n_id=f'row{row}_col{col}_var{var}', label=labels[var],
                    x=xPos + x[var], y=yPos + y[var],
                    color=colors['neurVar'], size=size
                )

    ##### Synaptic connection #####
    indexes = [(r, c) for r in range(variables) for c in range(variables)]
    constraints = []
    for i1, row in enumerate(indexes):
        for i2, col in enumerate(indexes):
            if (row[0] == col[0] or row[1] == col[1]) and i1 != i2:
                constraints.append((row, col))
            if (row[0] // 3 == col[0] // 3 and row[1] // 3 == col[1] // 3) and (
                    row[0] != col[0] and row[1] != col[1]) and i1 != i2:
                constraints.append((row, col))

    for so, to in constraints:
        if so == (0, 0):
            for k in range(variables):
                graph.add_edge(
                    source=f'row{so[0]}_col{so[1]}_var{k}', to=f'row{to[0]}_col{to[1]}_var{k}',
                    color=colors['synInhiLat'], width=1
                )

    for i1 in range(variables):
        for i2 in range(variables):
            for var1 in range(variables):
                for var2 in range(variables):
                    if var1 != var2:
                        graph.add_edge(
                            source=f'row{i1}_col{i2}_var{var1}', to=f'row{i1}_col{i2}_var{var2}',
                            color=colors['synInhiInt'], width=1
                        )

    for row in range(variables):
        for col in range(variables):
            if sudoku[row, col] == 0:
                for var in range(variables):
                    graph.add_edge(
                        source=f'row{row}_col{col}_stim{var}', to=f'row{row}_col{col}_var{var}',
                        color=colors['synExci'], width=5
                    )
            else:
                var = sudoku[row, col] - 1
                graph.add_edge(
                    source=f'row{row}_col{col}_stim{var}', to=f'row{row}_col{col}_var{var}',
                    color=colors['synExci'], width=5
                )

    graph.toggle_physics(False)
    graph.show('sudokuNetPlot.html')

    return 0


def baseNodesPlot(colors):
    ####################################
    # ##### Graph representation ##### #
    ####################################
    ##### Graph settings #####
    graph = Network(height='550px', width='50%', bgcolor='#F7F7F7', font_color='#000000')

    variables = 9
    radiusStim, radiusVar, phi = 150, 250, -np.pi / 2 + 0.17
    synStim, synInhi, = 7, 3
    angles = np.linspace(0, 2 * np.pi, variables, endpoint=False)

    size = 25
    offsetSecond = 700

    x0 = np.cos(angles + phi) + np.cos(phi - 0.17)
    y0 = np.sin(angles + phi)

    ##### First Node ####
    ##### Stimulation neurons #####
    x = radiusStim * x0
    y = radiusStim * y0
    labels = [i + 1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f's1{labels[i]}', label=f' ',  # {labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurStim'], size=size,  # shape='circle'
        )
    ##### Variables neurons #####
    x = radiusVar * x0
    y = radiusVar * y0
    labels = [i + 1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v1{labels[i]}', label=f' ',  # {labels[i]}',
            x=x[i], y=y[i],
            color=colors['neurVar'], size=size,  # shape='circle'
        )

    ##### Second Node ####
    ##### Stimulation neurons #####
    x = radiusStim * x0
    y = radiusStim * y0
    labels = [i + 1 for i in range(variables)]
    indexStim2 = 4
    for i in range(variables):
        if i == indexStim2 - 1:
            graph.add_node(
                n_id=f's2{labels[i]}', label=f' ',  # {labels[i]}',
                x=x[i] + offsetSecond, y=y[i],
                color=colors['neurStim'], size=size,  # shape='circle'
            )
    ##### Variables neurons #####
    x = radiusVar * x0
    y = radiusVar * y0
    labels = [i + 1 for i in range(variables)]
    for i in range(variables):
        graph.add_node(
            n_id=f'v2{labels[i]}', label=f' ',  # {labels[i]}',
            x=x[i] + offsetSecond, y=y[i],
            color=colors['neurVar'], size=size,  # shape='circle'
        )

    ##### Synapses first node #####
    ##### Synapses neuron variable inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v1{so}', to=f'v1{to}',
                    width=synInhi, color=colors['synInhiInt']
                )
    ##### Synapses neuron stimulus excitation #####
    for i in labels:
        graph.add_edge(
            source=f's1{i}', to=f'v1{i}',
            width=synStim, color=colors['synExci']
        )

    ##### Synapses second node #####
    ##### Synapses neuron variable inhibition #####
    for so in labels:
        for to in labels:
            if so != to:
                graph.add_edge(
                    source=f'v2{so}', to=f'v2{to}',
                    width=synInhi, color=colors['synInhiInt']
                )
    ##### Synapses neuron stimulus excitation #####
    for i in labels:
        if i == indexStim2:
            graph.add_edge(
                source=f's2{i}', to=f'v2{i}',
                width=synStim, color=colors['synExci']
            )

    ##### Synapses between node #####
    for i in labels:
        graph.add_edge(
            source=f'v1{i}', to=f'v2{i}',
            width=synInhi, color=colors['synInhiLat']
        )

    graph.toggle_physics(False)
    # graph.show('baseNodes.html')

    plt.figure(figsize=(6.4 * 2, 4.8 * 1.4))
    img = mpimg.imread('img/baseNode.png')
    plt.imshow(img)

    yCenter, xCenter = img.shape[0:2]
    yCenter, xCenter = yCenter / 2 - 6, xCenter / 2
    yLength, xLength = 240, 40

    plt.scatter(x=xCenter+230, y=yCenter, marker=f'$4$', color='#000000', s=70000, alpha=0.3)

    ##### Label left side #####
    radius, phi = 102, -np.pi/2+0.173
    angles = np.linspace(0, 2*np.pi, variables, endpoint=False)
    x = radius*(np.cos(angles+phi)+np.cos(phi-0.173))
    y = radius*np.sin(angles+phi)
    for i in range(variables):
        xPos = x[i]+xCenter-244
        yPos = y[i]+yCenter
        plt.scatter(x=xPos, y=yPos, marker=f'${i+1}$', color='#000000', s=200, alpha=1)
    radius, phi = 169.8, -np.pi/2+0.173
    angles = np.linspace(0, 2*np.pi, variables, endpoint=False)
    x = radius*(np.cos(angles+phi)+np.cos(phi-0.173))
    y = radius*np.sin(angles+phi)
    for i in range(variables):
        xPos = x[i]+xCenter-244
        yPos = y[i]+yCenter
        plt.scatter(x=xPos, y=yPos, marker=f'${i+1}$', color='#000000', s=200, alpha=1)

    ##### Label right side #####
    radius, phi = 102, -np.pi/2+0.173
    angles = np.linspace(0, 2*np.pi, variables, endpoint=False)
    x = radius*(np.cos(angles+phi)+np.cos(phi-0.173))
    y = radius*np.sin(angles+phi)
    for i in range(variables):
        if i == indexStim2-1:
            xPos = x[i]+xCenter+233
            yPos = y[i]+yCenter
            plt.scatter(x=xPos, y=yPos, marker=f'${i+1}$', color='#000000', s=200, alpha=1)
    radius, phi = 169.8, -np.pi/2+0.173
    angles = np.linspace(0, 2*np.pi, variables, endpoint=False)
    x = radius*(np.cos(angles+phi)+np.cos(phi-0.173))
    y = radius*np.sin(angles+phi)
    for i in range(variables):
        xPos = x[i]+xCenter+233
        yPos = y[i]+yCenter
        plt.scatter(x=xPos, y=yPos, marker=f'${i + 1}$', color='#000000', s=200, alpha=1)

    plt.vlines(x=0, ymin=yCenter-yLength, ymax=yCenter+yLength, linewidth=2, colors='#000000')
    plt.vlines(x=xCenter, ymin=yCenter-yLength, ymax=yCenter+yLength, linewidth=2, colors='#000000')
    plt.vlines(x=xCenter*2, ymin=yCenter-yLength, ymax=yCenter+yLength, linewidth=2, colors='#000000')

    plt.hlines(y=yCenter-200, xmin=0-xLength, xmax=xCenter*2+xLength, linewidth=2, colors='#000000')
    plt.hlines(y=yCenter+200, xmin=0-xLength, xmax=xCenter*2+xLength, linewidth=2, colors='#000000')

    plt.xticks([])
    plt.xlim([-100, xCenter * 2 + 100])
    plt.yticks([])
    plt.box(False)
    plt.tight_layout()

    plt.savefig('img/sudokuPlotDetail.png')

    # plt.show()

    return 0


if __name__ == '__main__':
    variables = 9

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

    colors = {
        'neurStim': '#FF5733',
        'neurVar': '#C9E3AC',

        'synExci': '#FA4D56',

        'synInhiInt': '#1192E8',
        'synInhiLat': '#877EFF',

    }

    sudokuPuzzlePlot(sudokuEasy)
    sudokuNetPlot(colors, sudokuEasy)
    baseNodesPlot(colors)
