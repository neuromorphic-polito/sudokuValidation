import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.image as mpimg
import csv


def successRatePlot(dataset):
    ############################
    # ##### Success rate ##### #
    ############################
    successRate = pd.pivot_table(
        dataset,
        index='puzzle', columns=['difficulty', 'bugFix', 'enhanced'],
        values='flagSol',
        aggfunc='sum'
    )
    print(successRate)

    configurations = [[b, e] for e in [0, 1] for b in [0, 1]]
    confusionMatrix = np.zeros((4, 3))
    for (r, (b, e)) in enumerate(configurations):
        for c, puzzle in enumerate(['easy', 'med', 'hard']):
                confusionMatrix[r, c] = np.sum(successRate[puzzle, b, e])/9

    plt.figure(figsize=(6.4, 4.8))

    xLabel = ['$Easy$', '$Medium$', '$Hard$']
    yLabel = ['$a$', '$b$', '$c$', '$d$']
    ax = plt.subplot(1, 1, 1)
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Greens, aspect='auto', vmin=0, vmax=100)
    for r in range(len(yLabel)):
        for c in range(len(xLabel)):
            value = round(confusionMatrix[r, c], 2)
            color = '#000000'
            if value > 50:
                color = '#FFFFFF'
            plt.text(c, r, f'{value:.02f}%', va='center', ha='center', size=baseFontSize-2, color=color)

    ax.set_xlabel('\n$Class$', fontsize=baseFontSize+1)
    ax.set_xticks(np.arange(len(xLabel)))
    ax.set_xticklabels(xLabel, fontsize=baseFontSize, rotation=0)
    ax.set_ylabel('$Strategy$\n', fontsize=baseFontSize+1)
    ax.set_yticks(np.arange(len(yLabel)))
    ax.set_yticklabels(yLabel, fontsize=baseFontSize)

    plt.tight_layout()
    plt.savefig('../article/image/GENN-SuccessRateMatrix.png')
    
    return 0


def spikeCountPlot(dataset):
    ##################################
    # ##### Spike Count Median ##### #
    ##################################
    spikeCountSol = pd.pivot_table(
        dataset,
        columns=['difficulty', 'enhanced', 'bugFix'],
        values='spikeSol',
        aggfunc='median'
    )
    print(spikeCountSol)
    spikeCountCheck = pd.pivot_table(
        dataset,
        columns=['difficulty', 'enhanced', 'bugFix'],
        values='spikeCheck',
        aggfunc='median'
    )
    spikeCountIf = pd.pivot_table(
        dataset,
        columns=['difficulty', 'enhanced', 'bugFix'],
        values='spikeIf',
        aggfunc='median'
    )
    spikeCountMem = pd.pivot_table(
        dataset,
        columns=['difficulty', 'enhanced', 'bugFix'],
        values='spikeMem',
        aggfunc='median'
    )

    xLabel = ['$Easy$', '$Medium$', '$Hard$']
    yLabel = ['$a$', '$b$', '$c$', '$d$']
    labels = ['$CSP$ $Solver$', '$Polisher+NetChecker$', '$If$', '$Memory$']
    # colors = ['#08519C', '#3182BD', '#6BAED6', '#BDD7E7']
    # colors = ['#003F5C', '#58508D', '#BC5090', '#FFA600']
    # colors = ['#D62828', '#EF9034', '#64CFDF', '#245484'][::-1]
    # colors = ['#003049', '#D62828', '#F77F00', '#FCBF49']
    colors = ['#264653', '#2A9D8F', '#E76F51', '#F4A261']
    legend(colors, labels)

    frame = plt.figure(figsize=(6.4*3.5, 4.8))
    for c, puzzle in enumerate(['easy', 'med', 'hard']):
        base = np.zeros(4)
        ax = plt.subplot(1, 3, c+1)

        plt.title(f'{xLabel[c]}', fontdict={'fontsize': baseFontSize+3})

        values = spikeCountSol[puzzle].values[0]
        plt.bar([0, 1, 2, 3], height=values, bottom=base, color=colors[0], zorder=3)
        base += values

        values = spikeCountCheck[puzzle].values[0]
        plt.bar([0, 1, 2, 3], height=values, bottom=base, color=colors[1], zorder=3)
        base += values

        values = spikeCountIf[puzzle].values[0]
        plt.bar([0, 1, 2, 3], height=values, bottom=base, color=colors[2], zorder=3)
        base += values

        values = spikeCountMem[puzzle].values[0]
        plt.bar([0, 1, 2, 3], height=values, bottom=base, color=colors[3], zorder=3)
        base += values

        plt.xticks([0, 1, 2, 3], labels=yLabel, fontsize=baseFontSize)
        plt.xlabel('$Strategy$', fontdict={'fontsize': baseFontSize+1})

        plt.ylim([0, 1.3e7])
        if c == 0:
            plt.ylabel('$Median$ $spike$ $count$\n', fontdict={'fontsize': baseFontSize+1})

        # plt.legend(, loc="upper center", ncol=4, fontsize=baseFontSize-4.5)

        ax.set_facecolor('#F7F7F7')

        ax.yaxis.set_major_locator(MultipleLocator(2e6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(zorder=0, which='major', color='#CCCCCC', linestyle='solid')
        ax.grid(zorder=0, which='minor', color='#CCCCCC', linestyle='dashed')

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        if c > 0:
            ax.yaxis.set_major_formatter(NullFormatter())

    img = mpimg.imread('../article/image/legend.png')
    newax = frame.add_axes([0.32, 0.0, 0.4, 0.4], anchor='S', zorder=3)
    newax.imshow(img)
    newax.axis('off')
    plt.subplots_adjust(bottom=0.32)

    plt.savefig('../article/image/GENN-SpikeCountMedian.png', bbox_inches='tight')

    return 0


def legend(colors, labels):

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls='none')[0]
    handles = [f('s', colors[i]) for i in range(len(colors))]
    legend = plt.legend(handles, labels, loc='center', ncol=4, fontsize=baseFontSize-4.5, frameon=True)
    plt.box(False)
    plt.xticks([])
    plt.yticks([])

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents+np.array([-5, -5, 5, 5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('../article/image/legend.png', dpi='figure', bbox_inches=bbox)
    plt.close()


if __name__ == '__main__':
    resultsGENN = pd.read_csv('../results/resultsGENN.csv')

    baseFontSize = 15

    # successRatePlot(resultsGENN)
    spikeCountPlot(resultsGENN)

