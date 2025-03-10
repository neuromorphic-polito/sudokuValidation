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
        index='puzzle', columns=['difficulty', 'enhanced'],
        values='flagSol',
        aggfunc='sum'
    )
    print(successRate)

    configurations = [e for e in [0, 1]]
    confusionMatrix = np.zeros((2, 3))
    for (r, e) in enumerate(configurations):
        for c, puzzle in enumerate(['easy', 'med', 'hard']):
                confusionMatrix[r, c] = np.sum(successRate[puzzle, e])/3/100*100

    plt.figure(figsize=(6.4, 4.8*0.62))
    
    xLabel = ['$Easy$', '$Medium$', '$Hard$']
    yLabel = ['$a$', '$d$']
    ax = plt.subplot(1, 1, 1)
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Greens, aspect='auto', vmin=0, vmax=100)
    for r in range(len(yLabel)):
        for c in range(len(xLabel)):
            value = round(confusionMatrix[r, c], 2)
            color = '#000000'
            if value > 50:
                color = '#FFFFFF'
            plt.text(c, r, f'{value:.02f}%', va='center', ha='center', size=baseFontSize, color=color)

    ax.set_xlabel('\n$Class$', fontsize=baseFontSize+1)
    ax.set_xticks(np.arange(len(xLabel)))
    ax.set_xticklabels(xLabel, fontsize=baseFontSize, rotation=0)
    ax.set_ylabel('$Strategy$\n', fontsize=baseFontSize+1)
    ax.set_yticks(np.arange(len(yLabel)))
    ax.set_yticklabels(yLabel, fontsize=baseFontSize)

    plt.tight_layout()
    plt.savefig('../article/image/SpiNNaker-SuccessRateMatrix.png')
    # plt.show()
    
    return 0


def reductionPerformance(dataset):
    spikeCountBase = pd.pivot_table(
        dataset[(dataset['enhanced'] == 0)],
        columns='difficulty',
        values='spikeCount',
        aggfunc='median'
    )
    spikeCountEnhanced = pd.pivot_table(
        dataset[(dataset['enhanced'] == 1)],
        columns='difficulty',
        values='spikeCount',
        aggfunc='median'
    )

    timeExtractionBase = pd.pivot_table(
        dataset[(dataset['enhanced'] == 0)],
        columns='difficulty',
        values='timeExtraction',
        aggfunc='median'
    )
    timeExtractionEnhanced = pd.pivot_table(
        dataset[(dataset['enhanced'] == 1)],
        columns='difficulty',
        values='timeExtraction',
        aggfunc='median'
    )

    xLabel = ['$Easy$', '$Medium$', '$Hard$']
    frame = plt.figure(figsize=(6.4*2, 4.8))
    ax = plt.subplot(1, 2, 1)

    for c, puzzle in enumerate(['easy', 'med', 'hard']):
        a = spikeCountBase[puzzle].values[0]
        d = spikeCountEnhanced[puzzle].values[0]
        plt.bar(c*4-0.7, a, width=1.3, color='#8E3B46', zorder=2)
        plt.bar(c*4+0.7, d, width=1.3, color='#226F54', zorder=2)
        plt.bar(c*4+0.7, a, width=1.3, color='#DEDEDE', zorder=1, alpha=0.5)
        position = (a+d)/2-3e5  # position = 10**((np.log10(a)+np.log10(d))/2-0.12)
        ax.text(c*4+0.7, position, f'{(a-d)/a*100:.2f}%', ha='center', va='bottom', fontsize=baseFontSize-4)

    ax.title.set_text(f'$Extracted$ $spikes$\n')
    ax.title.set_size(baseFontSize+3)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(4*np.array([0, 1, 2]), labels=xLabel, fontsize=baseFontSize)
    plt.xlabel('\n$Class$', fontdict={'fontsize': baseFontSize+1})
    plt.ylim([0, 8e6])
    plt.yticks(fontsize=baseFontSize-4)
    plt.ylabel('$Median$ $value$\n', fontdict={'fontsize': baseFontSize+1}, rotation=90)
    plt.legend(['Strategy $a$', 'Strategy $d$'], loc="upper right", ncol=1, fontsize=baseFontSize-3.5)
    ax.set_facecolor('#F7F7F7')
    ax.yaxis.set_major_locator(MultipleLocator(1e6))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(zorder=0, which='major', color='#CCCCCC', linestyle='solid')
    ax.grid(zorder=0, which='minor', color='#CCCCCC', linestyle='dashed')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    # ax.set_yscale('log')

    ax = plt.subplot(1, 2, 2)
    for c, puzzle in enumerate(['easy', 'med', 'hard']):
        a = timeExtractionBase[puzzle].values[0]
        d = timeExtractionEnhanced[puzzle].values[0]
        plt.bar(c*4-0.7, a, width=1.3, color='#8E3B46', zorder=2)
        plt.bar(c*4+0.7, d, width=1.3, color='#226F54', zorder=2)
        plt.bar(c*4+0.7, a, width=1.3, color='#DEDEDE', zorder=1, alpha=0.5)
        position = (a+d)/2-0.25  # position = 10**((np.log10(a)+np.log10(d))/2-0.12)
        ax.text(c*4+0.7, position, f'{(a-d)/a*100:.2f}%', ha='center', va='bottom', fontsize=baseFontSize-4)

    ax.title.set_text(f'$Extraction$ $time$\n')
    ax.title.set_size(baseFontSize + 3)
    plt.xticks(4 * np.array([0, 1, 2]), labels=xLabel, fontsize=baseFontSize)
    plt.xlabel('\n$Class$', fontdict={'fontsize': baseFontSize + 1})
    plt.ylim([0, 8])
    plt.yticks(fontsize=baseFontSize-2)
    plt.ylabel('\n\n$Median$ $value$ ($s$)\n', fontdict={'fontsize': baseFontSize + 1}, rotation=90)
    plt.legend(['Strategy $a$', 'Strategy $d$'], loc="upper right", ncol=1, fontsize=baseFontSize-3.5)
    ax.set_facecolor('#F7F7F7')
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(zorder=0, which='major', color='#CCCCCC', linestyle='solid')
    ax.grid(zorder=0, which='minor', color='#CCCCCC', linestyle='dashed')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('../article/image/SpiNNaker-ReductionPerformance.png')
    plt.show()


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


def timeSimulationPerformance(dataset):
    ############################
    # ##### Success rate ##### #
    ############################
    successRate = pd.pivot_table(
        dataset,
        index='puzzle', columns=['difficulty', 'enhanced'],
        values='timeSimulation',
        aggfunc='mean'
    )
    print(successRate)

    configurations = [e for e in [0, 1]]
    confusionMatrix = np.zeros((2, 3))
    for (r, e) in enumerate(configurations):
        for c, puzzle in enumerate(['easy', 'med', 'hard']):
            confusionMatrix[r, c] = np.mean(successRate[puzzle, e])/60
    print(np.mean(confusionMatrix))

    plt.figure(figsize=(6.4, 4.8 * 0.62))

    xLabel = ['$Easy$', '$Medium$', '$Hard$']
    yLabel = ['$a$', '$d$']
    ax = plt.subplot(1, 1, 1)
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Greens, aspect='auto', vmin=0, vmax=100)
    for r in range(len(yLabel)):
        for c in range(len(xLabel)):
            value = round(confusionMatrix[r, c], 2)
            color = '#000000'
            if value > 50:
                color = '#FFFFFF'
            plt.text(c, r, f'{value:.02f}', va='center', ha='center', size=baseFontSize, color=color)

    ax.set_xlabel('\n$Class$', fontsize=baseFontSize + 1)
    ax.set_xticks(np.arange(len(xLabel)))
    ax.set_xticklabels(xLabel, fontsize=baseFontSize, rotation=0)
    ax.set_ylabel('$Strategy$\n', fontsize=baseFontSize + 1)
    ax.set_yticks(np.arange(len(yLabel)))
    ax.set_yticklabels(yLabel, fontsize=baseFontSize)

    plt.tight_layout()
    plt.savefig('../article/image/SpiNNaker-TimeSimulationPerformanceMatrix.png')
    # plt.show()

    return 0


if __name__ == '__main__':
    resultsSpiNNaker = pd.read_csv('../results/resultsSpiNNaker.csv')

    baseFontSize = 14

    successRatePlot(resultsSpiNNaker)
    reductionPerformance(resultsSpiNNaker)
    timeSimulationPerformance(resultsSpiNNaker)
