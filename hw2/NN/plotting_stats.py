from __future__ import division
from __future__ import print_function
from util import LoadData, Load, Save, DisplayPlot
import matplotlib.pyplot as plt
plt.ion()


stats = Load('3.2_nn_stats_momentum_0.9_1.npz')
DisplayPlot(stats['train_ce'], stats['valid_ce'], 'Cross Entropy', number=0)
DisplayPlot(stats['train_acc'], stats['valid_acc'], 'Accuracy', number=1)
raw_input('Press Enter.')