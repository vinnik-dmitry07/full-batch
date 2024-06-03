import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def smooth(scalars: list[float], weight: float = 0.9) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars[::-1]:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed[::-1]


plt.rcParams['figure.dpi'] = 600
# plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)
plt.plot(smooth(pd.read_csv('sgd_256_sched.csv')['Value']), label='SGD Batch=256 + OneCycleLR')
plt.plot(smooth(pd.read_csv('madgrad_256_sched_d700.csv')['Value']), label='MADGRAD Batch=256 + OneCLR, LR/700')
plt.plot(smooth(pd.read_csv('adamw_256_sched_01.csv')['Value']), label='AdamW Batch=256 + OneCycleLR, LR/10')
plt.plot(smooth(pd.read_csv('lamb_16k_sched.csv')['Value'][::-1][::20][::-1]),
         label='LAMB Batch=16k + OneCLR x20 epochs')
plt.plot(smooth(pd.read_csv('sophia_256_sched_d16.csv')['Value']), label='Sophia Batch=256 + OneCycleLR, LR/16')
plt.plot(smooth(pd.read_csv('lion_256_sched_001_x10.csv')['Value']), label='Lion Batch=256 + OneCycleLR, LR/100')
plt.plot(smooth(pd.read_csv('adam_256_sched_001.csv')['Value']), label='Adam Batch=256 + OneCycleLR, LR/100')
plt.plot(smooth(pd.read_csv('lamb_sched.csv')['Value']), label='LAMB Batch=4k + OneCycleLR')
plt.plot(smooth(pd.read_csv('lion_256_nosched_001_x10.csv')['Value']), '--', label='Lion Batch=256, LR/100')
plt.plot(smooth(pd.read_csv('accum_nosched.csv')['Value']), '--', label='SGD Grad. accum. Steps=4k//256, BS=256')
plt.plot(smooth(pd.read_csv('sgd_sched.csv')['Value']), '--', label='SGD Batch=4k + OneCycleLR')
plt.plot(smooth(pd.read_csv('accum_sched.csv')['Value']), '--', label='SGD G. accum. S=4k//256, BS=256 + OCLR')
plt.plot(smooth(pd.read_csv('lamb_nosched.csv')['Value']), '--', label='LAMB Batch=4k')
plt.plot(smooth(pd.read_csv('sgd_256_nosched.csv')['Value']), '--', label='SGD Batch=4k')
plt.yscale('function', functions=(partial(np.power, 100.0), partial(np.emath.logn, 100.0)))
plt.title('Super-Convergence on CIFAR10, torchvision.resnet18\n16b Mixed Precision, ~15 min each, RTX3080 Ti Mobile')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')

# plt.ylim((0.3, 0.95))
plt.gca().add_patch(Rectangle((48, 0.917), 10, 0.007, linewidth=2, edgecolor='r', facecolor='none'))
plt.ylim((0.9, 0.95))

plt.legend(prop={'size': 7})
plt.tight_layout()
plt.show()
