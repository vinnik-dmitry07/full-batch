import math
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.rcsetup import cycler


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
os.chdir('data')


def f1():
    plt.ylim((0.3, 0.95))


def f2():
    plt.gca().add_patch(Rectangle((48, 0.917), 10, 0.007, linewidth=2, edgecolor='r', facecolor='none'))
    plt.ylim((0.9, 0.95))


for f in f1, f2:
    plt.plot(smooth(pd.read_csv('sgd_256_sched.csv')['Value']), label='SGD Batch=256 + OneCycleLR')
    plt.plot(smooth(pd.read_csv('124_madgrad_256_sched_d700.csv')['Value']), label='MADGRAD Batch=256 + OneCLR, LR/700')
    plt.plot(smooth(pd.read_csv('adamw_256_sched_01.csv')['Value']), label='AdamW Batch=256 + OneCycleLR, LR/10')
    plt.plot(smooth(pd.read_csv('lamb_16k_sched.csv')['Value'][::-1][::20][::-1]), label='LAMB Batch=16k + OneCLR x20 epochs')
    plt.plot(smooth(pd.read_csv('112_sophia_256_sched_d16.csv')['Value']), label='Sophia Batch=256 + OneCycleLR, LR/16')
    plt.plot(smooth(pd.read_csv('96_lion_256_sched_d300_x10.csv')['Value']), label='Lion Batch=256 + OneCycleLR, LR/300')
    plt.plot(smooth(pd.read_csv('adam_256_sched_001.csv')['Value']), label='Adam Batch=256 + OneCycleLR, LR/100')
    plt.plot(smooth(pd.read_csv('lamb_sched.csv')['Value']), label='LAMB Batch=4k + OneCycleLR')
    plt.plot(smooth(pd.read_csv('209_adamwschedulefree_sched.csv')['Value']), '--', label='AdamWScheduleFree Batch=256 + OCLR')
    plt.plot(smooth(pd.read_csv('102_lion_256_nosched_d1000_x10.csv')['Value']), '--', label='Lion Batch=256, LR/1000')
    plt.plot(smooth(pd.read_csv('accum_nosched.csv')['Value']), '--', label='SGD Grad. accum. Steps=4k//256, BS=256')
    plt.plot(smooth(pd.read_csv('212_adamwschedulefree_nosched.csv')['Value']), '--', label='AdamWScheduleFree Batch=256')
    plt.plot(smooth(pd.read_csv('sgd_sched.csv')['Value']), '--', label='SGD Batch=4k + OneCycleLR')
    plt.plot(smooth(pd.read_csv('accum_sched.csv')['Value']), '--', label='SGD G. accum. S=4k//256, BS=256 + OCLR')
    plt.plot(smooth(pd.read_csv('200_sgdschedulefree_sched.csv')['Value']), '--', label='SGDScheduleFree Batch=256 + OCLR')
    plt.plot(smooth(pd.read_csv('88_lamb_nosched.csv')['Value']), '--', label='LAMB Batch=4k')
    plt.plot(smooth(pd.read_csv('201_sgdschedulefree_nosched.csv')['Value']), '--', label='SGDScheduleFree Batch=256')
    plt.plot(smooth(pd.read_csv('95_sgd_4k_nosched.csv')['Value']), '--', label='SGD Batch=4k')

    plt.yscale('function', functions=(partial(np.power, 100.0), partial(np.emath.logn, 100.0)))
    plt.title('Super-Convergence on CIFAR10, torchvision.resnet18\n16b Mixed Precision, ~15 min each, RTX3080 Ti Mobile')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend(prop={'size': 7})

    f()

    plt.tight_layout()
    plt.show()

# tail -q -n1 epoch/*.csv >epoch.csv
# tail -q -n1 acc/*.csv > acc.csv
x = pd.read_csv('epoch.csv', header=None).values[:, 2]
y = pd.read_csv('acc.csv', header=None).values[:, 2]
o = np.argsort(x)
o = np.r_[o[1:4], o[5:]]
x = x[o]
y = y[o]

colors = list(plt.get_cmap('tab20').colors)
colors = colors[::2] + colors[1::2]
colors[3], colors[9] = colors[9], colors[3]
plt.rcParams['axes.prop_cycle'] = cycler('color', colors)

for xi, yi in zip(x, y):
    plt.scatter(xi, yi, s=60 if yi == max(y) else 30, marker='*' if yi == max(y) else 'o')
plt.yscale('function', functions=(partial(np.power, 1e5), partial(np.emath.logn, 1e5)))
plt.xscale('log')
plt.ylim((0.75, 0.96))
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.tight_layout()
plt.show()
