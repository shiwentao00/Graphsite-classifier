import matplotlib
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

cm = [
    [0.89, 0.002, 0.018, 0.002, 0.01, 0.01, 0.041, 0.009,
        0.016, 0.001, 0.002, 0.001, 0.01, 0.001],  # class 0

    [0.034, 0.88, 0.004, 0.004, 0.03, 0.022, 0.004,
        0, 0.013, 0, 0.004, 0, 0.009, 0],  # class 1

    [0.05, 0, 0.8, 0.012, 0.002, 0.008, 0.062, 0, 0.002,
        0.003, 0.007, 0.002, 0.048, 0.002],  # class 2

    [0.024, 0, 0.043, 0.8, 0, 0.033, 0.062, 0.01,
        0, 0, 0.01, 0.005, 0.019, 0],  # class 3

    [0.01, 0.062, 0, 0, 0.85, 0.067, 0, 0, 0.015, 0, 0, 0, 0, 0],  # class 4

    [0.029, 0.008, 0.008, 0.016, 0.045, 0.84, 0.016,
        0, 0.024, 0, 0, 0.003, 0.011, 0],  # class 5

    [0.12, 0.006, 0.045, 0.03, 0, 0.054, 0.62, 0, 0.009,
        0.012, 0.015, 0.003, 0.06, 0.018],  # class 6

    [0.19, 0, 0.008, 0.008, 0, 0, 0.041, 0.75, 0, 0, 0, 0, 0, 0],  # class 7

    [0.1, 0.026, 0.017, 0, 0, 0.061, 0.009, 0,
        0.77, 0, 0.009, 0, 0, 0.009],  # class 8

    [0.009, 0, 0, 0, 0, 0, 0.12, 0, 0, 0.84, 0, 0.009, 0.009, 0.009],  # class 9

    [0.022, 0.006, 0.022, 0.028, 0.011, 0.017, 0.094,
        0, 0.011, 0, 0.77, 0, 0.011, 0.006],  # class 10

    [0.13, 0.012, 0, 0.012, 0, 0.012, 0.071, 0,
        0.012, 0, 0, 0.73, 0.024, 0],  # class 11

    [0.17, 0.013, 0.21, 0.053, 0, 0.0270, 0.13, 0.027,
        0, 0, 0.04, 0.013, 0.31, 0],  # class 12

    [0.059, 0, 0.044, 0, 0.029, 0, 0.26, 0, 0,
        0.015, 0, 0, 0.029, 0.56]  # class 13
]

cm = np.array(cm)

for i in range(14):
    #print('length of class {}: {}'.format(i, len(cm[i])))
    assert(len(cm[i]) == 14)
    print('sum of class {}: {}'.format(i, sum(cm[i])))
    print('------------------------------------')

font = {'size': 8}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

#colors= "coolwarm"
#colors = "summer"
#colors = "viridis"
#colors = "inferno"
#colors = "magma"
#colors = "cividis"
#cmap = "YlGnBu"
cmap = sns.light_palette("green", as_cmap=True)
ax.set_title('Normalized confusion matrix')
ax = sns.heatmap(cm, 
                 annot=True, fmt='.2',
                 #linewidths=.5, 
                 cmap=cmap)
ax.set(xlabel='Predicted label', ylabel='True label')
plt.savefig('./confusion_matrix.png', bbox_inches='tight')
plt.savefig('./confusion_matrix.tif', bbox_inches='tight')
