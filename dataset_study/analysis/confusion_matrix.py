# encoding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Ref. https://seaborn.pydata.org/generated/seaborn.heatmap.html
def plot(y, labels, path, name=None, y_comp=None):
    plt.figure(figsize=(20, 20))
    if y_comp is None: y_comp=y
    sns.heatmap(confusion_matrix(y, y_comp), 
                annot= True, fmt='',
                xticklabels=labels,yticklabels=labels, cmap='Blues').set(title=f'{name}_confusion_matrix')
    plt.savefig(path)
