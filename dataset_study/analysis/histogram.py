# encoding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(y, path, name=None):
    plt.figure(figsize=(20, 15))
    sns.set_theme(style='ticks')
    sns.histplot(pd.DataFrame(y.T),
                legend = False,
                edgecolor='.3',
                kde=True,
                stat = 'density',
                linewidth=1).set(title=f'{name} label Distribuition')
    plt.savefig(path)
