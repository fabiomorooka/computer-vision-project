# encoding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from utils import plot_helper


# Ref. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
def transform(y, label_list, name):
    enc = OneHotEncoder(handle_unknown='ignore')
    gt = enc.fit_transform(y.reshape(-1, 1)).toarray()
    print(f'The training label set has {len(label_list)} different labels')
    print(f'Shape encoded {name} label : {gt.shape}')
    ind = np.random.randint(100, size=1)
    print(f'\nOneHotEncoder {name} set {gt[ind]} | label: {y[ind]} that is: {label_list[int(y[ind])]}')

    return gt

def plot(gt_predicions, predicions, label_list, graph_path, legend_path):
    '''This function that plot the precision and recall curve

    Input
    -----
    gt_predicions :
        The ground truth predictions of the dataset
    predicions :
        The predictions made for the dataset

    Output
    ------
    None
    '''

    # Calculate precision and recall in valid dataset
    precision = dict()
    recall = dict()
    average_precision = dict()

    plt.figure(figsize=(15,7)) # specifying the overall grid size
    for i in range(len(label_list)):
        precision[i], recall[i], _ = precision_recall_curve(gt_predicions[:, i], predicions[:,i])
        average_precision[i] = average_precision_score(gt_predicions[:, i], predicions[:, i])


    for i in range(len(label_list)):
        plt.plot(recall[i], precision[i],
                label='Precision-recall curve of class {0} (area = {1:0.2f})'
                        ''.format(label_list[i], average_precision[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall curve')
    plt.savefig(graph_path)
    legend = plt.legend(loc='best',framealpha=1, frameon=True)
    plot_helper.export_legend(legend, legend_path)
    

