# encoding: utf-8
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


# Ref. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
def plot(gt_values, pca_x, y_train, model, label_list, path, name=None):
  '''This function that plot the roc curve

  Input
  -----
  gt_values :
    The ground truth predictions of the dataset
  pca_x :
    The pca input values of the dataset

  Output
  ------
  None
  '''

  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(len(set(y_train))):
      fpr[i], tpr[i], _ = roc_curve(gt_values[:, i], model.predict(pca_x)[:,i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  fig, axs = plt.subplots(10,4,figsize=(30,25))
  pos = 0
  for i in range(10):
    for j in range(4):
      axs[i,j].plot(fpr[pos], tpr[pos], label='ROC curve (area = %0.2f)' % roc_auc[pos])
      axs[i,j].plot([0, 1], [0, 1], 'k--')
      axs[i,j].set_title('Category ' + label_list[pos] + ' ROC Curve')
      axs[i,j].set_xlabel('False Positive Rate')
      axs[i,j].set_ylabel('True Positive Rate')
      axs[i,j].legend(loc='lower right')
      pos += 1

  plt.title('ROC curve and area for {name} set')
  plt.savefig(path)
