# encoding: utf-8
from analysis import (confusion_matrix, examples, histogram, pca,
                      precision_recall, roc)
from models.multi_class_logistic_regression import MultiClassLogisticRegression
from services.plot.parser import PlotParametersParser
from utils import configuration as config
from utils import file_read_handler as read
from utils import image_processing as img

TRAFFIC_SIGNS_DIRECTORY = 'traffic_signs'
TRAIN_PICKLE_NAME = 'train'
valid_PICKLE_NAME = 'valid'
TEST_PICKLE_NAME = 'test'
LABEL_CSV_NAME = 'labels'

PLOT_PARAMS = PlotParametersParser()._get_plot_parameters()

def regression(x_train, y_train, x_valid, y_valid, x_test, y_test, label_list):
    x_train = pca.scale(x_train, 'train')
    x_valid = pca.scale(x_valid, 'valid')
    x_test = pca.scale(x_test, 'test')

    pca_x_train = pca.analyse(x_train, 'train')
    pca_x_valid = pca.analyse(x_valid, 'valid', x_train)
    pca_x_test = pca.analyse(x_test, 'test', x_train)

    gt_valid = precision_recall.transform(y_valid, label_list, 'valid')
    gt_test = precision_recall.transform(y_test, label_list, 'test')

    model = MultiClassLogisticRegression(y_train, label_list)
    model.fit(pca_x_train)
    model.plot_confusion_matrix(pca_x_train, y_train, 'train')
    model.plot_confusion_matrix(pca_x_valid, y_valid, 'valid')
    model.plot_confusion_matrix(pca_x_test, y_test, 'test')

    print(f'Train Accuracy: {round(100*model.calculate_score(pca_x_train, y_train), 4)}%')
    print(f'Valid Accuracy: {round(100*model.calculate_score(pca_x_valid, y_valid), 4)}%')
    print(f'Test Accuracy: {round(100*model.calculate_score(pca_x_test, y_test), 4)}%')
  
    model.calculate_final_score(pca_x_valid, y_valid)
    model.calculate_final_score(pca_x_test, y_test)

    if eval(PLOT_PARAMS['PrecisionRecall']):
        precision_recall.plot(gt_valid, model.predict(pca_x_valid), label_list, config.get_plot_path('valid', 'precision_recall'), config.get_plot_path('valid_leg', 'precision_recall'))
        precision_recall.plot(gt_test, model.predict(pca_x_test), label_list, config.get_plot_path('test', 'precision_recall'), config.get_plot_path('test_leg', 'precision_recall'))

    if eval(PLOT_PARAMS['ROC']):
        roc.plot(gt_valid, pca_x_valid, y_train, model, label_list, config.get_plot_path('valid', 'roc'), 'valid')
        roc.plot(gt_test, pca_x_test, y_train, model, label_list, config.get_plot_path('test', 'roc'), 'test') 

def main():
    train_path = config.get_pickle_path(TRAIN_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    valid_path = config.get_pickle_path(valid_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    test_path = config.get_pickle_path(TEST_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    labels_path = config.get_csv_path(LABEL_CSV_NAME, TRAFFIC_SIGNS_DIRECTORY)

    x_train, y_train, _, _ = read.read_rgb_dataset(train_path)
    x_valid, y_valid, _, _ = read.read_rgb_dataset(valid_path)
    x_test, y_test, _, _ = read.read_rgb_dataset(test_path)
    label_list = read.read_csv(labels_path)

    print(f'There are {len(label_list)} categories in the dataset')
    print(f'Shape train images:{x_train.shape} | Shape train labels:{y_train.shape}')
    print(f'Shape val images:{x_valid.shape} | Shape val labels:{y_valid.shape}')
    print(f'Shape test images:{x_test.shape} | Shape test labels:{y_test.shape}')

    if eval(PLOT_PARAMS['Examples']):
        examples.plot(x_train, y_train, label_list, config.get_plot_path('examples', 'examples'))
        gray_x_train = img.to_gray(x_train)
        examples.plot_gray(gray_x_train, y_train, label_list, config.get_plot_path('examples_gray', 'examples'))

    if eval(PLOT_PARAMS['ConfusionMatrix']):
        confusion_matrix.plot(y_train, label_list, config.get_plot_path('train', 'confusion_matrix'), 'train')
        confusion_matrix.plot(y_valid, label_list, config.get_plot_path('valid', 'confusion_matrix'), 'valid')
        confusion_matrix.plot(y_test, label_list, config.get_plot_path('test', 'confusion_matrix'), 'test')

    if eval(PLOT_PARAMS['Histogram']):
        histogram.plot(y_train, config.get_plot_path('train', 'histogram'), 'train')
        histogram.plot(y_valid, config.get_plot_path('valid', 'histogram'), 'valid')
        histogram.plot(y_test, config.get_plot_path('test', 'histogram'), 'test')

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    print(f'After flattening the X datasets')
    print(f'Shape train images:{x_train.shape} | Shape train labels:{y_train.shape}')
    print(f'Shape val images:{x_valid.shape} | Shape val labels:{y_valid.shape}')
    print(f'Shape test images:{x_test.shape} | Shape test labels:{y_test.shape}')

    if eval(PLOT_PARAMS['PCA']):
        pca.plot(x_train, config.get_plot_path('train', 'pca'), 'train')
    
    
    if eval(PLOT_PARAMS['PrecisionRecall']) or eval(PLOT_PARAMS['ROC']):
        regression(x_train, y_train, x_valid, y_valid, x_test, y_test, label_list)

if __name__ == '__main__':
    main()
