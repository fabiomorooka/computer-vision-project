# encoding: utf-8
from utils import configuration as config
from utils import file_read_handler as read

from analysis import confusion_matrix
from analysis import histogram
from analysis import pca
from analysis import examples
from analysis import precision_recall
from analysis import roc
from sklearn.linear_model import LogisticRegression

def main():
    # Spliting into train, validation and test set
    TRAFFIC_SIGNS_DIRECTORY = "traffic_signs"
    TRAIN_PICKLE_NAME = "train"
    VALIDATION_PICKLE_NAME = "valid"
    TEST_PICKLE_NAME = "test"
    LABEL_CSV_NAME = 'labels'
    
    train_path = config.get_pickle_path(TRAIN_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    validation_path = config.get_pickle_path(VALIDATION_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    test_path = config.get_pickle_path(TEST_PICKLE_NAME, TRAFFIC_SIGNS_DIRECTORY)
    labels_path = config.get_csv_path(LABEL_CSV_NAME, TRAFFIC_SIGNS_DIRECTORY)


    x_train, y_train, _, _ = read.read_rgb_dataset(train_path)
    x_validation, y_validation, _, _ = read.read_rgb_dataset(validation_path)
    x_test, y_test, _, _ = read.read_rgb_dataset(test_path)
    label_list = read.read_csv(labels_path)

    print(f'There are {len(label_list)} categories in the dataset')
    print(f'Shape train images:{x_train.shape} | Shape train labels:{y_train.shape}')
    print(f'Shape val images:{x_validation.shape} | Shape val labels:{y_validation.shape}')
    print(f'Shape test images:{x_test.shape} | Shape test labels:{y_test.shape}')

    #examples.plot(x_train, y_train, label_list, config.get_plot_path("examples", "examples"))

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_validation = x_validation.reshape(x_validation.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    print('After flattening the X datasets')
    print(f'Shape train images:{x_train.shape} | Shape train labels:{y_train.shape}')
    print(f'Shape val images:{x_validation.shape} | Shape val labels:{y_validation.shape}')
    print(f'Shape test images:{x_test.shape} | Shape test labels:{y_test.shape}')

    x_train = pca.scale(x_train, 'train')
    x_validation = pca.scale(x_validation, 'validation')
    x_test = pca.scale(x_test, 'test')

    pca_x_train = pca.analyse(x_train, 'training')
    pca_x_validation = pca.analyse(x_validation, 'validation', x_train)
    pca_x_test = pca.analyse(x_test, 'test', x_train)

    gt_train = precision_recall.transform(y_train, label_list, 'training')
    gt_validation = precision_recall.transform(y_validation, label_list, 'validation')
    gt_test = precision_recall.transform(y_test, label_list, 'test')

    # Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    model = LogisticRegression(multi_class='multinomial', solver = 'saga', penalty = 'l2').fit(pca_x_train, y_train)
    y_train_pred = model.predict(pca_x_train)
    y_validation_pred = model.predict(pca_x_validation)
    y_test_pred = model.predict(pca_x_test)

    confusion_matrix.plot(y_train_pred, label_list, config.get_plot_path("regression_training", "confusion_matrix"), "training", y_train)
    confusion_matrix.plot(y_validation_pred, label_list, config.get_plot_path("regression_validation", "confusion_matrix"), "validation", y_validation)
    confusion_matrix.plot(y_test_pred, label_list, config.get_plot_path("regression_test", "confusion_matrix"), "test", y_test)

    print(f'Training Accuracy: {round(100*model.score(pca_x_train, y_train), 4)}%')
    print(f'Validation Accuracy: {round(100*model.score(pca_x_validation, y_validation), 4)}%')
    print(f'Test Accuracy: {round(100*model.score(pca_x_test, y_test), 4)}%')

    precision_recall.plot(gt_validation, model.predict_proba(pca_x_validation), label_list, config.get_plot_path("validation", "precision_recall"), config.get_plot_path("validation_leg", "precision_recall"))
    precision_recall.plot(gt_test, model.predict_proba(pca_x_test), label_list, config.get_plot_path("test", "precision_recall"), config.get_plot_path("test_leg", "precision_recall"))

    precision_recall.calculate_final_score(y_validation, y_validation_pred)
    precision_recall.calculate_final_score(y_test, y_test_pred)

    roc.plot(gt_validation, pca_x_validation, y_train, model, label_list, config.get_plot_path("validation", "roc"), 'validation')
    roc.plot(gt_test, pca_x_test, y_train, model, label_list, config.get_plot_path("test", "roc"), 'test') 

if __name__ == '__main__':
    main()

'''
    pca.plot(x_train, config.get_plot_path("training", "pca"), 'training')
    pca.analyse(x_train, 'training')
    pca.analyse(x_validation, 'validation', x_train)
    pca.analyse(x_test, 'test', x_train)

    confusion_matrix.plot(y_train, label_list, config.get_plot_path("training", "confusion_matrix"), "training")
    confusion_matrix.plot(y_validation, label_list, config.get_plot_path("validation", "confusion_matrix"), "validation")
    confusion_matrix.plot(y_test, label_list, config.get_plot_path("test", "confusion_matrix"), "test")

    histogram.plot(y_train, config.get_plot_path("training", "histogram"), "training")
    histogram.plot(y_validation, config.get_plot_path("validation", "histogram"), "validation")
    histogram.plot(y_test, config.get_plot_path("test", "histogram"), "test")
'''
