# encoding: utf-8
from utils.path import build_path, get_parent_directory


def get_pickle_path(dataset, group=None):
    if group is None:
        return build_path(get_parent_directory(__file__, 2), 'data', dataset + '.pickle')
    else:
        return build_path(get_parent_directory(__file__, 2), 'data', group, dataset + '.pickle')

def get_csv_path(dataset, group=None):
    if group is None:
        return build_path(get_parent_directory(__file__, 2), 'data', dataset + '.csv')
    else:
        return build_path(get_parent_directory(__file__, 2), 'data', group, dataset + '.csv')

def get_plot_path(dataset, group=None):
    if group is None:
        return build_path(get_parent_directory(__file__, 2), 'plot', dataset + '.jpg')
    else:
        return build_path(get_parent_directory(__file__, 2), 'plot', group, dataset + '.jpg')

