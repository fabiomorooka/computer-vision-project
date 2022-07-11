# encoding: utf-8
from os.path import abspath, join

from unipath import Path


def get_parent_directory(f, level=1):
    return str(Path(abspath(f)).ancestor(level))


def build_path(*args):
    return join(*args)
