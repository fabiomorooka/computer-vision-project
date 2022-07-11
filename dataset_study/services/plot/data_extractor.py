# encoding: utf-8
import json

from utils.configuration import get_resources_folder
from utils.path import build_path


class ParametersDataExtractor(object):
    def extract_json_from_file(self):
        """
        Reads the json file containing hparams data
        :return: a dictionary containing the key "ParametersSettings" which has the two following keys in its
        inner dictionary:
            - "Training"
            - "Testing"
        """
        json_file = self._get_file_from_local_directory()
        return json_file

    @staticmethod
    def _get_file_from_local_directory():
        with open(build_path(get_resources_folder(), 'plot_configuration.json'), 'r') as plot_infomation_file:
            return json.load(plot_infomation_file)
