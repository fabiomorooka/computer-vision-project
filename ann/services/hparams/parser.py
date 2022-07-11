# encoding: utf-8
from services.hparams.data_extractor import ParametersDataExtractor


class RawFieldNames(object):
    PARAMETERS_KEY = 'ParametersSettings'
    TRAINING_INFO = 'TrainingParameters'
    TEST_INFO = 'TestingParameters'


class TrainParametersParser(object):

    def __init__(self):
        self.parameters_data_extractor = ParametersDataExtractor()
        self.raw_field_names = RawFieldNames
        self.raw_data = self._get_raw_data()

    def _get_raw_data(self):
        return self.parameters_data_extractor.extract_json_from_file()

    def parse(self):
        return self._get_training_parameters()

    def parse_train_parameters(self, training_name):
        """
        :return: List containing all brand identifiers in lower case
        """
        train_parameters_infos = self._get_training_parameters()
        return train_parameters_infos[training_name]

    def _get_training_parameters(self):
        return self.raw_data[self.raw_field_names.PARAMETERS_KEY][self.raw_field_names.TRAINING_INFO]

class TestParametersParser(object):

    def __init__(self):
        self.parameters_data_extractor = ParametersDataExtractor()
        self.raw_field_names = RawFieldNames
        self.raw_data = self._get_raw_data()

    def _get_raw_data(self):
        return self.parameters_data_extractor.extract_json_from_file()

    def parse(self):
        return self._get_testing_parameters()

    def parse_test_parameters(self, test_name):
        """
        :return: List containing all brand identifiers in lower case
        """
        test_parameters_infos = self._get_testing_parameters()
        return test_parameters_infos[test_name]

    def _get_testing_parameters(self):
        return self.raw_data[self.raw_field_names.PARAMETERS_KEY][self.raw_field_names.TEST_INFO]
