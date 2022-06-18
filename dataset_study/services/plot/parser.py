# encoding: utf-8
from services.plot.data_extractor import ParametersDataExtractor


class RawFieldNames(object):
    PARAMETERS_KEY = 'ParametersSettings'
    PLOT_INFO = 'PlotParamenters'


class PlotParametersParser(object):

    def __init__(self):
        self.parameters_data_extractor = ParametersDataExtractor()
        self.raw_field_names = RawFieldNames
        self.raw_data = self._get_raw_data()

    def _get_raw_data(self):
        return self.parameters_data_extractor.extract_json_from_file()

    def parse(self):
        return self._get_plot_parameters()

    def parse_plot_parameters(self, plot):
        """
        :return: List containing all brand identifiers in lower case
        """
        plot_parameters_infos = self._get_plot_parameters()
        return plot_parameters_infos[plot]

    def _get_plot_parameters(self):
        return self.raw_data[self.raw_field_names.PARAMETERS_KEY][self.raw_field_names.PLOT_INFO]
