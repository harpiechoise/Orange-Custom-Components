from tensorflow import keras
from keras.layers import Input as kinput
from keras.layers import Dense
import Orange.data
from tensorflow import dtypes
from orangewidget.widget import OWBaseWidget, Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget import gui, settings
from pprint import pformat


class InputLayer(OWBaseWidget):
    name = 'Input Layer'
    description = 'Data input for a deep learning model'
    icon = 'icons/input.svg'
    priority = 1

    class Outputs:
        layer = Output('Input layer', object)
    input_shape = "27, 27, 3"
    batch_size = 32
    layer_name = "Layer Name"
    dtype = 'float32'
    sparse = False
    ragged = False

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.optionBox = gui.widgetBox(
            self.controlArea, 'Input Layer Configuration')
        gui.lineEdit(self.optionBox, self, 'input_shape',
                     'Input shape (ex. 64,64,1)', orientation=gui.Qt.Horizontal, callbackOnType=self.change)
        gui.spin(self.optionBox, self, 'batch_size', minv=1, maxv=2147483640,
                 label='Batch Size', callback=self.change)
        gui.lineEdit(self.optionBox, self, 'layer_name',
                     'Layer Name:', orientation=gui.Qt.Horizontal, callbackOnType=self.change)
        gui.comboBox(self.optionBox, self, 'dtype', label='Data Type',
                     items=('int8', 'int16', 'int32', 'int64',
                            'float16', 'float32', 'float64', 'bool'),
                     orientation=gui.Qt.Horizontal, callback=self.change)
        self.sparse_check = gui.checkBox(
            self.optionBox, self, 'sparse', 'Sparse?', callback=self.sparse_checked)
        self.ragged_check = gui.checkBox(
            self.optionBox, self, 'ragged', 'Ragged?', callback=self.change)

        self.ragged_check.setDisabled(True)
        self.compile = gui.button(
            self.optionBox, self, 'Done', callback=self.compile_input)

        self.infoBox = gui.widgetBox(
            self.controlArea, 'Input Layer compilation information',
        )

        self.label = gui.widgetLabel(self.infoBox, 'Not compiled yet!')

    def change(self):
        self.compile.setDisabled(False)

    def sparse_checked(self):
        self.compile.setDisabled(False)

        if self.sparse:
            self.ragged_check.setDisabled(False)

    def compile_input(self):
        # Parse all arguments
        try:
            input_shape = tuple(map(int, str(self.input_shape).split(',')))
        except ValueError as _:
            self.error('Input shape must be a comma separated list of values')

        if self.batch_size < 0:
            self.error('Batch size must be a positive value')

        if self.layer_name == '':
            self.error('Layer name cannot be a empty string')

        arguments = {
            'shape': input_shape,
            'batch_size': self.batch_size,
            'name': self.layer_name,
            'dtype': self.dtype,
            'sparse': self.sparse,
            'ragged': self.ragged
        }
        keras_input = kinput(**arguments)

        self.Outputs.layer.send(keras_input)

        self.compile.setDisabled(True)
        self.label.setText(
            f'Compiled with arguments:\n{pformat(arguments, indent=2)}')
