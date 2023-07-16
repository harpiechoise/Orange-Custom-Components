
from tensorflow import keras
from keras.layers import Input as kinput
from keras.layers import Dense
import Orange.data
from tensorflow import dtypes
from orangewidget.widget import OWBaseWidget, Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget import gui, settings
from pprint import pformat


class DenseLayer(OWBaseWidget):
    name = 'Dense Layer'
    description = 'Fully Connected Layer'
    icon = 'icons/dense.svg'
    priority = 2

    class Inputs:
        input_layer = Input('Previous Layer', object)

    class Outputs:
        output_layer = Output('Output layer', object)

    want_main_area = False
    activation = ''

    def __init__(self) -> None:
        super().__init__()
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(
            box, 'No data on input yet, waiting to get something.')
        self.combo = gui.comboBox(box, self, 'activation', items=(
            'Random Normal', 'Random Uniform'), callback=self.set_activation)

    def set_activation(self):
        print(self.activation)
        print("CHANGED !")

    @Inputs.input_layer
    def set_layer(self, layer):
        self.infoa.setText(str(layer))
