
from tensorflow import keras
from keras.layers import Input as kinput
from keras.layers import Dense
import Orange.data
from tensorflow import dtypes
from orangewidget.widget import OWBaseWidget, Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget import gui, settings
from pprint import pformat
from .utils import KernelInitalizer


class DenseLayer(OWBaseWidget, KernelInitalizer):
    name = 'Dense Layer'
    description = 'Fully Connected Layer'
    icon = 'icons/dense.svg'
    priority = 2

    class Inputs:
        input_layer = Input('Previous Layer', object)

    class Outputs:
        output_layer = Output('Output layer', object)

    want_main_area = False
    units = 0

    def __init__(self) -> None:
        OWBaseWidget.__init__(self)
        box = gui.widgetBox(self.controlArea, "Info")

        gui.spin(box, self, value='units', minv=0,
                 maxv=10_000_000, step=1, spinType=int, label='Number of Units ')

        KernelInitalizer.__init__(self, box, self, self.controlArea)

    @Inputs.input_layer
    def set_layer(self, layer):
        self.infoa.setText(str(layer))
