
from tensorflow import keras
from keras.layers import Input as kinput
from keras.layers import Dense
import Orange.data
from tensorflow import dtypes
from orangewidget.widget import OWBaseWidget, Input, Output
from orangewidget import gui, settings
from pprint import pformat
from .utils import CommonControls, BiasInitalizer, Regularizers, BiasRegularizers


class DenseLayer(OWBaseWidget, CommonControls, BiasInitalizer, Regularizers, BiasRegularizers):
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
    use_bias = True

    def __init__(self) -> None:
        OWBaseWidget.__init__(self)
        box = gui.widgetBox(self.controlArea, "Info")

        gui.spin(box, self, value='units', minv=0,
                 maxv=10_000_000, step=1, spinType=int, label='Number of Units ')
        gui.checkBox(box, self, 'use_bias', label='Use Bias?')

        # KernelInitalizer.__init__(self, box, self, self.controlArea)
        # ActivationsGui.__init__(
        #    self, box, self, self.controlArea, self.set_initalizer)

        CommonControls.__init__(self, box, self, self.controlArea)
        BiasInitalizer.__init__(self, box, self, self.controlArea)
        Regularizers.__init__(self, box, self, self.controlArea)
        BiasRegularizers.__init__(self, box, self, self.controlArea)

    @Inputs.input_layer
    def set_layer(self, layer):
        self.infoa.setText(str(layer))
