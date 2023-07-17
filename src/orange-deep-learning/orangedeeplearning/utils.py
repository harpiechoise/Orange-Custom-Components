from orangewidget import gui
from PyQt5 import QtWidgets
from tensorflow import keras
from keras import initializers
from keras.layers import Activation, Lambda
from keras import activations


def check_widgets(control):
    if len(control.findChildren(QtWidgets.QWidget)) > 0:
        return True
    return False


class KernelInitalizer:
    initalizer = ''
    mean = 0.
    stddev = 0.05
    minval = 0.
    maxval = 1.

    def __init__(self, widget, master, control_area):
        self.master = master
        self.control_area = control_area

        self.initalizer_combo = gui.comboBox(
            widget, self.master, 'initalizer',
            items=('Random Normal',
                   'Random Uniform',
                   'Truncated Normal',
                   'Zeros',
                   'Ones',
                   'Glorot Normal',
                   'Glorot Uniform'),
            label='Kernel Initalizer',
            callback=self.set_initalizer,
        )
        self.empty_options = gui.widgetBox(
            self.control_area, 'Kernel Initalizer Settings'
        )

        # EMPTY
        self.empty_options.setVisible(False)

        # RANDOM NORMAL
        self.random_normal_options = gui.widgetBox(
            self.control_area, 'Kernel Initalizer Settings'
        )
        self.random_normal_options.setVisible(False)

        # RANDOM UNIFORM
        self.random_uniform_options = gui.widgetBox(
            self.control_area, 'Kernel Initalizer Settings'
        )
        self.random_uniform_options.setVisible(False)

        self.random_normal_options_gui()

    def set_initalizer(self):
        self.clear_all_initalizers()
        if self.initalizer == 0:
            self.random_normal_options_gui()
        elif self.initalizer == 1:
            self.random_uniform_options_gui()
        elif self.initalizer == 2:
            self.random_normal_options_gui()
        elif self.initalizer == 3:
            self.empty_options.setVisible(True)
        elif self.initalizer == 4:
            self.empty_options.setVisible(True)
        elif self.initalizer == 5:
            self.empty_options.setVisible(True)
        elif self.initalizer == 6:
            self.empty_options.setVisible(True)

    def return_initalizer(self):
        if self.initalizer == 0:
            return initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        elif self.initalizer == 1:
            return initializers.RandomUniform(minval=self.minval, maxval=self.maxval)
        elif self.initalizer == 2:
            return initializers.TruncatedNormal(mean=self.mean, stddev=self.stddev)
        elif self.initalizer == 3:
            return initializers.Zeros()
        elif self.initalizer == 4:
            return initializers.Ones()
        elif self.initalizer == 5:
            return initializers.GlorotNormal()
        elif self.initalizer == 6:
            return initializers.GlorotUniform()

    def clear_all_initalizers(self):
        # It works
        self.random_normal_options.setVisible(False)
        self.empty_options.setVisible(False)
        self.random_uniform_options.setVisible(False)

    def random_uniform_options_gui(self):
        if not check_widgets(self.random_uniform_options):
            gui.spin(self.random_uniform_options, self.master, 'minval', minv=0,
                     maxv=10000, step=0.01, spinType=float, label='Minimum Value')
            gui.spin(self.random_uniform_options, self.master, 'maxval', minv=0,
                     maxv=10000, step=0.01, spinType=float, label='Maximum Value')
        self.random_uniform_options.setVisible(True)

    def random_normal_options_gui(self):
        if not check_widgets(self.random_uniform_options):
            gui.spin(self.random_normal_options, self.master, 'mean',
                     minv=0, maxv=10000, step=0.01, spinType=float, label='Mean')
            gui.spin(self.random_normal_options, self.master, 'stddev',
                     label='Standard deviation', minv=0, maxv=10000, spinType=float, step=.1)
        self.random_normal_options.setVisible(True)

# Duplicated Code Superclass Overrides


class BiasInitalizer:
    bias_initalizer = ''
    bias_mean = 0.
    bias_stddev = 0.05
    bias_minval = 0.
    bias_maxval = 1.

    def __init__(self, widget, master, control_area):
        self.master = master
        self.control_area = control_area

        gui.comboBox(
            widget, self.master, 'bias_initalizer',
            items=('Random Normal',
                   'Random Uniform',
                   'Truncated Normal',
                   'Zeros',
                   'Ones',
                   'Glorot Normal',
                   'Glorot Uniform'),
            label='Bias initalizer',
            callback=self.set_bias_initalizer,
        )
        self.bias_empty_options = gui.widgetBox(
            self.control_area, 'Bias initalizer settings'
        )

        # EMPTY
        self.bias_empty_options.setVisible(False)

        # RANDOM NORMAL
        self.bias_random_normal_options = gui.widgetBox(
            self.control_area, 'Bias initalizer settings'
        )
        self.bias_random_normal_options.setVisible(False)

        # RANDOM UNIFORM
        self.bias_random_uniform_options = gui.widgetBox(
            self.control_area, 'Bias initalizer settings'
        )
        self.bias_random_uniform_options.setVisible(False)

        self.bias_random_normal_options_gui()

    def set_bias_initalizer(self):
        self.bias_clear_all_initalizers()
        if self.bias_initalizer == 0:
            self.bias_random_normal_options_gui()
        elif self.bias_initalizer == 1:
            self.bias_random_uniform_options_gui()
        elif self.bias_initalizer == 2:
            self.bias_random_normal_options_gui()
        elif self.bias_initalizer == 3:
            self.bias_empty_options.setVisible(True)
        elif self.bias_initalizer == 4:
            self.bias_empty_options.setVisible(True)
        elif self.bias_initalizer == 5:
            self.bias_empty_options.setVisible(True)
        elif self.bias_initalizer == 6:
            self.bias_empty_options.setVisible(True)

    def return_initalizer(self):
        if self.bias_initalizer == 0:
            return initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        elif self.bias_initalizer == 1:
            return initializers.RandomUniform(minval=self.minval, maxval=self.maxval)
        elif self.bias_initalizer == 2:
            return initializers.TruncatedNormal(mean=self.mean, stddev=self.stddev)
        elif self.bias_initalizer == 3:
            return initializers.Zeros()
        elif self.bias_initalizer == 4:
            return initializers.Ones()
        elif self.bias_initalizer == 5:
            return initializers.GlorotNormal()
        elif self.bias_initalizer == 6:
            return initializers.GlorotUniform()

    def bias_clear_all_initalizers(self):
        # It works
        self.bias_random_normal_options.setVisible(False)
        self.bias_empty_options.setVisible(False)
        self.bias_random_uniform_options.setVisible(False)

    def bias_random_uniform_options_gui(self):
        if not check_widgets(self.bias_random_uniform_options):
            gui.spin(self.bias_random_uniform_options, self.master, 'bias_minval', minv=0,
                     maxv=10000, step=0.01, spinType=float, label='Minimum Value')
            gui.spin(self.bias_random_uniform_options, self.master, 'bias_maxval', minv=0,
                     maxv=10000, step=0.01, spinType=float, label='Maximum Value')
        self.bias_random_uniform_options.setVisible(True)

    def bias_random_normal_options_gui(self):
        if not check_widgets(self.bias_random_uniform_options):
            gui.spin(self.bias_random_normal_options, self.master, 'bias_mean',
                     minv=0, maxv=10000, step=0.01, spinType=float, label='Mean')
            gui.spin(self.bias_random_normal_options, self.master, 'bias_stddev',
                     label='Standard deviation', minv=0, maxv=10000, spinType=float, step=.1)
        self.bias_random_normal_options.setVisible(True)


class ActivationsGui:
    activation = ''

    def __init__(self, widget, master, control_area) -> None:
        self.master = master
        self.control_area = control_area
        self.combo_activation = gui.comboBox(widget, self.master, 'activation',
                                             items=(
                                                 'ReLU',
                                                 'Sigmoid',
                                                 'Softmax',
                                                 'SoftPlus',
                                                 'SoftSign',
                                                 'Tanh',
                                                 'Elu',
                                                 'Selu',
                                                 'Exponential'
                                             ),
                                             label='Choose an activation Function')

    def compile_activation(self):
        if self.activation == 0:
            return activations.relu
        elif self.activation == 1:
            return activations.sigmoid
        elif self.activation == 2:
            return activations.softmax
        elif self.activation == 3:
            return activations.softplus
        elif self.activation == 4:
            return activations.softsign
        elif self.activation == 5:
            return activations.tanh
        elif self.activation == 6:
            return activations.elu
        elif self.activation == 7:
            return activations.selu
        elif self.activation == 8:
            return activations.exponential


class CommonControls(KernelInitalizer, ActivationsGui):
    def __init__(self, widget, master, control_area) -> None:
        KernelInitalizer.__init__(
            self, widget, master, control_area)
        ActivationsGui.__init__(self, widget, master,
                                control_area)
