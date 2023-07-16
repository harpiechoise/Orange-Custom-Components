from orangewidget import gui
from PyQt5 import QtWidgets
from tensorflow import keras
from keras import initializers


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
            label='Layer Initalier',
            callback=self.set_initalizer,
        )
        self.empty_options = gui.widgetBox(
            self.control_area, 'Kernel Initalizer Settings'
        )

        self.empty_options.setVisible(False)

        # RANDOM NORMAL
        self.random_normal_options = gui.widgetBox(
            self.control_area, 'Kernel Initalizer Settings')
        self.random_normal_options.setVisible(False)

        # RANDOM UNIFORM
        self.random_uniform_options = gui.widgetBox(
            self.control_area, 'Kernel Initalizer Settings'
        )
        self.random_uniform_options.setVisible(False)

        self.random_normal_options_gui()

    def set_initalizer(self):
        print("Setting initalizer")
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
        print("CLEAN")
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


class ActivationsGui:
    activation = ''

    # RELU
    alpha = 0.0
    max_value = -1.0
    threshold = 0.0
    axis = -1

    def __init__(self, widget, master, control_area) -> None:
        self.master = master
        self.control_area = control_area
        self.combo_activation = gui.comboBox(widget, self.master, 'activation',
                                             items=(
                                                 'ReLU',
                                                 'Sigmoid',
                                                 'Softmax',
                                                 'SoftPlus',
                                             ), callback=self.set_activation,
                                             label='Choose an activation Function')

        # ReLU
        self.relu_options = gui.widgetBox(self.control_area, 'ReLU Options')

        # Sigmoid
        self.sigmoid_options = gui.widgetBox(
            self.control_area, 'Sigmoid Options')
        self.sigmoid_options.setVisible(False)

        # Softmax
        self.softmax_options = gui.widgetBox(
            self.control_area, 'Softmax Options'
        )

        # SoftPlus
        self.softplus = gui.widgetBox(
            self.control_area, 'Softplus options'
        )
        self.softmax_options.setVisible(False)

        self.relu_options_gui()

    def set_activation(self):
        print("Setting activation")
        self.clear_all_activations()
        if self.activation == 0:
            self.relu_options_gui()
        if self.activation == 1:
            self.sigmoid_options_gui()
        if self.activation == 2:
            self.softmax_options_gui()

    def clear_all_activations(self):
        self.relu_options.setVisible(False)
        self.sigmoid_options.setVisible(False)
        self.softmax_options.setVisible(False)

    def sigmoid_options_gui(self):
        self.sigmoid_options.setVisible(True)

    def softmax_options_gui(self):
        if not check_widgets(self.softmax_options):
            gui.spin(self.softmax_options, self.master, 'axis',
                     minv=-1, maxv=1, step=1, label='Axis', spinType=int)
        self.softmax_options.setVisible(True)

    def relu_options_gui(self):
        if not check_widgets(self.relu_options):
            gui.spin(self.relu_options, self.master, 'alpha', minv=0.0,
                     maxv=10_000_000.0, step=0.01, label='Alpha', spinType=float)
            gui.spin(self.relu_options, self.master, 'max_value', minv=-1.0,
                     maxv=10_000_000.0, step=0.01, label='Max Value', spinType=float)
            gui.spin(self.relu_options, self.master, 'threshold', minv=0.0,
                     maxv=10_000_000.0, step=0.01, label='Threshold', spinType=float)
        self.relu_options.setVisible(True)


class CommonControls(KernelInitalizer, ActivationsGui):
    def __init__(self, widget, master, control_area) -> None:
        KernelInitalizer.__init__(
            self, widget, master, control_area)
        ActivationsGui.__init__(self, widget, master,
                                control_area)
