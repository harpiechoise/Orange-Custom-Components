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
        self.activation_combo = gui.comboBox(
            widget, self.master, 'initalizer',
            items=('Random Normal',
                   'Random Uniform',
                   'Truncated Normal',
                   'Zeros',
                   'Ones',
                   'Glorot Normal',
                   'Glorot Uniform'),
            label='Layer Activation',
            callback=self.set_initalizer
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
        self.clear_all()
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

    def clear_all(self):
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
