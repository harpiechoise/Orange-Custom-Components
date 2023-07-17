from orangewidget import gui
from PyQt5 import QtWidgets
from tensorflow import keras
from keras import initializers, regularizers
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


class Regularizers:
    regularizer = 0
    l1 = 0
    l2 = 0
    factor = 0
    mode = 0

    def __init__(self, widget, master, control_area) -> None:
        self.master = master
        self.control_area = control_area
        gui.comboBox(widget, master, 'regularizer', items=(
            'None',
            'L1',
            'L2',
            'L1L2',
            'Orthogonal Regularizer'
        ), label='Regularizer', callback=self.set_regularizer)

        self.l1_options = gui.widgetBox(
            self.control_area, 'L1 Regularization Settings'
        )
        self.l1_options.setVisible(False)

        self.l2_options = gui.widgetBox(
            self.control_area, 'L2 Regularization Settings'
        )
        self.l2_options.setVisible(False)

        self.l1l2_options = gui.widgetBox(
            self.control_area, 'L1L2 Regularization Settings'
        )
        self.l1l2_options.setVisible(False)

        self.orthogonal_options = gui.widgetBox(
            self.control_area, 'Orthogonal Regularizer Settings'
        )

        self.orthogonal_options.setVisible(False)

    def set_regularizer(self):
        self.clear_regularizer_ui()
        if self.regularizer == 1:
            self.l1_options_gui()
        elif self.regularizer == 2:
            self.l2_options_gui()
        elif self.regularizer == 3:
            self.l1l2_options_gui()
        elif self.regularizer == 4:
            self.orthogonal_options_gui()

    def make_regularizer(self):
        if self.regularizer == 1:
            return regularizers.l1(l1=self.l1)
        elif self.regularizer == 2:
            return regularizers.l2(l2=self.l2)
        elif self.regularizer == 3:
            return regularizers.l1_l2(l1=self.l1, l2=self.l2)
        elif self.regularizer == 4:
            if self.mode == 0:
                mode = 'rows'
            else:
                mode = 'columns'
            return regularizers.orthogonal_regularizer(self.factor, mode=mode)

    def clear_regularizer_ui(self):
        self.l1_options.setVisible(False)
        self.l2_options.setVisible(False)
        self.l1l2_options.setVisible(False)
        self.orthogonal_options.setVisible(False)

    def l1_options_gui(self):
        if not check_widgets(self.l1_options):
            gui.spin(self.l1_options, self.master, 'l1', minv=0,
                     maxv=10_000_000, step=.01, label='L1', spinType=float)
        self.l1_options.setVisible(True)

    def l2_options_gui(self):
        if not check_widgets(self.l2_options):
            gui.spin(self.l2_options, self.master, 'l2', minv=0,
                     maxv=10_000_000, step=.01, label='L2', spinType=float)
        self.l2_options.setVisible(True)

    def l1l2_options_gui(self):
        if not check_widgets(self.l1l2_options):
            gui.spin(self.l1l2_options, self.master, 'l1', minv=0,
                     maxv=10_000_000, step=.01, label='L1', spinType=float)
            gui.spin(self.l1l2_options, self.master, 'l2', minv=0,
                     maxv=10_000_000, step=.01, label='L2', spinType=float)
        self.l1l2_options.setVisible(True)

    def orthogonal_options_gui(self):
        if not check_widgets(self.orthogonal_options):
            gui.spin(self.orthogonal_options, self.master, 'factor',
                     minv=0, maxv=10_000_000, step=.01, label='Factor', spinType=float)
            gui.comboBox(self.orthogonal_options, self.master,
                         'mode', items=('Rows', 'Columns'), label='Mode')

        self.orthogonal_options.setVisible(True)


class BiasRegularizers:
    bias_regularizer = 0
    bias_l1 = 0.01
    bias_l2 = 0.01
    bias_factor = 0.01
    bias_mode = 0

    def __init__(self, widget, master, control_area) -> None:
        self.master = master
        self.control_area = control_area
        gui.comboBox(widget, master, 'bias_regularizer', items=(
            'None',
            'L1',
            'L2',
            'L1L2',
            'Orthogonal Regularizer'
        ), label='Bias Regularized', callback=self.bias_set_regularizer)

        self.bias_l1_options = gui.widgetBox(
            self.control_area, 'L1 Bias Regularization Settings'
        )
        self.bias_l1_options.setVisible(False)

        self.bias_l2_options = gui.widgetBox(
            self.control_area, 'L2 Bias Regularization Settings'
        )
        self.bias_l2_options.setVisible(False)

        self.bias_l1l2_options = gui.widgetBox(
            self.control_area, 'L1L2 Bias Regularization Settings'
        )
        self.bias_l1l2_options.setVisible(False)

        self.bias_orthogonal_options = gui.widgetBox(
            self.control_area, 'Orthogonal Bias Regularizer Settings'
        )

        self.bias_orthogonal_options.setVisible(False)

    def bias_set_regularizer(self):
        self.bias_clear_regularizer_ui()
        if self.bias_regularizer == 1:
            self.bias_l1_options_gui()
        elif self.bias_regularizer == 2:
            self.bias_l2_options_gui()
        elif self.bias_regularizer == 3:
            self.bias_l1l2_options_gui()
        elif self.bias_regularizer == 4:
            self.bias_orthogonal_options_gui()

    def make_bias_regularized(self):
        if self.bias_regularizer == 1:
            return regularizers.l1(l1=self.bias_l1)
        elif self.bias_regularizer == 2:
            return regularizers.l2(l2=self.bias_l2)
        elif self.bias_regularizer == 3:
            return regularizers.l1_l2(l1=self.bias_l1, l2=self.bias_l2)
        elif self.bias_regularizer == 4:
            if self.bias_mode == 0:
                mode = 'rows'
            else:
                mode = 'columns'
            return regularizers.orthogonal_regularizer(self.bias_factor, mode=mode)

    def bias_clear_regularizer_ui(self):
        self.bias_l1_options.setVisible(False)
        self.bias_l2_options.setVisible(False)
        self.bias_l1l2_options.setVisible(False)
        self.bias_orthogonal_options.setVisible(False)

    def bias_l1_options_gui(self):
        if not check_widgets(self.bias_l1_options):
            gui.spin(self.bias_l1_options, self.master, 'bias_l1', minv=0,
                     maxv=10_000_000, step=.01, label='L1', spinType=float)
        self.bias_l1_options.setVisible(True)

    def bias_l2_options_gui(self):
        if not check_widgets(self.bias_l2_options):
            gui.spin(self.bias_l2_options, self.master, 'bias_l2', minv=0,
                     maxv=10_000_000, step=.01, label='L2', spinType=float)
        self.bias_l2_options.setVisible(True)

    def bias_l1l2_options_gui(self):
        if not check_widgets(self.bias_l1l2_options):
            gui.spin(self.bias_l1l2_options, self.master, 'bias_l1', minv=0,
                     maxv=10_000_000, step=.01, label='L1', spinType=float)
            gui.spin(self.bias_l1l2_options, self.master, 'bias_l2', minv=0,
                     maxv=10_000_000, step=.01, label='L2', spinType=float)
        self.bias_l1l2_options.setVisible(True)

    def bias_orthogonal_options_gui(self):
        if not check_widgets(self.bias_orthogonal_options):
            gui.spin(self.bias_orthogonal_options, self.master, 'bias_factor',
                     minv=0, maxv=10_000_000, step=.01, label='Factor', spinType=float)
            gui.comboBox(self.bias_orthogonal_options, self.master,
                         'bias_mode', items=('Rows', 'Columns'), label='Mode')

        self.bias_orthogonal_options.setVisible(True)


class ActivityRegularizer:
    activity_regularizer = 0
    activity_l1 = 0.01
    activity_l2 = 0.01
    activity_factor = 0.01
    activity_mode = 0

    def __init__(self, widget, master, control_area) -> None:
        self.master = master
        self.control_area = control_area
        gui.comboBox(widget, master, 'activity_regularizer', items=(
            'None',
            'L1',
            'L2',
            'L1L2',
            'Orthogonal Regularizer'
        ), label='Activity Regularizer', callback=self.activity_set_regularizer)

        self.activity_l1_options = gui.widgetBox(
            self.control_area, 'L1 Activity Regularization Settings'
        )
        self.activity_l1_options.setVisible(False)

        self.activity_l2_options = gui.widgetBox(
            self.control_area, 'L2 Activity Regularization Settings'
        )
        self.activity_l2_options.setVisible(False)

        self.activity_l1l2_options = gui.widgetBox(
            self.control_area, 'L1L2 Activity Regularization Settings'
        )
        self.activity_l1l2_options.setVisible(False)

        self.activity_orthogonal_options = gui.widgetBox(
            self.control_area, 'Orthogonal Activity Regularizer Settings'
        )

        self.activity_orthogonal_options.setVisible(False)

    def activity_set_regularizer(self):
        self.activity_clear_regularizer_ui()
        if self.activity_regularizer == 1:
            self.activity_l1_options_gui()
        elif self.activity_regularizer == 2:
            self.activity_l2_options_gui()
        elif self.activity_regularizer == 3:
            self.activity_l1l2_options_gui()
        elif self.activity_regularizer == 4:
            self.activity_orthogonal_options_gui()

    def make_activity_regularized(self):
        if self.activity_regularizer == 1:
            return regularizers.l1(l1=self.bias_l1)
        elif self.activity_regularizer == 2:
            return regularizers.l2(l2=self.bias_l2)
        elif self.activity_regularizer == 3:
            return regularizers.l1_l2(l1=self.bias_l1, l2=self.bias_l2)
        elif self.activity_regularizer == 4:
            if self.activity_mode == 0:
                mode = 'rows'
            else:
                mode = 'columns'
            return regularizers.orthogonal_regularizer(self.activity_factor, mode=mode)

    def activity_clear_regularizer_ui(self):
        self.activity_l1_options.setVisible(False)
        self.activity_l2_options.setVisible(False)
        self.activity_l1l2_options.setVisible(False)
        self.activity_orthogonal_options.setVisible(False)

    def activity_l1_options_gui(self):
        if not check_widgets(self.activity_l1_options):
            gui.spin(self.activity_l1_options, self.master, 'activity_l1', minv=0,
                     maxv=10_000_000, step=.01, label='L1', spinType=float)
        self.activity_l1_options.setVisible(True)

    def activity_l2_options_gui(self):
        if not check_widgets(self.activity_l2_options):
            gui.spin(self.activity_l2_options, self.master, 'activity_l2', minv=0,
                     maxv=10_000_000, step=.01, label='L2', spinType=float)
        self.activity_l2_options.setVisible(True)

    def activity_l1l2_options_gui(self):
        if not check_widgets(self.activity_l1l2_options):
            gui.spin(self.activity_l1l2_options, self.master, 'activity_l1', minv=0,
                     maxv=10_000_000, step=.01, label='L1', spinType=float)
            gui.spin(self.activity_l1l2_options, self.master, 'activity_l2', minv=0,
                     maxv=10_000_000, step=.01, label='L2', spinType=float)
        self.activity_l1l2_options.setVisible(True)

    def activity_orthogonal_options_gui(self):
        if not check_widgets(self.activity_orthogonal_options):
            gui.spin(self.activity_orthogonal_options, self.master, 'activity_factor',
                     minv=0, maxv=10_000_000, step=.01, label='Factor', spinType=float)
            gui.comboBox(self.activity_orthogonal_options, self.master,
                         'activity_mode', items=('Rows', 'Columns'), label='Mode')

        self.activity_orthogonal_options.setVisible(True)


class CommonControls(KernelInitalizer, ActivationsGui):
    def __init__(self, widget, master, control_area) -> None:
        KernelInitalizer.__init__(
            self, widget, master, control_area)
        ActivationsGui.__init__(self, widget, master,
                                control_area)
