from orangewidget import gui


class KernelInitalizer:
    activation = ''

    def __init__(self, widget, master):
        self.activation_combo = gui.comboBox(
            widget, master, 'activation',
            items=('Random Normal', 'Random Uniform'), label='Layer Activation',
            callback=self.set_activation
        )

    def set_activation(self):
        print(self.activation)
        print('Changed')
