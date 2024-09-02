import TermTk as ttk

class NNCustomization:
    def __init__(self):
        self.layout = ttk.TTkGridLayout()
        self.setup_widgets()

    def setup_widgets(self):
        nn_cust_title = ttk.TTkLabel(text="NN Customization")
        self.layout.addWidget(nn_cust_title, 0, 0, 1, 2)

        # Learning Rate
        lr_label = ttk.TTkLabel(text="Learning Rate:")
        self.lr_input = ttk.TTkLineEdit(text="0.001")
        self.layout.addWidget(lr_label, 1, 0)
        self.layout.addWidget(self.lr_input, 1, 1)

        # Hidden Dimension
        hidden_dim_label = ttk.TTkLabel(text="Hidden Dimension: ")
        self.hidden_dim_input = ttk.TTkLineEdit(text="256")
        self.layout.addWidget(hidden_dim_label, 2, 0)
        self.layout.addWidget(self.hidden_dim_input, 2, 1)

        # Gamma
        gamma_label = ttk.TTkLabel(text="Gamma:")
        self.gamma_input = ttk.TTkLineEdit(text="0.9")
        self.layout.addWidget(gamma_label, 3, 0)
        self.layout.addWidget(self.gamma_input, 3, 1)

        # Apply Button
        apply_button = ttk.TTkButton(text="Apply Changes")
        apply_button.clicked.connect(self.apply_changes)
        self.layout.addWidget(apply_button, 4, 0, 1, 2)

    def get_layout(self):
        return self.layout

    def add_widget(self, widget, row, column, rowspan=1, colspan=1):
        self.layout.addWidget(widget, row, column, rowspan, colspan)

    def apply_changes(self):
        lr = float(self.lr_input.text())
        hidden_dim = int(self.hidden_dim_input.text())
        gamma = float(self.gamma_input.text())
        
        # TODO: Apply changes to the model