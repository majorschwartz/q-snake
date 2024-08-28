import TermTk as ttk

class NNCustomization:
    def __init__(self):
        self.layout = ttk.TTkGridLayout()
        self.setup_widgets()

    def setup_widgets(self):
        self.layout.addWidget(ttk.TTkButton(border=True, text="NN Customization"), 0, 0)

    def get_layout(self):
        return self.layout

    def add_widget(self, widget, row, column, rowspan=1, colspan=1):
        self.layout.addWidget(widget, row, column, rowspan, colspan)