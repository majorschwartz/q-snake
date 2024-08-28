import TermTk as ttk

class RightWrapper:
    def __init__(self):
        self.layout = ttk.TTkGridLayout()
        self.setup_widgets()

    def setup_widgets(self):
        self.layout.addWidget(ttk.TTkButton(border=True, text="Right 1"), 0, 0)
        self.layout.addWidget(ttk.TTkButton(border=True, text="Right 2"), 1, 0)

    def get_layout(self):
        return self.layout

    def add_widget(self, widget, row, column, rowspan=1, colspan=1):
        self.layout.addWidget(widget, row, column, rowspan, colspan)