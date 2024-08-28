import TermTk as ttk

class Graphs:
    def __init__(self):
        self.layout = ttk.TTkHBoxLayout()
        self.setup_widgets()

    def setup_widgets(self):
        self.layout.addWidget(ttk.TTkButton(border=True, text="Graphs 1"), 0, 0)
        self.layout.addWidget(ttk.TTkButton(border=True, text="Graphs 2"), 0, 1)

    def get_layout(self):
        return self.layout

    def add_widget(self, widget, row, column, rowspan=1, colspan=1):
        self.layout.addWidget(widget, row, column, rowspan, colspan)