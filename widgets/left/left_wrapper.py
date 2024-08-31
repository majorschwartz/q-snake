import TermTk as ttk
from widgets.left.display_and_stats import DisplayAndStats
from widgets.left.graphs import Graphs

class LeftWrapper:
    def __init__(self):
        self.layout = ttk.TTkGridLayout()
        self.display_and_stats = DisplayAndStats()
        self.graphs = Graphs()
        self.setup_widgets()

    def setup_widgets(self):
        self.layout.addItem(self.display_and_stats.get_layout(), 1, 0, 2, 1)
        self.layout.addItem(self.graphs.get_layout(), 3, 0, 1, 1)

    def get_layout(self):
        return self.layout

    def add_widget(self, widget, row, column, rowspan=1, colspan=1):
        self.layout.addWidget(widget, row, column, rowspan, colspan)