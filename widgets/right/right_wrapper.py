import TermTk as ttk
from widgets.right.nn_cust import NNCustomization
from widgets.right.snake_cust import SnakeCustomization

class RightWrapper:
    def __init__(self):
        self.layout = ttk.TTkGridLayout()
        self.nn_cust = NNCustomization()
        self.snake_cust = SnakeCustomization()
        self.setup_widgets()

    def setup_widgets(self):
        self.layout.addItem(self.nn_cust.get_layout(), 0, 0)
        self.layout.addItem(self.snake_cust.get_layout(), 1, 0)

    def get_layout(self):
        return self.layout

    def add_widget(self, widget, row, column, rowspan=1, colspan=1):
        self.layout.addWidget(widget, row, column, rowspan, colspan)