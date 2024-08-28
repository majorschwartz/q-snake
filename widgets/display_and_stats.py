import TermTk as ttk

class DisplayAndStats:
	def __init__(self):
		self.layout = ttk.TTkHBoxLayout()
		self.setup_widgets()

	def setup_widgets(self):
		display = ttk.TTkLabel(border=True, text="Display")
		stats = ttk.TTkLabel(border=True, text="Stats")
		self.layout.addWidget(display, 0, 1, 1, 1)
		self.layout.addWidget(stats, 0, 2, 1, 1)

	def get_layout(self):
		return self.layout

	def add_widget(self, widget, row, column, rowspan=1, colspan=1):
		self.layout.addWidget(widget, row, column, rowspan, colspan)