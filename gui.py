import TermTk as ttk
from TermTk.TTkCore.TTkTerm.term import TTkTerm
import time

root = ttk.TTk()

def wait_for_resize(widget: ttk.TTkLabel):
	term_size = TTkTerm.getTerminalSize()
	if term_size.lines < 15:
		widget.setText(f"Please make the terminal window larger ({term_size.lines}x{term_size.columns})")
		return False
	return True

gridLayout = ttk.TTkGridLayout()
root.setLayout(gridLayout)

label = ttk.TTkLabel(parent=root, text="Loading...")

ready = wait_for_resize(label)
if ready:
	label.setText("Ready")

root.mainloop()