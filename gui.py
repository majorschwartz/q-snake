import TermTk as ttk
from widgets.left.left_wrapper import LeftWrapper
from widgets.right.right_wrapper import RightWrapper

gridLayout = ttk.TTkGridLayout()
root = ttk.TTk(layout=gridLayout)

LWrap = LeftWrapper()
RWrap = RightWrapper()

gridLayout.addItem(LWrap.get_layout(), 0, 0, 1, 3)
gridLayout.addItem(RWrap.get_layout(), 0, 3, 1, 1)


root.mainloop()