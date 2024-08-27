import TermTk as ttk

gridLayout = ttk.TTkGridLayout()
root = ttk.TTk(layout=gridLayout)

LWrap = ttk.TTkGridLayout()
RWrap = ttk.TTkGridLayout(columnMinWidth=10)

gridLayout.addItem(LWrap, 0, 0)

LWrap.addWidget(ttk.TTkButton(border=True, text="Button1"), 0, 0)
LWrap.addWidget(ttk.TTkButton(border=True, text="Button2"), 1, 0)

gridLayout.addItem(RWrap, 0, 1)

RWrap.addWidget(ttk.TTkButton(border=True, text="Button3"), 0, 0, 1, 2)
RWrap.addWidget(ttk.TTkButton(border=True, text="Button4"), 1, 0, 1, 1)

root.mainloop()