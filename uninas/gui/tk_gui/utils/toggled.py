"""
adapted from https://stackoverflow.com/questions/13141259/expandable-and-contracting-frame-in-tkinter
"""

import tkinter as tk
from tkinter import ttk
from uninas.gui.tk_gui.utils.tooltip import CreateToolTip


class ToggledFrame(tk.Frame):

    def __init__(self, parent, text="", *args, relief=tk.RAISED, borderwidth=1, show=False, **options):
        tk.Frame.__init__(self, parent, *args, relief=relief, borderwidth=borderwidth, **options)

        self.show = tk.IntVar()
        self.show.set(show)

        self.text = text
        self._was_highlighted = False
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=tk.X, expand=1)

        self.toggle_button = ttk.Checkbutton(self.title_frame, width=2, text=text, command=self.toggle,
                                             variable=self.show, style='Toolbutton')
        self.toggle_button.pack(side=tk.LEFT, expand=1, fill=tk.X)
        CreateToolTip(self.toggle_button, "Show/Hide")

        self.sub_frame = tk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        self.toggle()

    def toggle(self):
        if bool(self.show.get()):
            self.sub_frame.pack(fill=tk.X, expand=1)
        else:
            self.sub_frame.forget()


if __name__ == "__main__":
    root = tk.Tk()

    t = ToggledFrame(root, text='Rotate')
    t.pack(fill="x", expand=1, pady=2, padx=2, anchor="n")

    ttk.Label(t.sub_frame, text='Rotation [deg]:').pack(side="left", fill="x", expand=1)
    ttk.Entry(t.sub_frame).pack(side="left")

    t2 = ToggledFrame(root, text='Resize')
    t2.pack(fill="x", expand=1, pady=2, padx=2, anchor="n")

    for i in range(10):
        ttk.Label(t2.sub_frame, text='Test' + str(i)).pack()

    t3 = ToggledFrame(root, text='Fooo')
    t3.pack(fill="x", expand=1, pady=2, padx=2, anchor="n")

    for i in range(10):
        ttk.Label(t3.sub_frame, text='Bar' + str(i)).pack()

    root.mainloop()
