from tkinter import *
from start_live_feed import live
window = Tk()
window.geometry("400x400")

li=Label(window,text="abnormal event detect",font=15)
li.grid(row=0,column=0)

btn = Button(window,text="start",command=live)
btn.grid(row=3,column=2)

window.mainloop()

