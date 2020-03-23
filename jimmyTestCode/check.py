import tkinter
import tkinter.filedialog
from tkinter.filedialog import askopenfilename
filename = askopenfilename()

#def center_window(width, height):
#    # get screen width and height
#    screen_width = root.winfo_screenwidth()
#    screen_height = root.winfo_screenheight()
#
#    # calculate position x and y coordinates
#    x = (screen_width/2) - (width/2)
#    y = (screen_height/2) - (height/2)
#    root.geometry('%dx%d+%d+%d' % (width, height, x, y))
#
#def OnButtonClick(self):
#    self.entryVariable.set( tkinter.filedialog.askopenfilename() )
#    self.entry.focus_set()
#    self.entry.selection_range(0, tkinter.END)
#
#
#root = tkinter.Tk()
#center_window(400, 300)
#root.title("File System")
#root.entryVariable = tkinter.StringVar()
#frame=tkinter.Frame(root)
#root.entry = tkinter.Entry(frame,textvariable=root.entryVariable)
#B = tkinter.Button(frame, text ="Choose", command=lambda: OnButtonClick(root))
#root.entry.grid(column=0,row=0)
#B.grid(column=1,row=0)
#frame.pack(pady=100)  #Change this number to move the frame up or down
#root.mainloop()
