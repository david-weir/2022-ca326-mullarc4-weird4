import tkinter as tk
from tkinter import ttk
from webcamrecognition import live_attendance
# from dataset_generator import generate
from feature_extract import extract


root = tk.Tk()
root.title('Class Attendance and Face Mask Detectection')

window_width = 900
window_height = 600

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
# root.iconbitmap('./logo/mask.ico')


message = tk.Label(root, text=
          "\tCA326: Third Year Project: Class Attendance using Facial Recognition and Mask Detection\n"
          "\n"
          "\tWelcome to the user interface of our 3rd Year Project\n"
          "\tUse the available commands below or see the user manual\n"
          "\n"
          "\tStudents:\n"
          "\tDavid Weir (19433086)\n"
          "\tCian Mullarkey (19763555)\n")
message.pack()

# label = ttk.Label(root)
# label['text'] = 'Hi, there'
# label.pack()
#
def data_gen():
    print("Generating Dataset")
# generate()

def feat_ext():
    print("Extracting Features")
    extract()
    print("Extracted facial features successfully")

def attend():
    live_attendance()

data_button = ttk.Button(root, text='Generate Dataset (eventually)', command=data_gen)
extract_button = ttk.Button(root, text='Build Recognition Model', command=feat_ext)
attend_button = ttk.Button(root, text='Attendance', command=attend)

data_button.pack()
extract_button.pack()
attend_button.pack()

# def select(option):
#     print(option)
#
#
# ttk.Button(root, text='Generate New Dataset', command=lambda: select('Generat New Dataset')).pack()
# ttk.Button(root, text='Build Recognition Model',command=lambda: select('Build Recognition Model')).pack()
# ttk.Button(root, text='Attendance', command=lambda: select('Attendance')).pack()
# #
# def return_pressed(event):
#     print('Return key pressed.')
# btn = ttk.Button(root, text='Save')
# btn.bind('<Return>', return_pressed)
#
#
# btn.focus()
# btn.pack(expand=True)



root.mainloop()