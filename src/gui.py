import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror, showwarning, showinfo
from ctypes import windll
from webcamrecognition import live_attendance
from dataset_generator import generate
from feature_extract import extract


root = tk.Tk()
root.title('Class Attendance and Face Mask Detectection')

windll.shcore.SetProcessDpiAwareness(1)

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

root.iconbitmap('C:/Users/cianm/Third_Year_Project/2022-ca326-mullarc4-weird4/src/logo/face_mask.ico')

# Create Label and add some text
heading = tk.Label(root, text="CA326: Third Year Project: Class Attendance using Facial Recognition and Mask Detection",
                   font=("Arial", 14))

# using place method we can set the position of label
heading.place(relx=0.0,
                  rely=0.0,
                  anchor='nw')

message = tk.Label(root,
                   text="Welcome to the user interface of our 3rd Year Project\n"
                        "Use the available commands below or see the user manual\n"
                        "\n"
                        "Students:\n"
                        "David Weir (19433086)\n"
                        "Cian Mullarkey (19763555)\n",
                   font=("Arial", 12))

message.place(relx = 0.5,
                   rely = 0.2,
                   anchor = 'center')

def data_gen():
    # declaring string variable
    # for storing name and password
    first_name_var = tk.StringVar()
    last_name_var = tk.StringVar()

    # defining a function that will
    # get the name and
    # print it on the screen
    def submit():
        first_name = first_name_var.get()
        last_name = last_name_var.get()
        if len(first_name) == 0 or len(last_name) == 0:
            showerror(title='Error', message='A first and second name must be entered.')
        else:
            print("The dataset is called : " + first_name + " " + last_name)
            print("Generating dataset")
            generate(first_name, last_name)
            print("Dataset generated")

            first_name_var.set("")
            last_name_var.set("")

    # creating a label for
    # name using widget Label
    first_name_label = tk.Label(root, text='First Name', font=('calibre', 10, 'bold'))

    # creating a entry for input
    # name using widget Entry
    first_name_entry = tk.Entry(root, textvariable=first_name_var, font=('calibre', 10, 'normal'))

    # creating a label for password
    last_name_label = tk.Label(root, text='Last Name', font=('calibre', 10, 'bold'))

    # creating a entry for password
    last_name_entry = tk.Entry(root, textvariable=last_name_var, font=('calibre', 10, 'normal'))

    # creating a button using the widget
    # Button that will call the submit function
    sub_btn = tk.Button(root, text='Submit', command=submit)


    first_name_label.place(relx = 0.4, rely = 0.7, anchor ='center')
    first_name_entry.place(relx = 0.6, rely = 0.7, anchor ='center')
    last_name_label.place(relx = 0.4, rely = 0.8, anchor ='center')
    last_name_entry.place(relx = 0.6, rely = 0.8, anchor ='center')
    sub_btn.place(relx = 0.5, rely = 0.9, anchor ='center')
#CLICLKING GENERATE BUTTON SHOWS/HIDES TEXT PROMPT !!!!!!!

def feat_ext():
    print("Extracting Features")
    extract()
    print("Extracted facial features successfully")

def attend():
    live_attendance()

# def user_manual():

usrman_button = ttk.Button(root, text='User Manual', command=data_gen)
data_button = ttk.Button(root, text='Generate Dataset', command=data_gen)
extract_button = ttk.Button(root, text='Extract Features', command=feat_ext)
attend_button = ttk.Button(root, text='Attendance', command=attend)


usrman_button.place(relx = 1.0,
                   rely = 1.0,
                   anchor ='se')

attend_button.place(relx = 0.5,
                   rely = 0.4,
                   anchor ='center')
extract_button.place(relx = 0.5,
                   rely = 0.5,
                   anchor = 'center')
data_button.place(relx = 0.5,
                   rely = 0.6,
                   anchor = 'center')

root.mainloop()