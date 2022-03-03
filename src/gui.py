import tkinter as tk
import subprocess
from tkinter import ttk, Canvas, messagebox
from tkinter.messagebox import showerror
from ctypes import windll
from PIL import ImageTk, Image
from webcamrecognition import live_attendance
from dataset_generator import generate
from feature_extract import extract

# fix text bluriness on Windows
windll.shcore.SetProcessDpiAwareness(1)

# window title
root = tk.Tk()
root.title('Class Attendance and Face Mask Detectection')


def set_window_size(window):
    # set window size
    window_width = 1200
    window_height = 800

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')


# set window icon
root.iconbitmap('./icon/face_mask.ico')

# create canvas
canvas = Canvas(root, width=400, height=400)
canvas.pack(fill="both", expand=True)

# set window background
bg = ImageTk.PhotoImage(Image.open('./background/masks.jpg'))  # PIL solution
canvas.create_image(0, 0, anchor="nw", image=bg)

# create header label
heading = tk.Label(root,
                   text="CA326:\n"
                        "Third Year Project: Class Attendance using\n"
                        "Facial Recognition and Mask Detection",
                   font=("Arial", 22),
                   bg="white")

# set the position of label
heading.place(relx=0.5, rely=0.1, anchor='center')

# set welcome intro label
message = tk.Label(root,
                   text="Welcome to the user interface of our 3rd Year Project\n"
                        "Use the available commands below or see the user manual\n"
                        "\n"
                        "\n"
                        "Students:\n"
                        "David Weir (19433086)\n"
                        "Cian Mullarkey (19763555)\n",
                   font=("Arial", 12),
                   bg="white")

message.place(relx=0.5, rely=0.4, anchor='center')


def user_manual():
    path = "../docs/3-final-reports/User Manual.pdf"
    subprocess.Popen([path], shell=True)


# defining a function that will get the fname and lname from the user and pass them as
# parameters through to the dataset generator function
def data_gen():
    first_name = first_name_var.get()
    last_name = last_name_var.get()

    if len(first_name) == 0 or len(last_name) == 0:
        showerror(title='Error', message='A first and last name must be entered.')
    else:
        MsgBox = tk.messagebox.askquestion('Generate Dataset',
                                           'Do you wish to build a new dataset?',
                                           icon='info')
        if MsgBox == 'yes':
            print("Generating dataset")
            print("The dataset is called: " + first_name + " " + last_name)
            generate(first_name, last_name)
            messagebox.showinfo("showinfo", "Dataset generated.")
            print("Dataset generated")

        first_name_var.set("")
        last_name_var.set("")


# declaring string variables for storing fname and lname
first_name_var = tk.StringVar()
last_name_var = tk.StringVar()


# create labels and entries for fname and lname
first_name_label = tk.Label(root, text='First Name', font=('calibre', 10, 'bold'))
first_name_label.place(relx=0.4, rely=0.6, anchor='center')


first_name_entry = tk.Entry(root, textvariable=first_name_var, font=('calibre', 10, 'normal'))
first_name_entry.place(relx=0.6, rely=0.6, anchor='center')


last_name_label = tk.Label(root, text='Last Name', font=('calibre', 10, 'bold'))
last_name_label.place(relx=0.4, rely=0.65, anchor='center')

last_name_entry = tk.Entry(root, textvariable=last_name_var, font=('calibre', 10, 'normal'))
last_name_entry.place(relx=0.6, rely=0.65, anchor='center')


# calls feature extract program
def feat_ext():
    MsgBox = tk.messagebox.askquestion('Feature Extract', 'Do you wish to begin extracting features from the '
                                                          'datasets?\n '
                                                          'This may take up to a minute or two.',
                                       icon='info')
    if MsgBox == 'yes':
        print("Extracting features")
        extract()
        messagebox.showinfo("Feature Extract", "Extracted facial features successfully.")
        print("Extracted facial features successfully")


# calls webcam recognition program
def attend():
    # print("Streaming")
    MsgBox = tk.messagebox.askquestion('Live Attendance',
                                       'Do you wish to begin taking live attendance and mask detection?\n',
                                       icon='info')
    if MsgBox == 'yes':
        # print("Streaming")
        messagebox.showinfo("Live Attendance", "Beginning Stream.\n" "Press 'Q' to end stream.")
        live_attendance()
        messagebox.showinfo("Live Attendance", "Stream Ended.")
        print("Stream Ended")


usrman_button = ttk.Button(root, text='User Manual', command=user_manual)
usrman_button.place(relx=0.5, rely=0.37, anchor='center')

data_gen_btn = tk.Button(root, text='Generate New Dataset', command=data_gen)
data_gen_btn.place(relx=0.5, rely=0.7, anchor='center')

attend_button = ttk.Button(root, text='Attendance', command=attend)
attend_button.place(relx=0.5, rely=0.9, anchor='center')

extract_button = ttk.Button(root, text='Extract Features', command=feat_ext)
extract_button.place(relx=0.5, rely=0.8, anchor='center')

set_window_size(root)
root.mainloop()
