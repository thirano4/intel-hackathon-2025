import tkinter as tk
from tkinter import *
from tkinter import ttk,  messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import time
from main import *

def dbar():
    for i in range(5):
        time.sleep(0.5)  
        progress['value'] = i*25
        window.update_idletasks()
    progress['value'] = 0

window = tk.Tk()
window.geometry("640x320")
window.resizable(False, False)
window.title("")
window.config(background="white")

def open():
    global image_path
    l2.config(text="")
    image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
    image = Image.open(image_path)
    image = image.resize((128, 128))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo
    
def scan():
    dbar()
    arr = identify(image_path)
    l2.config(text=arr[1])
    
vLink = StringVar()

l1=Label(window, text="",padx=64,pady=18,font="SegoeUI 18",bg="white",fg="black")
l1.pack()
b1=Button(window, text='Scan', command=scan, width=8).place(x=16, y=36)
b2=Button(window, text="Open", command=open, width=8).place(x=16, y=8)
progress = ttk.Progressbar(window, orient="horizontal", length=200, mode="determinate")
progress.place(x=224, y=280, width=192)
image_label = tk.Label(window, bg="white")
image_label.place(x=256, y=100)

l2 = tk.Label(window, bg="white", fg="black", font="SegoeUI 11")
l2.place(x=272, y=250)

window.mainloop()
 