import customtkinter as ctk
import ttkbootstrap as ttk
from settings import *
from panels import *
from tkinter import filedialog, Canvas

class ImportTxtFile(ctk.CTkFrame):
    def __init__(self, parent, import_func):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(fill = 'x', pady = 4, ipady = 8)
        self.import_func = import_func

        ImportTxtButton(self, 'بارگذاری فایل متنی', self.open_file_dialog)

    def open_file_dialog(self):
        path = filedialog.askopenfile().name
        self.import_func(path)
