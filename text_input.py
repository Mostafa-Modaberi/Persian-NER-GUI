from settings import *
import customtkinter as ctk
import tkinter as tk
from panels import *
from settings import *
from bidi.algorithm import get_display

class InputPanel(ctk.CTkFrame):
    def __init__(self, parent, text_var, model):
        super().__init__(master=parent)
        self.grid(row=0, column=0, padx=10, pady=30, ipadx = 10, ipady = 10, sticky='snew')
        self.text_var = text_var
        self.model = model
        model_name = ctk.StringVar()
        model_name.set(f"مدل: {self.model}")
        # textbox   
        self.textbox = CustomTextbox(self)

        

        if self.text_var.get():
            self.textbox.insert("1.0", self.text_var.get())

        # clear button
        # Replace the button with a label
        self.clear_label = ctk.CTkLabel(
            self, 
            text="پاک کردن", 
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE),
            fg_color=CLEAR_BUTTON_BG,
            text_color=GREY,
        )
        self.clear_label.place(relx=0.97, rely=0.95, anchor='se')  # Adjust grid placement as needed


        # Bind the click event to the label

        self.clear_label.bind("<Button-1>", lambda event: self.clear_button_command())
   
    def get_text(self):
        return self.textbox.get("1.0", "end-1c")

    def delete_text(self):
        self.textbox.delete("1.0", "end")

    def clear_button_command(self):
        self.textbox.delete("1.0", "end")
        self.text_var.set("")
