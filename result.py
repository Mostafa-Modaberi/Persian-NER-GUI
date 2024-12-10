from settings import *
import customtkinter as ctk
import tkinter as tk
from panels import *
from display_tokens import *
import numpy as np

class ShowResult(ctk.CTkFrame):
    def __init__(self, parent, labeled_tokens):
        super().__init__(master=parent)
        self.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        self.labeled_tokens = labeled_tokens
        self.token_display = TokenDisplay(self)
        self.token_display.display_tokens(self.labeled_tokens)
