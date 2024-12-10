import customtkinter as ctk
from settings import *
import darkdetect
from tkinter import messagebox
from menu import Menu
from result import ShowResult
from text_input import InputPanel
from preprocess import PersianTextProcessor
from bidi.algorithm import get_display
import arabic_reshaper
from predicter import *
from PIL import Image, ImageTk
import webbrowser
