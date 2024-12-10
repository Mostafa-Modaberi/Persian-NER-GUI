import customtkinter as ctk
from settings import *
from tkinter import filedialog, Canvas
import tkinter as tk

class Panel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill = 'x', pady = 4, ipady = 8)

class SegmentedPanel(Panel):
    def __init__(self, parent, text, data_var, options):
        super().__init__(parent = parent)

        self.title = ctk.CTkLabel(self, text=text, text_color=WHITE, font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE))
        self.title.pack(fill='x', padx = 8)

        self.segmented_button = ctk.CTkSegmentedButton(
            self, 
            values=options, 
            variable=data_var,
            font = ctk.CTkFont(FONT_ENGLISH, size=NORMAL_FONT_SIZE),
            dynamic_resizing = False,
            orientation='vertical'
        )
        self.segmented_button.pack(fill = 'x', expand = True, padx = 8)

class LabelPanel(ctk.CTkLabel):
    def __init__(self, parent, text, side='top'):
        super().__init__(parent, text=text, text_color=WHITE, font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE))
        self.pack(side=side, fill='x', padx = 8)

class DropdownPanel(ctk.CTkOptionMenu):
    def __init__(self, parent, data_var, options):
        super().__init__(
            master = parent, 
            values=options, 
            variable=data_var,
            font=ctk.CTkFont(FONT_ENGLISH, size=NORMAL_FONT_SIZE),
            dropdown_font=ctk.CTkFont(FONT_ENGLISH, size=NORMAL_FONT_SIZE),
            fg_color=DARK_GREY,
            button_color=DROPDOWM_MAIN_COLOR,
            button_hover_color=DROPDOWM_HOVER_COLOR
        )
        self.pack(fill='x', pady=4)

class CustomTextbox(ctk.CTkTextbox):
    def __init__(self, parent):
        super().__init__(
            master=parent,
            font=ctk.CTkFont(family=FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE),
            border_width=2,
            fg_color=DARK_GREY,
            border_color=DARK_GREY,
            text_color=WHITE
        )

        self.pack(padx=10, pady=10, fill="both", expand=True)
        self._textbox.configure(wrap="word")
        self._textbox.tag_configure("rtl", justify="right")
        self._textbox.tag_add("rtl", "1.0", "end")

class ImportTxtButton(ctk.CTkButton):
    def __init__(self, parent, text, command):
        super().__init__(
            master=parent, 
            text=text, 
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE),
            text_color=WHITE,
            command=command
        )
        self.pack(fill='x')

class ClearButton(ctk.CTkButton):
    def __init__(self, parent, command, text):
        super().__init__(
            master=parent, 
            text=text, 
            command=command,
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE),
            text_color=GREY,
            fg_color=DARK_GREY,
            hover_color=DARK_GREY,
        )
        self.place(relx=0.99, rely=0.99, anchor='se')

class FileNamePanel(Panel):
    def __init__(self, parent, name_string, file_string):
        super().__init__(parent = parent)

        self.columnconfigure((0,1,2,3), weight=1, uniform='a')
        self.rowconfigure((0,1), weight=1, uniform='a')
        english_font = ctk.CTkFont(FONT_ENGLISH, size=NORMAL_FONT_SIZE)
        persian_font = ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE)
        # data
        self.name_string = name_string
        self.name_string.trace('w', self.update_text)
        self.file_string = file_string

        # check boxes for the file format
        self.name_entry = ctk.CTkEntry(
            self, 
            textvariable=self.name_string, 
            font=english_font,
            placeholder_text='file name',
            placeholder_text_color=GREY
        )
        self.name_entry.grid(row=0, column=0, padx=4, columnspan=2, sticky='ew')

        txt_checkbox = ctk.CTkCheckBox(
            self, 
            text='txt',
            command=lambda: self.click('txt'), 
            variable=file_string, 
            onvalue='txt', 
            offvalue='csv', 
            border_width=1 ,
            font=english_font,
            checkbox_height=20,
            checkbox_width=20
        )
        csv_checkbox = ctk.CTkCheckBox(
            self, 
            text='csv',
            command=lambda: self.click('csv'), 
            variable=file_string, 
            onvalue='csv', 
            offvalue='txt', 
            border_width=1 ,
            font=english_font,
            checkbox_height=20,
            checkbox_width=20
        )

        txt_checkbox.grid(padx = 4,row=0, column=2, sticky='ew')
        csv_checkbox.grid(padx = 4,row=0, column=3, sticky='ew')

        # previwe the text
        self.output = ctk.CTkLabel(self, text = '', font=english_font)
        self.output.grid(row=1, column=0, padx=4, columnspan=4)

    def update_text(self, *args):
        if self.name_string.get():
            text = self.name_string.get().replace(' ', '_') + '.' + self.file_string.get()
            self.output.configure(text=text)
        else:
            self.output.configure(text='')

    def click(self, value):
        self.file_string.set(value)
        self.update_text()

class FilePathPanel(Panel):
    def __init__(self, parent, path_string):
        super().__init__(parent = parent)

        self.columnconfigure((0,1,2), weight=1, uniform='a')
        self.rowconfigure(0, weight=1)
        persian_font = ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE)
        english_font = ctk.CTkFont(FONT_ENGLISH, size=NORMAL_FONT_SIZE)
        self.path_string = path_string

        self.button = ctk.CTkButton(self, text='انتخاب پوشه', command=self.browse, font=persian_font)
        self.button.grid(row=0, column=2, pady=4, padx=4, sticky='ew')
        # path entry
        self.path_entry = ctk.CTkEntry(
            self, 
            textvariable=self.path_string, 
            font=english_font,
            placeholder_text='path to save',
            placeholder_text_color=GREY
        )
        self.path_entry.grid(row=0, column=0, columnspan=2, padx=4, sticky='ew')

    def browse(self):
        path = filedialog.askdirectory()
        self.path_string.set(path)

class ExportButton(ctk.CTkButton):
    def __init__(self, parent, export_image, name_string, file_format, path_string):
        super().__init__(
            master=parent, 
            text='ذخیره کن', 
            command=self.save, 
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE),
            text_color=WHITE
        )
        self.export_image = export_image
        self.name_string = name_string
        self.file_format = file_format
        self.path_string = path_string
        self.pack(pady=4, fill='x')

    def save(self):
        self.export_image(name = self.name_string.get(), path = self.path_string.get(), file_format = self.file_format.get())

class SwitchPanel(Panel):
    def __init__(self, parent, *args): #((var, text), (var, text), ...)
        super().__init__(parent=parent)

        for var, text in args:
            # Create a frame to hold the switch and label
            frame = ctk.CTkFrame(self, fg_color='transparent')
            frame.pack(side='left', expand=True, fill='both', padx=4, pady=4)

            # Create the switch
            self.switch = ctk.CTkSwitch(
                frame, 
                variable=var, 
                button_color=BLUE, 
                fg_color=SLIDER_BG, 
                progress_color=LIGHT_BLUE,
                button_hover_color=LIGHT_BLUE,
                text='',
                font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE)
            )
            self.switch.pack(side='left')

            # Create the label
            label = ctk.CTkLabel(
                frame, 
                text=text, 
                text_color=WHITE, 
                font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE)
            )
            label.pack(side='right', padx=4)

class SecondayLabelPanel(Panel):
    def __init__(self, parent, text,score):
        super().__init__(parent = parent)
        self.text = text
        self.score = score
        
        ctk.CTkLabel(
            self, 
            text=self.score, 
            text_color=LIGHT_GREY, 
            font=ctk.CTkFont(FONT_ENGLISH, size=SMALL_FONT_SIZE), 
            justify = 'left',
            anchor = 'w'
        ).pack(side='left', expand=True, fill='both', padx = 4)
        
        ctk.CTkLabel(
            self, 
            text=self.text, 
            text_color=LIGHT_GREY, 
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=SMALL_FONT_SIZE), 
            justify = 'right',
            anchor = 'e'
        ).pack(side='right', expand=True, fill='both', padx = 4)
        
        self.pack(side='bottom', fill = 'x', pady = 4, ipady = 8,ipadx = 4)

class ImageOutput(ctk.CTkLabel):
    def __init__(self, parent, photo):
        super().__init__(
            master=parent,
            image=photo, 
            text=""
        )
        self.pack(pady=8, fill='x')

class ProducerPanel(Panel):
    def __init__(self, parent, text):
        super().__init__(parent = parent)
        ctk.CTkLabel(self, text=text, text_color=WHITE, font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=SMALL_FONT_SIZE), justify='right').pack(expand=True, fill='both', padx = 4,ipadx = 4)

class CompileButton(ctk.CTkButton):
    def __init__(self, parent, text, command):
        super().__init__(
            master=parent, 
            text=text, 
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE),
            text_color=WHITE,
            command=command
        )
        self.pack(side='bottom', pady=8, fill='x')