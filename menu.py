from settings import *
import customtkinter as ctk
import tkinter as tk
from panels import *
from import_txt_file import ImportTxtFile
from PIL import Image, ImageTk

class Menu(ctk.CTkTabview):
    def __init__(self, parent, model_vars, model_options, predict_function, import_txt_file, export_labeled_sentences, open_github):
        super().__init__(
            master=parent,
            text_color=WHITE,
        )
        self.model_vars = model_vars
        self.model_options = model_options
        self.grid(row=0, column=1, padx=10, pady=10,rowspan=2, sticky='nsew')

        self.add('تنظیمات')
        self.add('توضیحات')
        self.add('تولید کننده')  

        # configure font
        for child in self._segmented_button._buttons_dict.values():
            child.configure(font=ctk.CTkFont(FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE))

        # widgets
        SettingsFrame(self.tab('تنظیمات'), self.model_vars, predict_function, import_txt_file, export_labeled_sentences)
        DescriptionFrame(self.tab('توضیحات'))
        ProducerFrame(self.tab('تولید کننده'), open_github)

class SettingsFrame(ctk.CTkFrame):
    def __init__(self, parent, model_vars, predict_function, import_txt_file, export_labeled_sentences):
        super().__init__(master=parent, fg_color='transparent')
        self.import_txt_file = import_txt_file
        self.pack(expand=True, fill='both', padx=10, pady=10)
        model_validation_score = MODEL_PATH[model_vars['model'].get()]['model_validation_score']
        model_validation_text = MODEL_PATH[model_vars['model'].get()]['model_validation_text']


        LabelPanel(self, 'مدل')
        DropdownPanel(self, model_vars['model'], list(MODEL_PATH.keys()))
        SwitchPanel(self, (model_vars['spell_check_var'], 'بررسی املایی'))
        ImportTxtFile(self, import_txt_file)
        LabelPanel(self, 'ذخیره')
        ExportPanel(self, export_labeled_sentences)
        CompileButton(self, 'پیشبینی کن', predict_function)
        SecondayLabelPanel(self, score = model_validation_score, text = model_validation_text)
        LabelPanel(self, 'مشخصات مدل', side='bottom')

class DescriptionFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both', padx=10, pady=10)

        LabelPanel(self, 'این برنامه چگونه کار می‌کند')
        photo = self.read_image(IMAGE_PATHES['explanation'], (388, 514))
        ImageOutput(self, photo)

        CompileButton(self, 'مقایسه مدل‌ها', self.new_window)

    def new_window(self):
        self.new_window = ctk.CTkToplevel(self, fg_color=BACKGROUND)
        self.new_window.geometry(f'{APP_SIZE[0]}x{APP_SIZE[1]}')
        self.new_window.resizable(False, False)
        self.new_window.iconbitmap(ICON_PATH)
        
        self.new_window.attributes('-topmost', True)

        self.new_window.title('Model Comparison')

        photo = self.read_image(IMAGE_PATHES['comparison'], (APP_SIZE[0] + int(APP_SIZE[0]*0.2), APP_SIZE[1] + int(APP_SIZE[1]*0.2)))
        ImageOutput(self.new_window, photo)

    def read_image(self, path, size):
        image = Image.open(path)
        resized_image = image.resize(size) 
        photo = ImageTk.PhotoImage(resized_image)
        return photo

class ProducerFrame(ctk.CTkFrame):
    def __init__(self, parent, open_github):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both', padx=10, pady=10)

        explanation_app = (
            "این اپلیکیشن، برای تشخیص موجودیت‌های نامدار در جملات\n"
            "فارسی با تمرکز بر روی نام و نام‌خانوادگی ایجاد شده‌است و با\n"
            "استفاده از مدل‌های یادگیری ماشین، این اپلیکیشن می‌تواند\n"
            "اسامی را به صورت خودکار از درون متن پیش‌بینی کند"
        )

        explanation_model = (
            "تشخیص اسامی نامدار فارسی، به ویژه نام و نام‌خانوادگی\n"
            "در پردازش زبان طبیعی اهمیت زیادی دارد\n"
            "این فرآیند به بهبود دقت و کارایی سیستم‌های جستجو و\n"
            "بازیابی اطلاعات کمک می‌کند\n"
            "همچنین، در کاربردهای امنیتی و نظارتی، شناسایی دقیق اسامی\n"
            "می‌تواند به تشخیص هویت و جلوگیری از تقلب کمک کند\n"
            "در حوزه تحلیل داده‌های اجتماعی و بازاریابی، تشخیص اسامی\n"
            "به تحلیل رفتار کاربران و شخصی‌سازی خدمات منجر می‌شود\n"
            "در نهایت، این تکنولوژی می‌تواند به توسعه ابزارهای \n"
            "ترجمه و تعاملات بین‌المللی کمک کند"
        )

        LabelPanel(self, 'درباره اپلیکیشن')
        ProducerPanel(self, explanation_app)
        LabelPanel(self, 'درباره هدف')
        ProducerPanel(self, explanation_model)

        LabelPanel(self, 'درباره تولید کننده')
        ProducerPanel(self, 'مصطفی مدبری')
        ProducerPanel(self, 'دانشگاه شهید رجایی تهران، دانشکده مهندسی کامپیوتر')
        ProducerPanel(self, 'دی ماه 1403')

        CompileButton(self, 'باز کردن گیت‌ پروژه', open_github)

class ExportPanel(ctk.CTkFrame):
    def __init__(self, parent, export_labeled_sentences):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(side='top', expand=True, fill='both') 

        self.name_string = ctk.StringVar(value='')
        self.file_string = ctk.StringVar(value='txt')
        self.path_string = ctk.StringVar(value='')
        
        # widgets
        FileNamePanel(self, self.name_string, self.file_string)
        FilePathPanel(self, self.path_string)

        ExportButton(self, export_labeled_sentences, self.name_string, self.file_string, self.path_string)

