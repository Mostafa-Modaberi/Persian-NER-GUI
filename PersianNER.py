from imports import *


class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=(WHITE, BACKGROUND))

        # setup theme
        ctk.set_appearance_mode("dark")

        # setup window
        self.geometry(f'{APP_SIZE[0]}x{APP_SIZE[1]}')
        self.minsize(APP_MIN_SIZE[0], APP_MIN_SIZE[1])
        self.maxsize(APP_MAX_SIZE[0], APP_MAX_SIZE[1])
        self.title(APP_TITLE)
        self.iconbitmap(ICON_PATH)

        self.init_parameters()

        # layout
        self.grid_rowconfigure((0,1), weight=1, uniform='a')
        self.grid_columnconfigure(0, weight=2, uniform='a')
        self.grid_columnconfigure(1, weight=1, uniform='a')

        self.init_parameters()
        self.init_widgets()

    def init_parameters(self):
        # variables
        self.model_options = list(MODEL_PATH.keys())   
        self.model_path = MODEL_PATH[self.model_options[0]]
        self.model_pathes_options = list(MODEL_PATH.keys())
        self.model_pathes = list(MODEL_PATH.values())
        self.model_vars = {
            'model': ctk.StringVar(value=self.model_options[0]),
            'spell_check_var': ctk.BooleanVar(value=SPELL_CHECK_DEFAULT)
            }
        self.input_text = ctk.StringVar(value="")
    
        self.sentences = ''
        self.labeled_sentences = ''
        self.txt_path = ''
        
        self.model_vars['model'].trace_add('write', lambda *args: self.update_model_path())
        # self.model_vars['spell_check_var'].trace_add('write', lambda *args: self.update_spell_check())
        # self.labeled_sentences.trace_add('write', lambda *args: self.update_result_panel())

        # print(MODEL_PATH[self.model_vars['model'].get()]['model_validation'])
    
    def update_model_path(self):
        self.model_path = MODEL_PATH[self.model_vars['model'].get()]
        print('model path')
        print(self.model_path)
        self.update_menu()

    def update_input_panel(self):
        self.input_panel = InputPanel(self, self.input_text, self.model_vars['model'].get())

    def update_menu(self):
        self.panel = Menu(self, self.model_vars, self.model_options, self.predict_tags, self.import_txt_file, self.export_labeled_sentences, self.open_github)

    def update_result_panel(self):
        ShowResult(self, self.labeled_sentences)

    def import_txt_file(self, txt_path):
        if txt_path:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                self.txt_path = txt_path
                self.input_text.set(text)
                self.update_menu()
                self.update_input_panel()
                
    def init_widgets(self):
        self.import_txt_file(self.txt_path)
        # widgets
        self.panel = Menu(self, self.model_vars, self.model_options, self.predict_tags, self.import_txt_file, self.export_labeled_sentences, self.open_github)
        self.input_panel = InputPanel(self, self.input_text, self.model_vars['model'].get())
        self.result_panel = ShowResult(self, self.labeled_sentences)

        self.limitation_label = ctk.CTkLabel(
            self, 
            text=LIMIATION, 
            wraplength=APP_SIZE[0]-10, 
            justify='center', 
            anchor='center',
            font=ctk.CTkFont(FONT_NORMAL_PERSIAN, SMALL_FONT_SIZE),
            text_color=GREY,
            fg_color='transparent'
        )
        self.limitation_label.place(relx=0.5, rely=0.485, anchor='center')
       
    def predict_tags(self):
        # Get text from input panel
        input_text = self.input_panel.get_text()
        
        # Process the text
        preprocessor = PersianTextProcessor(self.model_vars['spell_check_var'].get())
        processed_text = preprocessor.process_text(input_text)

        self.sentences = processed_text['sentences']

        # Predict tags
        predictor = NERModel(self.model_vars['model'].get(),self.model_path['training_data'], self.model_path['weights'], self.sentences)
        # print(predictor.results)
        self.labeled_sentences = predictor.results
        print(self.labeled_sentences)

        self.update_result_panel()

        # # Process each sentence separately since self.tokens is a list of sentences
        # all_predictions = []
        # for sentence in self.sentences:
        #     predicted_tags = predictor.predict(sentence)
        #     all_predictions.append(predicted_tags)
        #     print(f"Sentence predictions: {predicted_tags}")  # Print each sentence's predictions

        # # You might want to store all_predictions as a class attribute for later use
        # self.predictions = all_predictions

        # reshaped_text = arabic_reshaper.reshape(str(result))
        # bidi_text = get_display(reshaped_text)

    def export_labeled_sentences(self, name, path, file_format):
        export_string = f'{path}/{name}.{file_format}'
        if not self.labeled_sentences:
            messagebox.showerror(title="خطا", message="در ابتدا پیش بینی کنید")
        else:
            try:
                if file_format.lower() == 'txt':
                    with open(export_string, 'w', encoding='utf-8') as file:
                        # Process each sentence in the list
                        for sentence in self.labeled_sentences:
                            # Process each token-label pair in the sentence
                            for token, label in sentence:
                                file.write(f"{token} {label}\n")
                            file.write("\n")  # Add blank line between sentences
                
                elif file_format.lower() == 'csv':
                    import csv
                    with open(export_string, 'w', encoding='utf-8-sig', newline='') as file:  # Note the utf-8-sig encoding
                        writer = csv.writer(file)
                        writer.writerow(['Token', 'Label'])  # Header
                        for sentence in self.labeled_sentences:
                            for token, label in sentence:
                                # Reshape and handle bidirectional text
                                reshaped_token = arabic_reshaper.reshape(token)
                                bidi_token = get_display(reshaped_token)
                                writer.writerow([bidi_token, label])
                            writer.writerow([])
                                
                messagebox.showinfo(title="موفقیت", message="فایل با موفقیت ذخیره شد")
            except Exception as e:
                messagebox.showerror(title="خطا", message=f"خطا در ذخیره‌سازی فایل: {str(e)}")
        
    def open_github(self):
        webbrowser.open(GITHUB_URL)
  

if __name__ == '__main__':
    app = App()
    app.mainloop()