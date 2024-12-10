import customtkinter as ctk
import arabic_reshaper
from bidi.algorithm import get_display
from settings import *

class TokenDisplay:
    def __init__(self, master):
        # Create the main text display
        self.text_widget = ctk.CTkTextbox(
            master,
            wrap="word",
            text_color=GREY
        )
        self.text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure text widget
        self.text_widget._textbox.configure(wrap="word")
        self.text_widget._textbox.tag_configure("rtl", justify="right")
        self.text_widget._textbox.tag_add("rtl", "1.0", "end")
        
        # Set fonts
        self.default_font = ctk.CTkFont(family=FONT_NORMAL_PERSIAN, size=NORMAL_FONT_SIZE)
        self.bold_font = ctk.CTkFont(family=FONT_NORMAL_PERSIAN,size = OUTPUT_FONT_SIZE, weight="bold")
        
        # Apply fonts
        self.text_widget.configure(font=self.default_font)
        
        # Configure name tag for special styling
        self.text_widget._textbox.tag_configure(
            "name",
            font=self.bold_font,
            foreground=LIGHT_BLUE
        )

    def display_tokens(self, tokens):

        # Clear previous content
        self.text_widget._textbox.delete("1.0", "end")
        
        for line_num, token_list in enumerate(tokens, start=1):
            # First combine all tokens with spaces
            text = " ".join(token for token, _ in token_list)
            
            # Reshape and handle bidirectional text for the whole sentence
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            # Insert the whole sentence
            line_start = self.text_widget._textbox.index("end-1c")
            self.text_widget._textbox.insert("end", bidi_text + "\n", ("rtl", "right"))
            
            # Now find and style each token that should be highlighted
            current_pos = line_start
            for token, label in token_list:
                if label == 1:
                    reshaped_token = arabic_reshaper.reshape(token)
                    bidi_token = get_display(reshaped_token)
                    
                    # Search within current line only
                    token_index = self.text_widget._textbox.search(
                        bidi_token,
                        current_pos,
                        f"{line_num}.end",
                        nocase=True
                    )
                    
                    if token_index:
                        end_index = f"{token_index}+{len(bidi_token)}c"
                        self.text_widget._textbox.tag_add("name", token_index, end_index)
        
        # Apply RTL tag to entire text
        self.text_widget._textbox.tag_add("rtl", "1.0", "end")
        self.text_widget.configure(state="disabled")  # Disable text editing

