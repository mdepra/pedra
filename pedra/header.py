import tkinter as tk
import threading
from IPython.display import display, clear_output
import ipywidgets as widgets


class HeaderWindow:
    r'''Class to visualize a FITS image header window.'''

    def __init__(self, img):
        self.img = img
        self._running = True
        self._widget_initialized = False
        self.match_indices = []
        self.match_index = 0
        self.last_keyword = None
        # Inicialize interface in separated thread
        self.thread = threading.Thread(target=self._create_interface)
        self.thread.daemon = True
        self.thread.start()
        # Show close button on Jupyter cell 
        self._show_close_button()

    def _create_interface(self):
        r"""Creates Header window interface in separated thread"""
        try:
            self.root = tk.Tk()
            self.root.withdraw()  
            self.window = tk.Toplevel(self.root)
            self.window.title("Header Viewer")
            self.window.geometry("600x600")
            self.window.protocol("WM_DELETE_WINDOW", self._safe_close)
            # Frame for text widget and scrollbar
            text_frame = tk.Frame(self.window)
            self.text_widget = tk.Text(text_frame, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(text_frame, command=self.text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.text_widget.pack(fill=tk.BOTH, expand=True)
            self.text_widget.config(yscrollcommand=scrollbar.set)
            self.text_widget.insert(tk.END, self.img.hdr.__repr__())
            text_frame.pack(fill=tk.BOTH, expand=True)
            # Frame for search control
            control_frame = tk.Frame(self.window)
            control_frame.pack(pady=10, fill=tk.X)
            self.search_box = tk.Entry(control_frame, width=40)
            self.search_box.pack(side=tk.LEFT, padx=5)
            self.search_box.bind("<Return>", lambda event: self._search_key())
            search_btn = tk.Button(control_frame, text="Buscar", command=self._search_key)
            search_btn.pack(side=tk.LEFT, padx=5)
            # Label to show match count and current match
            self.match_label = tk.Label(control_frame, text="")
            self.match_label.pack(side=tk.LEFT, padx=5)
            self.window.deiconify()
            # Keep tkinter alive
            while self._running:
                self.root.update()
                self.root.after(100)
        except Exception as e:
            print(f"Could not create Tkinter Header interface: {e}")
        finally:
            self._cleanup_tk()

    def _search_key(self):
        r"""Search and highlight the keyword in the text widget, iterating with each Enter key."""
        keyword = self.search_box.get().strip()
        # Clear previous highlights
        self.text_widget.tag_remove("highlight", "1.0", tk.END)
        if not keyword or not hasattr(self, 'text_widget'):
            # Clear matches when input is empty
            self.match_label.config(text="")
            self.match_indices = []
            self.match_index = 0
            self.last_keyword = None
            return
        content = self.text_widget.get("1.0", tk.END)
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        # Recalculate matches if keyword changed
        if keyword_lower != self.last_keyword:
            self.match_indices = []
            self.match_index = 0
            self.last_keyword = keyword_lower
            start = 0
            while True:
                idx = content_lower.find(keyword_lower, start)
                if idx == -1:
                    break
                self.match_indices.append(idx)
                start = idx + len(keyword)
        if not self.match_indices:
            self.match_label.config(text="0 matches")
            return
        # Highlight current match
        idx = self.match_indices[self.match_index]
        line = content.count('\n', 0, idx) + 1
        col = idx - content.rfind('\n', 0, idx) - 1
        start_pos = f"{line}.{col}"
        end_pos = f"{line}.{col + len(keyword)}"
        self.text_widget.tag_add("highlight", start_pos, end_pos)
        self.text_widget.tag_config("highlight", background="chartreuse", foreground="black")
        self.text_widget.see(start_pos)
        # Show match index info
        self.match_label.config(text=f"{self.match_index + 1}/{len(self.match_indices)}")
        # Move to next match
        self.match_index = (self.match_index + 1) % len(self.match_indices)

    def _show_close_button(self):
        r"""Show button to close interface safely."""
        self._widget_initialized = True
        self.close_btn = widgets.Button(
            description="Close Header",
            button_style='danger',
            icon='times'
        )
        self.close_btn.on_click(lambda b: self._safe_close())
        display(self.close_btn)

    def _cleanup_tk(self):
        r"""Safely clean Tkinter resources."""
        try:
            if hasattr(self, 'window') and self.window.winfo_exists():
                self.window.destroy()
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except Exception as e:
            print(f"Error while cleaning Tkinter resources: {e}")

    def _safe_close(self):
        r"""Close all Tkinter resources."""
        self._running = False
        self._cleanup_tk()
        if self._widget_initialized:
            try:
                clear_output(wait=True)
            except Exception as e:
                print(f"Erro ao limpar o widget: {e}")
            self._widget_initialized = False

    def __del__(self):
        r"""Makes sure object is destroyed"""
        self._safe_close()
