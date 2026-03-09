import customtkinter as ctk
from tkinter import filedialog, messagebox
import sys
import os
import threading
from pathlib import Path

try:
    from Pipeline import Pipeline
    from Config import Config
    from FileUtils import FileUtils
except ImportError as e:
    print(f"Warning: Import error - {e}")
    Pipeline = None
    Config = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class ConsoleRedirect:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    
    def write(self, text):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", text)
        self.text_widget.configure(state="disabled")
        self.text_widget.see("end")
    
    def flush(self):
        pass

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("maxiM-Ze")
        self.geometry("1200x900")
        
        self.input_files = []
        self.library_file = None
        self.output_folder = None
        self.use_study_design = False
        self.study_groups = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#0B1F3B")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="maxiM-Ze", 
                                      font=ctk.CTkFont(size=28, weight="bold"),
                                      text_color="#6EE7B7")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.home_btn = ctk.CTkButton(self.sidebar, text="Setup", 
                                     command=lambda: self.show_page("home"),
                                     fg_color="#2BB673", hover_color="#6EE7B7")
        self.home_btn.grid(row=1, column=0, padx=20, pady=10)
        
        self.settings_btn = ctk.CTkButton(self.sidebar, text="Parameters",
                                         command=lambda: self.show_page("settings"),
                                         fg_color="#2BB673", hover_color="#6EE7B7")
        self.settings_btn.grid(row=2, column=0, padx=20, pady=10)
        
        self.about_btn = ctk.CTkButton(self.sidebar, text="About",
                                      command=lambda: self.show_page("about"),
                                      fg_color="#2BB673", hover_color="#6EE7B7")
        self.about_btn.grid(row=3, column=0, padx=20, pady=10)
        
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.create_pages()
        self.show_page("home")
        
    def create_pages(self):
        self.pages = {}
        self.pages["home"] = self.create_home_page()
        self.pages["settings"] = self.create_settings_page()
        self.pages["about"] = self.create_about_page()
        
    def create_home_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame)
        
        title = ctk.CTkLabel(page, text="Analysis Setup", 
                            font=ctk.CTkFont(size=32, weight="bold"))
        title.pack(pady=(0, 20))
        
        # File Input Section
        file_frame = ctk.CTkFrame(page)
        file_frame.pack(fill="x", pady=(0, 15))
        
        file_title = ctk.CTkLabel(file_frame, text="1. Input Files (mzXML/mzML)",
                                 font=ctk.CTkFont(size=18, weight="bold"))
        file_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        file_btn = ctk.CTkButton(file_frame, text="Select Input Files",
                                command=self.select_input_files, height=40)
        file_btn.pack(fill="x", padx=20, pady=5)
        
        self.file_status = ctk.CTkLabel(file_frame, text="No files selected")
        self.file_status.pack(pady=(5, 15), padx=20, anchor="w")
        
        # Study Design Section
        design_frame = ctk.CTkFrame(page)
        design_frame.pack(fill="x", pady=(0, 15))
        
        design_title = ctk.CTkLabel(design_frame, text="2. Study Design",
                                   font=ctk.CTkFont(size=18, weight="bold"))
        design_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        self.study_design_var = ctk.BooleanVar(value=False)
        study_check = ctk.CTkCheckBox(design_frame, text="Use Study Design (define experimental groups)",
                                     variable=self.study_design_var,
                                     command=self.toggle_study_design)
        study_check.pack(padx=20, pady=5, anchor="w")
        
        self.study_info = ctk.CTkLabel(design_frame, 
                                      text="Without study design: random sample selection for mass grouping",
                                      text_color="gray")
        self.study_info.pack(pady=(0, 15), padx=20, anchor="w")
        
        # Output Folder Section
        output_frame = ctk.CTkFrame(page)
        output_frame.pack(fill="x", pady=(0, 15))
        
        output_title = ctk.CTkLabel(output_frame, text="3. Output Folder",
                                   font=ctk.CTkFont(size=18, weight="bold"))
        output_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        output_btn = ctk.CTkButton(output_frame, text="Select Output Folder",
                                  command=self.select_output_folder, height=40)
        output_btn.pack(fill="x", padx=20, pady=5)
        
        self.output_status = ctk.CTkLabel(output_frame, text="No folder selected")
        self.output_status.pack(pady=(5, 15), padx=20, anchor="w")
        
        # Library Section
        library_frame = ctk.CTkFrame(page)
        library_frame.pack(fill="x", pady=(0, 15))
        
        library_title = ctk.CTkLabel(library_frame, text="4. Compound Library (Optional)",
                                    font=ctk.CTkFont(size=18, weight="bold"))
        library_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        library_btn = ctk.CTkButton(library_frame, text="Select Library File (CSV/XLSX)",
                                   command=self.select_library_file, height=40)
        library_btn.pack(fill="x", padx=20, pady=5)
        
        self.library_status = ctk.CTkLabel(library_frame, text="No library file selected")
        self.library_status.pack(pady=(5, 15), padx=20, anchor="w")
        
        # Run Button
        self.run_btn = ctk.CTkButton(page, text="Run Analysis", 
                                    command=self.run_analysis,
                                    height=60, font=ctk.CTkFont(size=18, weight="bold"),
                                    fg_color="#2BB673", hover_color="#6EE7B7")
        self.run_btn.pack(fill="x", pady=20)
        
        # Console Output
        console_frame = ctk.CTkFrame(page)
        console_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        console_title = ctk.CTkLabel(console_frame, text="Console Output",
                                    font=ctk.CTkFont(size=18, weight="bold"))
        console_title.pack(pady=(15, 10))
        
        self.console_output = ctk.CTkTextbox(console_frame, height=250, 
                                           font=ctk.CTkFont(family="Consolas", size=11))
        self.console_output.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        clear_btn = ctk.CTkButton(console_frame, text="Clear Console",
                                 command=self.clear_console)
        clear_btn.pack(pady=(0, 15))
        
        sys.stdout = ConsoleRedirect(self.console_output)
        
        return page
        
    def create_settings_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame)
        
        title = ctk.CTkLabel(page, text="Analysis Parameters", 
                            font=ctk.CTkFont(size=32, weight="bold"))
        title.pack(pady=(0, 20))
        
        # Library Matching Parameters
        library_frame = ctk.CTkFrame(page)
        library_frame.pack(fill="x", pady=(0, 15))
        
        library_title = ctk.CTkLabel(library_frame, text="Library Matching",
                                     font=ctk.CTkFont(size=18, weight="bold"))
        library_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        mz_tol_frame = ctk.CTkFrame(library_frame, fg_color="transparent")
        mz_tol_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(mz_tol_frame, text="Mass Tolerance (Da):").pack(side="left")
        self.mz_tolerance_var = ctk.DoubleVar(value=0.0005)
        self.mz_tolerance_entry = ctk.CTkEntry(mz_tol_frame, width=120, textvariable=self.mz_tolerance_var)
        self.mz_tolerance_entry.pack(side="right")
        
        rt_tol_frame = ctk.CTkFrame(library_frame, fg_color="transparent")
        rt_tol_frame.pack(fill="x", padx=20, pady=(5, 15))
        
        ctk.CTkLabel(rt_tol_frame, text="RT Tolerance (min):").pack(side="left")
        self.rt_tolerance_var = ctk.DoubleVar(value=0.1)
        self.rt_tolerance_entry = ctk.CTkEntry(rt_tol_frame, width=120, textvariable=self.rt_tolerance_var)
        self.rt_tolerance_entry.pack(side="right")
        
        # Mass Grouping Parameters
        grouping_frame = ctk.CTkFrame(page)
        grouping_frame.pack(fill="x", pady=(0, 15))
        
        grouping_title = ctk.CTkLabel(grouping_frame, text="Mass Grouping",
                                      font=ctk.CTkFont(size=18, weight="bold"))
        grouping_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        noise_frame = ctk.CTkFrame(grouping_frame, fg_color="transparent")
        noise_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(noise_frame, text="Noise Level:").pack(side="left")
        self.noise_var = ctk.DoubleVar(value=5000.0)
        self.noise_entry = ctk.CTkEntry(noise_frame, width=120, textvariable=self.noise_var)
        self.noise_entry.pack(side="right")
        
        min_scans_frame = ctk.CTkFrame(grouping_frame, fg_color="transparent")
        min_scans_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(min_scans_frame, text="Min Consecutive Scans:").pack(side="left")
        self.min_scans_var = ctk.IntVar(value=7)
        self.min_scans_entry = ctk.CTkEntry(min_scans_frame, width=120, textvariable=self.min_scans_var)
        self.min_scans_entry.pack(side="right")
        
        group_size_frame = ctk.CTkFrame(grouping_frame, fg_color="transparent")
        group_size_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(group_size_frame, text="Min Group Size:").pack(side="left")
        self.min_group_var = ctk.IntVar(value=3)
        self.min_group_entry = ctk.CTkEntry(group_size_frame, width=120, textvariable=self.min_group_var)
        self.min_group_entry.pack(side="right")
        
        max_group_frame = ctk.CTkFrame(grouping_frame, fg_color="transparent")
        max_group_frame.pack(fill="x", padx=20, pady=(5, 15))
        
        ctk.CTkLabel(max_group_frame, text="Max Group Size:").pack(side="left")
        self.max_group_var = ctk.IntVar(value=5)
        self.max_group_entry = ctk.CTkEntry(max_group_frame, width=120, textvariable=self.max_group_var)
        self.max_group_entry.pack(side="right")
        
        # Processing Options
        options_frame = ctk.CTkFrame(page)
        options_frame.pack(fill="x", pady=(0, 15))
        
        options_title = ctk.CTkLabel(options_frame, text="Processing Options",
                                     font=ctk.CTkFont(size=18, weight="bold"))
        options_title.pack(pady=(15, 10), anchor="w", padx=20)
        
        self.rebuild_var = ctk.BooleanVar(value=True)
        rebuild_check = ctk.CTkCheckBox(options_frame, text="Rebuild mass groups (ignore cache)",
                                       variable=self.rebuild_var)
        rebuild_check.pack(padx=20, pady=5, anchor="w")
        
        self.verbose_var = ctk.BooleanVar(value=False)
        verbose_check = ctk.CTkCheckBox(options_frame, text="Verbose output",
                                       variable=self.verbose_var)
        verbose_check.pack(padx=20, pady=(5, 15), anchor="w")
        
        # Buttons
        btn_frame = ctk.CTkFrame(page, fg_color="transparent")
        btn_frame.pack(fill="x", pady=20)
        
        reset_btn = ctk.CTkButton(btn_frame, text="Reset to Defaults",
                                 command=self.reset_settings)
        reset_btn.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        save_btn = ctk.CTkButton(btn_frame, text="Apply Settings",
                                command=self.apply_settings,
                                fg_color="#2BB673", hover_color="#6EE7B7")
        save_btn.pack(side="right", padx=(10, 0), fill="x", expand=True)
        
        return page
        
    def create_about_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame)
        
        title = ctk.CTkLabel(page, text="About maxiM-Ze", 
                            font=ctk.CTkFont(size=32, weight="bold"))
        title.pack(pady=(0, 20))
        
        info_frame = ctk.CTkFrame(page)
        info_frame.pack(fill="x", pady=(0, 15))
        
        version_label = ctk.CTkLabel(info_frame, text="Version: 1.0.0",
                                    font=ctk.CTkFont(size=16, weight="bold"))
        version_label.pack(pady=(20, 10))
        
        desc_label = ctk.CTkLabel(info_frame, 
                                 text="A Novel Image Recognition Approach for Visualizing and\nProcessing Mass Spectrometry Based Metabolomics Data",
                                 wraplength=500, justify="center")
        desc_label.pack(pady=10)
        
        features_text = """Features:
• Support for mzXML and mzML file formats
• Dynamic mass grouping from sample data
• Optional study design with experimental groups
• Compound library matching with configurable tolerances
• Automated peak detection and clustering
• Comprehensive visualization and reporting
• Multi-threaded processing for performance"""
        
        features_label = ctk.CTkLabel(info_frame, text=features_text, justify="left")
        features_label.pack(pady=(10, 20))
        
        return page
        
    def show_page(self, page_name):
        for page in self.pages.values():
            page.pack_forget()
        
        self.pages[page_name].pack(fill="both", expand=True, padx=20, pady=20)
        
    def select_input_files(self):
        files = filedialog.askopenfilenames(
            title="Select Input Files",
            filetypes=[("MS files", "*.mzxml *.mzml"), ("mzXML files", "*.mzxml"), 
                      ("mzML files", "*.mzml"), ("All files", "*.*")]
        )
        
        if files:
            self.input_files = list(files)
            self.file_status.configure(text=f"Selected {len(files)} file(s)")
            print(f"Selected {len(files)} input files")
            
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        
        if folder:
            self.output_folder = folder
            self.output_status.configure(text=f"Output: {Path(folder).name}")
            print(f"Output folder: {folder}")
            
    def select_library_file(self):
        file = filedialog.askopenfilename(
            title="Select Library File",
            filetypes=[("Spreadsheet files", "*.csv *.xlsx"), ("CSV files", "*.csv"),
                      ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file:
            self.library_file = file
            self.library_status.configure(text=f"Library: {Path(file).name}")
            print(f"Library file: {file}")
            
    def toggle_study_design(self):
        self.use_study_design = self.study_design_var.get()
        if self.use_study_design:
            self.study_info.configure(text="Study design enabled: configure groups in MassGrouping.py")
        else:
            self.study_info.configure(text="Without study design: random sample selection for mass grouping")
            
    def apply_settings(self):
        if Config:
            Config.LIBRARY_MATCH_MZ_TOL = self.mz_tolerance_var.get()
            Config.LIBRARY_MATCH_RT_TOL = self.rt_tolerance_var.get()
            Config.GROUP_NOISE_LEVEL = self.noise_var.get()
            Config.GROUP_MIN_CONSEC_SCANS = self.min_scans_var.get()
            Config.GROUP_MIN_GROUP_SIZE = self.min_group_var.get()
            Config.GROUP_MAX_GROUP_SIZE = self.max_group_var.get()
            Config.REBUILD_MASS_GROUPS = self.rebuild_var.get()
            Config.GROUPING_VERBOSE = self.verbose_var.get()
            
        messagebox.showinfo("Settings Applied", "Parameters have been updated successfully.")
        print("Settings applied to Config")
        
    def reset_settings(self):
        self.mz_tolerance_var.set(0.0005)
        self.rt_tolerance_var.set(0.1)
        self.noise_var.set(5000.0)
        self.min_scans_var.set(7)
        self.min_group_var.set(3)
        self.max_group_var.set(5)
        self.rebuild_var.set(True)
        self.verbose_var.set(False)
        messagebox.showinfo("Settings Reset", "All parameters reset to defaults.")
        
    def run_analysis(self):
        if not self.input_files:
            messagebox.showwarning("Missing Input", "Please select input files first")
            return
            
        if not self.output_folder:
            messagebox.showwarning("Missing Output", "Please select an output folder")
            return
            
        self.run_btn.configure(state="disabled", text="Running Analysis...")
        self.clear_console()
        
        def analysis_thread():
            try:
                if not Pipeline or not Config:
                    messagebox.showerror("Error", "Pipeline modules not found.")
                    return
                
                # Apply settings
                self.apply_settings()
                
                # Update Config with GUI selections
                Config.BASE_DIR = Path(self.output_folder)
                if self.library_file:
                    Config.LIB_FILE = self.library_file
                
                # Override FileUtils to use selected files
                original_get_file_paths = FileUtils.get_file_paths
                FileUtils.get_file_paths = lambda: self.input_files
                
                print("=" * 70)
                print("Starting maxiM-Ze Analysis Pipeline")
                print("=" * 70)
                print(f"Input files: {len(self.input_files)}")
                print(f"Output folder: {self.output_folder}")
                if self.library_file:
                    print(f"Library file: {self.library_file}")
                print("=" * 70)
                
                # Run pipeline
                pipeline = Pipeline()
                pipeline.run()
                
                # Restore original method
                FileUtils.get_file_paths = original_get_file_paths
                
                messagebox.showinfo("Success", "Analysis completed successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
                print(f"\nError: {str(e)}")
            finally:
                self.run_btn.configure(state="normal", text="Run Analysis")
        
        threading.Thread(target=analysis_thread, daemon=True).start()
        
    def clear_console(self):
        self.console_output.configure(state="normal")
        self.console_output.delete("1.0", "end")
        self.console_output.configure(state="disabled")

def main():
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
