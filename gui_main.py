import sys
import os

# Fix for PyInstaller
if getattr(sys, 'frozen', False):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk
from tkinter import filedialog, messagebox
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

ctk.set_appearance_mode("light")
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
        self.configure(fg_color="#2BB673")

        self.input_files = []
        self.library_file = None
        self.output_folder = None

        self.use_study_design = False
        self.study_groups = {}
        self.target_files = []
        self.use_target = False

        # Mass groups JSON import
        self.import_json_path = None      # path string or None

        self._group_frames = []
        self._target_frame_data = None

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#0B1F3B")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        try:
            from PIL import Image as PilImage
            _logo_path = Path(__file__).parent / "logo.png"
            pil_img = PilImage.open(_logo_path)
            self._logo_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(170, 18))
            self.logo_label = ctk.CTkLabel(self.sidebar, image=self._logo_img, text="", fg_color="transparent")
        except Exception:
            self.logo_label = ctk.CTkLabel(self.sidebar, text="maxiM-Ze",
                                           font=ctk.CTkFont(size=28, weight="bold"), text_color="#6EE7B7")
        self.logo_label.grid(row=0, column=0, padx=15, pady=(20, 10))

        self.home_btn = ctk.CTkButton(self.sidebar, text="Setup",
                                      command=lambda: self.show_page("home"),
                                      fg_color="#2BB673", hover_color="#6EE7B7")
        self.home_btn.grid(row=1, column=0, padx=20, pady=10)

        self.about_btn = ctk.CTkButton(self.sidebar, text="About",
                                       command=lambda: self.show_page("about"),
                                       fg_color="#2BB673", hover_color="#6EE7B7")
        self.about_btn.grid(row=2, column=0, padx=20, pady=10)

        self.main_frame = ctk.CTkFrame(self, fg_color="#FFFFFF")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.create_pages()
        self.show_page("home")

    def create_pages(self):
        self.pages = {}
        self.pages["home"] = self.create_home_page()
        self.pages["about"] = self.create_about_page()

    def create_home_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame, fg_color="#FFFFFF")

        ctk.CTkLabel(page, text="Analysis Setup",
                     font=ctk.CTkFont(size=32, weight="bold"), text_color="#0B1F3B").pack(pady=(0, 20))

        # ── 1. Import Files / Study Design ────────────────────────────
        design_outer = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        design_outer.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(design_outer, text="1. Import mzXML or mzML Files",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#0B1F3B").pack(pady=(15, 5), anchor="w", padx=20)

        mode_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        mode_row.pack(fill="x", padx=20, pady=(0, 10))

        self.study_design_var = ctk.BooleanVar(value=False)
        self._mode_check = ctk.CTkCheckBox(
            mode_row,
            text="Option A — Use Predefined Study Groups\ni.e. Group 1: Drug Treated, Group 2: Placebo, etc.\nSamples will be pulled from each group for mass grouping",
            variable=self.study_design_var,
            command=self._on_mode_toggle,
            text_color="#0B1F3B",
        )
        self._mode_check.pack(side="left")

        self.study_info = ctk.CTkLabel(
            design_outer,
            text="Option B active — add your files below and samples will be randomly selected for mass grouping",
            text_color="gray",
        )
        self.study_info.pack(anchor="w", padx=20, pady=(0, 5))

        # n-groups dropdown — not packed yet, shown only in Option A
        self._n_groups_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        ctk.CTkLabel(self._n_groups_row, text="Number of experimental groups:").pack(side="left")
        self._n_groups_var = ctk.IntVar(value=2)
        ctk.CTkOptionMenu(
            self._n_groups_row,
            values=[str(i) for i in range(1, 11)],
            variable=ctk.StringVar(value="2"),
            command=self._on_n_groups_changed,
            width=80,
        ).pack(side="left", padx=(10, 0))

        self._design_controls = ctk.CTkFrame(design_outer, fg_color="transparent")
        self._design_controls.pack(fill="x")
        self._build_study_design_controls(self._design_controls)

        # ── Noise Level ───────────────────────────────────────────────
        noise_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        noise_row.pack(fill="x", padx=20, pady=(5, 5))
        ctk.CTkLabel(noise_row, text="Noise Level:", text_color="#0B1F3B").pack(side="left")
        self.noise_var = ctk.DoubleVar(value=5000.0)
        ctk.CTkEntry(noise_row, width=120, textvariable=self.noise_var).pack(side="right")

        # ── Max Groups to Run ─────────────────────────────────────────
        max_groups_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        max_groups_row.pack(fill="x", padx=20, pady=(5, 5))
        ctk.CTkLabel(max_groups_row, text="Max Groups to Run:", text_color="#0B1F3B").pack(side="left")
        self.max_groups_var = ctk.StringVar(value="")
        ctk.CTkEntry(max_groups_row, width=120, textvariable=self.max_groups_var,
                     placeholder_text="all").pack(side="right")

        ctk.CTkLabel(design_outer,
                     text="Leave blank to run all groups. Enter a number (e.g. 30) to limit for testing.",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20, pady=(0, 15))

        # ── 2. Output Folder + Run Name ───────────────────────────────
        output_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        output_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(output_frame, text="2. Output Folder & Run Name",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#0B1F3B").pack(pady=(15, 10), anchor="w", padx=20)

        ctk.CTkButton(output_frame, text="Select Output Folder",
                      command=self.select_output_folder, height=40).pack(fill="x", padx=20, pady=5)

        self.output_status = ctk.CTkLabel(output_frame, text="No folder selected", text_color="#555")
        self.output_status.pack(pady=(5, 8), padx=20, anchor="w")

        # Run name field
        run_name_row = ctk.CTkFrame(output_frame, fg_color="transparent")
        run_name_row.pack(fill="x", padx=20, pady=(0, 5))
        ctk.CTkLabel(run_name_row, text="Run Name:", text_color="#0B1F3B",
                     font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.run_name_var = ctk.StringVar(value="")
        self.run_name_entry = ctk.CTkEntry(
            run_name_row,
            textvariable=self.run_name_var,
            placeholder_text="e.g.  MyStudy_POS_Run1",
            width=340,
        )
        self.run_name_entry.pack(side="left", padx=(12, 0))

        ctk.CTkLabel(output_frame,
                     text="A subfolder with this name will be created inside your output folder.",
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20, pady=(0, 15))

        # ── 3. Mass Groups JSON (optional import) ─────────────────────
        json_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        json_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(json_frame, text="3. Mass Groups (Detection)",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#0B1F3B").pack(pady=(15, 5), anchor="w", padx=20)

        # Default: fresh detection every run
        self.import_json_var = ctk.BooleanVar(value=False)
        self._json_check = ctk.CTkCheckBox(
            json_frame,
            text="Import previous MassGroups_Cache.json from same sample set\n"
                 "(skips re-detection — use when re-running with identical samples)",
            variable=self.import_json_var,
            command=self._on_json_toggle,
            text_color="#0B1F3B",
        )
        self._json_check.pack(anchor="w", padx=20, pady=(0, 8))

        # JSON import row — hidden until checkbox ticked
        self._json_import_row = ctk.CTkFrame(json_frame, fg_color="transparent")
        self._json_btn = ctk.CTkButton(
            self._json_import_row, text="Select MassGroups_Cache.json",
            command=self.select_mass_groups_json, height=36,
        )
        self._json_btn.pack(side="left", padx=(0, 12))
        self._json_status = ctk.CTkLabel(self._json_import_row, text="No file selected", text_color="#555")
        self._json_status.pack(side="left")

        self._json_default_label = ctk.CTkLabel(
            json_frame,
            text="Default: mass groups will be freshly detected from your input files each run.",
            text_color="gray", font=ctk.CTkFont(size=11),
        )
        self._json_default_label.pack(anchor="w", padx=20, pady=(0, 15))

        # ── 4. Compound Library ───────────────────────────────────────
        library_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        library_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(library_frame, text="4. Compound Library Import",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#0B1F3B").pack(pady=(15, 5), anchor="w", padx=20)

        ctk.CTkLabel(library_frame, text="CSV and XLSX accepted (optional)",
                     font=ctk.CTkFont(size=13), text_color="#0B1F3B").pack(anchor="w", padx=20, pady=(0, 10))

        ctk.CTkButton(library_frame, text="Select Library File",
                      command=self.select_library_file, height=40).pack(fill="x", padx=20, pady=5)

        self.library_status = ctk.CTkLabel(library_frame, text="No library file selected", text_color="#555")
        self.library_status.pack(pady=(5, 10), padx=20, anchor="w")

        mz_tol_row = ctk.CTkFrame(library_frame, fg_color="transparent")
        mz_tol_row.pack(fill="x", padx=20, pady=3)
        ctk.CTkLabel(mz_tol_row, text="Mass Tolerance (Da):", text_color="#0B1F3B").pack(side="left")
        self.mz_tolerance_var = ctk.DoubleVar(value=0.0005)
        ctk.CTkEntry(mz_tol_row, width=120, textvariable=self.mz_tolerance_var).pack(side="right")

        rt_tol_row = ctk.CTkFrame(library_frame, fg_color="transparent")
        rt_tol_row.pack(fill="x", padx=20, pady=(3, 15))
        ctk.CTkLabel(rt_tol_row, text="Retention Time Tolerance (min):", text_color="#0B1F3B").pack(side="left")
        self.rt_tolerance_var = ctk.DoubleVar(value=0.1)
        ctk.CTkEntry(rt_tol_row, width=120, textvariable=self.rt_tolerance_var).pack(side="right")

        # ── Run ───────────────────────────────────────────────────────
        self.run_btn = ctk.CTkButton(page, text="Run Analysis", command=self.run_analysis,
                                     height=60, font=ctk.CTkFont(size=18, weight="bold"),
                                     fg_color="#2BB673", hover_color="#6EE7B7")
        self.run_btn.pack(fill="x", pady=20)

        # ── Console ───────────────────────────────────────────────────
        console_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=4, border_color="#D0E8F5")
        console_frame.pack(fill="both", expand=True, pady=(0, 20))

        ctk.CTkLabel(console_frame, text="Console Output",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

        self.console_output = ctk.CTkTextbox(console_frame, height=250,
                                             font=ctk.CTkFont(family="Consolas", size=11))
        self.console_output.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        ctk.CTkButton(console_frame, text="Clear Console", command=self.clear_console).pack(pady=(0, 15))

        sys.stdout = ConsoleRedirect(self.console_output)
        return page

    # ------------------------------------------------------------------
    # Study design sub-UI
    # ------------------------------------------------------------------

    def _build_study_design_controls(self, parent):
        self._groups_scroll = ctk.CTkFrame(parent, fg_color="transparent")
        self._groups_scroll.pack(fill="x", padx=20, pady=(0, 10))

        target_toggle_row = ctk.CTkFrame(parent, fg_color="transparent")
        target_toggle_row.pack(fill="x", padx=20, pady=(0, 5))

        self._use_target_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(target_toggle_row,
                        text="Include target samples (i.e. pooled plasma controls)",
                        variable=self._use_target_var,
                        command=self._on_target_toggle).pack(side="left")

        self._target_slot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._build_target_slot(self._target_slot_frame)
        self._render_group_cards(1)

    def _render_group_cards(self, n_groups: int):
        for widget in self._groups_scroll.winfo_children():
            widget.destroy()
        self._group_frames = []
        for i in range(1, n_groups + 1):
            self._add_group_card(i)

    def _add_group_card(self, group_number: int):
        card = ctk.CTkFrame(self._groups_scroll, border_width=2, border_color="#2BB673")
        card.pack(fill="x", pady=5)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        name_var = ctk.StringVar(value=f"Group {group_number}")
        ctk.CTkEntry(header, textvariable=name_var, width=160, placeholder_text="Group name").pack(side="left")
        file_count_label = ctk.CTkLabel(header, text="0 file(s)", text_color="gray")
        file_count_label.pack(side="right")

        file_listbox = ctk.CTkTextbox(card, height=60, font=ctk.CTkFont(family="Consolas", size=10), state="disabled")
        file_listbox.pack(fill="x", padx=10, pady=(0, 6))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        group_data = {"name_var": name_var, "files": [], "file_listbox": file_listbox, "file_count_label": file_count_label}
        self._group_frames.append(group_data)

        ctk.CTkButton(btn_row, text="Add Files", width=100, height=28,
                      command=lambda gd=group_data: self._add_files_to_group(gd)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="Clear", width=80, height=28, fg_color="#555", hover_color="#777",
                      command=lambda gd=group_data: self._clear_group(gd)).pack(side="left")

    def _build_target_slot(self, parent):
        card = ctk.CTkFrame(parent, border_width=2, border_color="#2BB673")
        card.pack(fill="x", padx=20, pady=4)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(header, text="Target Samples  (pooled controls)",
                     font=ctk.CTkFont(weight="bold"), text_color="#2BB673").pack(side="left")
        self._target_count_label = ctk.CTkLabel(header, text="0 file(s)", text_color="gray")
        self._target_count_label.pack(side="right")

        self._target_listbox = ctk.CTkTextbox(card, height=55, font=ctk.CTkFont(family="Consolas", size=10), state="disabled")
        self._target_listbox.pack(fill="x", padx=10, pady=(0, 6))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkButton(btn_row, text="Add Files", width=100, height=28, command=self._add_target_files).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="Clear", width=80, height=28, fg_color="#555", hover_color="#777",
                      command=self._clear_target).pack(side="left")

    # ------------------------------------------------------------------
    # Group / target helpers
    # ------------------------------------------------------------------

    def _add_files_to_group(self, group_data: dict):
        files = filedialog.askopenfilenames(
            title=f"Select files for {group_data['name_var'].get()}",
            filetypes=[("MS files", "*.mzxml *.mzml"), ("All files", "*.*")])
        if files:
            group_data["files"].extend(list(files))
            self._refresh_group_display(group_data)

    def _clear_group(self, group_data: dict):
        group_data["files"] = []
        self._refresh_group_display(group_data)

    def _refresh_group_display(self, group_data: dict):
        lb = group_data["file_listbox"]
        lb.configure(state="normal")
        lb.delete("1.0", "end")
        for f in group_data["files"]:
            lb.insert("end", Path(f).name + "\n")
        lb.configure(state="disabled")
        group_data["file_count_label"].configure(text=f"{len(group_data['files'])} file(s)")

    def _add_target_files(self):
        files = filedialog.askopenfilenames(
            title="Select Target / Pooled Control Files",
            filetypes=[("MS files", "*.mzxml *.mzml"), ("All files", "*.*")])
        if files:
            self.target_files.extend(list(files))
            self._refresh_target_display()

    def _clear_target(self):
        self.target_files = []
        self._refresh_target_display()

    def _refresh_target_display(self):
        self._target_listbox.configure(state="normal")
        self._target_listbox.delete("1.0", "end")
        for f in self.target_files:
            self._target_listbox.insert("end", Path(f).name + "\n")
        self._target_listbox.configure(state="disabled")
        self._target_count_label.configure(text=f"{len(self.target_files)} file(s)")

    # ------------------------------------------------------------------
    # Toggle callbacks
    # ------------------------------------------------------------------

    def _on_mode_toggle(self):
        self.use_study_design = self.study_design_var.get()
        if self.use_study_design:
            self.study_info.configure(
                text="Option A active — assign files to each experimental group below.",
                text_color="#2BB673")
            self._design_controls.pack_forget()
            self._n_groups_row.pack(fill="x", padx=20, pady=(5, 10))
            self._design_controls.pack(fill="x")
            self._render_group_cards(self._n_groups_var.get())
        else:
            self.study_info.configure(
                text="Option B active — add your files below and samples will be randomly selected for mass grouping",
                text_color="gray")
            self._n_groups_row.pack_forget()
            self._render_group_cards(1)

    def _on_n_groups_changed(self, value: str):
        self._render_group_cards(int(value))
        if self._use_target_var.get():
            self._target_slot_frame.pack(fill="x")

    def _on_target_toggle(self):
        self.use_target = self._use_target_var.get()
        if self.use_target:
            self._target_slot_frame.pack(fill="x", padx=0, pady=(0, 5))
        else:
            self._target_slot_frame.pack_forget()

    def _on_json_toggle(self):
        """Show or hide the JSON file picker based on the checkbox state."""
        if self.import_json_var.get():
            self._json_default_label.pack_forget()
            self._json_import_row.pack(anchor="w", padx=20, pady=(0, 8))
            self._json_default_label.pack(anchor="w", padx=20, pady=(0, 15))
        else:
            self._json_import_row.pack_forget()
            self.import_json_path = None
            self._json_status.configure(text="No file selected")

    def _collect_study_groups(self) -> dict:
        out = {}
        for gd in self._group_frames:
            name = gd["name_var"].get().strip() or f"Group {self._group_frames.index(gd)+1}"
            out[name] = gd["files"]
        if self.use_target and self.target_files:
            out["target"] = self.target_files
        return out

    # ------------------------------------------------------------------
    # File / folder pickers
    # ------------------------------------------------------------------

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
                       ("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file:
            self.library_file = file
            self.library_status.configure(text=f"Library: {Path(file).name}")
            print(f"Library file: {file}")

    def select_mass_groups_json(self):
        """Let the user pick a MassGroups_Cache.json from a previous run."""
        file = filedialog.askopenfilename(
            title="Select MassGroups_Cache.json from previous run",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file:
            self.import_json_path = file
            self._json_status.configure(
                text=f"{Path(file).name}  ({Path(file).parent.name})",
                text_color="#2BB673",
            )
            print(f"Mass groups JSON: {file}")

    # ------------------------------------------------------------------
    # About page
    # ------------------------------------------------------------------

    def create_about_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame, fg_color="#FFFFFF")

        ctk.CTkLabel(page, text="About maxiM-Ze",
                     font=ctk.CTkFont(size=32, weight="bold"), text_color="#0B1F3B").pack(pady=(0, 20))

        info_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=2, border_color="#D0E8F5")
        info_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(info_frame, text="Version: 1.0.0",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))

        ctk.CTkLabel(info_frame,
                     text="A Novel Image Recognition Approach for Visualizing and\nProcessing Mass Spectrometry Based Metabolomics Data",
                     wraplength=500, justify="center").pack(pady=10)

        ctk.CTkLabel(info_frame, justify="left", text=(
            "Features:\n"
            "• Support for mzXML and mzML file formats\n"
            "• Dynamic mass grouping from sample data\n"
            "• Optional study design with experimental groups\n"
            "• Compound library matching with configurable tolerances\n"
            "• Automated peak detection and clustering\n"
            "• Comprehensive visualization and reporting\n"
            "• Multi-threaded processing for performance"
        )).pack(pady=(10, 20))

        return page

    # ------------------------------------------------------------------
    # Page navigation
    # ------------------------------------------------------------------

    def show_page(self, page_name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[page_name].pack(fill="both", expand=True, padx=20, pady=20)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _apply_config(self):
        if Config:
            Config.LIBRARY_MATCH_MZ_TOL = self.mz_tolerance_var.get()
            Config.LIBRARY_MATCH_RT_TOL = self.rt_tolerance_var.get()
            Config.GROUP_NOISE_LEVEL = self.noise_var.get()
            raw = self.max_groups_var.get().strip()
            Config.MAX_GROUPS_TO_RUN = int(raw) if raw.isdigit() else None

            # Custom run name — fall back to a sensible default if blank
            run_name = self.run_name_var.get().strip()
            if not run_name:
                from datetime import datetime
                run_name = "maxiMZe_Run_" + datetime.now().strftime("%Y%m%d_%H%M")
            Config.ANALYSIS_FOLDER = run_name

            # Mass groups: fresh detection by default; import JSON if requested
            if self.import_json_var.get() and self.import_json_path:
                Config.REBUILD_MASS_GROUPS = False   # we are supplying groups externally
                Config._imported_json_path = self.import_json_path
            else:
                Config.REBUILD_MASS_GROUPS = True
                Config._imported_json_path = None

            # Target files — always pass through so MassGrouping guarantees one is picked
            Config.TARGET_FILES = list(self.target_files) if self.use_target else []

    def run_analysis(self):
        if not self.output_folder:
            messagebox.showwarning("Missing Output", "Please select an output folder")
            return

        # Validate JSON import if checkbox is ticked
        if self.import_json_var.get() and not self.import_json_path:
            messagebox.showwarning(
                "Missing JSON",
                "You checked 'Import previous MassGroups_Cache.json' but haven't selected a file.\n"
                "Please select a JSON file or uncheck the option to detect groups fresh.",
            )
            return

        if self.use_study_design:
            study_groups = self._collect_study_groups()
            empty_groups = [name for name, files in study_groups.items()
                            if name != "target" and not files]
            if empty_groups:
                messagebox.showwarning("Empty Groups",
                                       "The following groups have no files assigned:\n" + "\n".join(empty_groups))
                return
            self.study_groups = study_groups
            self.input_files = [f for name, files in study_groups.items() if name != "target" for f in files]
            if self.use_target:
                self.input_files += self.target_files
        else:
            study_groups = self._collect_study_groups()
            all_files = [f for files in study_groups.values() for f in files]
            if not all_files:
                messagebox.showwarning("Missing Input", "Please add at least one input file to the group.")
                return
            self.input_files = all_files
            self.study_groups = {}

        if not self.input_files:
            messagebox.showwarning("Missing Input", "Please select input files first")
            return

        self.run_btn.configure(state="disabled", text="Running Analysis...")
        self.clear_console()

        def analysis_thread():
            try:
                if not Pipeline or not Config:
                    messagebox.showerror("Error", "Pipeline modules not found.")
                    return

                self._apply_config()

                Config.USE_STUDY_DESIGN = self.use_study_design
                if self.use_study_design:
                    Config.SAMPLE_GROUPS = self.study_groups
                    Config.TARGET_GROUP = "target" if self.use_target and self.target_files else None

                Config.BASE_DIR = Path(self.output_folder)
                if self.library_file:
                    Config.LIB_FILE = self.library_file

                original_get_file_paths = FileUtils.get_file_paths
                FileUtils.get_file_paths = lambda: self.input_files

                print("=" * 70)
                print("Starting maxiM-Ze Analysis Pipeline")
                print("=" * 70)
                print(f"Run name:      {Config.ANALYSIS_FOLDER}")
                print(f"Study design:  {'enabled' if self.use_study_design else 'disabled (random)'}")
                print(f"Input files:   {len(self.input_files)}")
                print(f"Output folder: {self.output_folder}")
                raw = self.max_groups_var.get().strip()
                print(f"Max groups to run: {int(raw) if raw.isdigit() else 'all'}")
                if self.import_json_var.get() and self.import_json_path:
                    print(f"Mass groups:   imported from {self.import_json_path}")
                else:
                    print("Mass groups:   freshly detected from input files")
                if self.library_file:
                    print(f"Library file:  {self.library_file}")
                if self.use_study_design:
                    for gname, gfiles in self.study_groups.items():
                        print(f"  [{gname}]: {len(gfiles)} file(s)")
                print("=" * 70)

                # Pass import_json_path into initialize_mass_groups
                imported_json = getattr(Config, "_imported_json_path", None)
                pipeline = Pipeline(import_json_path=imported_json)
                pipeline.run()

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
    import multiprocessing
    multiprocessing.freeze_support()
    main()