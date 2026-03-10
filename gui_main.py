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

        # Study design state
        self.use_study_design = False
        self.study_groups = {}          # {"Group 1": [file, ...], ...}
        self.target_files = []          # files assigned as target/pooled controls
        self.use_target = False

        # UI tracking for dynamic group widgets
        self._group_frames = []         # list of dicts per group
        self._target_frame_data = None  # dict for the target slot

        self.setup_ui()

    # ------------------------------------------------------------------
    # UI skeleton
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#0B1F3B")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        try:
            from PIL import Image as PilImage
            _logo_path = Path(__file__).parent / "logo.png"
            pil_img = PilImage.open(_logo_path)
            self._logo_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                          size=(170, 18))
            self.logo_label = ctk.CTkLabel(self.sidebar, image=self._logo_img,
                                           text="", fg_color="transparent")
        except Exception:
            self.logo_label = ctk.CTkLabel(
                self.sidebar, text="maxiM-Ze",
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color="#6EE7B7",
            )
        self.logo_label.grid(row=0, column=0, padx=15, pady=(20, 10))

        self.home_btn = ctk.CTkButton(
            self.sidebar, text="Setup",
            command=lambda: self.show_page("home"),
            fg_color="#2BB673", hover_color="#6EE7B7",
        )
        self.home_btn.grid(row=1, column=0, padx=20, pady=10)

        self.about_btn = ctk.CTkButton(
            self.sidebar, text="About",
            command=lambda: self.show_page("about"),
            fg_color="#2BB673", hover_color="#6EE7B7",
        )
        self.about_btn.grid(row=2, column=0, padx=20, pady=10)

        # Main frame
        self.main_frame = ctk.CTkFrame(self, fg_color="#FFFFFF")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.create_pages()
        self.show_page("home")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def create_pages(self):
        self.pages = {}
        self.pages["home"] = self.create_home_page()
        self.pages["about"] = self.create_about_page()

    def create_home_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame, fg_color="#FFFFFF")

        title = ctk.CTkLabel(
            page, text="Analysis Setup",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#0B1F3B",
        )
        title.pack(pady=(0, 20))

        # ── 1. Study Design ───────────────────────────────────────────
        design_outer = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        design_outer.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            design_outer, text="1. Study Design",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#0B1F3B",
        ).pack(pady=(15, 5), anchor="w", padx=20)

        # Mode toggle row
        mode_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        mode_row.pack(fill="x", padx=20, pady=(0, 10))

        self.study_design_var = ctk.BooleanVar(value=False)
        self._mode_check = ctk.CTkCheckBox(
            mode_row,
            text="Use Study Design  (define experimental groups)",
            variable=self.study_design_var,
            command=self._on_mode_toggle,
            text_color="#0B1F3B",
        )
        self._mode_check.pack(side="left")

        self.study_info = ctk.CTkLabel(
            design_outer,
            text="Option B active — add your files below; samples will be randomly selected for mass grouping.",
            text_color="gray",
        )
        self.study_info.pack(anchor="w", padx=20, pady=(0, 5))

        # Number of groups row — packed here so it sits above the group cards
        self._n_groups_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        # DO NOT pack yet — _on_mode_toggle will pack/unpack it
        ctk.CTkLabel(self._n_groups_row, text="Number of experimental groups:").pack(side="left")
        self._n_groups_var = ctk.IntVar(value=2)
        ctk.CTkOptionMenu(
            self._n_groups_row,
            values=[str(i) for i in range(1, 11)],
            variable=ctk.StringVar(value="2"),
            command=self._on_n_groups_changed,
            width=80,
        ).pack(side="left", padx=(10, 0))

        # Container for group cards and target slot — packed AFTER _n_groups_row
        self._design_controls = ctk.CTkFrame(design_outer, fg_color="transparent")
        self._design_controls.pack(fill="x")

        self._build_study_design_controls(self._design_controls)

        # Bottom padding inside the outer card
        noise_row = ctk.CTkFrame(design_outer, fg_color="transparent")
        noise_row.pack(fill="x", padx=20, pady=(5, 15))
        ctk.CTkLabel(noise_row, text="Noise Level:", text_color="#0B1F3B").pack(side="left")
        self.noise_var = ctk.DoubleVar(value=5000.0)
        ctk.CTkEntry(noise_row, width=120, textvariable=self.noise_var).pack(side="right")

        # ── 2. Output Folder ──────────────────────────────────────────
        output_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        output_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            output_frame, text="2. Output Folder",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#0B1F3B",
        ).pack(pady=(15, 10), anchor="w", padx=20)

        ctk.CTkButton(
            output_frame, text="Select Output Folder",
            command=self.select_output_folder, height=40,
        ).pack(fill="x", padx=20, pady=5)

        self.output_status = ctk.CTkLabel(output_frame, text="No folder selected", text_color="#555")
        self.output_status.pack(pady=(5, 15), padx=20, anchor="w")

        # ── 3. Compound Library ───────────────────────────────────────
        library_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=1, border_color="#FFFFFF")
        library_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            library_frame, text="3. Compound Library (Optional)",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#0B1F3B",
        ).pack(pady=(15, 10), anchor="w", padx=20)

        ctk.CTkButton(
            library_frame, text="Select Library File (CSV/XLSX)",
            command=self.select_library_file, height=40,
        ).pack(fill="x", padx=20, pady=5)

        self.library_status = ctk.CTkLabel(library_frame, text="No library file selected", text_color="#555")
        self.library_status.pack(pady=(5, 10), padx=20, anchor="w")

        mz_tol_row = ctk.CTkFrame(library_frame, fg_color="transparent")
        mz_tol_row.pack(fill="x", padx=20, pady=3)
        ctk.CTkLabel(mz_tol_row, text="Mass Tolerance (Da):", text_color="#0B1F3B").pack(side="left")
        self.mz_tolerance_var = ctk.DoubleVar(value=0.0005)
        ctk.CTkEntry(mz_tol_row, width=120, textvariable=self.mz_tolerance_var).pack(side="right")

        rt_tol_row = ctk.CTkFrame(library_frame, fg_color="transparent")
        rt_tol_row.pack(fill="x", padx=20, pady=(3, 15))
        ctk.CTkLabel(rt_tol_row, text="RT Tolerance (min):", text_color="#0B1F3B").pack(side="left")
        self.rt_tolerance_var = ctk.DoubleVar(value=0.1)
        ctk.CTkEntry(rt_tol_row, width=120, textvariable=self.rt_tolerance_var).pack(side="right")

        # ── Run ───────────────────────────────────────────────────────
        self.run_btn = ctk.CTkButton(
            page, text="Run Analysis",
            command=self.run_analysis,
            height=60, font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#2BB673", hover_color="#6EE7B7",
        )
        self.run_btn.pack(fill="x", pady=20)

        # ── Console ───────────────────────────────────────────────────
        console_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=4, border_color="#D0E8F5")
        console_frame.pack(fill="both", expand=True, pady=(0, 20))

        ctk.CTkLabel(
            console_frame, text="Console Output",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(15, 10))

        self.console_output = ctk.CTkTextbox(
            console_frame, height=250,
            font=ctk.CTkFont(family="Consolas", size=11),
        )
        self.console_output.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        ctk.CTkButton(
            console_frame, text="Clear Console",
            command=self.clear_console,
        ).pack(pady=(0, 15))

        sys.stdout = ConsoleRedirect(self.console_output)

        return page

    # ------------------------------------------------------------------
    # Study design sub-UI
    # ------------------------------------------------------------------

    def _build_study_design_controls(self, parent):
        """Populate the group cards and target slot."""

        # ── plain frame for group cards (expands naturally, no scroll) ─
        self._groups_scroll = ctk.CTkFrame(parent, fg_color="transparent")
        self._groups_scroll.pack(fill="x", padx=20, pady=(0, 10))

        # ── target sample toggle ──────────────────────────────────────
        target_toggle_row = ctk.CTkFrame(parent, fg_color="transparent")
        target_toggle_row.pack(fill="x", padx=20, pady=(0, 5))

        self._use_target_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            target_toggle_row,
            text="Include target samples  (e.g. pooled plasma controls)",
            variable=self._use_target_var,
            command=self._on_target_toggle,
        ).pack(side="left")

        # Container for the target file slot (hidden until toggled on)
        self._target_slot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        # NOT packed yet

        self._build_target_slot(self._target_slot_frame)

        # Render initial group cards — 1 for Option B default
        self._render_group_cards(1)

    def _render_group_cards(self, n_groups: int):
        """Destroy old group cards and build n_groups fresh ones."""
        for widget in self._groups_scroll.winfo_children():
            widget.destroy()
        self._group_frames = []

        for i in range(1, n_groups + 1):
            self._add_group_card(i)

    def _add_group_card(self, group_number: int):
        """Add a single group card to the scroll area."""
        card = ctk.CTkFrame(self._groups_scroll, border_width=2, border_color="#2BB673")
        card.pack(fill="x", pady=5)

        # Header row
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        name_var = ctk.StringVar(value=f"Group {group_number}")
        name_entry = ctk.CTkEntry(header, textvariable=name_var, width=160,
                                  placeholder_text="Group name")
        name_entry.pack(side="left")

        file_count_label = ctk.CTkLabel(header, text="0 file(s)", text_color="gray")
        file_count_label.pack(side="right")

        # File list box
        file_listbox = ctk.CTkTextbox(card, height=60,
                                      font=ctk.CTkFont(family="Consolas", size=10),
                                      state="disabled")
        file_listbox.pack(fill="x", padx=10, pady=(0, 6))

        # Buttons row
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        group_data = {
            "name_var": name_var,
            "files": [],
            "file_listbox": file_listbox,
            "file_count_label": file_count_label,
        }
        self._group_frames.append(group_data)

        # Capture group_data by reference so buttons update the right group
        ctk.CTkButton(
            btn_row, text="Add Files", width=100, height=28,
            command=lambda gd=group_data: self._add_files_to_group(gd),
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Clear", width=80, height=28,
            fg_color="#555", hover_color="#777",
            command=lambda gd=group_data: self._clear_group(gd),
        ).pack(side="left")

    def _build_target_slot(self, parent):
        """Build the target sample file slot inside parent."""
        card = ctk.CTkFrame(parent, border_width=2, border_color="#2BB673")
        card.pack(fill="x", padx=20, pady=4)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            header, text="Target Samples  (pooled controls)",
            font=ctk.CTkFont(weight="bold"), text_color="#2BB673",
        ).pack(side="left")

        self._target_count_label = ctk.CTkLabel(header, text="0 file(s)", text_color="gray")
        self._target_count_label.pack(side="right")

        self._target_listbox = ctk.CTkTextbox(
            card, height=55,
            font=ctk.CTkFont(family="Consolas", size=10),
            state="disabled",
        )
        self._target_listbox.pack(fill="x", padx=10, pady=(0, 6))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkButton(
            btn_row, text="Add Files", width=100, height=28,
            command=self._add_target_files,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Clear", width=80, height=28,
            fg_color="#555", hover_color="#777",
            command=self._clear_target,
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Group / target interaction helpers
    # ------------------------------------------------------------------

    def _add_files_to_group(self, group_data: dict):
        files = filedialog.askopenfilenames(
            title=f"Select files for {group_data['name_var'].get()}",
            filetypes=[("MS files", "*.mzxml *.mzml"), ("All files", "*.*")],
        )
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
        group_data["file_count_label"].configure(
            text=f"{len(group_data['files'])} file(s)"
        )

    def _add_target_files(self):
        files = filedialog.askopenfilenames(
            title="Select Target / Pooled Control Files",
            filetypes=[("MS files", "*.mzxml *.mzml"), ("All files", "*.*")],
        )
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
                text_color="#2BB673",
            )
            # Unpack _design_controls, insert _n_groups_row, then repack _design_controls
            self._design_controls.pack_forget()
            self._n_groups_row.pack(fill="x", padx=20, pady=(5, 10))
            self._design_controls.pack(fill="x")
            self._render_group_cards(self._n_groups_var.get())
        else:
            self.study_info.configure(
                text="Option B active — add your files below; samples will be randomly selected for mass grouping.",
                text_color="gray",
            )
            self._n_groups_row.pack_forget()
            self._render_group_cards(1)

    def _on_n_groups_changed(self, value: str):
        self._render_group_cards(int(value))
        # Re-show target slot if it was visible
        if self._use_target_var.get():
            self._target_slot_frame.pack(fill="x")

    def _on_target_toggle(self):
        self.use_target = self._use_target_var.get()
        if self.use_target:
            self._target_slot_frame.pack(fill="x", padx=0, pady=(0, 5))
        else:
            self._target_slot_frame.pack_forget()

    # ------------------------------------------------------------------
    # Helper: collect study groups dict from UI state
    # ------------------------------------------------------------------

    def _collect_study_groups(self) -> dict:
        """Return {"Group 1": [paths], "Group 2": [paths], "target": [paths]}."""
        out = {}
        for gd in self._group_frames:
            name = gd["name_var"].get().strip() or f"Group {self._group_frames.index(gd)+1}"
            out[name] = gd["files"]
        if self.use_target and self.target_files:
            out["target"] = self.target_files
        return out

    # ------------------------------------------------------------------
    # File / folder pickers (global)
    # ------------------------------------------------------------------

    def select_input_files(self):
        files = filedialog.askopenfilenames(
            title="Select Input Files",
            filetypes=[
                ("MS files", "*.mzxml *.mzml"),
                ("mzXML files", "*.mzxml"),
                ("mzML files", "*.mzml"),
                ("All files", "*.*"),
            ],
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
            filetypes=[
                ("Spreadsheet files", "*.csv *.xlsx"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*"),
            ],
        )
        if file:
            self.library_file = file
            self.library_status.configure(text=f"Library: {Path(file).name}")
            print(f"Library file: {file}")

    def create_about_page(self):
        page = ctk.CTkScrollableFrame(self.main_frame, fg_color="#FFFFFF")

        ctk.CTkLabel(
            page, text="About maxiM-Ze",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#0B1F3B",
        ).pack(pady=(0, 20))

        info_frame = ctk.CTkFrame(page, fg_color="#FFFFFF", border_width=2, border_color="#D0E8F5")
        info_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            info_frame, text="Version: 1.0.0",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(20, 10))

        ctk.CTkLabel(
            info_frame,
            text=(
                "A Novel Image Recognition Approach for Visualizing and\n"
                "Processing Mass Spectrometry Based Metabolomics Data"
            ),
            wraplength=500, justify="center",
        ).pack(pady=10)

        features_text = (
            "Features:\n"
            "• Support for mzXML and mzML file formats\n"
            "• Dynamic mass grouping from sample data\n"
            "• Optional study design with experimental groups\n"
            "• Compound library matching with configurable tolerances\n"
            "• Automated peak detection and clustering\n"
            "• Comprehensive visualization and reporting\n"
            "• Multi-threaded processing for performance"
        )
        ctk.CTkLabel(info_frame, text=features_text, justify="left").pack(pady=(10, 20))

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

    def run_analysis(self):
        if not self.output_folder:
            messagebox.showwarning("Missing Output", "Please select an output folder")
            return

        # Collect files depending on mode
        if self.use_study_design:
            study_groups = self._collect_study_groups()
            empty_groups = [name for name, files in study_groups.items()
                            if name != "target" and not files]
            if empty_groups:
                messagebox.showwarning(
                    "Empty Groups",
                    "The following groups have no files assigned:\n"
                    + "\n".join(empty_groups),
                )
                return
            self.study_groups = study_groups
            # Flatten all non-target files as the input list
            self.input_files = [
                f for name, files in study_groups.items()
                if name != "target" for f in files
            ]
            if self.use_target:
                self.input_files += self.target_files
        else:
            # Option B: user must have added files via the single group slot
            study_groups = self._collect_study_groups()
            all_files = [f for files in study_groups.values() for f in files]
            if not all_files:
                messagebox.showwarning(
                    "Missing Input",
                    "Please add at least one input file to the group.",
                )
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

                # Propagate study design state to Config
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
                print(f"Study design: {'enabled' if self.use_study_design else 'disabled (random)'}")
                print(f"Input files:  {len(self.input_files)}")
                print(f"Output folder: {self.output_folder}")
                if self.library_file:
                    print(f"Library file: {self.library_file}")
                if self.use_study_design:
                    for gname, gfiles in self.study_groups.items():
                        print(f"  [{gname}]: {len(gfiles)} file(s)")
                print("=" * 70)

                pipeline = Pipeline()
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
    main()