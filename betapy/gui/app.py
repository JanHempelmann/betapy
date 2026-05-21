"""
betapy GUI — top-level application window.
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QStatusBar, QMessageBox,
)
from PyQt5.QtCore import Qt

from betapy.core.settings import Settings
from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS
from betapy.core.structure import Supercell


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('betapy — Projected Force Constant Analysis')
        self.resize(1200, 800)

        self.settings  = Settings()
        self.supercell = None
        self.fc_data   = None

        self._build_ui()
        self._autoload_cwd()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_load_bar())

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        from betapy.gui.pfc_viewer  import PFCViewerWidget
        from betapy.gui.site_picker import SitePickerWidget

        self.pfc_viewer  = PFCViewerWidget()
        self.site_picker = SitePickerWidget()

        self.tabs.addTab(self.pfc_viewer,  'pFC Viewer')
        self.tabs.addTab(self.site_picker, 'Reference Site Picker')

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            'Open SPOSCAR and FORCE_CONSTANTS to run analysis, '
            'or load a settings file, '
            'or use "Open existing pFCs CSV" in the pFC Viewer tab.'
        )

    def _build_load_bar(self):
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(4, 4, 4, 4)

        btn_settings = QPushButton('Load settings…')
        btn_settings.clicked.connect(self._load_settings)
        self.lbl_settings = QLabel('No settings file')

        btn_sposcar = QPushButton('Open SPOSCAR…')
        btn_sposcar.clicked.connect(self._load_sposcar)
        self.lbl_sposcar = QLabel('SPOSCAR: not loaded')

        btn_fc = QPushButton('Open FORCE_CONSTANTS…')
        btn_fc.clicked.connect(self._load_fc)
        self.lbl_fc = QLabel('FORCE_CONSTANTS: not loaded')

        self.btn_run = QPushButton('Analyse')
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_analysis)

        for w in [btn_settings, self.lbl_settings,
                  btn_sposcar,  self.lbl_sposcar,
                  btn_fc,       self.lbl_fc,
                  self.btn_run]:
            row.addWidget(w)
        row.addStretch()
        return bar

    # ------------------------------------------------------------------
    # Auto-loading
    # ------------------------------------------------------------------

    def _autoload_cwd(self):
        """
        On startup, check the current working directory for standard
        Phonopy/betapy output files and load whatever is present.
        Any combination of these three can be present; missing files are
        silently skipped.
        """
        cwd      = Path.cwd()
        loaded   = []
        messages = []

        sposcar_path = cwd / 'SPOSCAR'
        if sposcar_path.exists():
            try:
                self._do_load_sposcar(sposcar_path)
                loaded.append('SPOSCAR')
            except Exception as e:
                messages.append(f'SPOSCAR auto-load failed: {e}')

        csv_path = cwd / 'unique_pFCs.csv'
        if csv_path.exists():
            try:
                self.pfc_viewer.load_from_csv(str(csv_path))
                loaded.append('unique_pFCs.csv')
            except Exception as e:
                messages.append(f'unique_pFCs.csv auto-load failed: {e}')

        fc_path = cwd / 'FORCE_CONSTANTS'
        if fc_path.exists():
            try:
                self._do_load_fc(fc_path)
                loaded.append('FORCE_CONSTANTS')
            except Exception as e:
                messages.append(f'FORCE_CONSTANTS auto-load failed: {e}')

        refpos_path = cwd / 'REFPOS'
        if refpos_path.exists():
            self.site_picker.load_refpos(str(refpos_path))
            loaded.append('REFPOS')

        if loaded:
            self.status.showMessage(
                f'Auto-loaded from {cwd}: {", ".join(loaded)}'
            )
        if messages:
            self.status.showMessage(' | '.join(messages))

    # ------------------------------------------------------------------
    # Settings file
    # ------------------------------------------------------------------

    def _load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load settings file', '',
            'YAML files (*.yaml *.yml);;All files (*)',
        )
        if not path:
            return
        try:
            self.settings = Settings.from_yaml(path)
            self.lbl_settings.setText(f'Settings: {Path(path).name}  ✓')
            self.status.showMessage(f'Loaded settings from {path}')
            self._try_autoload_from_settings()
        except Exception as e:
            QMessageBox.critical(self, 'Error loading settings', str(e))

    def _try_autoload_from_settings(self):
        sp = Path(self.settings.sposcar)
        fc = Path(self.settings.force_constants)
        if sp.exists():
            try:
                self._do_load_sposcar(sp)
            except Exception:
                pass
        if fc.exists():
            try:
                self._do_load_fc(fc)
            except Exception:
                pass
        if sp.exists() and fc.exists():
            self.status.showMessage(
                f'Auto-loaded {sp.name} and {fc.name} from settings.'
            )

    # ------------------------------------------------------------------
    # File loading — internal helpers
    # ------------------------------------------------------------------

    def _do_load_sposcar(self, path):
        """
        Load a SPOSCAR file and push the supercell to all tools that need it.
        Raises on error so callers can handle it appropriately.
        """
        path = Path(path)
        self.supercell = Supercell(read_SPOSCAR(path))
        self.settings.sposcar = str(path)
        self.lbl_sposcar.setText(f'SPOSCAR: {path.name}  ✓')

        # Push supercell to pFC viewer's structure panel immediately —
        # no analysis needed to render the structure
        self.pfc_viewer.set_supercell(self.supercell)
        self.site_picker.load_supercell(
            self.supercell,
            self.fc_data,   # may be None if FC not loaded yet, that's fine
        )

        # Auto-load REFPOS from the same directory as the SPOSCAR
        refpos_path = path.parent / 'REFPOS'
        if refpos_path.exists():
            self.site_picker.load_refpos(str(refpos_path))

        self._check_ready()

    def _do_load_fc(self, path):
        """Load a FORCE_CONSTANTS file. Raises on error."""
        path = Path(path)
        self.fc_data = read_FORCE_CONSTANTS(path)
        self.settings.force_constants = str(path)
        n = len(self.fc_data['atomic_pairs'])
        self.lbl_fc.setText(
            f'FORCE_CONSTANTS: {path.name}  ({n} pairs) ✓'
        )
        # Update site picker with FC data if supercell already loaded
        if self.supercell is not None:
            self.site_picker.load_supercell(self.supercell, self.fc_data)
        self._check_ready()

    # ------------------------------------------------------------------
    # Manual file loading (button handlers — thin wrappers)
    # ------------------------------------------------------------------

    def _load_sposcar(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open SPOSCAR', '',
            'SPOSCAR files (SPOSCAR);;All files (*)',
        )
        if not path:
            return
        try:
            self._do_load_sposcar(path)
            self.status.showMessage(str(self.supercell))
        except Exception as e:
            QMessageBox.critical(self, 'Error loading SPOSCAR', str(e))

    def _load_fc(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open FORCE_CONSTANTS', '',
            'FORCE_CONSTANTS files (FORCE_CONSTANTS);;All files (*)',
        )
        if not path:
            return
        try:
            self._do_load_fc(path)
        except Exception as e:
            QMessageBox.critical(self, 'Error loading FORCE_CONSTANTS', str(e))

    def _check_ready(self):
        if self.supercell is not None and self.fc_data is not None:
            self.btn_run.setEnabled(True)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _run_analysis(self):
        from betapy.core.projection import compute_bulk_pfcs, unique_pfcs

        self.status.showMessage('Running analysis…')
        QApplication.processEvents()

        try:
            results, onsite, _ = compute_bulk_pfcs(
                self.supercell,
                self.fc_data['atomic_pairs'],
                self.fc_data['force_matrices'],
            )
            df_unique = unique_pfcs(results)
        except Exception as e:
            QMessageBox.critical(self, 'Analysis error', str(e))
            self.status.showMessage('Analysis failed.')
            return

        self.pfc_viewer.load_data(df_unique, results, supercell=self.supercell)
        self.site_picker.load_supercell(self.supercell, self.fc_data)

        self.status.showMessage(
            f'Analysis complete — {len(results)} off-site pairs, '
            f'{len(df_unique)} unique pFC values.'
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
