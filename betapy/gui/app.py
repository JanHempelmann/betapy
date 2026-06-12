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
    QDialog, QFormLayout, QDialogButtonBox, QComboBox,
    QTabBar, QMenu, QProgressBar,
)
from PyQt5.QtCore import Qt, QSettings, QEvent, QTimer, QThread, QPoint, pyqtSignal
from PyQt5.QtGui import QIcon

from betapy.core.settings import Settings
from betapy.core.io import (
    read_SPOSCAR, read_FORCE_CONSTANTS,
    write_bulk_pfcs, read_bulk_pfcs,
)
from betapy.core.structure import Supercell

# QSettings identifiers — platform-native storage location
_ORG  = 'betapy'
_APP  = 'betapy'

# Tab preference values
_AUTO   = 'auto'
_ALWAYS = 'always'
_NEVER  = 'never'



class _TabBar(QTabBar):
    """
    QTabBar subclass whose sizeHint equals the sum of natural tab widths —
    no expansion.  This keeps the allocated bar width tight against the tabs
    so the '+' overlay button (a child of QTabWidget) can sit in the empty
    background area immediately to their right.
    """

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(self._natural_tabs_width())
        return hint

    def minimumSizeHint(self):
        return self.sizeHint()

    def _natural_tabs_width(self):
        return sum(self.tabSizeHint(i).width() for i in range(self.count()))

    def natural_tabs_width(self):
        return self._natural_tabs_width()


class PreferencesDialog(QDialog):
    """
    Controls which optional tabs are visible.

    Each tab has three modes:
      Auto   — show when relevant files / CLI flags are detected
      Always — show unconditionally
      Never  — hide unconditionally (override auto-detection)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('betapy — Preferences')
        self.setFixedWidth(420)
        self._qs = QSettings(_ORG, _APP)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lbl = QLabel('Optional tab visibility')
        lbl.setStyleSheet('font-weight: bold;')
        layout.addWidget(lbl)

        note = QLabel(
            'Auto conditions:\n'
            '  Ref. Site Projection — REFPOS present in working directory, '
            'or --refsite flag used\n'
            '  Stiffness Shift — settings file contains stiffness_shift: section, '
            'or --stiffness-shift flag used\n'
            '  Multicenter Bonding  (β) — must be opened manually via the + menu'
        )
        note.setWordWrap(True)
        note.setStyleSheet('color: #666; font-size: 11px; padding: 4px 0 8px 0;')
        layout.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._refsite_combo = self._make_combo('tab/refsite')
        self._shift_combo   = self._make_combo('tab/stiffness_shift')
        self._mc_combo      = self._make_combo('tab/multicenter')
        form.addRow('Ref. Site Projection:', self._refsite_combo)
        form.addRow('Stiffness Shift:',      self._shift_combo)
        form.addRow('Multicenter Bonding  (β):',  self._mc_combo)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _make_combo(self, key):
        combo = QComboBox()
        combo.addItems(['Auto', 'Always', 'Never'])
        current = self._qs.value(key, _AUTO)
        combo.setCurrentText(current.capitalize())
        return combo

    def _save_and_accept(self):
        self._qs.setValue('tab/refsite',
                          self._refsite_combo.currentText().lower())
        self._qs.setValue('tab/stiffness_shift',
                          self._shift_combo.currentText().lower())
        self._qs.setValue('tab/multicenter',
                          self._mc_combo.currentText().lower())
        self.accept()


class _AnalysisWorker(QThread):
    """Background thread for compute_bulk_pfcs so the GUI stays responsive."""
    progress = pyqtSignal(int, int)   # (current_pair, total_pairs)
    finished = pyqtSignal(list, list) # (results, onsite)
    error    = pyqtSignal(str)

    def __init__(self, supercell, fc_data):
        super().__init__()
        self._supercell = supercell
        self._fc_data   = fc_data

    def run(self):
        try:
            from betapy.core.projection import compute_bulk_pfcs
            results, onsite, _ = compute_bulk_pfcs(
                self._supercell,
                self._fc_data['atomic_pairs'],
                self._fc_data['force_matrices'],
                show_progress=False,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(results, onsite)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):

    def __init__(self, splash=None, cli_args=None):
        super().__init__()
        self._splash   = splash
        self._cli_args = cli_args   # argparse.Namespace from Settings.from_cli()

        self.setWindowTitle('betapy — Projected Force Constant Analysis')
        self.resize(1200, 800)

        self.settings        = Settings()
        self.supercell       = None
        self.fc_data         = None
        self._lobster_pairs  = None
        self._lobster_dir    = None
        self._bulk_results   = None

        if self._splash:
            self._splash.set_status('Initializing interface…')
        self._build_ui()
        self._autoload_cwd()
        self._update_tab_visibility()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_load_bar())

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Custom tab bar: sizeHint = natural tab widths only, so the allocated
        # bar stays tight against the tabs and the '+' overlay has room beside them.
        self.tabs.setTabBar(_TabBar())

        # '+' button overlay: child of QTabWidget so it isn't clipped by the
        # narrow QTabBar geometry; repositioned after every tab change.
        self._plus_btn = QPushButton('+', self.tabs)
        self._plus_btn.setFixedSize(30, 24)
        self._plus_btn.setFlat(True)
        self._plus_btn.setToolTip('Add tab')
        self._plus_btn.setStyleSheet(
            'QPushButton { border:1px solid #aaa; border-radius:3px;'
            '              background:#dadada; font-size:15px; font-weight:bold; }'
            'QPushButton:hover  { background:#c4c4c4; }'
            'QPushButton:pressed{ background:#b0b0b0; }'
        )
        self._plus_btn.clicked.connect(self._on_plus_clicked)
        self.tabs.tabBar().installEventFilter(self)

        from betapy.gui.pfc_viewer             import PFCViewerWidget
        from betapy.gui.site_picker            import SitePickerWidget
        from betapy.gui.stiffness_shift_widget import StiffnessShiftWidget
        from betapy.gui.lt_viewer              import LTDecompositionWidget

        self.pfc_viewer      = PFCViewerWidget()
        self.site_picker     = SitePickerWidget()
        self.stiffness_shift = StiffnessShiftWidget()
        self.lt_viewer       = LTDecompositionWidget()
        # Created lazily the first time the user opens the tab via the + menu,
        # so their VTK contexts (expensive) are not allocated on every startup.
        self.multicenter     = None
        self.badger          = None

        # pFC Viewer is always present; track permanent tabs (no close button).
        self.tabs.addTab(self.pfc_viewer, 'pFC Viewer')
        self._permanent_widgets = {self.pfc_viewer}

        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)

        self._sync_close_buttons()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            'Open SPOSCAR and FORCE_CONSTANTS to run analysis, '
            'or load a settings file, '
            'or use "Open existing pFCs CSV" in the pFC Viewer tab.'
        )

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(220)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()
        self.status.addPermanentWidget(self._progress_bar)
        self._worker = None

        # Apply saved unit preference to all freshly created widgets
        self._set_unit(self._unit_combo.currentData())

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

        btn_prefs = QPushButton('⚙')
        btn_prefs.setFixedWidth(28)
        btn_prefs.setToolTip('Preferences')
        btn_prefs.clicked.connect(self._open_preferences)

        for w in [btn_settings, self.lbl_settings,
                  btn_sposcar,  self.lbl_sposcar,
                  btn_fc,       self.lbl_fc,
                  self.btn_run]:
            row.addWidget(w)
        row.addStretch()
        row.addWidget(QLabel('Units:'))
        self._unit_combo = QComboBox()
        self._unit_combo.addItem('eV/Å²', 'eV/Ang2')
        self._unit_combo.addItem('N/m',   'N/m')
        saved_unit = QSettings(_ORG, _APP).value('display/unit', 'eV/Ang2')
        idx = self._unit_combo.findData(saved_unit)
        if idx >= 0:
            self._unit_combo.setCurrentIndex(idx)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        row.addWidget(self._unit_combo)
        row.addWidget(btn_prefs)
        return bar

    # ------------------------------------------------------------------
    # Tab visibility management
    # ------------------------------------------------------------------

    def _update_tab_visibility(self):
        """Add or remove optional tabs based on preferences + auto-detection."""
        qs = QSettings(_ORG, _APP)
        pref_refsite = qs.value('tab/refsite',         _AUTO)
        pref_shift   = qs.value('tab/stiffness_shift', _AUTO)
        pref_mc      = qs.value('tab/multicenter',     _AUTO)

        show_refsite = (
            pref_refsite == _ALWAYS or
            (pref_refsite == _AUTO and self._should_show_refsite_tab())
        )
        show_shift = (
            pref_shift == _ALWAYS or
            (pref_shift == _AUTO and self._should_show_stiffness_tab())
        )
        show_mc = (
            pref_mc == _ALWAYS or
            (pref_mc == _AUTO and self._should_show_multicenter_tab())
        )

        self._set_tab_visible(self.multicenter,     'Multicenter Bonding  (β)',  1, show_mc)
        self._set_tab_visible(self.site_picker,     'Ref. Site Projection', 2, show_refsite)
        self._set_tab_visible(self.stiffness_shift, 'Stiffness Shift',      3, show_shift)
        self._sync_close_buttons()

    def _should_show_refsite_tab(self):
        """Auto condition: REFPOS in CWD, or --refsite flag given."""
        if self._cli_args and getattr(self._cli_args, 'refsite', None) is not None:
            return True
        if (Path.cwd() / 'REFPOS').exists():
            return True
        return False

    def _should_show_stiffness_tab(self):
        """Auto condition: settings carry a stiffness_shift section, or --stiffness-shift flag given."""
        if self._cli_args and getattr(self._cli_args, 'stiffness_shift', False):
            return True
        if self.settings.stiffness_shift is not None:
            return True
        return False

    def _should_show_multicenter_tab(self):
        """Auto condition: disabled — tab must be opened manually (beta feature)."""
        return False

    def _set_tab_visible(self, widget, label, preferred_idx, show):
        """Insert or remove a tab without destroying the widget."""
        if widget is None:
            return
        current_idx = self.tabs.indexOf(widget)
        if show and current_idx == -1:
            idx = min(preferred_idx, self.tabs.count())
            self.tabs.insertTab(idx, widget, label)
        elif not show and current_idx != -1:
            self.tabs.removeTab(current_idx)

    def _open_preferences(self):
        dlg = PreferencesDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self._update_tab_visibility()

    def _on_unit_changed(self):
        unit = self._unit_combo.currentData()
        QSettings(_ORG, _APP).setValue('display/unit', unit)
        self._set_unit(unit)

    def _set_unit(self, unit: str):
        from betapy.gui.pfc_viewer import PFCViewerWidget
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if isinstance(w, PFCViewerWidget):
                w.set_unit(unit)
        self.site_picker.set_unit(unit)
        self.stiffness_shift.set_unit(unit)
        self.lt_viewer.set_unit(unit)
        if self.badger is not None:
            self.badger.set_unit(unit)

    def _ensure_multicenter(self):
        """Create MulticenterWidget on first use and push any existing data."""
        if self.multicenter is None:
            from betapy.gui.multicenter_viewer import MulticenterWidget
            self.multicenter = MulticenterWidget()
            if self._bulk_results is not None and self.supercell is not None:
                self.multicenter.load_data(
                    self._bulk_results, self.supercell,
                    lobster_dir=self._lobster_dir,
                )
        return self.multicenter

    def _ensure_badger(self):
        """Create BadgerWidget on first use and push any existing data."""
        if self.badger is None:
            from betapy.gui.badger_viewer import BadgerWidget
            self.badger = BadgerWidget()
            self.badger.set_unit(self._unit_combo.currentData())
            if self._bulk_results is not None:
                self.badger.load_data(
                    self._bulk_results,
                    reliability_cutoff=self.pfc_viewer._reliability_cutoff,
                    supercell=self.supercell,
                )
        return self.badger

    # ------------------------------------------------------------------
    # Tab bar — "+" overlay button and close handling
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Reposition the '+' button whenever the tab bar lays out."""
        if obj is self.tabs.tabBar() and event.type() in (
            QEvent.Resize, QEvent.LayoutRequest
        ):
            QTimer.singleShot(0, self._reposition_plus_btn)
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_plus_btn()

    def _reposition_plus_btn(self):
        """Place the '+' button immediately after the last tab's natural width."""
        bar = self.tabs.tabBar()
        if not isinstance(bar, _TabBar) or bar.count() == 0:
            self._plus_btn.hide()
            return
        natural_x = bar.natural_tabs_width()
        tab_h     = bar.height()
        btn_h     = self._plus_btn.height()
        origin = bar.mapTo(self.tabs, QPoint(0, 0))
        x = origin.x() + natural_x + 4
        y = origin.y() + max(0, (tab_h - btn_h) // 2)
        self._plus_btn.move(x, y)
        self._plus_btn.raise_()
        self._plus_btn.show()

    def _on_plus_clicked(self):
        pos = self._plus_btn.mapToGlobal(self._plus_btn.rect().bottomLeft())
        self._show_add_tab_menu(pos)

    def _sync_close_buttons(self):
        """Remove close buttons from permanent tabs after any tab insertion."""
        bar = self.tabs.tabBar()
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) in self._permanent_widgets:
                bar.setTabButton(i, QTabBar.RightSide, None)
                bar.setTabButton(i, QTabBar.LeftSide,  None)
        QTimer.singleShot(0, self._reposition_plus_btn)

    def _close_tab(self, idx):
        widget = self.tabs.widget(idx)
        if widget in self._permanent_widgets:
            return
        self.tabs.removeTab(idx)
        # Optional singleton tabs are kept alive for re-adding; extra pFC
        # viewers are independent instances and can be destroyed.
        if widget not in (self.site_picker, self.stiffness_shift,
                          self.lt_viewer, self.multicenter, self.badger):
            widget.deleteLater()
        QTimer.singleShot(0, self._reposition_plus_btn)

    def _show_add_tab_menu(self, pos):
        menu = QMenu(self)

        menu.addAction('New pFC Viewer', self._add_pfc_viewer_tab)
        menu.addSeparator()

        has_refsite = self.tabs.indexOf(self.site_picker)     != -1
        has_shift   = self.tabs.indexOf(self.stiffness_shift) != -1
        has_lt      = self.tabs.indexOf(self.lt_viewer)       != -1
        has_mc      = (self.multicenter is not None and
                       self.tabs.indexOf(self.multicenter) != -1)
        has_badger  = (self.badger is not None and
                       self.tabs.indexOf(self.badger) != -1)

        label_ref    = ('• ' if has_refsite else '  ') + 'Ref. Site Projection'
        label_shift  = ('• ' if has_shift   else '  ') + 'Stiffness Shift'
        label_lt     = ('• ' if has_lt      else '  ') + 'LT Decomposition  (β)'
        label_mc     = ('• ' if has_mc      else '  ') + 'Multicenter Bonding  (β)'
        label_badger = ('• ' if has_badger  else '  ') + 'Badger Analysis  (β)'

        menu.addAction(label_mc,     lambda: self._add_optional_tab(
            self._ensure_multicenter(), 'Multicenter Bonding  (β)'))
        menu.addAction(label_badger, lambda: self._add_optional_tab(
            self._ensure_badger(), 'Badger Analysis  (β)'))
        menu.addAction(label_ref,    lambda: self._add_optional_tab(
            self.site_picker, 'Ref. Site Projection'))
        menu.addAction(label_shift,  lambda: self._add_optional_tab(
            self.stiffness_shift, 'Stiffness Shift'))
        menu.addAction(label_lt,     lambda: self._add_optional_tab(
            self.lt_viewer, 'LT Decomposition  (β)'))

        menu.exec_(pos)

    def _add_optional_tab(self, widget, label):
        """Show an optional singleton tab, or focus it if already present."""
        idx = self.tabs.indexOf(widget)
        if idx != -1:
            self.tabs.setCurrentIndex(idx)
        else:
            self.tabs.addTab(widget, label)
            self.tabs.setCurrentWidget(widget)
            self._sync_close_buttons()

    def _add_pfc_viewer_tab(self):
        """Open an additional independent pFC Viewer tab."""
        from betapy.gui.pfc_viewer import PFCViewerWidget
        viewer = PFCViewerWidget()
        if self.supercell is not None:
            viewer.set_supercell(self.supercell)
        if self._lobster_pairs is not None:
            viewer.set_lobster_pairs(self._lobster_pairs)
        if self._lobster_dir is not None:
            viewer.set_lobster_dir(self._lobster_dir)
        viewer.set_unit(self._unit_combo.currentData())
        n = sum(1 for i in range(self.tabs.count())
                if isinstance(self.tabs.widget(i), PFCViewerWidget))
        self.tabs.addTab(viewer, f'pFC Viewer ({n + 1})')
        self.tabs.setCurrentWidget(viewer)
        QTimer.singleShot(0, self._reposition_plus_btn)

    # ------------------------------------------------------------------
    # Auto-loading
    # ------------------------------------------------------------------

    def _autoload_cwd(self):
        """
        On startup, check the working directory for standard Phonopy/betapy
        files and load whatever is present. Missing files are silently skipped;
        files that fail to parse raise a warning dialog.
        """
        cwd    = Path.cwd()
        loaded = []
        errors = []

        sposcar_path = cwd / 'SPOSCAR'
        if sposcar_path.exists():
            if self._splash:
                self._splash.set_status('Loading SPOSCAR…')
            try:
                self._do_load_sposcar(sposcar_path)
                loaded.append('SPOSCAR')
            except Exception as e:
                errors.append(f'SPOSCAR: {e}')

        csv_path = cwd / 'unique_pFCs.csv'
        if csv_path.exists():
            if self._splash:
                self._splash.set_status('Loading pFCs CSV…')
            try:
                self.pfc_viewer.load_from_csv(str(csv_path))
                loaded.append('unique_pFCs.csv')
            except Exception as e:
                errors.append(f'unique_pFCs.csv: {e}')

        bulk_path = cwd / 'bulk_pFCs.csv'
        if bulk_path.exists():
            if self._splash:
                self._splash.set_status('Loading bulk pFCs…')
            try:
                results = read_bulk_pfcs(str(bulk_path))
                self._populate_from_bulk_results(results)
                loaded.append('bulk_pFCs.csv')
            except Exception as e:
                errors.append(f'bulk_pFCs.csv: {e}')

        fc_path = cwd / 'FORCE_CONSTANTS'
        if fc_path.exists():
            if self._splash:
                self._splash.set_status('Loading FORCE_CONSTANTS…')
            try:
                self._do_load_fc(fc_path)
                loaded.append('FORCE_CONSTANTS')
            except Exception as e:
                errors.append(f'FORCE_CONSTANTS: {e}')

        refpos_path = cwd / 'REFPOS'
        if refpos_path.exists():
            if self._splash:
                self._splash.set_status('Loading REFPOS…')
            self.site_picker.load_refpos(str(refpos_path))
            loaded.append('REFPOS')

        refsite_csv_path = cwd / 'refsite_pFCs.csv'
        if refsite_csv_path.exists():
            if self._splash:
                self._splash.set_status('Loading refsite pFCs CSV…')
            try:
                self.site_picker.load_refsite_csv(str(refsite_csv_path))
                loaded.append('refsite_pFCs.csv')
            except Exception as e:
                errors.append(f'refsite_pFCs.csv: {e}')

        if loaded:
            self.status.showMessage(
                f'Auto-loaded from {cwd}: {", ".join(loaded)}'
            )
        if errors:
            QMessageBox.warning(
                self, 'Auto-load error',
                'Some files could not be loaded automatically:\n\n'
                + '\n'.join(errors),
            )

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
            self._update_tab_visibility()   # stiffness_shift section may now be present
        except Exception as e:
            QMessageBox.critical(self, 'Error loading settings', str(e))

    def _try_autoload_from_settings(self):
        sp = Path(self.settings.sposcar)
        fc = Path(self.settings.force_constants)
        loaded = []
        errors = []
        if sp.exists():
            try:
                self._do_load_sposcar(sp)
                loaded.append(sp.name)
            except Exception as e:
                errors.append(f'{sp.name}: {e}')
        if fc.exists():
            try:
                self._do_load_fc(fc)
                loaded.append(fc.name)
            except Exception as e:
                errors.append(f'{fc.name}: {e}')
        if errors:
            QMessageBox.warning(
                self, 'Auto-load error',
                'Could not load the following files from settings:\n\n'
                + '\n'.join(errors),
            )
        elif loaded:
            self.status.showMessage(
                f'Auto-loaded {" and ".join(loaded)} from settings.'
            )

    # ------------------------------------------------------------------
    # File loading — internal helpers
    # ------------------------------------------------------------------

    def _try_load_lobster(self, sposcar_dir: Path):
        """Silently attempt to discover and load a sibling LOBSTER directory."""
        from betapy.core.lobster import find_lobster_dir, load_pairs as _lob_load
        ldir = find_lobster_dir(sposcar_dir)
        if ldir is None:
            return
        try:
            self._lobster_pairs = _lob_load(ldir)
            self._lobster_dir = ldir
            n = len(self._lobster_pairs)
            self.status.showMessage(
                f'LOBSTER: loaded {n} pair shells from {ldir.name}'
            )
        except Exception:
            self._lobster_pairs = None

    def _do_load_sposcar(self, path):
        """Load SPOSCAR and push supercell to all tools. Raises on error."""
        path = Path(path)
        self.supercell = Supercell(read_SPOSCAR(path))
        self.settings.sposcar = str(path)
        self.lbl_sposcar.setText(f'SPOSCAR: {path.name}  ✓')

        self._try_load_lobster(path.parent)

        from betapy.gui.pfc_viewer import PFCViewerWidget
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if isinstance(w, PFCViewerWidget):
                w.set_supercell(self.supercell)
                if self._lobster_pairs is not None:
                    w.set_lobster_pairs(self._lobster_pairs)
                if self._lobster_dir is not None:
                    w.set_lobster_dir(self._lobster_dir)
        self.site_picker.load_supercell(
            self.supercell,
            self.fc_data,
        )

        # Co-located REFPOS auto-loads and may reveal the refsite tab
        refpos_path = path.parent / 'REFPOS'
        if refpos_path.exists():
            self.site_picker.load_refpos(str(refpos_path))
        self._update_tab_visibility()

        self._check_ready()

    def _do_load_fc(self, path):
        """Load FORCE_CONSTANTS. Raises on error."""
        path = Path(path)
        self.fc_data = read_FORCE_CONSTANTS(path)
        self.settings.force_constants = str(path)
        n = len(self.fc_data['atomic_pairs'])
        self.lbl_fc.setText(f'FORCE_CONSTANTS: {path.name}  ({n} pairs) ✓')
        if self.supercell is not None:
            self.site_picker.load_supercell(self.supercell, self.fc_data)
        self._check_ready()

    # ------------------------------------------------------------------
    # Manual file loading (button handlers)
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

    def _populate_from_bulk_results(self, results):
        """
        Push bulk pFC results to all downstream tabs.

        Called both after a fresh analysis and when restoring from
        bulk_pFCs.csv on startup, so the two paths stay in sync.
        """
        from betapy.core.projection import unique_pfcs
        df_unique = unique_pfcs(results)
        self.pfc_viewer.load_data(df_unique, results, supercell=self.supercell)
        if self._lobster_pairs is not None:
            self.pfc_viewer.set_lobster_pairs(self._lobster_pairs)
        if self._lobster_dir is not None:
            self.pfc_viewer.set_lobster_dir(self._lobster_dir)
        self.lt_viewer.load_data(
            results,
            reliability_cutoff=self.pfc_viewer._reliability_cutoff,
        )
        self._bulk_results = results
        if self.multicenter is not None:
            self.multicenter.load_data(
                results, self.supercell, lobster_dir=self._lobster_dir
            )
        if self.badger is not None:
            self.badger.load_data(
                results,
                reliability_cutoff=self.pfc_viewer._reliability_cutoff,
                supercell=self.supercell,
            )
        self._update_tab_visibility()

    def _run_analysis(self):
        if self._worker is not None and self._worker.isRunning():
            return

        n_pairs = len(self.fc_data['atomic_pairs'])
        self._progress_bar.setRange(0, n_pairs)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self.btn_run.setEnabled(False)
        self.status.showMessage(f'Analysing {n_pairs} pairs…')

        self._worker = _AnalysisWorker(self.supercell, self.fc_data)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_progress(self, n, total):
        self._progress_bar.setValue(n)

    def _on_analysis_done(self, results, onsite):
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._progress_bar.hide()
        self.btn_run.setEnabled(True)

        self._populate_from_bulk_results(results)
        self.site_picker.load_supercell(self.supercell, self.fc_data)

        try:
            write_bulk_pfcs(results)
        except Exception:
            pass

        self.status.showMessage(
            f'Analysis complete — {len(results)} off-site pairs.'
        )

    def _on_analysis_error(self, msg):
        self._progress_bar.hide()
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, 'Analysis error', msg)
        self.status.showMessage('Analysis failed.')


def main(cli_args=None):
    app = QApplication(sys.argv)
    app.setApplicationName('betapy')
    app.setApplicationDisplayName('betapy')
    app.setStyle('Fusion')

    _icon_path = Path(__file__).parent.parent / 'data' / 'icon.png'
    if _icon_path.exists():
        app.setWindowIcon(QIcon(str(_icon_path)))

    from betapy.gui.splash import BetapySplashScreen
    splash = BetapySplashScreen()
    splash.show()
    app.processEvents()

    window = MainWindow(splash=splash, cli_args=cli_args)
    window.show()
    splash.finish(window)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
