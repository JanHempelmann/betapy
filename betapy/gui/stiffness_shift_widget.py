"""
Stiffness Shift tab — side-by-side ΔpFC comparison of two structures.

Layout
------
Top bar   : directory paths for A and B, REFPOS, analysis settings
3D views  : side-by-side StructureView with refsite cube + connection toggles
Bottom    : overlay scatter (A ○, B △, matched connected, unmatched grey)
            + tabbed tables: "Matched" and "Unmatched"

Scatter interaction
-------------------
• Clicking a matched A/B point  → gold ring on both points, bond highlighted
  in both 3D views, Matched tab focused.
• Clicking an unmatched A/B point → gold ring on that point, bond highlighted
  in the relevant 3D view only, Unmatched tab focused.
"""

import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QDoubleSpinBox, QLineEdit,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFrame, QScrollArea, QGridLayout,
    QTabWidget, QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS, read_refpos
from betapy.core.structure import Supercell
from betapy.core.projection import (
    find_refsite_pairs,
    match_fc_pairs_direct,
    stiffness_shift_from_pairs,
)
from betapy.gui.structure_view import StructureView
from betapy.core.constants import EV_ANG2_TO_N_M, UNIT_LABEL, UNIT_EV


PICK_TOLERANCE = 0.025


class _StiffnessWorker(QThread):
    """Background thread for stiffness-shift computation."""
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, dir_a, dir_b, refpos_path_a, refpos_path_b,
                 cutoff, msd, tol, excl):
        super().__init__()
        self._dir_a        = Path(dir_a)
        self._dir_b        = Path(dir_b)
        self._refpos_path_a = Path(refpos_path_a)
        self._refpos_path_b = Path(refpos_path_b)
        self._cutoff       = cutoff
        self._msd          = msd
        self._tol          = tol
        self._excl         = excl
        # Outputs — read by the main thread after finished emits
        self.sc_a         = None
        self.sc_b         = None
        self.refsites_a   = []
        self.refsites_b   = []
        self.offsite_a    = []
        self.offsite_b    = []
        self.matched      = []
        self.unmatched_a  = []
        self.unmatched_b  = []
        self.excl_sp      = None

    def run(self):
        try:
            sc_a = Supercell(read_SPOSCAR(self._dir_a / 'SPOSCAR'))
            sc_b = Supercell(read_SPOSCAR(self._dir_b / 'SPOSCAR'))
            fc_a = read_FORCE_CONSTANTS(self._dir_a / 'FORCE_CONSTANTS')
            fc_b = read_FORCE_CONSTANTS(self._dir_b / 'FORCE_CONSTANTS')

            refsites_a = read_refpos(self._refpos_path_a)['positions']
            if not refsites_a:
                raise ValueError(f'No positions found in {self._refpos_path_a}')
            refsites_b = read_refpos(self._refpos_path_b)['positions']
            if not refsites_b:
                raise ValueError(f'No positions found in {self._refpos_path_b}')
            # (all_refsites_a/b used separately for 3D view markers per structure)

            excl_sp = None
            if self._excl:
                found = set()
                for frac_pos in refsites_b:
                    fp    = np.asarray(frac_pos)
                    dists = [sc_b.distance_to_point(k + 1, fp)
                             for k in range(sc_b.n_atoms)]
                    ni = min(range(sc_b.n_atoms), key=lambda k: dists[k])
                    if dists[ni] < self._msd:
                        found.add(sc_b.species(ni + 1))
                excl_sp = found if found else None

            sp_set = set(sc_a.chem_symbols) & set(sc_b.chem_symbols)

            offsite_a   = []
            offsite_b   = []
            matched     = []
            unmatched_a = []
            unmatched_b = []

            for ref_a, ref_b in zip(refsites_a, refsites_b):
                res_a, _ = find_refsite_pairs(
                    sc_a, fc_a['atomic_pairs'], fc_a['force_matrices'],
                    ref_a, cutoff=self._cutoff, min_distance=0.0,
                    exclude_species=excl_sp, show_progress=False,
                )
                res_b, _ = find_refsite_pairs(
                    sc_b, fc_b['atomic_pairs'], fc_b['force_matrices'],
                    ref_b, cutoff=self._cutoff, min_distance=self._msd,
                    exclude_species=excl_sp, show_progress=False,
                )

                # Drop pairs involving species absent from the other structure
                # (e.g. Li-containing pairs when A is the deintercalated phase)
                sub_a = [r for r in res_a if r['species1'] in sp_set and r['species2'] in sp_set]
                sub_b = [r for r in res_b if r['species1'] in sp_set and r['species2'] in sp_set]

                offsite_a.extend(res_a)
                offsite_b.extend(res_b)

                m, ua, ub = match_fc_pairs_direct(
                    sub_a, sub_b, sc_a, sc_b, ref_a, ref_b, tol=self._tol
                )
                matched.extend(m)
                unmatched_a.extend(ua)
                unmatched_b.extend(ub)

            self.sc_a       = sc_a
            self.sc_b       = sc_b
            self.refsites_a = refsites_a
            self.refsites_b = refsites_b
            self.offsite_a    = offsite_a
            self.offsite_b    = offsite_b
            self.matched      = matched
            self.unmatched_a  = unmatched_a
            self.unmatched_b  = unmatched_b
            self.excl_sp      = excl_sp
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class StiffnessShiftWidget(QWidget):
    """Tab for side-by-side stiffness-shift comparison of two structures."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker       = None
        self._sc_a         = None
        self._sc_b         = None
        self._refsites_a   = []
        self._refsites_b   = []
        self._offsite_a    = []
        self._offsite_b    = []
        self._matched      = []
        self._unmatched_a  = []
        self._unmatched_b  = []

        # Selection state — at most one of these is non-None at a time.
        # Unmatched selections are (ridx, record) where ridx is the index
        # into _unmatched_a / _unmatched_b.
        self._selected_midx = None
        self._selected_ua   = None   # (ridx, record) or None
        self._selected_ub   = None   # (ridx, record) or None

        # Scatter click-detection lists (rebuilt every _refresh_plot)
        self._scatter_pts = {}   # (sp1,sp2) -> [{xa,ya,xb,yb,midx}, ...]
        self._um_a_pts    = []   # [{x,y,ridx}, ...] visible unmatched A
        self._um_b_pts    = []   # [{x,y,ridx}, ...] visible unmatched B

        self._pair_types  = []
        self._checkboxes  = {}
        self._ax          = None
        self._unit        = UNIT_EV
        self._total_raw   = None
        self._excl_note   = ''
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def set_unit(self, unit: str):
        """Switch display unit ('eV/Ang2' or 'N/m') and redraw."""
        if unit != self._unit:
            self._unit = unit
            self._refresh_plot()
            self._refresh_table()
            self._refresh_unmatched_table()
            self._update_result_label()

    def _update_result_label(self):
        if self._total_raw is None:
            return
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        unit_lbl = UNIT_LABEL[self._unit]
        self._result_lbl.setText(
            f'Matched: {len(self._matched)}   '
            f'Unmatched A: {len(self._unmatched_a)}   '
            f'Unmatched B: {len(self._unmatched_b)}\n'
            f'{self._excl_note}'
            f'Σ ΔpFC (B-A): {self._total_raw * factor:+.5f} {unit_lbl}'
        )
        self._btn_copy_shift.setEnabled(True)

    def _copy_shift_result(self):
        if self._total_raw is None:
            return
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        QApplication.clipboard().setText(f'{self._total_raw * factor:.5f}')

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self._build_load_bar())

        self._v_splitter = QSplitter(Qt.Vertical)
        outer.addWidget(self._v_splitter)
        self._v_splitter.addWidget(self._build_views_panel())
        self._v_splitter.addWidget(self._build_bottom_panel())

        screen_h = QApplication.primaryScreen().availableGeometry().height()
        self._v_splitter.setSizes([int(screen_h * 0.35), int(screen_h * 0.30)])

        s = QSettings('betapy', 'StiffnessShift')
        ratios = s.value('v_splitter')
        if ratios:
            self._v_splitter.setSizes([int(float(r) * 1000) for r in ratios])
        self._v_splitter.splitterMoved.connect(self._save_splitter_state)

    def _build_load_bar(self):
        bar = QGroupBox('Structures and settings')
        outer_layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel('Structure A (deintercalated):'), 0, 0)
        self._edit_a = QLineEdit()
        self._edit_a.setPlaceholderText('directory containing SPOSCAR, FORCE_CONSTANTS')
        grid.addWidget(self._edit_a, 0, 1)
        btn_a = QPushButton('Browse…')
        btn_a.setFixedWidth(80)
        btn_a.clicked.connect(lambda: self._browse_dir(self._edit_a, self._edit_refpos_a))
        grid.addWidget(btn_a, 0, 2)

        grid.addWidget(QLabel('REFPOS A:'), 1, 0)
        self._edit_refpos_a = QLineEdit()
        self._edit_refpos_a.setPlaceholderText('auto-detected from Dir A, or browse')
        grid.addWidget(self._edit_refpos_a, 1, 1)
        btn_rpa = QPushButton('Browse…')
        btn_rpa.setFixedWidth(80)
        btn_rpa.clicked.connect(lambda: self._browse_file(
            self._edit_refpos_a, 'REFPOS files (REFPOS);;All files (*)'
        ))
        grid.addWidget(btn_rpa, 1, 2)

        grid.addWidget(QLabel('Structure B (intercalated):'), 2, 0)
        self._edit_b = QLineEdit()
        self._edit_b.setPlaceholderText('directory containing SPOSCAR, FORCE_CONSTANTS')
        grid.addWidget(self._edit_b, 2, 1)
        btn_b = QPushButton('Browse…')
        btn_b.setFixedWidth(80)
        btn_b.clicked.connect(lambda: self._browse_dir(self._edit_b, self._edit_refpos_b))
        grid.addWidget(btn_b, 2, 2)

        grid.addWidget(QLabel('REFPOS B:'), 3, 0)
        self._edit_refpos_b = QLineEdit()
        self._edit_refpos_b.setPlaceholderText('auto-detected from Dir B, or browse')
        grid.addWidget(self._edit_refpos_b, 3, 1)
        btn_rpb = QPushButton('Browse…')
        btn_rpb.setFixedWidth(80)
        btn_rpb.clicked.connect(lambda: self._browse_file(
            self._edit_refpos_b, 'REFPOS files (REFPOS);;All files (*)'
        ))
        grid.addWidget(btn_rpb, 3, 2)
        outer_layout.addLayout(grid)

        srow = QHBoxLayout()

        srow.addWidget(QLabel('Cutoff (Å):'))
        self._spin_cutoff = QDoubleSpinBox()
        self._spin_cutoff.setRange(0.5, 30.0)
        self._spin_cutoff.setSingleStep(0.5)
        self._spin_cutoff.setDecimals(2)
        self._spin_cutoff.setValue(6.0)
        self._spin_cutoff.setFixedWidth(72)
        srow.addWidget(self._spin_cutoff)

        srow.addWidget(QLabel('Min site dist (Å):'))
        self._spin_msd = QDoubleSpinBox()
        self._spin_msd.setRange(0.0, 5.0)
        self._spin_msd.setSingleStep(0.05)
        self._spin_msd.setDecimals(2)
        self._spin_msd.setValue(0.1)
        self._spin_msd.setFixedWidth(72)
        srow.addWidget(self._spin_msd)

        _tol_lbl = QLabel('Match tol (Å):')
        _tol_lbl.setToolTip(
            'Maximum RMS of (Δatom1_ref_dist, Δbond_length) in Å for two pairs\n'
            'to be considered equivalent across structures.\n'
            'Uses Cartesian distances only — invariant to cell origin, rotation,\n'
            'and inversion — so it works even when A and B were set up with\n'
            'different crystallographic origins or have significant lattice changes.\n'
            'Correct intercalation pairs differ by <0.2 Å; wrong pairs by >0.5 Å.\n'
            'Default 0.3 Å gives a safe margin for typical intercalation.'
        )
        srow.addWidget(_tol_lbl)
        self._spin_tol = QDoubleSpinBox()
        self._spin_tol.setRange(0.01, 2.0)
        self._spin_tol.setSingleStep(0.05)
        self._spin_tol.setDecimals(2)
        self._spin_tol.setValue(0.3)
        self._spin_tol.setToolTip(_tol_lbl.toolTip())
        self._spin_tol.setFixedWidth(72)
        srow.addWidget(self._spin_tol)

        self._chk_excl = QCheckBox('Exclude refsite species')
        self._chk_excl.setChecked(True)
        self._chk_excl.setToolTip(
            'Exclude off-site pairs where either atom is the species\n'
            'occupying the reference site in structure B.'
        )
        srow.addWidget(self._chk_excl)

        self._btn_run = QPushButton('Run analysis')
        self._btn_run.setFixedWidth(110)
        self._btn_run.clicked.connect(self._run_analysis)
        srow.addWidget(self._btn_run)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(120)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()
        srow.addWidget(self._progress_bar)

        srow.addStretch(1)

        self._result_lbl = QLabel('')
        self._result_lbl.setWordWrap(True)
        srow.addWidget(self._result_lbl)

        self._btn_copy_shift = QPushButton('Copy Σ ΔpFC')
        self._btn_copy_shift.setFixedWidth(100)
        self._btn_copy_shift.setEnabled(False)
        self._btn_copy_shift.clicked.connect(self._copy_shift_result)
        srow.addWidget(self._btn_copy_shift)

        outer_layout.addLayout(srow)
        bar.setLayout(outer_layout)
        return bar

    def _build_views_panel(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_single_view('a'))
        layout.addWidget(self._build_single_view('b'))
        self._view_a.colours_changed.connect(self._refresh_plot)
        self._view_b.colours_changed.connect(self._refresh_plot)
        return widget

    def _build_single_view(self, which):
        is_a = (which == 'a')
        w = QWidget()
        col = QVBoxLayout(w)

        lbl = QLabel('A — deintercalated' if is_a else 'B — intercalated')
        lbl.setStyleSheet('font-weight: bold; padding: 3px;')
        col.addWidget(lbl)

        view = StructureView(self)
        col.addWidget(view, stretch=1)

        conn_row = QHBoxLayout()
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 30.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(2)
        spin.setValue(6.0)
        spin.setFixedWidth(72)
        btn = QPushButton('Show connections')
        btn.setCheckable(True)
        btn.clicked.connect(
            lambda chk, v=view, s=spin, b=btn:
                self._toggle_connections(chk, v, s, b)
        )
        conn_row.addWidget(QLabel('Conn. cutoff (Å):'))
        conn_row.addWidget(spin)
        conn_row.addWidget(btn)
        conn_row.addStretch()
        col.addLayout(conn_row)

        if is_a:
            self._view_a      = view
            self._conn_spin_a = spin
            self._btn_conn_a  = btn
        else:
            self._view_b      = view
            self._conn_spin_b = spin
            self._btn_conn_b  = btn

        return w

    def _build_bottom_panel(self):
        h = QSplitter(Qt.Horizontal)

        # Left: scatter + filter checkboxes
        scatter_w = QWidget()
        sl = QVBoxLayout(scatter_w)
        sl.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._canvas.mpl_connect('button_press_event', self._on_scatter_click)
        sl.addWidget(self._toolbar)
        sl.addWidget(self._canvas, stretch=1)

        filter_group = QGroupBox('Pair types')
        self._filter_layout = QVBoxLayout()
        filter_group.setLayout(self._filter_layout)
        scroll = QScrollArea()
        scroll.setWidget(filter_group)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(90)
        sl.addWidget(scroll)

        h.addWidget(scatter_w)

        # Right: tabbed tables + shared selection bar
        right_w = QWidget()
        rl = QVBoxLayout(right_w)
        rl.setContentsMargins(0, 0, 0, 0)

        self._bottom_tabs = QTabWidget()

        # Tab 0 — Matched pairs
        matched_w = QWidget()
        ml = QVBoxLayout(matched_w)
        ml.setContentsMargins(4, 4, 4, 4)
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            'Species', 'Ref dist A (Å)', 'Ref dist B (Å)',
            'pFC A', 'pFC B', 'ΔpFC',
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_select)
        ml.addWidget(self._table)
        self._bottom_tabs.addTab(matched_w, 'Matched (0)')

        # Tab 1 — Unmatched pairs
        um_w = QWidget()
        ul = QVBoxLayout(um_w)
        ul.setContentsMargins(4, 4, 4, 4)
        self._table_um = QTableWidget()
        self._table_um.setColumnCount(6)
        self._table_um.setHorizontalHeaderLabels([
            'Struct.', 'Species', 'Atom 1', 'Atom 2', 'Ref dist (Å)', 'pFC',
        ])
        self._table_um.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table_um.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table_um.setSortingEnabled(True)
        self._table_um.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table_um.verticalHeader().setVisible(False)
        self._table_um.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table_um.horizontalHeader().setStretchLastSection(True)
        self._table_um.itemSelectionChanged.connect(self._on_unmatched_table_select)
        ul.addWidget(self._table_um)
        self._bottom_tabs.addTab(um_w, 'Unmatched (0)')

        rl.addWidget(self._bottom_tabs)

        self._sel_bar = QLabel('')
        self._sel_bar.setFrameStyle(QFrame.StyledPanel)
        self._sel_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._sel_bar.setFixedHeight(24)
        rl.addWidget(self._sel_bar)

        h.addWidget(right_w)
        screen_w = QApplication.primaryScreen().availableGeometry().width()
        h.setSizes([int(screen_w * 0.38), int(screen_w * 0.25)])

        s = QSettings('betapy', 'StiffnessShift')
        ratios = s.value('h_splitter')
        if ratios:
            h.setSizes([int(float(r) * 1000) for r in ratios])
        h.splitterMoved.connect(self._save_splitter_state)
        self._h_splitter = h
        return h

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _save_splitter_state(self):
        s = QSettings('betapy', 'StiffnessShift')
        for key, splitter in [('v_splitter', self._v_splitter),
                               ('h_splitter', self._h_splitter)]:
            sizes = splitter.sizes()
            total = sum(sizes)
            if total > 0:
                s.setValue(key, [sz / total for sz in sizes])

    def _browse_dir(self, edit, refpos_edit=None):
        path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if path:
            edit.setText(path)
            if refpos_edit is not None and not refpos_edit.text().strip():
                candidate = Path(path) / 'REFPOS'
                if candidate.exists():
                    refpos_edit.setText(str(candidate))

    def _browse_file(self, edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, 'Select file', '', filter_str)
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------
    # Connection toggle
    # ------------------------------------------------------------------

    def _toggle_connections(self, checked, view, spin, btn):
        if checked:
            btn.setText('Hide connections')
            view.set_refsite_bonds(spin.value())
        else:
            btn.setText('Show connections')
            view.set_refsite_bonds(None)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _run_analysis(self):
        if self._worker is not None and self._worker.isRunning():
            return

        dir_a_str    = self._edit_a.text().strip()
        dir_b_str    = self._edit_b.text().strip()
        refpos_a_str = self._edit_refpos_a.text().strip()
        refpos_b_str = self._edit_refpos_b.text().strip()

        if not dir_a_str or not dir_b_str:
            QMessageBox.warning(self, 'Missing input',
                                'Please specify both structure directories.')
            return

        dir_a = Path(dir_a_str)
        dir_b = Path(dir_b_str)

        refpos_path_a = Path(refpos_a_str) if refpos_a_str else dir_a / 'REFPOS'
        if not refpos_path_a.exists():
            QMessageBox.warning(self, 'Missing REFPOS',
                                f'REFPOS for structure A not found:\n{refpos_path_a}')
            return

        refpos_path_b = Path(refpos_b_str) if refpos_b_str else dir_b / 'REFPOS'
        if not refpos_path_b.exists():
            refpos_path_b = refpos_path_a  # fall back to A's REFPOS

        self._btn_run.setEnabled(False)
        self._progress_bar.setRange(0, 0)  # indeterminate / busy indicator
        self._progress_bar.show()

        self._worker = _StiffnessWorker(
            dir_a          = dir_a,
            dir_b          = dir_b,
            refpos_path_a  = refpos_path_a,
            refpos_path_b  = refpos_path_b,
            cutoff         = self._spin_cutoff.value(),
            msd            = self._spin_msd.value(),
            tol            = self._spin_tol.value(),
            excl           = self._chk_excl.isChecked(),
        )
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_done(self):
        self._progress_bar.hide()
        self._btn_run.setEnabled(True)

        w = self._worker
        sc_a        = w.sc_a
        sc_b        = w.sc_b
        refsites_a  = w.refsites_a
        refsites_b  = w.refsites_b
        matched      = w.matched
        unmatched_a  = w.unmatched_a
        unmatched_b  = w.unmatched_b
        excl_sp      = w.excl_sp

        self._sc_a       = sc_a
        self._sc_b       = sc_b
        self._refsites_a = refsites_a
        self._refsites_b = refsites_b
        self._offsite_a  = w.offsite_a
        self._offsite_b  = w.offsite_b
        self._matched      = matched
        self._unmatched_a  = unmatched_a
        self._unmatched_b  = unmatched_b
        self._selected_midx = None
        self._selected_ua   = None
        self._selected_ub   = None

        self._view_a.load_supercell(sc_a)
        self._view_a.set_ref_sites(refsites_a)
        self._view_b.load_supercell(sc_b)
        self._view_b.set_ref_sites(refsites_b)

        for btn, view in [(self._btn_conn_a, self._view_a),
                          (self._btn_conn_b, self._view_b)]:
            btn.setChecked(False)
            btn.setText('Show connections')
            view.set_refsite_bonds(None)

        self._bottom_tabs.setTabText(0, f'Matched ({len(matched)})')
        self._bottom_tabs.setTabText(
            1, f'Unmatched ({len(unmatched_a) + len(unmatched_b)})'
        )

        self._rebuild_checkboxes()
        self._refresh_plot()
        self._refresh_table()
        self._refresh_unmatched_table()

        _, total = stiffness_shift_from_pairs(matched)
        self._total_raw = total
        self._excl_note = f'  excl. {next(iter(excl_sp))} pairs\n' if excl_sp else ''
        self._update_result_label()

    def _on_analysis_error(self, msg):
        self._progress_bar.hide()
        self._btn_run.setEnabled(True)
        QMessageBox.critical(self, 'Analysis error', msg)

    # ------------------------------------------------------------------
    # Checkboxes
    # ------------------------------------------------------------------

    def _rebuild_checkboxes(self):
        while self._filter_layout.count():
            item = self._filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes = {}

        types = set()
        for r in self._matched:
            types.add((r['species1'], r['species2']))
        for r in self._unmatched_a + self._unmatched_b:
            types.add((r['species1'], r['species2']))
        self._pair_types = sorted(types)

        for pt in self._pair_types:
            c1, c2 = self._view_a.pair_colours_hex(pt[0], pt[1])
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            cb = QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_plot)
            lbl = QLabel(
                f'<b><span style="color:{c1}">{pt[0]}</span></b>'
                f'<span style="color:#888888"> – </span>'
                f'<b><span style="color:{c2}">{pt[1]}</span></b>'
            )
            row_l.addWidget(cb)
            row_l.addWidget(lbl)
            row_l.addStretch()
            self._filter_layout.addWidget(row_w)
            self._checkboxes[pt] = cb

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------

    def _draw_empty_plot(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No data — run stiffness-shift analysis',
                transform=ax.transAxes, ha='center', va='center',
                color='grey', fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _refresh_plot(self):
        if not self._matched and not self._unmatched_a and not self._unmatched_b:
            self._draw_empty_plot()
            return

        active = {pt for pt, cb in self._checkboxes.items() if cb.isChecked()}

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._ax = ax
        self._scatter_pts = {}
        self._um_a_pts    = []
        self._um_b_pts    = []
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0

        seen_species = set()

        for pt in self._pair_types:
            if pt not in active:
                continue
            c1, _ = self._view_a.pair_colours_hex(pt[0], pt[1])

            sub = [(i, m) for i, m in enumerate(self._matched)
                   if (m['species1'], m['species2']) == pt]
            if sub:
                midxs = [i for i, _ in sub]
                ms    = [m for _, m in sub]
                xa = np.array([m['atom1_ref_dist_a']         for m in ms])
                ya = np.array([m['mean_pfc_a'] * factor      for m in ms])
                xb = np.array([m['atom1_ref_dist_b']         for m in ms])
                yb = np.array([m['mean_pfc_b'] * factor      for m in ms])

                for k in range(len(ms)):
                    ax.plot([xa[k], xb[k]], [ya[k], yb[k]],
                            color='#999999', linewidth=0.6, alpha=0.45, zorder=1)

                ax.scatter(xa, ya, s=32, color=c1, marker='o',
                           alpha=0.85, edgecolors='none', zorder=3)
                ax.scatter(xb, yb, s=38, color=c1, marker='^',
                           alpha=0.65, edgecolors='none', zorder=3)

                self._scatter_pts[pt] = [
                    {'xa': xa[k], 'ya': ya[k],
                     'xb': xb[k], 'yb': yb[k],
                     'midx': midxs[k]}
                    for k in range(len(ms))
                ]
                seen_species.add(pt)

        # Unmatched A — store per-record ridx for click detection
        um_a_indexed = [(i, r) for i, r in enumerate(self._unmatched_a)
                        if (r['species1'], r['species2']) in active]
        if um_a_indexed:
            ax.scatter(
                [r.get('atom1_ref_dist', 0.0) for _, r in um_a_indexed],
                [r['mean_pfc'] * factor for _, r in um_a_indexed],
                s=28, color='#aaaaaa', marker='x', linewidths=1.2, zorder=2,
            )
            self._um_a_pts = [
                {'x': r.get('atom1_ref_dist', 0.0), 'y': r['mean_pfc'] * factor, 'ridx': i}
                for i, r in um_a_indexed
            ]

        # Unmatched B
        um_b_indexed = [(i, r) for i, r in enumerate(self._unmatched_b)
                        if (r['species1'], r['species2']) in active]
        if um_b_indexed:
            ax.scatter(
                [r.get('atom1_ref_dist', 0.0) for _, r in um_b_indexed],
                [r['mean_pfc'] * factor for _, r in um_b_indexed],
                s=22, color='#aaaaaa', marker='s', edgecolors='none', zorder=2,
            )
            self._um_b_pts = [
                {'x': r.get('atom1_ref_dist', 0.0), 'y': r['mean_pfc'] * factor, 'ridx': i}
                for i, r in um_b_indexed
            ]

        # Gold rings on selected points
        if self._selected_midx is not None and self._selected_midx < len(self._matched):
            m  = self._matched[self._selected_midx]
            pt = (m['species1'], m['species2'])
            if pt in active:
                for px, py in [(m['atom1_ref_dist_a'], m['mean_pfc_a'] * factor),
                               (m['atom1_ref_dist_b'], m['mean_pfc_b'] * factor)]:
                    ax.scatter([px], [py], s=140, facecolors='none',
                               edgecolors='#c8a000', linewidths=2.5, zorder=5)

        for sel, lst in [(self._selected_ua, self._unmatched_a),
                         (self._selected_ub, self._unmatched_b)]:
            if sel is not None:
                ridx, r = sel
                if (r['species1'], r['species2']) in active:
                    ax.scatter(
                        [r.get('atom1_ref_dist', 0.0)], [r['mean_pfc'] * factor],
                        s=140, facecolors='none', edgecolors='#c8a000',
                        linewidths=2.5, zorder=5,
                    )

        # Legend
        handles = []
        for pt in sorted(seen_species):
            c1, _ = self._view_a.pair_colours_hex(pt[0], pt[1])
            handles.append(Line2D([0], [0], linestyle='', marker='o',
                                  color=c1, markersize=7,
                                  label=f'{pt[0]}–{pt[1]}'))
        handles.append(Line2D([0], [0], linestyle='', marker='o',
                              color='#555555', markersize=7, label='A  (○)'))
        handles.append(Line2D([0], [0], linestyle='', marker='^',
                              color='#555555', markersize=7, label='B  (△)'))
        if um_a_indexed:
            handles.append(Line2D([0], [0], linestyle='', marker='x',
                                  color='#aaaaaa', markersize=7,
                                  markeredgewidth=1.4, label='Unmatched A'))
        if um_b_indexed:
            handles.append(Line2D([0], [0], linestyle='', marker='s',
                                  color='#aaaaaa', markersize=6,
                                  label='Unmatched B'))
        if handles:
            ax.legend(handles=handles, loc='upper right',
                      framealpha=0.9, fontsize=8)

        ax.set_xlabel('Atom 1 – refsite distance (Å)', fontsize=11)
        ax.set_ylabel(f'Projected force constant ({UNIT_LABEL[self._unit]})', fontsize=11)
        ax.set_title('Stiffness shift: A (○) vs B (△)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Matched-pair table
    # ------------------------------------------------------------------

    def _refresh_table(self):
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        unit_lbl = UNIT_LABEL[self._unit]
        self._table.setHorizontalHeaderItem(3, QTableWidgetItem(f'pFC A ({unit_lbl})'))
        self._table.setHorizontalHeaderItem(4, QTableWidgetItem(f'pFC B ({unit_lbl})'))
        self._table.setHorizontalHeaderItem(5, QTableWidgetItem(f'ΔpFC ({unit_lbl})'))

        for midx, m in sorted(enumerate(self._matched),
                               key=lambda p: abs(p[1]['delta_pfc']),
                               reverse=True):
            row = self._table.rowCount()
            self._table.insertRow(row)

            sp_item = QTableWidgetItem(f"{m['species1']}–{m['species2']}")
            sp_item.setFlags(sp_item.flags() & ~Qt.ItemIsEditable)
            sp_item.setData(Qt.UserRole, midx)
            self._table.setItem(row, 0, sp_item)

            for col, val in enumerate([
                m.get('atom1_ref_dist_a', 0.0),
                m.get('atom1_ref_dist_b', 0.0),
                m['mean_pfc_a'] * factor,
                m['mean_pfc_b'] * factor,
                m['delta_pfc']  * factor,
            ], start=1):
                fmt = f'{val:+.4f}' if col >= 3 else f'{val:.4f}'
                it  = _NumericItem(fmt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if col == 5:
                    it.setForeground(
                        QColor(30, 60, 160) if val >= 0 else QColor(180, 40, 30)
                    )
                self._table.setItem(row, col, it)

        self._table.setSortingEnabled(True)
        self._table.blockSignals(False)
        if self._selected_midx is not None:
            self._sync_table_selection()

    def _sync_table_selection(self):
        if self._selected_midx is None:
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.data(Qt.UserRole) == self._selected_midx:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                self._table.blockSignals(False)
                return

    # ------------------------------------------------------------------
    # Unmatched-pair table
    # ------------------------------------------------------------------

    def _refresh_unmatched_table(self):
        self._table_um.blockSignals(True)
        self._table_um.setSortingEnabled(False)
        self._table_um.setRowCount(0)
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        self._table_um.setHorizontalHeaderItem(
            5, QTableWidgetItem(f'pFC ({UNIT_LABEL[self._unit]})')
        )

        rows = (
            [('A', i, r) for i, r in enumerate(self._unmatched_a)] +
            [('B', i, r) for i, r in enumerate(self._unmatched_b)]
        )
        rows.sort(key=lambda t: abs(t[2]['mean_pfc']), reverse=True)

        for struct, ridx, r in rows:
            row = self._table_um.rowCount()
            self._table_um.insertRow(row)

            # Column 0: Struct. — UserRole stores (which, ridx) for lookup
            st_item = QTableWidgetItem(struct)
            st_item.setFlags(st_item.flags() & ~Qt.ItemIsEditable)
            st_item.setData(Qt.UserRole, (struct.lower(), ridx))
            self._table_um.setItem(row, 0, st_item)

            sp_item = QTableWidgetItem(f"{r['species1']}–{r['species2']}")
            sp_item.setFlags(sp_item.flags() & ~Qt.ItemIsEditable)
            self._table_um.setItem(row, 1, sp_item)

            a1_item = _NumericItem(str(int(r['atom1_idx'])))
            a1_item.setFlags(a1_item.flags() & ~Qt.ItemIsEditable)
            self._table_um.setItem(row, 2, a1_item)

            a2_item = _NumericItem(str(int(r['atom2_idx'])))
            a2_item.setFlags(a2_item.flags() & ~Qt.ItemIsEditable)
            self._table_um.setItem(row, 3, a2_item)

            rd_item = _NumericItem(f"{r.get('atom1_ref_dist', 0.0):.4f}")
            rd_item.setFlags(rd_item.flags() & ~Qt.ItemIsEditable)
            self._table_um.setItem(row, 4, rd_item)

            pfc_val  = r['mean_pfc'] * factor
            pfc_item = _NumericItem(f'{pfc_val:+.4f}')
            pfc_item.setFlags(pfc_item.flags() & ~Qt.ItemIsEditable)
            pfc_item.setForeground(
                QColor(30, 60, 160) if pfc_val >= 0 else QColor(180, 40, 30)
            )
            self._table_um.setItem(row, 5, pfc_item)

        self._table_um.setSortingEnabled(True)
        self._table_um.blockSignals(False)

        if self._selected_ua is not None or self._selected_ub is not None:
            self._sync_unmatched_table_selection()

    def _sync_unmatched_table_selection(self):
        if self._selected_ua is not None:
            target = ('a', self._selected_ua[0])
        elif self._selected_ub is not None:
            target = ('b', self._selected_ub[0])
        else:
            return
        for row in range(self._table_um.rowCount()):
            item = self._table_um.item(row, 0)
            if item and item.data(Qt.UserRole) == target:
                self._table_um.blockSignals(True)
                self._table_um.selectRow(row)
                self._table_um.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                self._table_um.blockSignals(False)
                return

    # ------------------------------------------------------------------
    # Scatter interaction
    # ------------------------------------------------------------------

    def _on_scatter_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return

        ax      = self._ax
        x_range = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1
        y_range = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1

        best_d    = float('inf')
        best_type = None
        best_id   = None

        for pts_list in self._scatter_pts.values():
            for pt_info in pts_list:
                for px, py in [(pt_info['xa'], pt_info['ya']),
                               (pt_info['xb'], pt_info['yb'])]:
                    dx = (px - cx) / x_range
                    dy = (py - cy) / y_range
                    d  = (dx**2 + dy**2) ** 0.5
                    if d < best_d:
                        best_d, best_type, best_id = d, 'matched', pt_info['midx']

        for pt_info in self._um_a_pts:
            dx = (pt_info['x'] - cx) / x_range
            dy = (pt_info['y'] - cy) / y_range
            d  = (dx**2 + dy**2) ** 0.5
            if d < best_d:
                best_d, best_type, best_id = d, 'unmatched_a', pt_info['ridx']

        for pt_info in self._um_b_pts:
            dx = (pt_info['x'] - cx) / x_range
            dy = (pt_info['y'] - cy) / y_range
            d  = (dx**2 + dy**2) ** 0.5
            if d < best_d:
                best_d, best_type, best_id = d, 'unmatched_b', pt_info['ridx']

        if best_type is None or best_d > PICK_TOLERANCE:
            return

        if best_type == 'matched':
            self._select_pair(best_id, source='scatter')
        elif best_type == 'unmatched_a':
            self._select_unmatched(best_id, 'a', source='scatter')
        else:
            self._select_unmatched(best_id, 'b', source='scatter')

    # ------------------------------------------------------------------
    # Table interaction
    # ------------------------------------------------------------------

    def _on_table_select(self):
        rows = self._table.selectedItems()
        if not rows:
            return
        item = self._table.item(rows[0].row(), 0)
        if item is None:
            return
        midx = item.data(Qt.UserRole)
        if midx is None:
            return
        self._select_pair(midx, source='table')

    def _on_unmatched_table_select(self):
        rows = self._table_um.selectedItems()
        if not rows:
            return
        item = self._table_um.item(rows[0].row(), 0)
        if item is None:
            return
        key = item.data(Qt.UserRole)
        if key is None:
            return
        which, ridx = key
        self._select_unmatched(ridx, which, source='table')

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_pair(self, midx: int, source: str = 'click'):
        if midx >= len(self._matched):
            return
        m = self._matched[midx]

        self._selected_midx = midx
        self._selected_ua   = None
        self._selected_ub   = None

        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        unit_lbl = UNIT_LABEL[self._unit]
        self._sel_bar.setText(
            f'  {m["species1"]}–{m["species2"]}'
            f'   A: {m["atom1_idx_a"]}→{m["atom2_idx_a"]}'
            f'  pFC = {m["mean_pfc_a"] * factor:+.5f}'
            f'   B: {m["atom1_idx_b"]}→{m["atom2_idx_b"]}'
            f'  pFC = {m["mean_pfc_b"] * factor:+.5f}'
            f'   ΔpFC = {m["delta_pfc"] * factor:+.5f} {unit_lbl}'
        )

        if source == 'scatter':
            self._bottom_tabs.setCurrentIndex(0)
        self._refresh_plot()
        if source != 'table':
            self._sync_table_selection()

        self._view_a.highlight_bond(m['atom1_idx_a'], m['atom2_idx_a'])
        self._view_b.highlight_bond(m['atom1_idx_b'], m['atom2_idx_b'])

    def _select_unmatched(self, ridx: int, which: str, source: str = 'click'):
        self._selected_midx = None

        if which == 'a':
            if ridx >= len(self._unmatched_a):
                return
            record = self._unmatched_a[ridx]
            self._selected_ua = (ridx, record)
            self._selected_ub = None
        else:
            if ridx >= len(self._unmatched_b):
                return
            record = self._unmatched_b[ridx]
            self._selected_ua = None
            self._selected_ub = (ridx, record)

        struct_lbl = 'A' if which == 'a' else 'B'
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        self._sel_bar.setText(
            f'  {record["species1"]}–{record["species2"]}'
            f'   [Unmatched {struct_lbl}]'
            f'   atom {int(record["atom1_idx"])}→{int(record["atom2_idx"])}'
            f'   ref dist = {record.get("atom1_ref_dist", 0.0):.3f} Å'
            f'   pFC = {record["mean_pfc"] * factor:+.5f} {UNIT_LABEL[self._unit]}'
        )

        if source == 'scatter':
            self._bottom_tabs.setCurrentIndex(1)
        self._refresh_plot()
        if source != 'table':
            self._sync_unmatched_table_selection()

        if which == 'a':
            self._view_a.highlight_bond(
                int(record['atom1_idx']), int(record['atom2_idx'])
            )
            self._view_b.clear_highlight()
        else:
            self._view_a.clear_highlight()
            self._view_b.highlight_bond(
                int(record['atom1_idx']), int(record['atom2_idx'])
            )
