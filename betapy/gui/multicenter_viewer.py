"""
Multicenter bonding detection viewer.

Three-panel layout:
  Left   : detection controls, species-pair checkboxes (grey-out style),
            unique cobiBetween directives.
  Middle : pFC vs r scatter in natural coordinates, Badger curve overlay
           per species pair (dashed), ±σ band shading, flagged points red.
  Right  : 3D structure view — clicking any point highlights the bond;
           clicking a flagged point also highlights the full chain.

Typical use
-----------
    widget = MulticenterWidget()
    widget.load_data(bulk_results, supercell, lobster_dir=ldir)
"""

import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QGroupBox, QListWidget, QProgressBar, QApplication,
    QMessageBox, QCheckBox, QScrollArea, QFrame, QDialog,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QStackedWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from betapy.gui.structure_view import StructureView


_GREY_COLOR  = '#aaaaaa'
_GREY_ALPHA  = 0.22
_NORM_ALPHA  = 0.75
_CHAIN_COLOR = '#e07820'   # amber — chain segment bonds not individually flagged
_CHAIN_ALPHA = 0.85
_FLAG_COLOR  = '#d03030'   # red   — statistically anomalous trigger pairs
_FLAG_ALPHA  = 0.90
_CURVE_ALPHA = 0.65
_BAND_ALPHA  = 0.10
_PICK_TOL    = 0.025     # fraction of axis range for click detection

_NC_COLOUR_BOND = '#4d94ff'
_NC_COLOUR_ANTI = '#ff6666'


# ---------------------------------------------------------------------------
# NcICOBI popup
# ---------------------------------------------------------------------------

class _NcCobiViewerWidget(QDialog):
    """
    Non-modal popup showing NcICOBI(N) value and the NcCOBI energy curve
    for a selected cobiBetween directive.

    Call show_result() when the user clicks a directive whose entry is found
    in NcICOBILIST.lobster.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('LOBSTER — Multicenter COBI')
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._title_label = QLabel('')
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setWordWrap(True)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._title_label.setFont(mono)
        layout.addWidget(self._title_label)

        self._icobi_label = QLabel('')
        self._icobi_label.setAlignment(Qt.AlignCenter)
        f = self._icobi_label.font()
        f.setPointSize(f.pointSize() + 1)
        self._icobi_label.setFont(f)
        layout.addWidget(self._icobi_label)

        self._figure = Figure(figsize=(4, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas, stretch=1)

        btn_row = QHBoxLayout()
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.hide)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.resize(420, 540)

    def show_result(self, directive, icobi, curves):
        """Update display and bring the window forward."""
        self._draw(directive, icobi, curves)
        if not self.isVisible():
            self._position_beside_parent()
        self.show()
        self.raise_()
        self.activateWindow()

    def _position_beside_parent(self):
        top = self.parent().window() if self.parent() else None
        if top is None:
            return
        pg = top.frameGeometry()
        x, y = pg.right() + 10, pg.top()
        screen = QApplication.screenAt(pg.center())
        if screen:
            sr = screen.availableGeometry()
            if x + self.width() > sr.right():
                x = max(sr.left(), pg.left() - self.width() - 10)
            y = max(sr.top(), min(y, sr.bottom() - self.height()))
        self.move(x, y)

    def _draw(self, directive, icobi, curves):
        body = directive[len('cobiBetween'):].strip() \
               if directive.startswith('cobiBetween') else directive
        self._title_label.setText(body)

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not curves:
            self._icobi_label.setText(f'NcICOBI(N) = {icobi:.5f}')
            ax.text(0.5, 0.5, 'no NcCOBICAR data',
                    transform=ax.transAxes,
                    ha='center', va='center', color='grey', fontsize=9)
        else:
            result = curves[0]   # show first (and usually only) match
            energy = result['energy']
            curve  = result['curve']
            ax.fill_betweenx(energy, 0, curve, where=(curve >= 0),
                             color=_NC_COLOUR_BOND, alpha=0.35, linewidth=0)
            ax.fill_betweenx(energy, 0, curve, where=(curve <= 0),
                             color=_NC_COLOUR_ANTI, alpha=0.35, linewidth=0)
            ax.plot(curve, energy, color='#111', linewidth=1.0, zorder=2)
            ef_val = result['ival_ef']
            self._icobi_label.setText(
                f'NcICOBI(N) = {icobi:.5f}   ·   NcICOBI(EF) = {ef_val:.5f}'
            )

        ax.axhline(0, color='#777', linestyle='--', linewidth=0.9, zorder=1)
        ax.axvline(0, color='#999', linestyle='-',  linewidth=0.5, zorder=1)
        ax.grid(True, linestyle=':', alpha=0.35)
        ax.set_xlabel('NcCOBI', fontsize=10)
        ax.set_ylabel('Energy (eV)', fontsize=10)
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _MulticenterWorker(QThread):
    """
    Runs Theil-Sen Badger fit and (optionally) the full directive pipeline
    in a background thread so the GUI stays responsive.
    """
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, bulk_results, supercell, poscar_path,
                 n_sigma, max_order, min_angle, reliability_cutoff=None):
        super().__init__()
        self._bulk              = bulk_results
        self._sc                = supercell
        self._poscar            = poscar_path
        self._sigma             = n_sigma
        self._order             = max_order
        self._angle             = min_angle
        self._reliability_cutoff = reliability_cutoff

    def run(self):
        try:
            from betapy.core.badger import compute_badger_quantities
            from scipy.stats import theilslopes, median_abs_deviation

            if self._reliability_cutoff is not None:
                reliability_limit = float(self._reliability_cutoff)
            else:
                L = self._sc.lattice
                a, b, c = L[0], L[1], L[2]
                V = abs(float(np.dot(a, np.cross(b, c))))
                reliability_limit = min(
                    V / np.linalg.norm(np.cross(b, c)),
                    V / np.linalg.norm(np.cross(a, c)),
                    V / np.linalg.norm(np.cross(a, b)),
                ) / 2.0
            # Expand compact FC files via crystal symmetry before filtering.
            try:
                from betapy.core.symmetry import expand_by_symmetry
                _bulk_expanded = expand_by_symmetry(self._bulk, self._sc)
            except Exception:
                _bulk_expanded = self._bulk

            reliable = [r for r in _bulk_expanded
                        if r['distance'] <= reliability_limit]

            reliable = compute_badger_quantities(reliable)

            # Deduplicate (full FC lists every bond twice)
            seen: set = set()
            deduped: list = []
            for r in reliable:
                key = (min(r['atom1_idx'], r['atom2_idx']),
                       max(r['atom1_idx'], r['atom2_idx']))
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)

            # Group by species pair; one global Theil-Sen fit per pair.
            by_pair: dict = defaultdict(list)
            for r in deduped:
                sp_key = tuple(sorted([r['species1'], r['species2']]))
                by_pair[sp_key].append(r)

            badger_data: dict = {}
            for sp_key, records in by_pair.items():
                pfcs  = np.array([r.get('phi_iso', float('nan')) for r in records])
                dists = np.array([r['distance'] for r in records])
                valid = np.isfinite(pfcs) & (pfcs > 0)
                v_pfcs    = pfcs[valid]
                v_dists   = dists[valid]
                v_records = [r for r, v in zip(records, valid) if v]

                slope = intercept = std = std_raw = float('nan')
                x_min = x_max = float('nan')
                if valid.sum() >= 4 and (
                        float(v_dists.max() - v_dists.min()) >= 0.05):
                    inv_cbrt = v_pfcs ** (-1.0 / 3.0)
                    _keys = np.stack([np.round(v_dists, 3),
                                      np.round(inv_cbrt, 4)], axis=1)
                    _, _ux = np.unique(_keys, axis=0, return_index=True)
                    if len(_ux) >= 4:
                        slope, intercept, *_ = theilslopes(
                            inv_cbrt[_ux], v_dists[_ux], method='joint')
                        pred_uniq = slope * v_dists[_ux] + intercept
                        log_ratio_uniq = 3.0 * np.log(
                            np.maximum(pred_uniq, 1e-12) / inv_cbrt[_ux])
                        std_raw = float(
                            median_abs_deviation(log_ratio_uniq) * 1.4826)
                        std   = max(std_raw, 1e-6)
                        x_min = float(v_dists.min())
                        x_max = float(v_dists.max())

                badger_data[sp_key] = {
                    'distances':   v_dists,
                    'pfcs':        v_pfcs,
                    'flagged':     np.zeros(valid.sum(), dtype=bool),
                    'n_flagged':   0,
                    'records':     v_records,
                    'slope':       float(slope),
                    'intercept':   float(intercept),
                    'std':         float(std),
                    'std_raw':     float(std_raw),   # MAD before 1e-6 floor
                    'x_min':       x_min,
                    'x_max':       x_max,
                }

            # Directive pipeline (needs POSCAR for chain detection)
            if self._poscar is not None:
                from betapy.core.multicenter import suggest_cobi_directives
                result = suggest_cobi_directives(
                    _bulk_expanded, self._sc, self._poscar,
                    n_sigma=self._sigma,
                    max_order=self._order,
                    min_angle_deg=self._angle,
                    detect_cutoff_frac=1.0,
                    _skip_symmetry_expand=True,
                )
            else:
                from betapy.core.multicenter import detect_anomalous_pairs
                flagged_pairs = detect_anomalous_pairs(
                    reliable,
                    n_sigma=self._sigma, value_key='phi_iso',
                    max_detect_dist=reliability_limit)
                result = {
                    'flagged_pairs': flagged_pairs,
                    'chains':        [],
                    'directives':    [],
                }

            result['badger_data'] = badger_data
            self.finished.emit(result)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class MulticenterWidget(QWidget):
    """
    Self-contained multicenter bonding detection tab.

    Call :meth:`load_data` after a bulk pFC analysis completes.
    Detection runs automatically; the Run button re-runs after parameter changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bulk_results    = None
        self._supercell       = None
        self._lobster_dir     = None
        self._poscar_path     = None
        self._worker          = None
        self._result          = None
        self._checkboxes      = {}   # pair_key -> QCheckBox
        self._pair_keys       = []   # sorted list of pair_keys from badger_data
        # flat list of (dist, pfc, record_dict, is_flagged, pair_key) for click lookup
        self._plot_points     = []
        self._directive_lookup  = {}   # directive string → list of SPOSCAR indices
        self._directive_trigger = {}   # directive string → trigger pair record
        self._selected_record   = None
        self._selected_chain_pairs: set = set()  # ALL (min,max) pairs within active chain
        self._selected_chain_consecutive_pairs: set = set()  # consecutive-only (for missing-FC)
        self._ax                = None
        # NcICOBI / NcCOBICAR
        self._lob_poscar         = None   # dict from parse_poscar_lobster()
        self._nc_icobi_records   = []     # list from parse_ncicobi_list()
        self._nc_cobicar_header  = None   # dict from parse_nccobicar_header(), or None
        self._nc_viewer          = None   # _NcCobiViewerWidget, created lazily
        self._build_ui()

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(190)
        left.setMaximumWidth(275)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(6)

        param_box = QGroupBox('Detection parameters')
        pf = QVBoxLayout(param_box)
        pf.setSpacing(4)

        def _spin_row(label, widget):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addStretch()
            row.addWidget(widget)
            pf.addLayout(row)

        self._spin_sigma = QDoubleSpinBox()
        self._spin_sigma.setRange(0.5, 20.0)
        self._spin_sigma.setSingleStep(0.5)
        self._spin_sigma.setValue(1.5)
        self._spin_sigma.setDecimals(1)
        self._spin_sigma.setFixedWidth(68)
        _spin_row('σ threshold:', self._spin_sigma)

        self._spin_order = QSpinBox()
        self._spin_order.setRange(3, 8)
        self._spin_order.setValue(5)
        self._spin_order.setFixedWidth(68)
        _spin_row('Max order:', self._spin_order)

        self._spin_angle = QDoubleSpinBox()
        self._spin_angle.setRange(90.0, 180.0)
        self._spin_angle.setSingleStep(5.0)
        self._spin_angle.setValue(150.0)
        self._spin_angle.setDecimals(0)
        self._spin_angle.setSuffix(' °')
        self._spin_angle.setFixedWidth(68)
        _spin_row('Min angle:', self._spin_angle)

        self._chk_badger_space = QCheckBox('Linearise (Φ⁻¹/³)')
        self._chk_badger_space.setChecked(False)
        self._chk_badger_space.setToolTip(
            'Show Φ_iso^{-1/3} vs r (Badger space) instead of Φ_iso vs r.\n'
            'In Badger space the baseline fit is a straight line.')
        self._chk_badger_space.stateChanged.connect(self._refresh_plot)
        pf.addWidget(self._chk_badger_space)

        lv.addWidget(param_box)

        self._btn_run = QPushButton('Run detection')
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run_detection)
        lv.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        lv.addWidget(self._progress)

        self._lbl_status = QLabel('Load SPOSCAR and FORCE_CONSTANTS,\nthen run analysis.')
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet('color: #666; font-size: 10px;')
        lv.addWidget(self._lbl_status)

        # ── Species-pair filter checkboxes ────────────────────────────
        pair_box = QGroupBox('Atom pair types')
        self._pair_layout = QVBoxLayout()
        self._pair_layout.setAlignment(Qt.AlignTop)
        pair_box.setLayout(self._pair_layout)
        pair_scroll = QScrollArea()
        pair_scroll.setWidget(pair_box)
        pair_scroll.setWidgetResizable(True)
        pair_scroll.setFixedHeight(110)
        lv.addWidget(pair_scroll)

        # ── Chains / Directives tabs ──────────────────────────────────
        self._left_tabs = QTabWidget()
        self._left_tabs.setDocumentMode(True)

        # Tab 0 — Chains ───────────────────────────────────────────────
        chains_tab = QWidget()
        ct = QVBoxLayout(chains_tab)
        ct.setContentsMargins(2, 4, 2, 2)
        ct.setSpacing(4)

        self._lbl_chains_count = QLabel('—')
        self._lbl_chains_count.setStyleSheet('font-size: 10px; color: #444;')
        ct.addWidget(self._lbl_chains_count)

        self._chains_tree = QTreeWidget()
        self._chains_tree.setHeaderHidden(True)
        self._chains_tree.setColumnCount(1)
        self._chains_tree.setIndentation(12)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(9)
        self._chains_tree.setFont(mono)
        self._chains_tree.setToolTip(
            'Click a chain to highlight it in the 3D view.\n'
            'Expanded sub-chains show individual COBI windows.')
        self._chains_tree.itemClicked.connect(self._on_chain_item_clicked)
        ct.addWidget(self._chains_tree, stretch=1)

        self._left_tabs.addTab(chains_tab, 'Chains')

        # Tab 1 — Directives ───────────────────────────────────────────
        dir_tab = QWidget()
        dv = QVBoxLayout(dir_tab)
        dv.setContentsMargins(2, 4, 2, 2)
        dv.setSpacing(4)

        self._dir_stack = QStackedWidget()

        # Page 0: directive list (POSCAR present)
        list_page = QWidget()
        lp = QVBoxLayout(list_page)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(4)

        self._lbl_count = QLabel('—')
        self._lbl_count.setStyleSheet('font-size: 10px; color: #444;')
        lp.addWidget(self._lbl_count)

        self._directive_list = QListWidget()
        mono2 = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono2.setPointSize(9)
        self._directive_list.setFont(mono2)
        self._directive_list.setToolTip(
            'Click a directive to highlight its chain in 3D.')
        self._directive_list.itemClicked.connect(self._on_directive_clicked)
        lp.addWidget(self._directive_list, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_copy = QPushButton('Copy')
        self._btn_copy.setEnabled(False)
        self._btn_copy.setToolTip('Copy directives to clipboard')
        self._btn_copy.clicked.connect(self._copy_directives)
        self._btn_save = QPushButton('Save…')
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip('Save directives to a text file')
        self._btn_save.clicked.connect(self._save_directives)
        btn_row.addWidget(self._btn_copy)
        btn_row.addWidget(self._btn_save)
        btn_row.addStretch()
        lp.addLayout(btn_row)
        self._dir_stack.addWidget(list_page)   # index 0

        # Page 1: no-POSCAR prompt
        no_poscar_page = QWidget()
        np_layout = QVBoxLayout(no_poscar_page)
        np_layout.setContentsMargins(6, 8, 6, 8)
        np_layout.setSpacing(10)
        np_layout.addStretch()

        _lbl_no_poscar = QLabel(
            'No LOBSTER directory found.\n\n'
            'cobiBetween directives require a POSCAR\n'
            'for SPOSCAR → unit-cell atom mapping.'
        )
        _lbl_no_poscar.setWordWrap(True)
        _lbl_no_poscar.setAlignment(Qt.AlignCenter)
        _lbl_no_poscar.setStyleSheet('color: #666; font-size: 10px;')
        np_layout.addWidget(_lbl_no_poscar)

        self._btn_browse = QPushButton('Browse for LOBSTER dir…')
        self._btn_browse.setToolTip(
            'Select a LOBSTER calculation directory containing a POSCAR')
        self._btn_browse.clicked.connect(self._browse_lobster_dir)
        np_layout.addWidget(self._btn_browse)
        np_layout.addStretch()
        self._dir_stack.addWidget(no_poscar_page)  # index 1

        dv.addWidget(self._dir_stack, stretch=1)
        self._left_tabs.addTab(dir_tab, 'Directives (—)')

        lv.addWidget(self._left_tabs, stretch=1)
        splitter.addWidget(left)

        # ── Middle panel — pFC vs r scatter ───────────────────────────
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.mpl_connect('button_press_event', self._on_scatter_click)
        self._mpl_toolbar = NavigationToolbar(self._canvas, self)
        mv.addWidget(self._mpl_toolbar)
        mv.addWidget(self._canvas, stretch=1)

        self._selection_bar = QLabel('')
        self._selection_bar.setFrameStyle(QFrame.StyledPanel)
        self._selection_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._selection_bar.setFixedHeight(28)
        mv.addWidget(self._selection_bar)

        splitter.addWidget(mid)

        # ── Right panel — 3D structure view ───────────────────────────
        self.structure_view = StructureView(self)
        self.structure_view.colours_changed.connect(self._on_colours_changed)
        splitter.addWidget(self.structure_view)

        splitter.setSizes([240, 600, 460])

        self._draw_placeholder()

    # ------------------------------------------------------------------ data

    def load_data(self, bulk_results, supercell, lobster_dir=None,
                  reliability_cutoff=None):
        """
        Supply analysis results and trigger automatic detection.

        Parameters
        ----------
        bulk_results : list of dicts from compute_bulk_pfcs()
        supercell    : Supercell
        lobster_dir  : Path or None — directory containing POSCAR
        """
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(500)

        self._bulk_results       = bulk_results
        self._supercell          = supercell
        self._lobster_dir        = lobster_dir
        self._reliability_cutoff = reliability_cutoff
        self._poscar_path        = None
        self._lob_poscar         = None
        self._nc_icobi_records   = []
        self._nc_cobicar_header  = None

        self.structure_view.load_supercell(supercell)

        if lobster_dir is not None:
            ldir = Path(lobster_dir)
            p = ldir / 'POSCAR'
            if p.exists():
                self._poscar_path = p

            from betapy.core.lobster import (
                parse_poscar_lobster, parse_ncicobi_list,
                parse_car_header,
            )
            for _pl_name in ('POSCAR.lobster', 'POSCAR.lobster.vasp', 'POSCAR'):
                _pl = ldir / _pl_name
                if _pl.exists():
                    try:
                        self._lob_poscar = parse_poscar_lobster(_pl)
                        break
                    except Exception:
                        pass

            ni = ldir / 'NcICOBILIST.lobster'
            if ni.exists():
                try:
                    self._nc_icobi_records = parse_ncicobi_list(ni)
                except Exception:
                    pass

            nc = ldir / 'COBICAR.lobster'
            if nc.exists():
                try:
                    self._nc_cobicar_header = parse_car_header(nc)
                except Exception:
                    pass

        if self._poscar_path is not None:
            self._lbl_status.setText(
                f'POSCAR: {self._poscar_path.parent.name}/POSCAR  ✓'
            )
        else:
            self._lbl_status.setText(
                '⚠ No POSCAR in LOBSTER directory —\n'
                'Badger plots available; directives require a POSCAR.'
            )

        self._btn_run.setEnabled(True)
        self._run_detection()

    # ------------------------------------------------------------------ run

    def _run_detection(self):
        if self._bulk_results is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._btn_run.setEnabled(False)
        self._btn_copy.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._progress.setVisible(True)

        self._worker = _MulticenterWorker(
            self._bulk_results,
            self._supercell,
            self._poscar_path,
            self._spin_sigma.value(),
            self._spin_order.value(),
            self._spin_angle.value(),
            reliability_cutoff=self._reliability_cutoff,
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._result = result

        # Red  — all statistically anomalous pairs (honest Badger detection output).
        # Amber — chain segment bonds (consecutive pairs in confirmed chains) that
        #         are not themselves individually flagged; shows the full geometry
        #         of the multicenter bond even when only the trigger bond diverges.
        # Trigger pairs that are also chain segments stay red (priority).
        trigger_keys = {
            (min(int(r['atom1_idx']), int(r['atom2_idx'])),
             max(int(r['atom1_idx']), int(r['atom2_idx'])))
            for r in result.get('flagged_pairs', [])
        }
        chain_segment_keys: set = set()
        for chain in result.get('chains', []):
            idxs = chain.get('full_chain', [])
            # Mark ALL pairs within the chain (not just consecutive neighbours)
            # so that e.g. Bi-Bi in a Bi-S-Bi-S chain gets an amber halo.
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = int(idxs[i]), int(idxs[j])
                    chain_segment_keys.add((min(a, b), max(a, b)))

        for bd in result['badger_data'].values():
            flagged      = []
            chain_member = []
            for rec in bd['records']:
                k = (min(int(rec['atom1_idx']), int(rec['atom2_idx'])),
                     max(int(rec['atom1_idx']), int(rec['atom2_idx'])))
                flagged.append(k in trigger_keys)
                chain_member.append(k in chain_segment_keys and k not in trigger_keys)
            bd['flagged']      = np.array(flagged)
            bd['chain_member'] = np.array(chain_member)
            bd['n_flagged']    = int(np.sum(flagged))

        # directive string → SPOSCAR atom indices + trigger pair record
        self._directive_lookup  = {}
        self._directive_trigger = {}
        for chain in result.get('chains', []):
            trigger = chain.get('trigger_pair', {})
            for sub in chain.get('sub_chains', []):
                d = sub.get('directive', '')
                if d and not d.startswith('#') and d not in self._directive_lookup:
                    self._directive_lookup[d]  = sub['indices']
                    self._directive_trigger[d] = trigger

        # Clear any selection from the previous run so stale rings don't persist.
        self._selected_chain_pairs             = set()
        self._selected_chain_consecutive_pairs = set()
        self._selected_record                  = None

        self._rebuild_checkboxes(result['badger_data'])
        self._refresh_plot()
        self._update_chains_tree(result)
        self._update_directives(result)

    def _on_error(self, msg):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        short = msg.strip().splitlines()[-1][:160]
        self._lbl_status.setText(f'Error: {short}')
        QMessageBox.critical(self, 'Multicenter detection error', msg)

    # ------------------------------------------------------------------ colours

    def _on_colours_changed(self):
        self._rebuild_checkboxes()
        self._refresh_plot()

    # ------------------------------------------------------------------ checkboxes

    def _rebuild_checkboxes(self, badger_data=None):
        if badger_data is None:
            if self._result is not None:
                badger_data = self._result.get('badger_data', {})
            else:
                return

        # Preserve checked state across rebuilds
        previously_checked = {pk for pk, cb in self._checkboxes.items()
                               if cb.isChecked()}

        while self._pair_layout.count():
            item = self._pair_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._checkboxes = {}
        self._pair_keys  = sorted(badger_data.keys())

        for pk in self._pair_keys:
            sp1, sp2 = pk
            if self._supercell is not None:
                c1, c2 = self.structure_view.pair_colours_hex(sp1, sp2)
            else:
                c1 = c2 = '#888888'

            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)

            cb = QCheckBox()
            # New pairs start checked; re-runs restore previous state
            cb.setChecked(pk in previously_checked if previously_checked
                          else True)
            cb.stateChanged.connect(self._refresh_plot)

            lbl = QLabel(
                f'<b><span style="color:{c1}">{sp1}</span></b>'
                f'<span style="color:#888"> – </span>'
                f'<b><span style="color:{c2}">{sp2}</span></b>'
            )
            rl.addWidget(cb)
            rl.addWidget(lbl)

            n_flag = badger_data[pk]['n_flagged']
            if n_flag:
                flag_lbl = QLabel(
                    f'<span style="color:#d03030">({n_flag}✶)</span>'
                )
                rl.addWidget(flag_lbl)

            rl.addStretch()
            self._pair_layout.addWidget(row)
            self._checkboxes[pk] = cb

    # ------------------------------------------------------------------ plot

    def _draw_placeholder(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5,
                'Run detection to see the Φ_iso vs r plot\nwith Badger curve overlay.',
                ha='center', va='center', transform=ax.transAxes,
                color='#bbb', fontsize=12)
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _refresh_plot(self):
        if self._result is None:
            self._draw_placeholder()
            return

        badger_data = self._result.get('badger_data', {})
        if not badger_data:
            self._draw_placeholder()
            return

        checked_pairs = {pk for pk, cb in self._checkboxes.items()
                         if cb.isChecked()}
        n_sigma      = self._spin_sigma.value()
        badger_mode  = self._chk_badger_space.isChecked()

        def _display_y(phi):
            return phi ** (-1.0 / 3.0) if badger_mode else phi

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._ax = ax
        self._plot_points = []

        # Separate points into rendering layers.
        # Halos (chain / trigger) are drawn behind species-coloured dots so
        # species information is always visible in the foreground.
        gray_xs:        list = []
        gray_ys:        list = []
        colored:        dict = defaultdict(lambda: {'xs': [], 'ys': []})
        halo_chain_xs:  list = []
        halo_chain_ys:  list = []
        halo_trigger_xs: list = []
        halo_trigger_ys: list = []

        for pk in self._pair_keys:
            if pk not in badger_data:
                continue
            bd           = badger_data[pk]
            dists        = bd['distances']
            pfcs         = bd['pfcs']
            flagged      = bd['flagged']
            chain_member = bd.get('chain_member',
                                  np.zeros(len(dists), dtype=bool))
            records      = bd['records']
            is_checked   = pk in checked_pairs

            for d, pfc, rec, fl, cm in zip(
                    dists, pfcs, records, flagged, chain_member):
                self._plot_points.append((d, pfc, rec, bool(fl), pk, bool(cm)))
                y = _display_y(pfc)
                if not is_checked:
                    gray_xs.append(d)
                    gray_ys.append(y)
                else:
                    colored[pk]['xs'].append(d)
                    colored[pk]['ys'].append(y)
                    if fl:
                        halo_trigger_xs.append(d)
                        halo_trigger_ys.append(y)
                    elif cm:
                        halo_chain_xs.append(d)
                        halo_chain_ys.append(y)

        # Layer 1: unchecked-pair points (grey, provides visual context)
        if gray_xs:
            ax.scatter(gray_xs, gray_ys, s=18, c=_GREY_COLOR,
                       alpha=_GREY_ALPHA, linewidths=0, zorder=1)

        # Layer 2: Badger curve + band (one per checked species pair, under dots)
        for pk in self._pair_keys:
            if pk not in checked_pairs or pk not in badger_data:
                continue
            bd = badger_data[pk]
            slope     = bd['slope']
            intercept = bd['intercept']
            std       = bd['std']
            x_min     = bd['x_min']
            x_max     = bd['x_max']
            if math.isnan(slope) or math.isnan(std):
                continue
            sp1, sp2 = pk
            c1, c2 = (self.structure_view.pair_colours_hex(sp1, sp2)
                      if self._supercell is not None else ('#888888', '#888888'))
            mixed = (c1 != c2)

            x_line   = np.linspace(x_min, x_max, 300)
            lin_base = slope * x_line + intercept   # Φ^{-1/3} linear fit
            # σ is in log(phi_iso) space, so bands are multiplicative in
            # Φ^{-1/3}: a ±σ_log shift in log(phi_iso) maps to
            # Φ^{-1/3}_band = Φ^{-1/3}_fit × exp(∓σ_log / 3).
            lin_strong = lin_base * np.exp(-n_sigma * std / 3.0)  # detection side
            ok = (lin_base > 0) & (lin_strong > 0)
            if not ok.any():
                continue
            xc     = x_line[ok]
            lin_c  = lin_base[ok]
            lin_lo = lin_c * np.exp(+n_sigma * std / 3.0)  # weaker (higher Φ^{-1/3})
            lin_hi = lin_strong[ok]                          # stronger (detection threshold)

            # Use the pre-floor MAD: if std_raw≈0 the band would be ~4µ wide
            # (invisible at plot scale) but has_band = std>1e-8 would still pass
            # because of the 1e-6 detection floor.  Gate on the actual scatter.
            has_band = bd.get('std_raw', 0.0) > 1e-8

            # Mixed species: alternate dash colors (phase-offset overlapping lines)
            # to mirror the split-circle scatter markers.
            def _draw_curve(xs, ys):
                if mixed:
                    # Phase-offset dashes: c1 at [0–6], gap, c2 at [8–14], gap, …
                    # Combined period=16, 2-unit gaps — visually matches '--'.
                    ax.plot(xs, ys, color=c1, lw=1.5, alpha=_CURVE_ALPHA,
                            linestyle=(0, (6, 10)), zorder=3)
                    ax.plot(xs, ys, color=c2, lw=1.5, alpha=_CURVE_ALPHA,
                            linestyle=(8, (6, 10)), zorder=3)
                else:
                    ax.plot(xs, ys, color=c1, lw=1.5, alpha=_CURVE_ALPHA,
                            linestyle='--', zorder=3)

            label_txt = f'{sp1}–{sp2}'
            if badger_mode:
                # Linearised view: straight line + ±nσ band
                _draw_curve(xc, lin_c)
                if has_band:
                    # Overlay both colors at half alpha for mixed pairs
                    ax.fill_between(xc, lin_lo, lin_hi,
                                    color=c1, alpha=_BAND_ALPHA, zorder=2)
                    if mixed:
                        ax.fill_between(xc, lin_lo, lin_hi,
                                        color=c2, alpha=_BAND_ALPHA, zorder=2)
                    # Label at the right edge of the band (mid-band y)
                    label_y = float(0.5 * (lin_lo[-1] + lin_hi[-1]))
                    ax.text(float(xc[-1]), label_y, f' {label_txt}',
                            color=c1, fontsize=7, va='center',
                            alpha=_CURVE_ALPHA, zorder=3)
                else:
                    # No band (zero scatter) — label the line itself
                    ax.text(float(xc[-1]), float(lin_c[-1]), f' {label_txt}',
                            color=c1, fontsize=7, va='center',
                            alpha=_CURVE_ALPHA, zorder=3)
            else:
                # Real-space view: dashed Badger curve + label at right end
                yc = lin_c ** -3
                _draw_curve(xc, yc)
                ax.text(float(xc[-1]), float(yc[-1]), f' {label_txt}',
                        color=c1, fontsize=7, va='center',
                        alpha=_CURVE_ALPHA, zorder=3)

        # Layer 3a: amber halos behind chain segment bonds
        if halo_chain_xs:
            ax.scatter(halo_chain_xs, halo_chain_ys, s=90,
                       facecolors=_CHAIN_COLOR, edgecolors='none',
                       alpha=0.35, linewidths=0, zorder=3)

        # Layer 3b: red halos behind anomalous trigger pairs
        if halo_trigger_xs:
            ax.scatter(halo_trigger_xs, halo_trigger_ys, s=110,
                       facecolors=_FLAG_COLOR, edgecolors='none',
                       alpha=0.45, linewidths=0, zorder=3)

        # Layer 4: checked-pair points (species colour) — drawn on top of halos
        for pk, data in colored.items():
            if not data['xs']:
                continue
            sp1, sp2 = pk
            c1, c2 = (self.structure_view.pair_colours_hex(sp1, sp2)
                      if self._supercell is not None else ('#888888', '#888888'))

            if c1 == c2:
                ax.scatter(data['xs'], data['ys'], s=25, c=c1,
                           alpha=_NORM_ALPHA, linewidths=0, zorder=4,
                           label=f'{sp1}–{sp2}')
            else:
                theta = np.linspace(0, np.pi, 60)
                top_v = np.column_stack([
                    np.concatenate([[0], np.cos(theta),        [0]]),
                    np.concatenate([[0], np.sin(theta),        [0]]),
                ])
                bot_v = np.column_stack([
                    np.concatenate([[0], np.cos(theta + np.pi), [0]]),
                    np.concatenate([[0], np.sin(theta + np.pi), [0]]),
                ])
                ax.scatter(data['xs'], data['ys'], marker=top_v, s=25, c=c1,
                           alpha=_NORM_ALPHA, linewidths=0, zorder=4)
                ax.scatter(data['xs'], data['ys'], marker=bot_v, s=25, c=c2,
                           alpha=_NORM_ALPHA, linewidths=0, zorder=4,
                           label=f'{sp1}–{sp2}')

        # Teal rings for non-trigger chain bonds when a chain is selected in the tree.
        # Dashed teal vertical lines for chain bonds absent from the scatter (no FC data).
        # Golden ring for the trigger (or any directly-clicked point).
        if self._selected_chain_pairs:
            trigger_key = None
            if self._selected_record is not None:
                sr = self._selected_record
                trigger_key = (min(int(sr['atom1_idx']), int(sr['atom2_idx'])),
                               max(int(sr['atom1_idx']), int(sr['atom2_idx'])))
            in_plot: set = set()
            for pt in self._plot_points:
                rec  = pt[2]
                rkey = (min(int(rec['atom1_idx']), int(rec['atom2_idx'])),
                        max(int(rec['atom1_idx']), int(rec['atom2_idx'])))
                in_plot.add(rkey)
                if rkey in self._selected_chain_pairs and rkey != trigger_key:
                    phi = rec.get('phi_iso', rec.get('mean_pfc', 0.0))
                    ax.plot(
                        rec['distance'], _display_y(phi),
                        'o', markersize=16,
                        markerfacecolor='none', markeredgecolor='#00b8b8',
                        markeredgewidth=2.0, zorder=6,
                    )
            # Consecutive chain bonds missing from the scatter (compact FC file):
            # draw a dashed vertical line at the geometric distance so the user
            # can see where the bond would sit. Non-consecutive pairs (A-C, A-D…)
            # are expected to be absent at longer range, so no dashed line for those.
            if self._supercell is not None:
                for pair in self._selected_chain_consecutive_pairs - in_plot:
                    if pair == trigger_key:
                        continue
                    i, j = pair
                    d = self._supercell.atom_distance(i, j)
                    ax.axvline(d, color='#00b8b8', linestyle='--',
                               linewidth=1.5, alpha=0.55, zorder=5)

        if self._selected_record is not None:
            sr  = self._selected_record
            phi = sr.get('phi_iso', sr.get('mean_pfc', 0.0))
            ax.plot(
                sr['distance'], _display_y(phi),
                'o', markersize=16,
                markerfacecolor='none', markeredgecolor='#c8a000',
                markeredgewidth=2.5, zorder=7,
            )

        # Pin y-axis to the actual data point range — the Badger curve can
        # reach extreme values near its left edge and would otherwise dominate.
        active_ys = (
            [_display_y(pt[1]) for pt in self._plot_points
             if pt[4] in checked_pairs]
            if self._plot_points else []
        )
        if active_ys:
            y_lo = min(active_ys)
            y_hi = max(active_ys)
            margin = (y_hi - y_lo) * 0.08 if y_hi > y_lo else abs(y_hi) * 0.1
            ax.set_ylim(max(0.0, y_lo - margin), y_hi + margin)

        ax.set_xlabel('Interatomic distance (Å)', fontsize=12)
        if badger_mode:
            ax.set_ylabel('Φ_iso^{-1/3}  ((eV/Å²)^{-1/3})', fontsize=12)
        else:
            ax.set_ylabel('Φ_iso  (eV/Å²)', fontsize=12)
        if badger_mode:
            title = 'Multicenter bonding — Φ_iso^{-1/3} vs r  (dashed: Badger fit, ±nσ band)'
        else:
            title = 'Multicenter bonding — Φ_iso vs r  (dashed: Badger fit)'
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)

        if colored or halo_chain_xs or halo_trigger_xs:
            import matplotlib.lines as mlines
            handles, labels = ax.get_legend_handles_labels()
            if halo_chain_xs:
                handles.append(mlines.Line2D(
                    [], [], marker='o', linestyle='none',
                    markersize=10, color=_CHAIN_COLOR, alpha=0.55,
                    label=f'chain bond ({len(halo_chain_xs)})'))
                labels.append(f'chain bond ({len(halo_chain_xs)})')
            if halo_trigger_xs:
                handles.append(mlines.Line2D(
                    [], [], marker='o', linestyle='none',
                    markersize=10, color=_FLAG_COLOR, alpha=0.65,
                    label=f'anomalous trigger ({len(halo_trigger_xs)})'))
                labels.append(f'anomalous trigger ({len(halo_trigger_xs)})')
            ax.legend(handles=handles, loc='upper right',
                      framealpha=0.9, fontsize=9)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------ click

    def _on_scatter_click(self, event):
        if event.inaxes is None or self._ax is None or not self._plot_points:
            return
        if event.button != 1:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        xs = (xlim[1] - xlim[0]) or 1.0
        ys = (ylim[1] - ylim[0]) or 1.0

        badger_mode = self._chk_badger_space.isChecked()
        best_d2, best_pt = float('inf'), None
        for pt in self._plot_points:
            dy = pt[1] ** (-1.0 / 3.0) if badger_mode else pt[1]
            d2 = ((pt[0] - cx) / xs) ** 2 + ((dy - cy) / ys) ** 2
            if d2 < best_d2:
                best_d2, best_pt = d2, pt

        if best_pt is None or best_d2 ** 0.5 > _PICK_TOL:
            return

        _, _, rec, is_flagged, _, is_chain = best_pt
        self._selected_record                  = rec
        self._selected_chain_pairs             = set()
        self._selected_chain_consecutive_pairs = set()

        a1  = int(rec['atom1_idx'])
        a2  = int(rec['atom2_idx'])
        sp1 = rec['species1']
        sp2 = rec['species2']
        d   = rec['distance']
        pfc = rec.get('phi_iso', rec.get('mean_pfc', float('nan')))

        if self._supercell is not None:
            self.structure_view.highlight_bond(a1, a2)

        if is_flagged:
            info = 'Anomalous trigger'
        elif is_chain:
            info = 'Chain bond'
        else:
            info = 'Selected'
        self._selection_bar.setText(
            f'{info}:  atom {a1} ({sp1}) – atom {a2} ({sp2})   '
            f'd = {d:.4f} Å   Φ_iso = {pfc:.6f} eV/Å²'
        )

        self._refresh_plot()

    # ------------------------------------------------------------------ directives

    def _update_directives(self, result):
        directives = result.get('directives', [])

        if self._poscar_path is None:
            self._dir_stack.setCurrentIndex(1)
            self._left_tabs.setTabText(1, 'Directives (—)')
            self._btn_copy.setEnabled(False)
            self._btn_save.setEnabled(False)
            return

        self._dir_stack.setCurrentIndex(0)
        self._directive_list.clear()
        for d in directives:
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(d)
            idxs = self._directive_lookup.get(d, [])
            if idxs and self._supercell is not None:
                try:
                    sp = [self._supercell.species(i) for i in idxs]
                    item.setToolTip(
                        f'{len(idxs)}-center: {" – ".join(sp)}\n'
                        f'SPOSCAR indices: {idxs}'
                    )
                except Exception:
                    pass
            self._directive_list.addItem(item)

        n = len(directives)
        self._lbl_count.setText(f'{n} unique directive(s)')
        self._left_tabs.setTabText(1, f'Directives ({n})')
        has = bool(directives)
        self._btn_copy.setEnabled(has)
        self._btn_save.setEnabled(has)

    def _update_chains_tree(self, result):
        chains        = result.get('chains', [])
        flagged_pairs = result.get('flagged_pairs', [])

        self._chains_tree.clear()

        # Group instances by canonical species chain (A-B-C == C-B-A).
        groups: dict = {}   # canon_key → list of chain dicts
        for chain in chains:
            sp = tuple(chain['species_chain'])
            canon = min(sp, sp[::-1])
            groups.setdefault(canon, []).append(chain)

        # Also surface sub-chains (shorter consecutive sub-sequences of each
        # detected chain) as independent top-level entries.  find_chains grows
        # each trigger into the longest possible chain, so 3-center and
        # 4-center chains only exist inside chain['sub_chains'] — they would
        # never appear in the tree at all without this step.
        # Deduplication is by canonical atom-index tuple so the same sub-chain
        # encountered from two different parent chains is shown only once.
        # Pre-seed with all full chains so independently-detected shorter chains
        # don't get duplicated when they also appear as a sub-chain of a longer one.
        seen_sub_keys: set = {
            min(tuple(c['full_chain']), tuple(reversed(c['full_chain'])))
            for c in chains
        }
        for chain in chains:
            full    = chain.get('full_chain', [])
            trigger = chain.get('trigger_pair', {})
            t1 = int(trigger.get('atom1_idx', -1))
            t2 = int(trigger.get('atom2_idx', -1))
            for sub in chain.get('sub_chains', []):
                if sub['order'] >= len(full):
                    continue                # skip the full chain (already in groups)
                idxs     = sub['indices']
                sub_key  = min(tuple(idxs), tuple(reversed(idxs)))
                if sub_key in seen_sub_keys:
                    continue
                seen_sub_keys.add(sub_key)
                if self._supercell is not None:
                    try:
                        sub_sp = [self._supercell.species(i) for i in idxs]
                    except Exception:
                        sub_sp = ['?'] * len(idxs)
                    pos0       = self._supercell.positions[idxs[0]  - 1]
                    posN       = self._supercell.positions[idxs[-1] - 1]
                    total_dist = float(np.linalg.norm(
                        self._supercell.cart_diff(pos0, posN)))
                else:
                    sub_sp     = ['?'] * len(idxs)
                    total_dist = 0.0
                sub_canon   = min(tuple(sub_sp), tuple(reversed(sub_sp)))
                sub_trigger = trigger if (t1 in idxs and t2 in idxs) else {}
                groups.setdefault(sub_canon, []).append({
                    'trigger_pair':   sub_trigger,
                    'full_chain':     idxs,
                    'species_chain':  sub_sp,
                    'total_distance': total_dist,
                    'sub_chains':     [],  # no further nesting at top level
                })

        # Sort groups: highest max-σ first (monotone chains treated as σ=0).
        def _max_sigma(chain_list):
            vals = []
            for c in chain_list:
                s = c.get('trigger_pair', {}).get('n_sigma', float('nan'))
                vals.append(s if not math.isnan(s) else 0.0)
            return max(vals, default=0.0)

        sorted_groups = sorted(groups.items(),
                               key=lambda kv: _max_sigma(kv[1]),
                               reverse=True)

        # Scatter pair keys — used to tiebreak representative selection so that
        # the chosen instance maximises the number of all-pair combinations
        # (including non-consecutive end-to-end pairs) that are visible in the
        # scatter plot.  Built once here; _refresh_plot() has already run.
        _scatter_keys = {
            (min(int(pt[2]['atom1_idx']), int(pt[2]['atom2_idx'])),
             max(int(pt[2]['atom1_idx']), int(pt[2]['atom2_idx'])))
            for pt in self._plot_points
        }

        def _scatter_coverage(c):
            idxs = c.get('full_chain', [])
            return sum(
                1 for ii in range(len(idxs))
                for jj in range(ii + 1, len(idxs))
                if (min(int(idxs[ii]), int(idxs[jj])),
                    max(int(idxs[ii]), int(idxs[jj]))) in _scatter_keys
            )

        for canon_key, chain_list in sorted_groups:
            order   = len(canon_key)
            sp_str  = '–'.join(canon_key)
            n_inst  = len(chain_list)

            # Representative: highest trigger σ, tiebroken by scatter coverage
            # so the chosen instance shows as many all-pair rings as possible.
            rep = max(
                chain_list,
                key=lambda c: (
                    c.get('trigger_pair', {}).get('n_sigma', float('nan'))
                    if not math.isnan(
                        c.get('trigger_pair', {}).get('n_sigma', float('nan')))
                    else -1.0,
                    _scatter_coverage(c),
                ),
            )
            rep_trigger = rep.get('trigger_pair', {})
            rep_sig     = rep_trigger.get('n_sigma', float('nan'))
            sig_str     = f'{rep_sig:.1f}σ' if not math.isnan(rep_sig) else 'monotone'

            # Distance range across all instances.
            distances = [c['total_distance'] for c in chain_list]
            d_lo, d_hi = min(distances), max(distances)
            dist_str = (f'{d_lo:.2f} Å' if abs(d_hi - d_lo) < 0.05
                        else f'{d_lo:.2f}–{d_hi:.2f} Å')

            count_str = f'  ×{n_inst}' if n_inst > 1 else ''
            label = (f'{sp_str}  ·  {order}-center  ·  '
                     f'{dist_str}  ·  {sig_str}{count_str}')

            top = QTreeWidgetItem([label])
            top.setData(0, Qt.UserRole, {
                'type':    'chain',
                'indices': rep['full_chain'],
                'chain':   rep,
            })

            # Tooltip on the parent shows the representative instance.
            phi = rep_trigger.get(
                'phi_iso', rep_trigger.get('mean_pfc', float('nan')))
            tip_lines = [
                f'Showing representative instance of {n_inst} total.',
                (f'Trigger: {rep_trigger.get("species1","")}'
                 f'-{rep_trigger.get("species2","")}  '
                 f'd={rep_trigger.get("distance", 0):.3f} Å  '
                 f'Φ_iso={phi:.4f}  [{rep_trigger.get("method","")}]'),
            ]
            if self._supercell is not None:
                segs = self._chain_segment_distances(rep['full_chain'])
                tip_lines.append(
                    f'Segments: {" + ".join(f"{d:.3f}" for d in segs)} Å')
            top.setToolTip(0, '\n'.join(tip_lines))

            # Children — instances (when multiple), then sub-chains.
            if n_inst > 1:
                for i, c in enumerate(chain_list, 1):
                    idxs  = c['full_chain']
                    tr    = c.get('trigger_pair', {})
                    ns    = tr.get('n_sigma', float('nan'))
                    s_str = f'{ns:.1f}σ' if not math.isnan(ns) else 'monotone'
                    idx_str = ' → '.join(str(x) for x in idxs)
                    child = QTreeWidgetItem(
                        [f'  #{i}  {idx_str}  ·  {c["total_distance"]:.2f} Å  ·  {s_str}'])
                    child.setData(0, Qt.UserRole, {
                        'type':    'chain',
                        'indices': idxs,
                        'chain':   c,
                    })
                    if self._supercell is not None:
                        segs = self._chain_segment_distances(idxs)
                        child.setToolTip(
                            0,
                            f'Segments: {" + ".join(f"{d:.3f}" for d in segs)} Å\n'
                            f'Trigger Φ_iso={tr.get("phi_iso", float("nan")):.4f}  [{tr.get("method","")}]'
                        )
                    self._add_subchain_children(child, c)
                    top.addChild(child)
            else:
                # Single instance: sub-chains as direct children of the top item.
                self._add_subchain_children(top, rep)

            self._chains_tree.addTopLevelItem(top)

        n_types = self._chains_tree.topLevelItemCount()
        n_total = len(chains)
        self._left_tabs.setTabText(0, f'Chains ({n_types})')
        if flagged_pairs:
            summary = f'{len(flagged_pairs)} flagged pair(s)  →  {n_types} unique type(s)'
            if n_total > n_types:
                summary += f'  ({n_total} instances)'
            self._lbl_chains_count.setText(summary)
        else:
            self._lbl_chains_count.setText('No anomalous pairs detected.')

        # Auto-select and highlight first chain type; expand to reveal sub-chains.
        if n_types > 0:
            first = self._chains_tree.topLevelItem(0)
            self._chains_tree.setCurrentItem(first)
            first.setExpanded(True)
            self._on_chain_item_clicked(first, 0)

    def _add_subchain_children(self, parent_item, chain_dict):
        """Attach sub-chain tree items as children of *parent_item*."""
        full       = chain_dict.get('full_chain', [])
        sub_chains = chain_dict.get('sub_chains', [])
        trigger    = chain_dict.get('trigger_pair', {})
        t1 = int(trigger.get('atom1_idx', -1))
        t2 = int(trigger.get('atom2_idx', -1))

        # Only sub-chains shorter than the full chain, sorted by order then
        # by their start position within the full chain.
        full_idx = {idx: pos for pos, idx in enumerate(full)}
        shorter  = [s for s in sub_chains if s['order'] < len(full)]
        shorter.sort(key=lambda s: (s['order'], full_idx.get(s['indices'][0], 0)))

        for sub in shorter:
            idxs  = sub['indices']
            order = sub['order']

            if self._supercell is not None:
                try:
                    parts = [f'{self._supercell.species(i)}({i})' for i in idxs]
                except Exception:
                    parts = [str(i) for i in idxs]
                segs = self._chain_segment_distances(idxs)
                dist = f'{sum(segs):.2f} Å'
            else:
                parts = [str(i) for i in idxs]
                dist  = ''

            label = (f'  {order}-center:  '
                     + '  →  '.join(parts)
                     + (f'   {dist}' if dist else ''))
            child = QTreeWidgetItem([label])

            # Pass the parent trigger only when both trigger atoms are in this
            # sub-chain, so the golden ring appears on the correct scatter point.
            sub_trigger = trigger if (t1 in idxs and t2 in idxs) else {}
            child.setData(0, Qt.UserRole, {
                'type':    'sub_chain',
                'indices': idxs,
                'chain':   {
                    'trigger_pair': sub_trigger,
                    'full_chain':   idxs,
                    'sub_chains':   [],
                },
            })
            parent_item.addChild(child)

    def _on_chain_item_clicked(self, item, _column):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return

        indices = data['indices']
        chain   = data['chain']

        if self._supercell is not None and len(indices) >= 2:
            pairs = [(indices[i], indices[i + 1])
                     for i in range(len(indices) - 1)]
            self.structure_view.highlight_bonds(
                pairs, center_on=indices[0], highlight_atoms=True)

        trigger = chain.get('trigger_pair', {})

        # Consecutive pairs — the bonds that chain detection actually traversed.
        # Used for the missing-FC warning and dashed fallback lines.
        self._selected_chain_consecutive_pairs = {
            (min(int(indices[i]), int(indices[i + 1])),
             max(int(indices[i]), int(indices[i + 1])))
            for i in range(len(indices) - 1)
        }
        # All pairwise combinations within the chain atoms — used for teal ring
        # highlighting so that e.g. A-C, A-D, B-D also get rings, not just A-B, B-C.
        self._selected_chain_pairs = {
            (min(int(indices[i]), int(indices[j])),
             max(int(indices[i]), int(indices[j])))
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        }

        if self._supercell is not None:
            try:
                sp = [self._supercell.species(i) for i in indices]
            except Exception:
                sp = ['?'] * len(indices)
            label   = '  →  '.join(f'{s}({i})' for s, i in zip(sp, indices))
            n_sig   = trigger.get('n_sigma', float('nan'))
            sig_str = f'{n_sig:.1f}σ' if not math.isnan(n_sig) else 'monotone'
            kind    = 'Sub-chain' if data['type'] == 'sub_chain' else 'Chain'
            # Count consecutive bonds missing from FC data (compact FORCE_CONSTANTS).
            plot_keys = {(min(int(pt[2]['atom1_idx']), int(pt[2]['atom2_idx'])),
                          max(int(pt[2]['atom1_idx']), int(pt[2]['atom2_idx'])))
                         for pt in self._plot_points}
            n_missing = len(self._selected_chain_consecutive_pairs - plot_keys)
            bar_text  = f'{kind} ({len(indices)}-center):  {label}   [{sig_str}]'
            if n_missing:
                bar_text += f'   ⚠ {n_missing} bond(s) missing from FC data'
            self._selection_bar.setText(bar_text)
        # Reset before search so a stale record never leaks into the new render.
        self._selected_record = None
        if trigger and self._plot_points:
            tkey = (min(int(trigger['atom1_idx']), int(trigger['atom2_idx'])),
                    max(int(trigger['atom1_idx']), int(trigger['atom2_idx'])))
            for pt in self._plot_points:
                rec  = pt[2]
                rkey = (min(int(rec['atom1_idx']), int(rec['atom2_idx'])),
                        max(int(rec['atom1_idx']), int(rec['atom2_idx'])))
                if rkey == tkey:
                    self._selected_record = rec
                    break
        self._refresh_plot()

    def _chain_segment_distances(self, indices):
        dists = []
        for i in range(len(indices) - 1):
            pos_a = self._supercell.positions[indices[i]     - 1]
            pos_b = self._supercell.positions[indices[i + 1] - 1]
            dists.append(
                float(np.linalg.norm(self._supercell.cart_diff(pos_a, pos_b))))
        return dists

    def _browse_lobster_dir(self):
        from PyQt5.QtWidgets import QFileDialog
        start = str(self._lobster_dir) if self._lobster_dir else str(Path.home())
        path  = QFileDialog.getExistingDirectory(
            self, 'Select LOBSTER calculation directory', start)
        if not path:
            return

        ldir   = Path(path)
        poscar = ldir / 'POSCAR'
        if not poscar.exists():
            QMessageBox.warning(
                self, 'No POSCAR found',
                f'No POSCAR file found in:\n{ldir}\n\n'
                'The LOBSTER directory must contain a POSCAR for atom mapping.')
            return

        self._lobster_dir = ldir
        self._poscar_path = poscar

        from betapy.core.lobster import (
            parse_poscar_lobster, parse_ncicobi_list, parse_car_header,
        )
        self._lob_poscar = None
        for _name in ('POSCAR.lobster', 'POSCAR.lobster.vasp', 'POSCAR'):
            _pl = ldir / _name
            if _pl.exists():
                try:
                    self._lob_poscar = parse_poscar_lobster(_pl)
                    break
                except Exception:
                    pass

        self._nc_icobi_records = []
        ni = ldir / 'NcICOBILIST.lobster'
        if ni.exists():
            try:
                self._nc_icobi_records = parse_ncicobi_list(ni)
            except Exception:
                pass

        self._nc_cobicar_header = None
        nc = ldir / 'COBICAR.lobster'
        if nc.exists():
            try:
                self._nc_cobicar_header = parse_car_header(nc)
            except Exception:
                pass

        self._lbl_status.setText(f'POSCAR: {ldir.name}/POSCAR  ✓')
        self._run_detection()

    def _on_directive_clicked(self, item):
        directive = item.text()
        idxs = self._directive_lookup.get(directive)
        if not idxs:
            self._selection_bar.setText(f'No chain data for: {directive}')
            return

        if self._supercell is not None:
            try:
                sp = [self._supercell.species(i) for i in idxs]
            except Exception:
                sp = ['?'] * len(idxs)
            label = '  →  '.join(f'{s}({i})' for s, i in zip(sp, idxs))
            self._selection_bar.setText(
                f'Directive ({len(idxs)}-center):  {label}'
            )

        if self._supercell is not None and len(idxs) >= 2:
            chain_pairs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
            self.structure_view.highlight_bonds(
                chain_pairs, center_on=idxs[0], highlight_atoms=True)

        # Highlight the trigger pair in the pFC scatter plot
        trigger = self._directive_trigger.get(directive)
        if trigger and self._plot_points:
            tkey = (min(int(trigger['atom1_idx']), int(trigger['atom2_idx'])),
                    max(int(trigger['atom1_idx']), int(trigger['atom2_idx'])))
            for pt in self._plot_points:
                rec = pt[2]
                rkey = (min(int(rec['atom1_idx']), int(rec['atom2_idx'])),
                        max(int(rec['atom1_idx']), int(rec['atom2_idx'])))
                if rkey == tkey:
                    self._selected_record = rec
                    break
            self._refresh_plot()

        # NcICOBI popup — only open if the directive is found in NcICOBILIST
        if self._nc_icobi_records and self._lob_poscar is not None:
            from betapy.core.lobster import lookup_ncicobi, load_nccobicar_curves
            icobi = lookup_ncicobi(
                self._nc_icobi_records, directive, self._lob_poscar)
            if icobi is not None:
                curves = []
                if self._nc_cobicar_header is not None:
                    nc_path = Path(self._lobster_dir) / 'COBICAR.lobster'
                    curves = load_nccobicar_curves(
                        nc_path, self._nc_cobicar_header,
                        directive, self._lob_poscar)
                self._ensure_nc_viewer().show_result(directive, icobi, curves)

    def _ensure_nc_viewer(self):
        if self._nc_viewer is None:
            self._nc_viewer = _NcCobiViewerWidget(self)
        return self._nc_viewer

    def _copy_directives(self):
        lines = [self._directive_list.item(i).text()
                 for i in range(self._directive_list.count())]
        text = '\n'.join(lines).strip()
        if text:
            QApplication.clipboard().setText(text)

    def _save_directives(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save directives',
            'multicenter_directives.txt',
            'Text files (*.txt);;All files (*)',
        )
        if path:
            lines = [self._directive_list.item(i).text()
                     for i in range(self._directive_list.count())]
            Path(path).write_text('\n'.join(lines).strip() + '\n')
