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
    QMessageBox, QCheckBox, QScrollArea, QFrame,
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
_FLAG_COLOR  = '#d03030'
_FLAG_ALPHA  = 0.90
_CURVE_ALPHA = 0.65
_BAND_ALPHA  = 0.10
_PICK_TOL    = 0.025     # fraction of axis range for click detection


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
                 n_sigma, max_order, min_angle):
        super().__init__()
        self._bulk   = bulk_results
        self._sc     = supercell
        self._poscar = poscar_path
        self._sigma  = n_sigma
        self._order  = max_order
        self._angle  = min_angle

    def run(self):
        try:
            from betapy.core.multicenter import _split_into_shells
            from scipy.stats import theilslopes, median_abs_deviation

            reliability_limit = (
                min(np.linalg.norm(v) for v in self._sc.lattice) / 2.0
            )
            reliable = [r for r in self._bulk
                        if r['distance'] <= reliability_limit]

            # Deduplicate (full FC lists every bond twice)
            seen: set = set()
            deduped: list = []
            for r in reliable:
                key = (min(r['atom1_idx'], r['atom2_idx']),
                       max(r['atom1_idx'], r['atom2_idx']))
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)

            # Group by species pair, split into distance shells, fit Theil-Sen.
            # Single-distance shells (x_range < 0.05 Å) and perfect-fit shells
            # (MAD ≈ 0) appear in the scatter but receive no Badger curve.
            by_pair: dict = defaultdict(list)
            for r in deduped:
                sp_key = tuple(sorted([r['species1'], r['species2']]))
                by_pair[sp_key].append(r)

            sp_acc: dict = defaultdict(lambda: {'records': [], 'badger_lines': []})

            for sp_key, records in by_pair.items():
                for shell_records in _split_into_shells(records):
                    pfcs  = np.array([r['mean_pfc'] for r in shell_records])
                    dists = np.array([r['distance']  for r in shell_records])
                    valid = pfcs > 0
                    if valid.sum() < 2:
                        continue
                    v_pfcs    = pfcs[valid]
                    v_dists   = dists[valid]
                    v_records = [shell_records[i]
                                 for i in range(len(shell_records)) if valid[i]]

                    acc = sp_acc[sp_key]
                    acc['records'].extend(v_records)

                    if float(v_dists.max() - v_dists.min()) < 0.05:
                        continue

                    inv_cbrt = v_pfcs ** (-1.0 / 3.0)
                    if valid.sum() >= 4:
                        slope, intercept, *_ = theilslopes(inv_cbrt, v_dists)
                        residuals = inv_cbrt - (slope * v_dists + intercept)
                        std_raw = float(median_abs_deviation(residuals) * 1.4826)
                        if std_raw < 1e-8:
                            continue  # perfect fit, no meaningful scatter
                        std = max(std_raw, 1e-6)
                    else:
                        slope = intercept = std = float('nan')

                    acc['badger_lines'].append({
                        'slope':     float(slope),
                        'intercept': float(intercept),
                        'std':       float(std),
                        'x_min':     float(v_dists.min()),
                        'x_max':     float(v_dists.max()),
                    })

            # Build final badger_data keyed by species pair
            badger_data: dict = {}
            for sp_key, acc in sp_acc.items():
                v_recs  = acc['records']
                v_dists = np.array([r['distance'] for r in v_recs])
                v_pfcs  = np.array([r['mean_pfc']  for r in v_recs])
                lines   = acc['badger_lines']
                first   = lines[0] if lines else {}
                badger_data[sp_key] = {
                    'distances':   v_dists,
                    'pfcs':        v_pfcs,
                    'flagged':     np.zeros(len(v_recs), dtype=bool),
                    'n_flagged':   0,
                    'records':     v_recs,
                    'badger_lines': lines,
                    # Legacy scalar fields kept for backward compatibility
                    'slope':       first.get('slope',     float('nan')),
                    'intercept':   first.get('intercept', float('nan')),
                    'std':         first.get('std',       float('nan')),
                }

            # Directive pipeline (needs POSCAR for chain detection)
            if self._poscar is not None:
                from betapy.core.multicenter import suggest_cobi_directives
                result = suggest_cobi_directives(
                    self._bulk, self._sc, self._poscar,
                    n_sigma=self._sigma,
                    max_order=self._order,
                    min_angle_deg=self._angle,
                )
            else:
                from betapy.core.multicenter import detect_anomalous_pairs
                flagged_pairs = detect_anomalous_pairs(
                    reliable, n_sigma=self._sigma)
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
        self._ax                = None
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
        self._spin_sigma.setValue(2.0)
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

        # ── Directives box ────────────────────────────────────────────
        dir_box = QGroupBox('cobiBetween directives')
        dv = QVBoxLayout(dir_box)
        dv.setSpacing(4)

        self._lbl_count = QLabel('—')
        self._lbl_count.setStyleSheet('font-size: 10px; color: #444;')
        dv.addWidget(self._lbl_count)

        self._directive_list = QListWidget()
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(9)
        self._directive_list.setFont(mono)
        self._directive_list.setToolTip('Click a directive to highlight its chain in 3D.')
        self._directive_list.itemClicked.connect(self._on_directive_clicked)
        dv.addWidget(self._directive_list, stretch=1)

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
        dv.addLayout(btn_row)

        lv.addWidget(dir_box, stretch=1)
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

    def load_data(self, bulk_results, supercell, lobster_dir=None):
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

        self._bulk_results = bulk_results
        self._supercell    = supercell
        self._lobster_dir  = lobster_dir
        self._poscar_path  = None

        self.structure_view.load_supercell(supercell)

        if lobster_dir is not None:
            p = Path(lobster_dir) / 'POSCAR'
            if p.exists():
                self._poscar_path = p

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
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._result = result

        # Override the worker's independent per-pair flagging with the
        # authoritative result from detect_anomalous_pairs.  The worker has no
        # monotone fallback (used when a species pair has < min_pairs valid
        # records), so bonds caught only by that path would otherwise appear in
        # the directives but not be coloured red in the scatter plot.
        flagged_keys = {
            (min(int(r['atom1_idx']), int(r['atom2_idx'])),
             max(int(r['atom1_idx']), int(r['atom2_idx'])))
            for r in result.get('flagged_pairs', [])
        }
        for bd in result['badger_data'].values():
            corrected = np.array([
                (min(int(rec['atom1_idx']), int(rec['atom2_idx'])),
                 max(int(rec['atom1_idx']), int(rec['atom2_idx']))) in flagged_keys
                for rec in bd['records']
            ])
            bd['flagged']   = corrected
            bd['n_flagged'] = int(corrected.sum())

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

        self._rebuild_checkboxes(result['badger_data'])
        self._refresh_plot()
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
                'Run detection to see the pFC vs r plot\nwith Badger curve overlay.',
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
        n_sigma = self._spin_sigma.value()
        _effective_n = n_sigma

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._ax = ax
        self._plot_points = []

        # Separate points into rendering layers
        gray_xs: list = []
        gray_ys: list = []
        colored: dict = defaultdict(lambda: {'xs': [], 'ys': []})
        flag_xs: list = []
        flag_ys: list = []

        for pk in self._pair_keys:
            if pk not in badger_data:
                continue
            bd        = badger_data[pk]
            dists     = bd['distances']
            pfcs      = bd['pfcs']
            flagged   = bd['flagged']
            records   = bd['records']
            is_checked = pk in checked_pairs

            for d, pfc, rec, fl in zip(dists, pfcs, records, flagged):
                self._plot_points.append((d, pfc, rec, bool(fl), pk))
                if not is_checked:
                    gray_xs.append(d)
                    gray_ys.append(pfc)
                elif fl:
                    flag_xs.append(d)
                    flag_ys.append(pfc)
                else:
                    colored[pk]['xs'].append(d)
                    colored[pk]['ys'].append(pfc)

        # Layer 1: unchecked-pair points (grey, provides visual context)
        if gray_xs:
            ax.scatter(gray_xs, gray_ys, s=18, c=_GREY_COLOR,
                       alpha=_GREY_ALPHA, linewidths=0, zorder=1)

        # Layer 2: Badger curves + bands (checked pairs only, under colored dots)
        for pk in self._pair_keys:
            if pk not in checked_pairs or pk not in badger_data:
                continue
            bd = badger_data[pk]
            sp1, sp2 = pk
            c1, _ = (self.structure_view.pair_colours_hex(sp1, sp2)
                     if self._supercell is not None else ('#888888', '#888888'))

            for bl in bd.get('badger_lines', []):
                slope, intercept, std = bl['slope'], bl['intercept'], bl['std']
                if math.isnan(slope) or math.isnan(std):
                    continue
                x_line   = np.linspace(bl['x_min'], bl['x_max'], 200)
                lin_base = slope * x_line + intercept
                # Clip to the x-range where both the Badger curve AND the
                # detection-band upper edge are positive.  Where lin_hi ≤ 0
                # the threshold maps to infinite pFC and is undefined.
                lin_hi_full = lin_base - _effective_n * std
                valid = (lin_base > 0) & (lin_hi_full > 0)
                if valid.any():
                    xc     = x_line[valid]
                    yc     = lin_base[valid] ** -3
                    lin_lo = lin_base[valid] + _effective_n * std
                    lin_hi = lin_hi_full[valid]
                    ax.plot(xc, yc, color=c1, lw=1.5, alpha=_CURVE_ALPHA,
                            linestyle='--', zorder=3)
                    ax.fill_between(xc, lin_lo ** -3, lin_hi ** -3,
                                    color=c1, alpha=_BAND_ALPHA, zorder=2)

        # Layer 3: checked-pair normal points (species color)
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

        # Layer 4: flagged (multicenter) points — float visibly above the curve
        if flag_xs:
            ax.scatter(flag_xs, flag_ys, s=60, c=_FLAG_COLOR,
                       alpha=_FLAG_ALPHA, linewidths=0.5,
                       edgecolors='#800000', zorder=5,
                       label=f'multicenter ({len(flag_xs)})')

        # Selection ring around the last clicked point
        if self._selected_record is not None:
            sr = self._selected_record
            ax.plot(
                sr['distance'], sr['mean_pfc'],
                'o', markersize=16,
                markerfacecolor='none', markeredgecolor='#c8a000',
                markeredgewidth=2.5, zorder=6,
            )

        # Pin y-axis to the actual data point range — the Badger curve can
        # reach extreme values near its left edge and would otherwise dominate.
        if self._plot_points:
            all_pfcs = [pt[1] for pt in self._plot_points]
            pfc_min  = min(all_pfcs)
            pfc_max  = max(all_pfcs)
            margin   = (pfc_max - pfc_min) * 0.08 if pfc_max > pfc_min else pfc_max * 0.1
            ax.set_ylim(max(0.0, pfc_min - margin), pfc_max + margin)

        ax.set_xlabel('Interatomic distance (Å)', fontsize=12)
        ax.set_ylabel('Projected force constant (eV/Å²)', fontsize=12)
        ax.set_title(
            'Multicenter bonding — pFC vs r  '
            '(dashed: Theil-Sen Badger fit, ±σ band)',
            fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)

        if colored or flag_xs:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

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

        best_d2, best_pt = float('inf'), None
        for pt in self._plot_points:
            d2 = ((pt[0] - cx) / xs) ** 2 + ((pt[1] - cy) / ys) ** 2
            if d2 < best_d2:
                best_d2, best_pt = d2, pt

        if best_pt is None or best_d2 ** 0.5 > _PICK_TOL:
            return

        _, _, rec, is_flagged, _ = best_pt
        self._selected_record    = rec

        a1  = int(rec['atom1_idx'])
        a2  = int(rec['atom2_idx'])
        sp1 = rec['species1']
        sp2 = rec['species2']
        d   = rec['distance']
        pfc = rec['mean_pfc']

        if self._supercell is not None:
            self.structure_view.highlight_bond(a1, a2)

        info = 'Multicenter' if is_flagged else 'Selected'
        self._selection_bar.setText(
            f'{info}:  atom {a1} ({sp1}) – atom {a2} ({sp2})   '
            f'd = {d:.4f} Å   pFC = {pfc:.6f} eV/Å²'
        )

        self._refresh_plot()

    # ------------------------------------------------------------------ directives

    def _update_directives(self, result):
        directives    = result.get('directives',    [])
        flagged_pairs = result.get('flagged_pairs', [])
        chains        = result.get('chains',        [])

        self._directive_list.clear()
        for d in directives:
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(d)
            # Tooltip: show order and species chain if known
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

        parts = [
            f'{len(flagged_pairs)} flagged pair(s)',
            f'{len(chains)} chain(s)',
            f'{len(directives)} unique directive(s)',
        ]
        if self._poscar_path is None:
            parts.append('(no POSCAR — directives suppressed)')
        self._lbl_count.setText('  ·  '.join(parts))

        has = bool(directives)
        self._btn_copy.setEnabled(has)
        self._btn_save.setEnabled(has)

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
