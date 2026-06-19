"""
Badger Analysis viewer — experimental feature.

2×2 layout:
  Top-left     : conventional Φ_p^{-1/3} vs r  (projected)
  Top-right    : isotropic   Φ_iso^{-1/3} vs r  (rotationally invariant)
  Bottom-left  : conventional scatter coloured by ξ = Φ_p / Φ_iso
  Bottom-right : 3D structure view — clicking a point in the ξ scatter
                 highlights the corresponding bond
"""

import math
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox, QProgressBar,
    QCheckBox, QScrollArea, QMessageBox, QFrame, QSpinBox,
)
from PyQt5.QtCore import Qt, QThread, QSize, pyqtSignal

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from betapy.core.constants import EV_ANG2_TO_N_M, UNIT_LABEL, UNIT_EV, UNIT_NM
from betapy.gui.structure_view import StructureView

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]
# Distinct palette for per-family Badger lines (warm/vivid, contrast with tab10)
_FAMILY_COLORS = [
    '#e41a1c', '#ff7f00', '#4daf4a', '#984ea3',
    '#a65628', '#f781bf', '#377eb8', '#999999',
]
_MARKERS      = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
_SCATTER_ALPHA = 0.55
_SCATTER_SIZE  = 10
_LINE_ALPHA    = 0.80
_LINE_WIDTH    = 1.5
_PICK_TOL      = 0.025   # fraction of axis range for click detection


def _sp_family_color(sp_key, fid, color_map, max_fid):
    """
    Derive a family-specific shade of the species-pair base color.

    Family 0 → darkest/most saturated; family max_fid → brightest/lightest.
    This lets both species-pair and family be read from color alone.
    """
    import colorsys
    base = color_map.get(sp_key, '#888888')
    r = int(base[1:3], 16) / 255.0
    g = int(base[3:5], 16) / 255.0
    b = int(base[5:7], 16) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    t = fid / max(max_fid, 1)          # 0 (stiffest family) → 1 (softest)
    v2 = 0.40 + 0.55 * t              # brightness ramp dark → light
    s2 = max(0.25, s * (1.0 - 0.4 * t))  # slight desaturation toward lighter end
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s2, v2)
    return '#{:02x}{:02x}{:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _BadgerWorker(QThread):
    finished = pyqtSignal(object)   # BadgerAnalysisResult
    error    = pyqtSignal(str)

    def __init__(self, bulk_results, reliability_cutoff, n_families):
        super().__init__()
        self._bulk       = bulk_results
        self._limit      = reliability_cutoff
        self._n_families = n_families

    def run(self):
        try:
            from betapy.core.badger import analyze_badger
            reliable = [r for r in self._bulk
                        if r['distance'] <= self._limit]
            self.finished.emit(analyze_badger(reliable,
                                              n_families=self._n_families))
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class BadgerWidget(QWidget):
    """
    Self-contained Badger Analysis tab.

    Call load_data() after a bulk pFC analysis completes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bulk_results       = None
        self._reliability_cutoff = float('inf')
        self._supercell          = None
        self._result             = None
        self._worker             = None
        self._color_map          = {}
        self._pair_checkboxes    = {}
        self._unit               = UNIT_EV
        self._cbar               = None
        self._plot_points_fam    = []   # (r, pfc_inv_cbrt, record) for click detection
        self._all_plot_points    = []   # same, all species (for fast visibility filter)
        self._selected_record    = None
        self._ax_fam_ref         = None
        # Caches rebuilt on each full refresh; used for fast visibility toggling
        self._artists_by_sp      = {}   # sp_key → [matplotlib artists]
        self._fam_y_by_sp        = {}   # sp_key → [y-values] for ax_conv ylim recompute
        self._sp_keys_cache      = []   # result of _all_sp_keys(), cleared on new result
        self._family_color_cache = {}   # (sp_key, fid) → hex color
        self._max_fid_by_sp      = {}   # sp_key → max family id
        self._build_ui()

    # ------------------------------------------------------------------ public

    def load_data(self, bulk_results, reliability_cutoff=None, supercell=None):
        self._bulk_results = bulk_results
        self._reliability_cutoff = (
            reliability_cutoff
            if reliability_cutoff is not None and math.isfinite(reliability_cutoff)
            else float('inf')
        )
        if supercell is not None:
            self._supercell = supercell
            self.structure_view.load_supercell(supercell)
        self._btn_run.setEnabled(bool(bulk_results))
        self._run_analysis()

    def set_unit(self, unit: str):
        if unit != self._unit:
            self._unit = unit
            if self._result is not None:
                self._refresh_plot()

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        outer = QSplitter(Qt.Horizontal)
        root.addWidget(outer)

        # ── Left panel: controls ──────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(170)
        left.setMaximumWidth(250)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(6)

        self._btn_run = QPushButton('Run analysis')
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run_analysis)
        lv.addWidget(self._btn_run)

        fam_row = QHBoxLayout()
        fam_row.addWidget(QLabel('Families:'))
        self._spin_families = QSpinBox()
        self._spin_families.setRange(2, 15)
        self._spin_families.setValue(5)
        self._spin_families.setFixedWidth(52)
        self._spin_families.setToolTip(
            'Number of Badger families (k-means k).\n'
            'Increase to split the cloud into more lines;\n'
            'decrease to merge closely spaced lines.'
        )
        fam_row.addWidget(self._spin_families)
        fam_row.addStretch()
        lv.addLayout(fam_row)

        self._chk_iso = QCheckBox('Isotropic (top-right)')
        self._chk_iso.setChecked(True)
        self._chk_iso.setToolTip(
            'Checked: top-right shows Φ_iso = (|φ_l|+2|φ_t|)/3\n'
            'Unchecked: top-right shows conventional Φ_p\n'
            '(use to compare grouping improvement directly)'
        )
        self._chk_iso.toggled.connect(self._refresh_plot)
        lv.addWidget(self._chk_iso)

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

        pair_box = QGroupBox('Atom pair types')
        self._pair_layout = QVBoxLayout()
        self._pair_layout.setAlignment(Qt.AlignTop)
        pair_box.setLayout(self._pair_layout)
        pair_scroll = QScrollArea()
        pair_scroll.setWidget(pair_box)
        pair_scroll.setWidgetResizable(True)
        pair_scroll.setFixedHeight(110)
        lv.addWidget(pair_scroll)

        stats_box = QGroupBox('Fit quality  R²')
        sv = QVBoxLayout(stats_box)
        sv.setSpacing(2)
        self._stats_label = QLabel('—')
        self._stats_label.setWordWrap(True)
        self._stats_label.setAlignment(Qt.AlignTop)
        self._stats_label.setStyleSheet('font-size: 10px;')
        sv.addWidget(self._stats_label)
        lv.addWidget(stats_box)

        lv.addStretch()
        outer.addWidget(left)

        # ── Right panel: 2×2 grid ─────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(2)

        # Top figure: conv + iso side by side
        self._fig = Figure(constrained_layout=True)
        self._ax_conv, self._ax_iso = self._fig.subplots(1, 2)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, right)
        self._toolbar.setIconSize(QSize(24, 24))

        rv.addWidget(self._toolbar)
        rv.addWidget(self._canvas, stretch=1)

        # Bottom: family-space scatter (left) + 3D structure view (right)
        bot = QSplitter(Qt.Horizontal)

        fam_widget = QWidget()
        fv = QVBoxLayout(fam_widget)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.setSpacing(0)

        self._fig_fam = Figure(constrained_layout=True)
        self._ax_fam  = self._fig_fam.add_subplot(111)
        self._canvas_fam = FigureCanvas(self._fig_fam)
        self._canvas_fam.mpl_connect('button_press_event', self._on_fam_click)
        fv.addWidget(self._canvas_fam, stretch=1)

        bot.addWidget(fam_widget)

        self.structure_view = StructureView(self)
        bot.addWidget(self.structure_view)
        bot.setSizes([500, 500])
        bot.setStretchFactor(0, 1)
        bot.setStretchFactor(1, 1)

        rv.addWidget(bot, stretch=1)

        # Selection info bar
        self._sel_bar = QLabel('')
        self._sel_bar.setFrameStyle(QFrame.StyledPanel)
        self._sel_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._sel_bar.setFixedHeight(24)
        self._sel_bar.setStyleSheet('font-size: 10px;')
        rv.addWidget(self._sel_bar)

        outer.addWidget(right)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)

        self._draw_empty()

    def _draw_empty(self):
        for ax, title in [
            (self._ax_conv, 'Conventional   $\\Phi_p$'),
            (self._ax_iso,  'Isotropic   $(|\\phi_l|+2|\\phi_t|)/3$'),
        ]:
            ax.cla()
            ax.set_title(title, fontsize=11)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='#aaaaaa', fontsize=13)
            ax.set_axis_off()
        self._canvas.draw()

        self._ax_fam.cla()
        self._ax_fam.set_title('Badger scatter  (cos²θ)', fontsize=11)
        self._ax_fam.text(0.5, 0.5, 'No data', transform=self._ax_fam.transAxes,
                          ha='center', va='center', color='#aaaaaa', fontsize=13)
        self._ax_fam.set_axis_off()
        self._canvas_fam.draw()

    # ------------------------------------------------------------------ worker

    def _run_analysis(self):
        if not self._bulk_results:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText('Fitting Badger lines…')
        self._worker = _BadgerWorker(self._bulk_results, self._reliability_cutoff,
                                     self._spin_families.value())
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._result = result
        self._sp_keys_cache = []   # force recompute for new result
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._build_color_map()
        self._rebuild_pair_checkboxes()
        self._update_stats_label()
        self._refresh_plot()
        self._lbl_status.setText(f'{len(result.records)} pairs analysed.')

    def _on_error(self, msg):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._lbl_status.setText('Analysis failed.')
        QMessageBox.critical(self, 'Badger analysis error', msg)

    # ------------------------------------------------------------------ helpers

    def _all_sp_keys(self):
        if self._result is None:
            return []
        if self._sp_keys_cache:
            return self._sp_keys_cache
        keys = set()
        for r in self._result.records:
            keys.add(tuple(sorted([r['species1'], r['species2']])))
        self._sp_keys_cache = sorted(keys)
        return self._sp_keys_cache

    def _active_sp_keys(self):
        return [k for k, cb in self._pair_checkboxes.items() if cb.isChecked()]

    def _build_color_map(self):
        for i, sp_key in enumerate(self._all_sp_keys()):
            if sp_key not in self._color_map:
                self._color_map[sp_key] = _COLORS[i % len(_COLORS)]

    def _rebuild_pair_checkboxes(self):
        while self._pair_layout.count():
            item = self._pair_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._pair_checkboxes.clear()

        for sp_key in self._all_sp_keys():
            color = self._color_map.get(sp_key, '#888')
            cb = QCheckBox(f'{sp_key[0]}–{sp_key[1]}')
            cb.setChecked(True)
            cb.setStyleSheet(f'color: {color}; font-weight: bold;')
            cb.stateChanged.connect(self._update_sp_visibility)
            self._pair_checkboxes[sp_key] = cb
            self._pair_layout.addWidget(cb)

    def _update_stats_label(self):
        if self._result is None:
            self._stats_label.setText('—')
            return
        lines = []
        for sp_key in self._all_sp_keys():
            label = f'{sp_key[0]}–{sp_key[1]}'
            conv_shells = self._result.conv_fits.get(sp_key, [])
            iso_shells  = self._result.iso_fits.get(sp_key, [])
            cr2 = conv_shells[0]['r2_robust'] if conv_shells else float('nan')
            ir2 = iso_shells[0]['r2_robust']  if iso_shells  else float('nan')
            if math.isfinite(cr2) and math.isfinite(ir2):
                delta = ir2 - cr2
                sign  = '+' if delta >= 0 else ''
                lines.append(
                    f'<b>{label}</b><br>'
                    f'&nbsp;&nbsp;conv R²={cr2:.3f}<br>'
                    f'&nbsp;&nbsp;iso&nbsp; R²={ir2:.3f}<br>'
                    f'&nbsp;&nbsp;Δ ={sign}{delta:.3f}'
                )
            else:
                lines.append(f'<b>{label}</b>  (no fit)')
        self._stats_label.setText('<br>'.join(lines))

    # ------------------------------------------------------------------ plot

    def _update_sp_visibility(self):
        """Fast path: toggle artist visibility without rebuilding anything."""
        if not self._artists_by_sp:
            return
        active = set(self._active_sp_keys())
        for sp_key, artists in self._artists_by_sp.items():
            vis = sp_key in active
            for a in artists:
                a.set_visible(vis)
        # Recompute ax_conv y-limits from visible scatter data only
        visible_ys = []
        for sp_key in active:
            visible_ys.extend(self._fam_y_by_sp.get(sp_key, []))
        if visible_ys:
            y_lo, y_hi = min(visible_ys), max(visible_ys)
            margin = (y_hi - y_lo) * 0.08
            self._ax_conv.set_ylim(y_lo - margin, y_hi + margin)
        # Update click-detection list to active species only
        self._plot_points_fam = [
            pt for pt in self._all_plot_points
            if tuple(sorted([pt[2]['species1'], pt[2]['species2']])) in active
        ]
        self._canvas.draw_idle()
        self._canvas_fam.draw_idle()

    def _refresh_plot(self):
        if self._result is None:
            return

        # Reset per-refresh caches
        self._artists_by_sp.clear()
        self._fam_y_by_sp.clear()
        self._family_color_cache.clear()

        from collections import defaultdict as _dd

        scale      = EV_ANG2_TO_N_M if self._unit == UNIT_NM else 1.0
        cbrt_scale = scale ** (-1.0 / 3.0)
        unit_lbl   = UNIT_LABEL[self._unit]
        y_label    = f'Φ$^{{-1/3}}$  [{unit_lbl}]$^{{-1/3}}$'
        active     = set(self._active_sp_keys())
        all_keys   = self._all_sp_keys()

        # Pre-group records by species pair once — avoids O(n²) repeated scans
        recs_by_sp = _dd(list)
        for r in self._result.records:
            recs_by_sp[tuple(sorted([r['species1'], r['species2']]))].append(r)

        # ── Top row: conventional and isotropic/reference Badger ──────────
        use_iso     = self._chk_iso.isChecked()
        iso_val_key = 'phi_iso'    if use_iso else 'mean_pfc'
        iso_fits    = self._result.iso_fits if use_iso else self._result.conv_fits
        iso_title   = ('Isotropic   $(|\\phi_l|+2|\\phi_t|)/3$'
                       if use_iso else 'Conventional   $\\Phi_p$  (reference)')

        for ax, fits, val_key, title, show_global_fit, draw_scatter in [
            (self._ax_conv, self._result.conv_fits, 'mean_pfc',
             'Conventional   $\\Phi_p$', False, False),
            (self._ax_iso,  iso_fits, iso_val_key, iso_title, True, True),
        ]:
            ax.cla()
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('r  (Å)')
            ax.set_ylabel(y_label)
            ax.tick_params(labelsize=9)

            for sp_key in all_keys:
                color = self._color_map.get(sp_key, '#888888')
                label = f'{sp_key[0]}–{sp_key[1]}'
                vis   = sp_key in active

                recs = [r for r in recs_by_sp[sp_key]
                        if math.isfinite(r.get(val_key, float('nan')))
                        and r[val_key] > 0]
                if not recs:
                    continue

                xs = np.array([r['distance']                          for r in recs])
                ys = np.array([r[val_key] ** (-1.0/3.0) * cbrt_scale for r in recs])

                if draw_scatter:
                    sc = ax.scatter(xs, ys, color=color, alpha=_SCATTER_ALPHA,
                                    s=_SCATTER_SIZE, label=label, zorder=3)
                    sc.set_visible(vis)
                    self._artists_by_sp.setdefault(sp_key, []).append(sc)

                if show_global_fit and sp_key in fits:
                    shell_fits = fits[sp_key]
                    if shell_fits:
                        sf = shell_fits[0]
                        x0, x1 = sf['r_min'], sf['r_max']
                        xx = np.linspace(x0, x1, 80)
                        yy = (sf['slope'] * xx + sf['intercept']) * cbrt_scale
                        ln, = ax.plot(xx, yy, color=color, alpha=_LINE_ALPHA * 0.6,
                                      lw=_LINE_WIDTH, ls='--', zorder=4)
                        ln.set_visible(vis)
                        self._artists_by_sp.setdefault(sp_key, []).append(ln)

            if draw_scatter:
                ax.legend(fontsize=8, markerscale=1.5, framealpha=0.7)

        # ── Conventional Badger: family-coloured scatter + extended fit lines ──
        if self._result.family_fits:
            from matplotlib.path import Path
            from matplotlib.lines import Line2D

            def _wedge_path(a1, a2, n=20):
                t = np.linspace(a1, a2, n)
                verts = ([(0., 0.)]
                         + list(zip(0.5 * np.cos(t), 0.5 * np.sin(t)))
                         + [(0., 0.)])
                codes = [Path.MOVETO] + [Path.LINETO] * n + [Path.CLOSEPOLY]
                return Path(verts, codes)

            # Precompute max family id and all family colors once
            max_fid_by_sp = _dd(int)
            for (sk, fid) in self._result.family_fits:
                max_fid_by_sp[sk] = max(max_fid_by_sp[sk], fid)
            self._max_fid_by_sp = dict(max_fid_by_sp)

            for (sp_key, fid) in self._result.family_fits:
                key = (sp_key, fid)
                if key not in self._family_color_cache:
                    self._family_color_cache[key] = _sp_family_color(
                        sp_key, fid, self._color_map, max_fid_by_sp[sp_key])

            # Deduplicate pairs; batch single-family points by (sp_key, fid)
            seen_pairs      = set()
            single_by_sp_fid = _dd(lambda: [[], []])
            multi_pts       = []

            for r in self._result.records:
                sp_key = tuple(sorted([r['species1'], r['species2']]))
                val    = r.get('mean_pfc', float('nan'))
                if not math.isfinite(val) or val <= 0:
                    continue
                pair_key = (min(r['atom1_idx'], r['atom2_idx']),
                            max(r['atom1_idx'], r['atom2_idx']))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                fids = [f for f in r.get('family_ids', [-1]) if f >= 0]
                y = val ** (-1.0/3.0) * cbrt_scale
                self._fam_y_by_sp.setdefault(sp_key, []).append(y)
                if len(fids) == 1:
                    single_by_sp_fid[(sp_key, fids[0])][0].append(r['distance'])
                    single_by_sp_fid[(sp_key, fids[0])][1].append(y)
                elif len(fids) > 1:
                    multi_pts.append((r['distance'], y, sp_key, fids))

            # Single-family scatter: one call per (sp_key, fid)
            for (sp_key, fid), xy in single_by_sp_fid.items():
                color = self._family_color_cache.get((sp_key, fid), '#888888')
                sc = self._ax_conv.scatter(xy[0], xy[1], color=color,
                                           alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE + 4,
                                           linewidths=0, zorder=3)
                sc.set_visible(sp_key in active)
                self._artists_by_sp.setdefault(sp_key, []).append(sc)

            # Multi-family wedge markers
            for x_pt, y_pt, sp_key, fids in multi_pts:
                n_col  = len(fids)
                angles = np.linspace(0, 2 * np.pi, n_col + 1)
                for i, fid in enumerate(fids):
                    color = self._family_color_cache.get((sp_key, fid), '#888888')
                    wp    = _wedge_path(angles[i], angles[i + 1])
                    sc    = self._ax_conv.scatter([x_pt], [y_pt], marker=wp,
                                                  color=color,
                                                  s=(_SCATTER_SIZE + 4) * 2,
                                                  alpha=_SCATTER_ALPHA,
                                                  linewidths=0, zorder=3)
                    sc.set_visible(sp_key in active)
                    self._artists_by_sp.setdefault(sp_key, []).append(sc)

            # Extended fit lines
            all_r   = [r['distance'] for r in self._result.records]
            x_right = (max(all_r) * 1.03) if all_r else 10.0

            legend_handles   = []
            legend_keys_seen = set()
            for (sp_key, fid), ffit in sorted(self._result.family_fits.items()):
                color  = self._family_color_cache.get((sp_key, fid), '#888888')
                r0, r1 = ffit['r_min'], ffit['r_max']
                a, b   = ffit['slope'], ffit['intercept']
                x_left = ffit.get('r_anchor', r0) * 0.90
                vis    = sp_key in active

                if x_left < r0:
                    xx = np.linspace(x_left, r0, 40)
                    ln, = self._ax_conv.plot(xx, (a * xx + b) * cbrt_scale,
                                             color=color, alpha=_LINE_ALPHA * 0.45,
                                             lw=1.5, ls='--', zorder=4)
                    ln.set_visible(vis)
                    self._artists_by_sp.setdefault(sp_key, []).append(ln)

                xx = np.linspace(r0, min(r1, x_right), 60)
                ln, = self._ax_conv.plot(xx, (a * xx + b) * cbrt_scale,
                                         color=color, alpha=_LINE_ALPHA,
                                         lw=2.0, zorder=5)
                ln.set_visible(vis)
                self._artists_by_sp.setdefault(sp_key, []).append(ln)

                if r1 < x_right:
                    xx = np.linspace(r1, x_right, 40)
                    ln, = self._ax_conv.plot(xx, (a * xx + b) * cbrt_scale,
                                             color=color, alpha=_LINE_ALPHA * 0.45,
                                             lw=1.5, ls='--', zorder=4)
                    ln.set_visible(vis)
                    self._artists_by_sp.setdefault(sp_key, []).append(ln)

                leg_key = (sp_key, fid)
                if leg_key not in legend_keys_seen:
                    legend_keys_seen.add(leg_key)
                    sp_label = f'{sp_key[0]}–{sp_key[1]}' if len(all_keys) > 1 else ''
                    label    = f'{sp_label} F{fid}' if sp_label else f'Fam {fid}'
                    legend_handles.append(
                        Line2D([0], [0], color=color, lw=2.0,
                               label=label, alpha=_LINE_ALPHA)
                    )

            if legend_handles:
                self._ax_conv.legend(handles=legend_handles, fontsize=7,
                                     framealpha=0.7, title='family',
                                     title_fontsize=7, ncol=max(1, len(all_keys)))

            # Clamp y-axis to visible scatter range
            visible_ys = []
            for sp_key in active:
                visible_ys.extend(self._fam_y_by_sp.get(sp_key, []))
            if visible_ys:
                y_lo, y_hi = min(visible_ys), max(visible_ys)
                margin = (y_hi - y_lo) * 0.08
                self._ax_conv.set_ylim(y_lo - margin, y_hi + margin)

        self._canvas.draw()

        # ── Bottom-left: Badger scatter coloured by cos²θ ────────────────
        ax = self._ax_fam
        ax.cla()
        ax.set_title('Badger scatter  (cos²θ)', fontsize=11)
        ax.set_xlabel('r  (Å)')
        ax.set_ylabel(y_label)
        ax.tick_params(labelsize=9)

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self._all_plot_points = []
        self._ax_fam_ref = ax

        fam_recs = [
            r for r in self._result.records
            if math.isfinite(r.get('cos2theta', float('nan')))
            and math.isfinite(r.get('mean_pfc',  float('nan')))
            and r.get('mean_pfc', 0) > 0
        ]

        if fam_recs:
            import matplotlib.colors as mcolors
            from matplotlib.lines import Line2D

            cmap     = 'plasma'
            all_cos2 = [r['cos2theta'] for r in fam_recs]
            c_lo     = float(np.percentile(all_cos2,  2))
            c_hi     = float(np.percentile(all_cos2, 98))
            if c_hi <= c_lo:
                c_lo, c_hi = min(all_cos2), max(all_cos2)
            norm_c = mcolors.Normalize(vmin=c_lo, vmax=c_hi)

            last_sc        = None
            legend_handles = []
            fam_recs_by_sp = _dd(list)
            for r in fam_recs:
                fam_recs_by_sp[tuple(sorted([r['species1'], r['species2']]))].append(r)

            for i, sp_key in enumerate(all_keys):
                recs = fam_recs_by_sp.get(sp_key, [])
                if not recs:
                    continue
                vis    = sp_key in active
                xs     = np.array([r['distance']                             for r in recs])
                ys     = np.array([r['mean_pfc'] ** (-1.0/3.0) * cbrt_scale  for r in recs])
                cs     = np.array([r['cos2theta']                             for r in recs])
                marker = _MARKERS[i % len(_MARKERS)]

                sc = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm_c,
                                marker=marker, alpha=_SCATTER_ALPHA,
                                s=_SCATTER_SIZE + 4, zorder=3)
                sc.set_visible(vis)
                self._artists_by_sp.setdefault(sp_key, []).append(sc)
                last_sc = sc

                for x_pt, y_pt, rec in zip(xs, ys, recs):
                    self._all_plot_points.append((float(x_pt), float(y_pt), rec))

                legend_handles.append(
                    Line2D([0], [0], marker=marker, color='w',
                           markerfacecolor='#666666', markersize=6,
                           label=f'{sp_key[0]}–{sp_key[1]}', alpha=0.85)
                )

            # Iso fit reference lines (dotted grey)
            for sp_key in all_keys:
                vis = sp_key in active
                for shell_fit in self._result.iso_fits.get(sp_key, []):
                    x0, x1 = shell_fit['r_min'], shell_fit['r_max']
                    xx = np.linspace(x0, x1, 80)
                    yy = (shell_fit['slope'] * xx + shell_fit['intercept']) * cbrt_scale
                    ln, = ax.plot(xx, yy, color='#888888', alpha=0.45,
                                  lw=1.0, ls=':', zorder=2)
                    ln.set_visible(vis)
                    self._artists_by_sp.setdefault(sp_key, []).append(ln)

            # Selection ring (not per-sp-key)
            if self._selected_record is not None:
                sr    = self._selected_record
                pfc_v = sr.get('mean_pfc', float('nan'))
                d_v   = sr.get('distance', float('nan'))
                if math.isfinite(pfc_v) and pfc_v > 0 and math.isfinite(d_v):
                    y_sel = pfc_v ** (-1.0/3.0) * cbrt_scale
                    ax.plot(d_v, y_sel, 'o', markersize=16,
                            markerfacecolor='none', markeredgecolor='#c8a000',
                            markeredgewidth=2.5, zorder=6)

            if last_sc is not None:
                self._cbar = self._fig_fam.colorbar(last_sc, ax=ax, pad=0.01,
                                                     fraction=0.035)
                self._cbar.set_label(f'cos²θ  [{c_lo:.2f}–{c_hi:.2f}]', fontsize=9)
                self._cbar.ax.tick_params(labelsize=8)

            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=8,
                          framealpha=0.7, title='pair type', title_fontsize=7)

        # Populate click-detection list for active species only
        self._plot_points_fam = [
            pt for pt in self._all_plot_points
            if tuple(sorted([pt[2]['species1'], pt[2]['species2']])) in active
        ]
        self._canvas_fam.draw()

    # ------------------------------------------------------------------ click

    def _on_fam_click(self, event):
        if event.inaxes is None or self._ax_fam_ref is None:
            return
        if event.button != 1 or not self._plot_points_fam:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return

        xlim = self._ax_fam_ref.get_xlim()
        ylim = self._ax_fam_ref.get_ylim()
        xs = (xlim[1] - xlim[0]) or 1.0
        ys = (ylim[1] - ylim[0]) or 1.0

        best_d2, best_pt = float('inf'), None
        for pt in self._plot_points_fam:
            d2 = ((pt[0] - cx) / xs) ** 2 + ((pt[1] - cy) / ys) ** 2
            if d2 < best_d2:
                best_d2, best_pt = d2, pt

        if best_pt is None or best_d2 ** 0.5 > _PICK_TOL:
            return

        _, _, rec = best_pt
        self._selected_record = rec

        a1    = int(rec['atom1_idx'])
        a2    = int(rec['atom2_idx'])
        sp1   = rec['species1']
        sp2   = rec['species2']
        d     = rec['distance']
        pfc   = rec['mean_pfc']
        cos2t = rec.get('cos2theta', float('nan'))
        xi    = rec.get('xi',        float('nan'))

        if self._supercell is not None:
            self.structure_view.highlight_bond(a1, a2)

        ct_str = f'{cos2t:.3f}' if math.isfinite(cos2t) else '—'
        xi_str = f'{xi:.3f}'    if math.isfinite(xi)    else '—'
        self._sel_bar.setText(
            f'atom {a1} ({sp1}) – atom {a2} ({sp2})   '
            f'd = {d:.4f} Å   '
            f'Φ_p = {pfc:.5f} {UNIT_LABEL[self._unit]}   '
            f'cos²θ = {ct_str}   ξ = {xi_str}'
        )

        self._refresh_plot()
