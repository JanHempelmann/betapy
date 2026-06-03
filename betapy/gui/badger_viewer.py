"""
Badger Analysis viewer — experimental feature.

2×2 layout:
  Top-left     : conventional Φ_p^{-1/3} vs r  (projected)
  Top-right    : isotropic   F_iso^{-1/3} vs r  (rotationally invariant)
  Bottom-left  : conventional scatter coloured by ξ = Φ_p / F_iso
  Bottom-right : 3D structure view — clicking a point in the ξ scatter
                 highlights the corresponding bond
"""

import math
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox, QProgressBar,
    QCheckBox, QScrollArea, QMessageBox, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

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
_MARKERS      = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
_SCATTER_ALPHA = 0.55
_SCATTER_SIZE  = 10
_LINE_ALPHA    = 0.80
_LINE_WIDTH    = 1.5
_PICK_TOL      = 0.025   # fraction of axis range for click detection


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _BadgerWorker(QThread):
    finished = pyqtSignal(object)   # BadgerAnalysisResult
    error    = pyqtSignal(str)

    def __init__(self, bulk_results, reliability_cutoff):
        super().__init__()
        self._bulk  = bulk_results
        self._limit = reliability_cutoff

    def run(self):
        try:
            from betapy.core.badger import analyze_badger
            reliable = [r for r in self._bulk
                        if r['distance'] <= self._limit]
            self.finished.emit(analyze_badger(reliable))
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
        self._plot_points_xi     = []   # (x, y_displayed, record) for click detection
        self._selected_record    = None
        self._ax_xi_ref          = None
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

        rv.addWidget(self._toolbar)
        rv.addWidget(self._canvas, stretch=1)

        # Bottom: ξ scatter (left) + 3D structure view (right)
        bot = QSplitter(Qt.Horizontal)

        xi_widget = QWidget()
        xv = QVBoxLayout(xi_widget)
        xv.setContentsMargins(0, 0, 0, 0)
        xv.setSpacing(0)

        self._fig_xi = Figure(constrained_layout=True)
        self._ax_xi  = self._fig_xi.add_subplot(111)
        self._canvas_xi = FigureCanvas(self._fig_xi)
        self._canvas_xi.mpl_connect('button_press_event', self._on_xi_click)
        xv.addWidget(self._canvas_xi, stretch=1)

        bot.addWidget(xi_widget)

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
            (self._ax_iso,  'Isotropic   $F_\\mathrm{iso}$'),
        ]:
            ax.cla()
            ax.set_title(title, fontsize=11)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='#aaaaaa', fontsize=13)
            ax.set_axis_off()
        self._canvas.draw()

        self._ax_xi.cla()
        self._ax_xi.set_title('Bond character   ξ  =  Φ_p / F_iso', fontsize=11)
        self._ax_xi.text(0.5, 0.5, 'No data', transform=self._ax_xi.transAxes,
                         ha='center', va='center', color='#aaaaaa', fontsize=13)
        self._ax_xi.set_axis_off()
        self._canvas_xi.draw()

    # ------------------------------------------------------------------ worker

    def _run_analysis(self):
        if not self._bulk_results:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText('Fitting Badger lines…')
        self._worker = _BadgerWorker(self._bulk_results, self._reliability_cutoff)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._result = result
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
        keys = set()
        for r in self._result.records:
            keys.add(tuple(sorted([r['species1'], r['species2']])))
        return sorted(keys)

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
            cb.stateChanged.connect(self._refresh_plot)
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

    def _refresh_plot(self):
        if self._result is None:
            return

        scale      = EV_ANG2_TO_N_M if self._unit == UNIT_NM else 1.0
        cbrt_scale = scale ** (-1.0 / 3.0)
        unit_lbl   = UNIT_LABEL[self._unit]
        y_label    = f'Φ$^{{-1/3}}$  [{unit_lbl}]$^{{-1/3}}$'
        active     = set(self._active_sp_keys())
        all_keys   = self._all_sp_keys()

        # ── Top row: conventional and isotropic Badger ─────────────────
        for ax, fits, val_key, title in [
            (self._ax_conv, self._result.conv_fits, 'mean_pfc',
             'Conventional   $\\Phi_p$'),
            (self._ax_iso,  self._result.iso_fits,  'f_iso',
             'Isotropic   $F_\\mathrm{iso}$'),
        ]:
            ax.cla()
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('r  (Å)')
            ax.set_ylabel(y_label)
            ax.tick_params(labelsize=9)

            for sp_key in all_keys:
                if sp_key not in active:
                    continue
                color = self._color_map.get(sp_key, '#888888')
                label = f'{sp_key[0]}–{sp_key[1]}'

                recs = [r for r in self._result.records
                        if tuple(sorted([r['species1'], r['species2']])) == sp_key
                        and math.isfinite(r.get(val_key, float('nan')))
                        and r[val_key] > 0]
                if not recs:
                    continue

                xs = np.array([r['distance']                          for r in recs])
                ys = np.array([r[val_key] ** (-1.0/3.0) * cbrt_scale for r in recs])
                ax.scatter(xs, ys, color=color, alpha=_SCATTER_ALPHA,
                           s=_SCATTER_SIZE, label=label, zorder=3)

                if sp_key in fits:
                    for shell_fit in fits[sp_key]:
                        x0, x1 = shell_fit['r_min'], shell_fit['r_max']
                        xx = np.linspace(x0, x1, 80)
                        yy = (shell_fit['slope'] * xx + shell_fit['intercept']) * cbrt_scale
                        ax.plot(xx, yy, color=color, alpha=_LINE_ALPHA,
                                lw=_LINE_WIDTH, ls='--', zorder=4)

            ax.legend(fontsize=8, markerscale=1.5, framealpha=0.7)

        self._canvas.draw()

        # ── Bottom-left: ξ scatter ─────────────────────────────────────
        ax = self._ax_xi
        ax.cla()
        ax.set_title('Bond character   ξ  =  Φ_p / F_iso', fontsize=11)
        ax.set_xlabel('r  (Å)')
        ax.set_ylabel(y_label)
        ax.tick_params(labelsize=9)

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self._plot_points_xi = []
        self._ax_xi_ref = ax

        all_xi = [
            r['xi'] for r in self._result.records
            if tuple(sorted([r['species1'], r['species2']])) in active
            and math.isfinite(r.get('xi', float('nan'))) and r['xi'] > 0
            and math.isfinite(r.get('mean_pfc', float('nan'))) and r['mean_pfc'] > 0
        ]

        if all_xi:
            import matplotlib.colors as mcolors
            from matplotlib.lines import Line2D

            xi_arr = np.array(all_xi)
            vmin = max(float(np.percentile(xi_arr,  2)), 1e-2)
            vmax = max(float(np.percentile(xi_arr, 98)), vmin * 2.0)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = 'plasma'

            last_sc      = None
            legend_handles = []

            for i, sp_key in enumerate(all_keys):
                if sp_key not in active:
                    continue
                recs = [r for r in self._result.records
                        if tuple(sorted([r['species1'], r['species2']])) == sp_key
                        and math.isfinite(r.get('mean_pfc', float('nan')))
                        and r['mean_pfc'] > 0
                        and math.isfinite(r.get('xi', float('nan')))
                        and r['xi'] > 0]
                if not recs:
                    continue

                xs     = np.array([r['distance']                              for r in recs])
                ys     = np.array([r['mean_pfc'] ** (-1.0/3.0) * cbrt_scale  for r in recs])
                xi_pts = np.array([r['xi']                                    for r in recs])
                marker = _MARKERS[i % len(_MARKERS)]

                last_sc = ax.scatter(xs, ys, c=xi_pts, cmap=cmap, norm=norm,
                                     marker=marker, alpha=_SCATTER_ALPHA,
                                     s=_SCATTER_SIZE + 4, zorder=3)

                for x_pt, y_pt, rec in zip(xs, ys, recs):
                    self._plot_points_xi.append((float(x_pt), float(y_pt), rec))

                legend_handles.append(
                    Line2D([0], [0], marker=marker, color='w',
                           markerfacecolor='#666666', markersize=6,
                           label=f'{sp_key[0]}–{sp_key[1]}', alpha=0.85)
                )

            # Selection ring for currently selected record
            if self._selected_record is not None:
                sr = self._selected_record
                if math.isfinite(sr.get('mean_pfc', float('nan'))) and sr['mean_pfc'] > 0:
                    sy = sr['mean_pfc'] ** (-1.0/3.0) * cbrt_scale
                    ax.plot(sr['distance'], sy, 'o', markersize=16,
                            markerfacecolor='none', markeredgecolor='#c8a000',
                            markeredgewidth=2.5, zorder=6)

            if last_sc is not None:
                self._cbar = self._fig_xi.colorbar(last_sc, ax=ax, pad=0.01,
                                                    fraction=0.035)
                self._cbar.set_label('ξ  (log scale)', fontsize=9)
                self._cbar.ax.tick_params(labelsize=8)

            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=8,
                          framealpha=0.7, title='pair type', title_fontsize=7)

        self._canvas_xi.draw()

    # ------------------------------------------------------------------ click

    def _on_xi_click(self, event):
        if event.inaxes is None or self._ax_xi_ref is None:
            return
        if event.button != 1 or not self._plot_points_xi:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return

        xlim = self._ax_xi_ref.get_xlim()
        ylim = self._ax_xi_ref.get_ylim()
        xs = (xlim[1] - xlim[0]) or 1.0
        ys = (ylim[1] - ylim[0]) or 1.0

        best_d2, best_pt = float('inf'), None
        for pt in self._plot_points_xi:
            d2 = ((pt[0] - cx) / xs) ** 2 + ((pt[1] - cy) / ys) ** 2
            if d2 < best_d2:
                best_d2, best_pt = d2, pt

        if best_pt is None or best_d2 ** 0.5 > _PICK_TOL:
            return

        _, _, rec = best_pt
        self._selected_record = rec

        a1  = int(rec['atom1_idx'])
        a2  = int(rec['atom2_idx'])
        sp1 = rec['species1']
        sp2 = rec['species2']
        d   = rec['distance']
        pfc = rec['mean_pfc']
        xi  = rec.get('xi', float('nan'))

        if self._supercell is not None:
            self.structure_view.highlight_bond(a1, a2)

        xi_str = f'{xi:.3f}' if math.isfinite(xi) else '—'
        self._sel_bar.setText(
            f'atom {a1} ({sp1}) – atom {a2} ({sp2})   '
            f'd = {d:.4f} Å   '
            f'Φ_p = {pfc:.5f} {UNIT_LABEL[self._unit]}   '
            f'ξ = {xi_str}'
        )

        self._refresh_plot()
