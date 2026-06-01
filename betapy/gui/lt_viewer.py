"""
LT Decomposition Viewer — longitudinal / transverse pFC decomposition.

Four-panel view for verification and exploration of bond-character anisotropy:
  Top-left  : φ_L (stretching) vs distance
  Top-right : φ_T (bending)    vs distance
  Bottom-left : θ = atan2(φ_T, φ_L)  vs distance  (bond-character angle)
  Bottom-right: φ_T vs φ_L cross-plot with ionic (A = -½) reference line
"""

import numpy as np
import matplotlib.transforms as mtransforms

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QScrollArea, QCheckBox,
    QPushButton, QFileDialog,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from betapy.core.constants import EV_ANG2_TO_N_M, UNIT_LABEL, UNIT_EV

# Reference angles for θ = atan2(φ_T, φ_L) in [0°, 360°)
# atan2 is scale-invariant so unit choice (eV/Å² vs N/m) doesn't matter
_TH_IONIC  = float(np.degrees(np.arctan2( 0.5, -1.0)))          # ≈ 153.43°  A = -½
_TH_RADIAL = 180.0                                                # 180.00°   A =  0
_TH_SP3    = float(np.degrees(np.arctan2(-0.5, -1.0))) + 360.0  # ≈ 206.57°  A = +½

# Matplotlib color cycle used consistently for each species pair
_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]


class LTDecompositionWidget(QWidget):
    """
    Tab for the φ_L / φ_T decomposition of bulk pFCs.

    Populated by MainWindow via load_data() after a successful analysis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results            = []
        self._pair_types         = []
        self._checkboxes         = {}
        self._color_map          = {}   # pair_type -> color string
        self._unit               = UNIT_EV
        self._reliability_cutoff = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        status_row = QHBoxLayout()
        self._status_label = QLabel('No data — run analysis first.')
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._chk_reliable = QCheckBox('Reliable only  (d ≤ L/2)')
        self._chk_reliable.setChecked(True)
        self._chk_reliable.setEnabled(False)
        self._chk_reliable.stateChanged.connect(self._refresh)
        btn_export = QPushButton('Export plot…')
        btn_export.clicked.connect(self._export_plot)
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        status_row.addWidget(self._chk_reliable)
        status_row.addWidget(btn_export)
        outer.addLayout(status_row)

        self.figure = Figure(figsize=(10, 8), tight_layout={'pad': 2.5})
        self.canvas = FigureCanvas(self.figure)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        outer.addWidget(self.mpl_toolbar)
        outer.addWidget(self.canvas, stretch=1)

        filter_group = QGroupBox('Atom pair types')
        self._filter_layout = QVBoxLayout()
        filter_group.setLayout(self._filter_layout)
        scroll = QScrollArea()
        scroll.setWidget(filter_group)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(100)
        outer.addWidget(scroll)

        self._draw_empty()

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def set_unit(self, unit: str):
        if unit != self._unit:
            self._unit = unit
            self._refresh()

    def load_data(self, results, reliability_cutoff=None):
        """
        Populate the viewer with results from compute_bulk_pfcs().

        Only records containing 'phi_l' / 'phi_t' keys are used;
        CSV-loaded results that predate this feature are silently skipped.
        """
        self._results = [r for r in results if 'phi_l' in r and 'phi_t' in r]
        if reliability_cutoff is not None:
            self._reliability_cutoff = reliability_cutoff
        self._chk_reliable.setEnabled(self._reliability_cutoff is not None)
        self._rebuild_checkboxes()
        self._refresh()
        n = len(self._results)
        if n:
            self._status_label.setText(
                f'{n} off-site pairs  —  φ_L / φ_T decomposition'
            )
        else:
            self._status_label.setText(
                'No LT data — re-run analysis (phi_l/phi_t require a fresh run).'
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_checkboxes(self):
        while self._filter_layout.count():
            item = self._filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes = {}
        self._color_map  = {}

        self._pair_types = sorted(set(
            (r['species1'], r['species2']) for r in self._results
        ))
        for i, pt in enumerate(self._pair_types):
            color = _COLORS[i % len(_COLORS)]
            self._color_map[pt] = color

            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(f'{pt[0]}-{pt[1]}')
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh)
            rl.addWidget(cb)
            rl.addStretch()
            self._filter_layout.addWidget(row)
            self._checkboxes[pt] = cb

    def _draw_empty(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No data — run analysis first.',
                transform=ax.transAxes, ha='center', va='center',
                color='grey', fontsize=13)
        ax.set_axis_off()
        self.canvas.draw()

    def _refresh(self):
        if not self._results:
            self._draw_empty()
            return

        # Apply reliability filter when checkbox is checked and cutoff is known
        reliable_only = (self._chk_reliable.isChecked()
                         and self._reliability_cutoff is not None)
        data = ([r for r in self._results
                 if r['distance'] <= self._reliability_cutoff]
                if reliable_only else self._results)

        active   = {pt for pt, cb in self._checkboxes.items() if cb.isChecked()}
        factor   = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        unit_lbl = UNIT_LABEL[self._unit]

        self.figure.clear()
        axes     = self.figure.subplots(2, 2)
        ax_l     = axes[0, 0]
        ax_t     = axes[0, 1]
        ax_a     = axes[1, 0]
        ax_cross = axes[1, 1]

        for ax in (ax_l, ax_t, ax_cross):
            ax.axhline(0, color='#cccccc', linewidth=0.8, zorder=0)
        ax_cross.axvline(0, color='#cccccc', linewidth=0.8, zorder=0)

        all_phi_l_flat = []

        for pt in self._pair_types:
            if pt not in active:
                continue
            sub = [r for r in data if (r['species1'], r['species2']) == pt]
            if not sub:
                continue

            color = self._color_map[pt]
            label = f'{pt[0]}-{pt[1]}'
            xs    = [r['distance'] for r in sub]
            pls   = [r['phi_l'] * factor for r in sub]
            pts_  = [r['phi_t'] * factor for r in sub]

            kw = dict(s=18, color=color, alpha=0.72, edgecolors='none', zorder=2)

            ax_l.scatter(xs, pls,  label=label, **kw)
            ax_t.scatter(xs, pts_, label=label, **kw)

            # θ = atan2(φ_T, φ_L) in [0°, 360°) — scale-invariant, no threshold needed
            th_xs, th_ys = [], []
            for x, pl, pt_val in zip(xs, pls, pts_):
                th = np.degrees(np.arctan2(pt_val, pl))
                th_xs.append(x)
                th_ys.append(th if th >= 0 else th + 360.0)
            if th_xs:
                ax_a.scatter(th_xs, th_ys, label=label, **kw)

            ax_cross.scatter(pls, pts_, label=label, **kw)
            all_phi_l_flat.extend(pls)

        # Cross-plot: A = -½ reference slope — labelled on the line, not in legend
        if all_phi_l_flat:
            span = max(abs(v) for v in all_phi_l_flat) * 1.3 or 1.0
            xr   = np.array([-span, span])
            ax_cross.plot(xr, -0.5 * xr,
                          color='#cc4444', linestyle='--', linewidth=1.3,
                          alpha=0.75, zorder=1)
            tx = -span * 0.7
            ax_cross.text(tx, -0.5 * tx, 'A = -½',
                          color='#cc4444', fontsize=7, alpha=0.85,
                          ha='center', va='bottom',
                          rotation=np.degrees(np.arctan(0.5)))

        # θ-vs-distance: reference lines — labelled on the lines, not in legend
        ax_a.axhline(_TH_IONIC,  color='#cc4444', linestyle='--', linewidth=1.3,
                     alpha=0.75)
        ax_a.axhline(_TH_RADIAL, color='#555555', linestyle=':',  linewidth=1.0,
                     alpha=0.55)
        ax_a.axhline(_TH_SP3,    color='#2266cc', linestyle='--', linewidth=1.0,
                     alpha=0.65)
        ax_a.set_ylim(90.0, 270.0)

        # Inline labels using a blended transform (x in axes coords, y in data coords)
        _xt = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
        ax_a.text(0.98, _TH_IONIC,  'A = -½  (ionic)',
                  transform=_xt, ha='right', va='bottom',
                  fontsize=7, color='#cc4444', alpha=0.85)
        ax_a.text(0.98, _TH_RADIAL, 'A = 0  (radial)',
                  transform=_xt, ha='right', va='bottom',
                  fontsize=7, color='#555555', alpha=0.75)
        ax_a.text(0.98, _TH_SP3,    'A = +½  (sp³)',
                  transform=_xt, ha='right', va='bottom',
                  fontsize=7, color='#2266cc', alpha=0.75)

        # Reliability shading on distance-axis panels (only when not already filtered)
        if not reliable_only:
            for ax in (ax_l, ax_t, ax_a):
                self._draw_reliability_line(ax)

        ax_l.set_xlabel('Distance (Å)')
        ax_l.set_ylabel(f'φ_L  ({unit_lbl})')
        ax_l.set_title('Longitudinal  (bond stretching)', fontsize=11)

        ax_t.set_xlabel('Distance (Å)')
        ax_t.set_ylabel(f'φ_T  ({unit_lbl})')
        ax_t.set_title('Transverse  (bond bending)', fontsize=11)

        ax_a.set_xlabel('Distance (Å)')
        ax_a.set_ylabel('θ = atan2(φ_T, φ_L)  (°)')
        ax_a.set_title('Bond-character angle  θ', fontsize=11)

        ax_cross.set_xlabel(f'φ_L  ({unit_lbl})')
        ax_cross.set_ylabel(f'φ_T  ({unit_lbl})')
        ax_cross.set_title('φ_T  vs  φ_L', fontsize=11)

        for ax in axes.flat:
            ax.grid(True, linestyle='--', alpha=0.35)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7,
                          loc='upper right', framealpha=0.85)

        self.canvas.draw_idle()

    def _draw_reliability_line(self, ax):
        if self._reliability_cutoff is None:
            return
        rc   = self._reliability_cutoff
        xlim = ax.get_xlim()
        ax.axvspan(rc * 0.85, rc,
                   color='#e6c800', alpha=0.12, zorder=0, linewidth=0)
        ax.axvspan(rc, max(xlim[1], rc) * 2,
                   color='#cc4444', alpha=0.09, zorder=0, linewidth=0)
        ax.axvline(rc, color='#cc4444', linestyle='--',
                   linewidth=1.2, alpha=0.65, zorder=1)
        ax.text(rc, 0.98, ' L/2', color='#cc4444', fontsize=7,
                va='top', ha='left', transform=ax.get_xaxis_transform())
        ax.set_xlim(xlim)

    def _export_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save LT decomposition plot', 'lt_decomposition.png',
            'PNG image (*.png);;PDF document (*.pdf);;SVG vector (*.svg)',
        )
        if path:
            self.figure.savefig(path, dpi=150, bbox_inches='tight')
