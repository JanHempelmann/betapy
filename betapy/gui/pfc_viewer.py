"""
pFC Viewer — Tool 1.

Split panel:
  Left  — scatter plot of pFC value vs interatomic distance,
           coloured by atom-pair species type.
  Right — 3D structure view (StructureView).

Clicking a point in the scatter plot highlights the corresponding
bond in the 3D view and shows atom details in the status bar.

Data can arrive two ways:
  1. From MainWindow after a fresh analysis (load_data).
  2. Directly from a CSV file (load_from_csv), so users can
     inspect previous results without re-running the analysis.
"""

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QCheckBox, QPushButton, QLabel, QFileDialog,
    QGroupBox, QScrollArea, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from betapy.gui.structure_view import StructureView
from betapy.core.constants import PFC_ROUNDING_DECIMALS, EV_ANG2_TO_N_M, UNIT_LABEL, UNIT_EV


# How close (in data units) a click must be to count as a point selection
PICK_TOLERANCE = 0.02    # fraction of axis range


class PFCViewerWidget(QWidget):
    """
    Self-contained pFC viewer tab.

    Signals
    -------
    pair_selected(int, int) : emitted when a scatter point is clicked,
        carrying the 1-based atom indices of the selected pair.
    """

    pair_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results              = []   # list of dicts from compute_bulk_pfcs
        self._pair_types           = []
        self._checkboxes           = {}
        self._scatter_collections  = {}  # pair_type -> (PathCollection, records/shells)
        self._supercell            = None
        self._selected_record      = None  # record dict for highlighted point
        self._unit                 = UNIT_EV
        self._shells               = []   # list of shell dicts from group_by_shells()
        self._view_mode            = 'individual'
        self._selected_shell       = None
        self._reliability_cutoff   = None  # half minimum perpendicular supercell width

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Top toolbar: load CSV button + status label
        toolbar_row = QHBoxLayout()
        btn_load_csv = QPushButton('Open existing pFCs CSV…')
        btn_load_csv.clicked.connect(self.load_from_csv)
        self._status_label = QLabel('No data loaded.')
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        btn_export = QPushButton('Export plot…')
        btn_export.clicked.connect(self._export_plot)
        toolbar_row.addWidget(btn_load_csv)
        toolbar_row.addWidget(btn_export)
        toolbar_row.addStretch()
        toolbar_row.addWidget(self._status_label)
        outer.addLayout(toolbar_row)

        # Main splitter: left = scatter + controls, right = 3D view
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # --- Left panel: scatter plot + filter checkboxes ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect('button_press_event', self._on_scatter_click)

        left_layout.addWidget(self.mpl_toolbar)
        left_layout.addWidget(self.canvas, stretch=1)

        # Filter checkboxes in a scrollable area
        filter_group = QGroupBox('Atom pair types')
        self._filter_layout = QVBoxLayout()
        filter_group.setLayout(self._filter_layout)
        scroll = QScrollArea()
        scroll.setWidget(filter_group)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(140)
        left_layout.addWidget(scroll)

        bottom_row = QHBoxLayout()
        self._btn_shell = QPushButton('Shell view')
        self._btn_shell.setCheckable(True)
        self._btn_shell.setEnabled(False)
        self._btn_shell.clicked.connect(self._toggle_view_mode)
        self.chk_unique = QCheckBox('Show unique pFCs only')
        self.chk_unique.stateChanged.connect(self._refresh_plot)
        bottom_row.addWidget(self._btn_shell)
        bottom_row.addWidget(self.chk_unique)
        bottom_row.addStretch()
        left_layout.addLayout(bottom_row)

        splitter.addWidget(left)

        # --- Right panel: 3D structure view ---
        self.structure_view = StructureView(self)
        # When the user changes a colour in the structure view picker,
        # refresh the scatter plot so it stays in sync
        self.structure_view.colours_changed.connect(self._on_colours_changed)
        splitter.addWidget(self.structure_view)

        splitter.setSizes([550, 550])

        # Selection info bar at the bottom
        self._selection_bar = QLabel('')
        self._selection_bar.setFrameStyle(QFrame.StyledPanel)
        self._selection_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._selection_bar.setFixedHeight(28)
        outer.addWidget(self._selection_bar)

        self._draw_empty_plot()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_supercell(self, supercell):
        """
        Push a supercell to the 3D structure view without touching pFC data.
        Called by MainWindow as soon as SPOSCAR is loaded, independently
        of whether any analysis has been run.
        """
        self._supercell = supercell
        self.structure_view.load_supercell(supercell)
        L = supercell.lattice
        a, b, c = L[0], L[1], L[2]
        V = abs(float(np.dot(a, np.cross(b, c))))
        self._reliability_cutoff = min(
            V / np.linalg.norm(np.cross(b, c)),
            V / np.linalg.norm(np.cross(a, c)),
            V / np.linalg.norm(np.cross(a, b)),
        ) / 2.0

    def set_unit(self, unit: str):
        """Switch display unit ('eV/Ang2' or 'N/m') and redraw."""
        if unit != self._unit:
            self._unit = unit
            self._refresh_plot()

    def _on_colours_changed(self):
        """Called when the structure view colour picker changes a species colour."""
        self._rebuild_checkboxes()
        self._refresh_plot()

    def load_data(self, df_unique, all_results, supercell=None):
        """
        Called by MainWindow after a fresh analysis.

        Parameters
        ----------
        df_unique   : DataFrame from unique_pfcs()
        all_results : list of dicts from compute_bulk_pfcs()
        supercell   : Supercell instance (optional, enables 3D view)
        """
        self._results  = all_results
        if supercell is not None:
            self.set_supercell(supercell)
        self._rebuild_checkboxes()
        self._compute_shells()
        self._refresh_plot()
        self._status_label.setText(
            f'{len(all_results)} off-site pairs loaded from analysis.'
        )

    def load_from_csv(self, path=None):
        """
        Load pFC data from a CSV file (unique_pFCs.csv format).
        Populates the scatter plot without needing a full analysis run.
        """
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, 'Open pFCs CSV', '',
                'CSV files (*.csv);;All files (*)',
            )
        if not path:
            return

        try:
            df = pd.read_csv(path)

            # Normalise column names to what we expect
            col_map = {
                'Index 1': 'atom1_idx', 'Atom 1': 'species1',
                'Index 2': 'atom2_idx', 'Atom 2': 'species2',
                'Distance (Angstr.)': 'distance',
                'pFC value': 'mean_pfc',
                # Also accept the full-results format
                'Atom1 Index': 'atom1_idx', 'Atom1 Type': 'species1',
                'Atom2 Index': 'atom2_idx', 'Atom2 Type': 'species2',
                'Atom-Atom Distance (Angstr.)': 'distance',
                'Mean pFC value': 'mean_pfc',
            }
            df = df.rename(columns=col_map)
            required = {'atom1_idx', 'species1', 'atom2_idx',
                        'species2', 'distance', 'mean_pfc'}
            if not required.issubset(df.columns):
                raise ValueError(
                    f'CSV missing columns. Found: {list(df.columns)}'
                )

            self._results = df.to_dict('records')
            self._rebuild_checkboxes()
            self._compute_shells()
            self._refresh_plot()
            struct_note = (
                '' if self._supercell is not None
                else ' (open SPOSCAR to enable 3D view)'
            )
            self._status_label.setText(
                f'{len(self._results)} pairs loaded from {path}{struct_note}'
            )
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, 'Error loading CSV', str(e))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _draw_empty_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No data — load a CSV or run analysis',
                transform=ax.transAxes, ha='center', va='center',
                color='grey', fontsize=13)
        ax.set_axis_off()
        self.canvas.draw()

    def _rebuild_checkboxes(self):
        """Recreate the species-pair filter checkboxes from current data."""
        while self._filter_layout.count():
            item = self._filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes = {}

        self._pair_types = sorted(set(
            (r['species1'], r['species2']) for r in self._results
        ))
        for pt in self._pair_types:
            if self._supercell is not None:
                c1, c2 = self.structure_view.pair_colours_hex(pt[0], pt[1])
            else:
                c1 = c2 = '#555555'

            # Use a QLabel for rich coloured text + a QCheckBox for the tick
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_plot)

            lbl = QLabel(
                f'<b><span style="color:{c1}">{pt[0]}</span></b>'
                f'<span style="color:#888888"> – </span>'
                f'<b><span style="color:{c2}">{pt[1]}</span></b>'
            )

            row_layout.addWidget(cb)
            row_layout.addWidget(lbl)
            row_layout.addStretch()
            self._filter_layout.addWidget(row)
            self._checkboxes[pt] = cb

    def _draw_reliability_line(self, ax):
        """
        Two-zone reliability shading around the half-cell cutoff (L/2).

        Yellow band from 0.85*L/2 to L/2 — caution, interactions are
        decaying and may be near the noise floor, but finite-size errors
        are typically small.  Red band beyond L/2 — strictly outside the
        minimum-image regime, though errors in practice are often tiny
        because real interactions have already decayed by this distance.
        """
        if self._reliability_cutoff is None:
            return
        rc  = self._reliability_cutoff
        xlim = ax.get_xlim()

        # Caution zone (yellow): 0.85*L/2 to L/2
        ax.axvspan(rc * 0.85, rc,
                   color='#e6c800', alpha=0.12, zorder=0, linewidth=0)

        # Unreliable zone (red): L/2 onward
        ax.axvspan(rc, xlim[1] * 2,
                   color='#cc4444', alpha=0.09, zorder=0, linewidth=0)

        # Boundary line at L/2
        ax.axvline(rc, color='#cc4444', linestyle='--',
                   linewidth=1.2, alpha=0.65, zorder=1)

        ax.text(rc, 0.98, f' L/2={rc:.2f} Å',
                color='#cc4444', fontsize=8, va='top', ha='left',
                transform=ax.get_xaxis_transform())

        # axvspan with a far-right edge can expand xlim; restore it
        ax.set_xlim(xlim)

    # ------------------------------------------------------------------
    # Shell computation and view mode
    # ------------------------------------------------------------------

    def _compute_shells(self):
        from betapy.core.projection import group_by_shells
        if self._results:
            self._shells = group_by_shells(
                self._results,
                max_distance=self._reliability_cutoff,
            )
            n = len(self._shells)
            tip = f'{n} shells'
            if self._reliability_cutoff is not None:
                tip += f' within L/2={self._reliability_cutoff:.2f} A'
            else:
                tip += ' (load SPOSCAR to apply L/2 cutoff)'
            self._btn_shell.setEnabled(True)
            self._btn_shell.setToolTip(tip)
        else:
            self._shells = []
            self._btn_shell.setEnabled(False)
            self._btn_shell.setToolTip('')

    def _toggle_view_mode(self, checked):
        self._view_mode      = 'shell' if checked else 'individual'
        self._selected_shell = None
        self._selected_record = None
        self.chk_unique.setEnabled(not checked)
        self._selection_bar.setText('')
        if not checked and self._supercell is not None:
            self.structure_view.clear_highlight()
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _refresh_plot(self):
        if not self._results:
            self._draw_empty_plot()
            return
        if self._view_mode == 'shell':
            self._refresh_shell_plot()
        else:
            self._refresh_individual_plot()

    def _refresh_individual_plot(self):
        use_unique   = self.chk_unique.isChecked()
        active_pairs = {pt for pt, cb in self._checkboxes.items()
                        if cb.isChecked()}

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._scatter_collections = {}
        self._ax = ax

        # Determine data to plot
        if use_unique:
            seen = set()
            deduped = []
            for r in self._results:
                key = round(r['mean_pfc'], PFC_ROUNDING_DECIMALS)
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            data = deduped
        else:
            data = self._results

        for pt in self._pair_types:
            if pt not in active_pairs:
                continue
            sub = [r for r in data
                   if (r['species1'], r['species2']) == pt]
            if not sub:
                continue
            factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
            xs = [r['distance']          for r in sub]
            ys = [r['mean_pfc'] * factor for r in sub]

            if self._supercell is not None:
                c1, c2 = self.structure_view.pair_colours_hex(pt[0], pt[1])
            else:
                c1 = c2 = '#888888'

            if c1 == c2:
                sc = ax.scatter(
                    xs, ys, label=f'{pt[0]}-{pt[1]}',
                    color=c1, s=35,
                    alpha=0.80, edgecolors='none',
                    picker=True, pickradius=6,
                )
            else:
                theta = np.linspace(0, np.pi, 60)
                top_verts = np.column_stack([
                    np.concatenate([[0], np.cos(theta),  [0]]),
                    np.concatenate([[0], np.sin(theta),  [0]]),
                ])
                bot_verts = np.column_stack([
                    np.concatenate([[0], np.cos(theta + np.pi), [0]]),
                    np.concatenate([[0], np.sin(theta + np.pi), [0]]),
                ])
                ax.scatter(xs, ys, marker=top_verts, s=35,
                           color=c1, alpha=0.80, edgecolors='none')
                sc = ax.scatter(xs, ys, marker=bot_verts, s=35,
                                color=c2, alpha=0.80, edgecolors='none',
                                label=f'{pt[0]}-{pt[1]}',
                                picker=True, pickradius=6)

            self._scatter_collections[pt] = (sc, sub)

        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        if self._selected_record is not None:
            sr = self._selected_record
            if (sr['species1'], sr['species2']) in active_pairs:
                ax.scatter(
                    [sr['distance']], [sr['mean_pfc'] * factor],
                    s=140, facecolors='none', edgecolors='#c8a000',
                    linewidths=2.5, zorder=5,
                )

        ax.set_xlabel('Interatomic distance (Å)', fontsize=12)
        ax.set_ylabel(f'Projected force constant ({UNIT_LABEL[self._unit]})', fontsize=12)
        ax.set_title('Projected force constants vs bond length', fontsize=13)
        if self._scatter_collections:
            legend = ax.legend(loc='upper right', framealpha=0.9)
            for legend_text, pt in zip(legend.get_texts(),
                                       [p for p in self._pair_types
                                        if p in active_pairs]):
                lc1 = (self.structure_view.pair_colours_hex(pt[0], pt[1])[0]
                       if self._supercell is not None else '#555555')
                legend_text.set_color(lc1)
        ax.grid(True, linestyle='--', alpha=0.4)
        self._draw_reliability_line(ax)
        self.canvas.draw_idle()

    def _refresh_shell_plot(self):
        active_pairs = {pt for pt, cb in self._checkboxes.items()
                        if cb.isChecked()}

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._scatter_collections = {}
        self._ax = ax

        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0

        for pt in self._pair_types:
            if pt not in active_pairs:
                continue
            shells_for_pt = [s for s in self._shells
                             if (s['species1'], s['species2']) == pt]
            if not shells_for_pt:
                continue

            if self._supercell is not None:
                c1, _ = self.structure_view.pair_colours_hex(pt[0], pt[1])
            else:
                c1 = '#888888'

            xs     = [s['distance_mean']       for s in shells_for_pt]
            ys     = [s['pfc_mean']   * factor for s in shells_for_pt]
            counts = [s['count']                for s in shells_for_pt]
            sizes  = [max(30, min(200, 30 + 40 * np.log1p(c))) for c in counts]

            sc = ax.scatter(
                xs, ys, s=sizes,
                color=c1, alpha=0.85, edgecolors='none',
                label=f'{pt[0]}-{pt[1]}',
                picker=True, pickradius=8, zorder=3,
            )

            for s_dict, x in zip(shells_for_pt, xs):
                ymin = s_dict['pfc_min'] * factor
                ymax = s_dict['pfc_max'] * factor
                ax.vlines(x, ymin, ymax, color=c1, alpha=0.45,
                          linewidth=1.5, zorder=2)

            self._scatter_collections[pt] = (sc, shells_for_pt)

        if self._selected_shell is not None:
            ss = self._selected_shell
            pt = (ss['species1'], ss['species2'])
            if pt in active_pairs:
                ax.scatter(
                    [ss['distance_mean']], [ss['pfc_mean'] * factor],
                    s=250, facecolors='none', edgecolors='#c8a000',
                    linewidths=2.5, zorder=5,
                )

        ax.set_xlabel('Interatomic distance (Å)', fontsize=12)
        ax.set_ylabel(f'Projected force constant ({UNIT_LABEL[self._unit]})', fontsize=12)
        ax.set_title('Projected force constants — shell view', fontsize=13)
        if self._scatter_collections:
            legend = ax.legend(loc='upper right', framealpha=0.9)
            for legend_text, pt in zip(legend.get_texts(),
                                       [p for p in self._pair_types
                                        if p in active_pairs]):
                lc1 = (self.structure_view.pair_colours_hex(pt[0], pt[1])[0]
                       if self._supercell is not None else '#555555')
                legend_text.set_color(lc1)
        ax.grid(True, linestyle='--', alpha=0.4)
        self._draw_reliability_line(ax)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Interaction: scatter click → 3D highlight
    # ------------------------------------------------------------------

    def _on_scatter_click(self, event):
        if event.inaxes is None or not self._scatter_collections:
            return
        if event.button != 1:
            return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None:
            return

        ax = self._ax
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        x_scale = x_range[1] - x_range[0] or 1
        y_scale = y_range[1] - y_range[0] or 1
        factor   = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        unit_lbl = UNIT_LABEL[self._unit]

        if self._view_mode == 'shell':
            best_dist  = float('inf')
            best_shell = None
            for pt, (sc_obj, shells) in self._scatter_collections.items():
                for s in shells:
                    dx = (s['distance_mean']     - click_x) / x_scale
                    dy = (s['pfc_mean'] * factor - click_y) / y_scale
                    d  = (dx**2 + dy**2) ** 0.5
                    if d < best_dist:
                        best_dist  = d
                        best_shell = s

            if best_shell is None or best_dist > PICK_TOLERANCE * 3:
                return

            self._selected_shell = best_shell
            # Pick the atom1 with most bonds in this shell as representative;
            # show only its bonds so we avoid overlapping images from multiple
            # symmetry-equivalent source atoms.
            from collections import Counter
            atom1_counts = Counter(int(r['atom1_idx']) for r in best_shell['records'])
            rep_atom1 = atom1_counts.most_common(1)[0][0]
            pairs = [(int(r['atom1_idx']), int(r['atom2_idx']))
                     for r in best_shell['records']
                     if int(r['atom1_idx']) == rep_atom1]
            if self._supercell is not None:
                self.structure_view.highlight_bonds(pairs, center_on=rep_atom1)

            sp1      = best_shell['species1']
            sp2      = best_shell['species2']
            n        = best_shell['count']
            d        = best_shell['distance_mean']
            pfc      = best_shell['pfc_mean'] * factor
            pfc_std  = best_shell['pfc_std']  * factor
            pfc_min  = best_shell['pfc_min']  * factor
            pfc_max  = best_shell['pfc_max']  * factor
            self._selection_bar.setText(
                f'Shell: {sp1}-{sp2}  d = {d:.4f} A  n = {n}  '
                f'pFC = {pfc:.5f} +/- {pfc_std:.5f} {unit_lbl}  '
                f'[{pfc_min:.5f} ... {pfc_max:.5f}]'
            )
            self._refresh_plot()

        else:
            best_dist   = float('inf')
            best_record = None
            for pt, (sc_obj, records) in self._scatter_collections.items():
                for r in records:
                    dx = (r['distance']          - click_x) / x_scale
                    dy = (r['mean_pfc'] * factor - click_y) / y_scale
                    d  = (dx**2 + dy**2) ** 0.5
                    if d < best_dist:
                        best_dist   = d
                        best_record = r

            if best_record is None or best_dist > PICK_TOLERANCE:
                return

            self._selected_record = best_record
            a1      = int(best_record['atom1_idx'])
            a2      = int(best_record['atom2_idx'])
            sp1     = best_record['species1']
            sp2     = best_record['species2']
            d       = best_record['distance']
            pfc_disp = best_record['mean_pfc'] * factor

            if self._supercell is not None:
                self.structure_view.highlight_bond(a1, a2)

            self._selection_bar.setText(
                f'Selected:  atom {a1} ({sp1}) - atom {a2} ({sp2})   '
                f'distance = {d:.4f} A   pFC = {pfc_disp:.6f} {unit_lbl}'
            )
            self.pair_selected.emit(a1, a2)
            self._refresh_plot()

    def _export_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save plot', 'pfc_plot.png',
            'PNG image (*.png);;PDF document (*.pdf);;SVG vector (*.svg)',
        )
        if path:
            self.figure.savefig(path, dpi=150, bbox_inches='tight')
