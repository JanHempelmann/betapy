"""
Refsite pFC viewer widget.

Scatter plot (x = atom1–refsite distance, y = pFC value) on the left,
sortable pair table on the right. Clicking either panel highlights the
corresponding atom pair in the linked StructureView.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QGroupBox,
    QScrollArea, QFrame, QCheckBox, QTabWidget,
    QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from betapy.core.constants import EV_ANG2_TO_N_M, UNIT_LABEL, UNIT_EV


PICK_TOLERANCE = 0.025   # fraction of axis range for scatter click detection


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class RefsitePFCWidget(QWidget):
    """
    Self-contained refsite pFC viewer: scatter plot + sortable table.

    Connect set_structure_view() to enable 3D pair highlighting.
    Load data via load_data() (from a fresh analysis) or load_from_csv().
    """

    pair_selected = pyqtSignal(int, int)   # 1-based atom indices

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results        = []    # list of dicts (offsite results)
        self._pair_types     = []    # sorted list of (sp1, sp2)
        self._checkboxes     = {}    # (sp1, sp2) -> QCheckBox
        self._scatter_data   = {}    # (sp1, sp2) -> (xs, ys, records)
        self._selected_pair  = None  # (a1, a2) or None
        self._structure_view = None
        self._unit           = UNIT_EV
        self._sum_raw        = None  # Σ pFC in eV/Å², set after each load
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Toolbar row
        toolbar = QHBoxLayout()
        btn_load = QPushButton('Load refsite_pFCs CSV…')
        btn_load.clicked.connect(self.load_from_csv)
        self._status_label = QLabel('No data — run refsite analysis or load CSV.')
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._btn_copy_sum = QPushButton('Copy Σ pFC')
        self._btn_copy_sum.setFixedWidth(90)
        self._btn_copy_sum.setEnabled(False)
        self._btn_copy_sum.clicked.connect(self._copy_sum)
        toolbar.addWidget(btn_load)
        toolbar.addStretch()
        toolbar.addWidget(self._status_label)
        toolbar.addWidget(self._btn_copy_sum)
        outer.addLayout(toolbar)

        # Main splitter: scatter (left) | table (right)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # --- Left: scatter plot ---
        scatter_widget = QWidget()
        scatter_layout = QVBoxLayout(scatter_widget)
        scatter_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        self.mpl_toolbar.setIconSize(QSize(24, 24))
        self.canvas.mpl_connect('button_press_event', self._on_scatter_click)
        scatter_layout.addWidget(self.mpl_toolbar)
        scatter_layout.addWidget(self.canvas, stretch=1)

        # Species-pair filter checkboxes
        filter_group = QGroupBox('Atom pair types')
        self._filter_layout = QVBoxLayout()
        filter_group.setLayout(self._filter_layout)
        scroll = QScrollArea()
        scroll.setWidget(filter_group)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(95)
        scatter_layout.addWidget(scroll)

        splitter.addWidget(scatter_widget)

        # --- Right: table ---
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel('Pairs  —  sorted by |pFC| descending')
        hdr.setStyleSheet('font-weight: bold; padding: 3px 0;')
        table_layout.addWidget(hdr)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            'Atom 1', 'Atom 2', 'Ref dist (Å)', 'Bond dist (Å)', 'pFC (eV/Å²)',
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        table_layout.addWidget(self.table)

        # Selection info bar
        self._selection_bar = QLabel('')
        self._selection_bar.setFrameStyle(QFrame.StyledPanel)
        self._selection_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._selection_bar.setFixedHeight(24)
        table_layout.addWidget(self._selection_bar)

        splitter.addWidget(table_widget)
        splitter.setSizes([500, 380])

        self._draw_empty_plot()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_structure_view(self, sv):
        """Link to a StructureView for 3D pair highlighting."""
        self._structure_view = sv

    def _update_status_label(self, source_note=''):
        """Rebuild the status label from current data and unit."""
        if not self._results:
            self._status_label.setText('No data — run refsite analysis or load CSV.')
            self._btn_copy_sum.setEnabled(False)
            return
        factor   = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        unit_lbl = UNIT_LABEL[self._unit]
        n        = len(self._results)
        if self._sum_raw is not None:
            sum_val = self._sum_raw * factor
            self._status_label.setText(
                f'{n} off-site pairs{source_note}   |   '
                f'Σ pFC = {sum_val:+.4f} {unit_lbl}'
            )
            self._btn_copy_sum.setEnabled(True)
        else:
            self._status_label.setText(f'{n} pairs{source_note}')

    def _copy_sum(self):
        if self._sum_raw is None:
            return
        from PyQt5.QtWidgets import QApplication
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        QApplication.clipboard().setText(f'{self._sum_raw * factor:.5f}')

    def set_unit(self, unit: str):
        """Switch display unit ('eV/Ang2' or 'N/m') and redraw."""
        if unit != self._unit:
            self._unit = unit
            self._refresh_plot()
            self._refresh_table()
            self._update_status_label()

    def load_data(self, results: list, supercell=None):
        """Load from a fresh refsite analysis (list of offsite result dicts)."""
        self._results       = results
        self._selected_pair = None
        self._sum_raw       = sum(r['mean_pfc'] for r in results) if results else None
        self._rebuild_checkboxes()
        self._refresh_plot()
        self._refresh_table()
        self._update_status_label()

    def load_from_csv(self, path=None):
        """Load from a refsite_pFCs.csv file."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, 'Open refsite pFCs CSV', '',
                'CSV files (*.csv);;All files (*)',
            )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            col_map = {
                'Atom1 Index':                 'atom1_idx',
                'Atom1 Type':                  'species1',
                'Atom2 Index':                 'atom2_idx',
                'Atom2 Type':                  'species2',
                'Atom1-Ref Distance (Angstr.)': 'atom1_ref_dist',
                'Atom-Atom Distance (Angstr.)': 'distance',
                'Mean pFC value':              'mean_pfc',
                'RMS pFC value':               'rms_pfc',
            }
            df = df.rename(columns=col_map)
            required = {'atom1_idx', 'species1', 'atom2_idx', 'species2',
                        'atom1_ref_dist', 'distance', 'mean_pfc'}
            if not required.issubset(df.columns):
                raise ValueError(
                    f'Missing columns. Found: {list(df.columns)}'
                )
            self._results       = df.to_dict('records')
            self._selected_pair = None
            self._sum_raw       = sum(r['mean_pfc'] for r in self._results)
            self._rebuild_checkboxes()
            self._refresh_plot()
            self._refresh_table()
            self._update_status_label(f' — loaded from {Path(path).name}')
        except Exception as e:
            QMessageBox.critical(self, 'Error loading CSV', str(e))

    def clear(self):
        self._results       = []
        self._selected_pair = None
        self._rebuild_checkboxes()
        self._draw_empty_plot()
        self._refresh_table()
        self._status_label.setText('No data.')

    # ------------------------------------------------------------------
    # Species colours (delegated to StructureView when available)
    # ------------------------------------------------------------------

    def _pair_colours_hex(self, sp1, sp2):
        if self._structure_view is not None:
            return self._structure_view.pair_colours_hex(sp1, sp2)
        return '#888888', '#888888'

    # ------------------------------------------------------------------
    # Checkboxes
    # ------------------------------------------------------------------

    def _rebuild_checkboxes(self):
        while self._filter_layout.count():
            item = self._filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes = {}

        self._pair_types = sorted(set(
            (r['species1'], r['species2']) for r in self._results
        ))
        for pt in self._pair_types:
            c1, c2 = self._pair_colours_hex(pt[0], pt[1])
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
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No data — run refsite analysis or load CSV',
                transform=ax.transAxes, ha='center', va='center',
                color='grey', fontsize=11)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _refresh_plot(self):
        if not self._results:
            self._draw_empty_plot()
            return

        active_pairs = {pt for pt, cb in self._checkboxes.items()
                        if cb.isChecked()}

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._scatter_data = {}
        self._ax = ax
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0

        for pt in self._pair_types:
            if pt not in active_pairs:
                continue
            sub = [r for r in self._results
                   if (r['species1'], r['species2']) == pt]
            if not sub:
                continue

            xs = np.array([r['atom1_ref_dist'] for r in sub])
            ys = np.array([r['mean_pfc'] * factor for r in sub])
            c1, c2 = self._pair_colours_hex(pt[0], pt[1])

            if c1 == c2:
                ax.scatter(xs, ys, label=f'{pt[0]}–{pt[1]}',
                           color=c1, s=32, alpha=0.78,
                           edgecolors='none', zorder=3)
            else:
                theta   = np.linspace(0, math.pi, 60)
                top_v   = np.column_stack([
                    np.concatenate([[0], np.cos(theta),         [0]]),
                    np.concatenate([[0], np.sin(theta),         [0]]),
                ])
                bot_v   = np.column_stack([
                    np.concatenate([[0], np.cos(theta + math.pi), [0]]),
                    np.concatenate([[0], np.sin(theta + math.pi), [0]]),
                ])
                ax.scatter(xs, ys, marker=top_v, s=32, color=c1,
                           alpha=0.78, edgecolors='none', zorder=3)
                ax.scatter(xs, ys, marker=bot_v, s=32, color=c2,
                           alpha=0.78, edgecolors='none',
                           label=f'{pt[0]}–{pt[1]}', zorder=3)

            self._scatter_data[pt] = (xs, ys, sub)

        # Gold ring on currently selected point
        if self._selected_pair is not None:
            a1, a2 = self._selected_pair
            for xs, ys, sub in self._scatter_data.values():
                for i, r in enumerate(sub):
                    if (int(r['atom1_idx']) == a1
                            and int(r['atom2_idx']) == a2):
                        ax.scatter([xs[i]], [ys[i]], s=140,
                                   facecolors='none',
                                   edgecolors='#c8a000',
                                   linewidths=2.5, zorder=5)
                        break

        ax.set_xlabel('Atom 1 – refsite distance (Å)', fontsize=11)
        ax.set_ylabel(f'Projected force constant ({UNIT_LABEL[self._unit]})', fontsize=11)
        ax.set_title('Refsite projected force constants', fontsize=12)
        if self._scatter_data:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.35)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
        self.table.setHorizontalHeaderItem(
            4, QTableWidgetItem(f'pFC ({UNIT_LABEL[self._unit]})')
        )

        sorted_results = sorted(
            self._results, key=lambda r: abs(r['mean_pfc']), reverse=True
        )

        for r in sorted_results:
            row  = self.table.rowCount()
            self.table.insertRow(row)
            a1   = int(r['atom1_idx'])
            a2   = int(r['atom2_idx'])
            sp1  = r['species1']
            sp2  = r['species2']

            # Atom-label cells (store index pair for lookup)
            lbl1 = QTableWidgetItem(f'{a1} ({sp1})')
            lbl1.setFlags(lbl1.flags() & ~Qt.ItemIsEditable)
            lbl1.setData(Qt.UserRole, (a1, a2))
            self.table.setItem(row, 0, lbl1)

            lbl2 = QTableWidgetItem(f'{a2} ({sp2})')
            lbl2.setFlags(lbl2.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, lbl2)

            for col, val in enumerate([
                r['atom1_ref_dist'],
                r['distance'],
                r['mean_pfc'] * factor,
            ], start=2):
                it = _NumericItem(f'{val:+.4f}' if col == 4 else f'{val:.4f}')
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if col == 4:   # pFC column: colour-code sign
                    it.setForeground(
                        QColor(30, 60, 160) if val >= 0 else QColor(180, 40, 30)
                    )
                self.table.setItem(row, col, it)

        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)

        if self._selected_pair is not None:
            self._sync_table_selection()

    def _sync_table_selection(self):
        """Scroll the table to and select the row for _selected_pair."""
        if self._selected_pair is None:
            return
        a1, a2 = self._selected_pair
        for row in range(self.table.rowCount()):
            pair = self.table.item(row, 0).data(Qt.UserRole)
            if pair == (a1, a2):
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.scrollToItem(
                    self.table.item(row, 0),
                    QAbstractItemView.PositionAtCenter,
                )
                self.table.blockSignals(False)
                return

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_scatter_click(self, event):
        if event.inaxes is None or not self._scatter_data or event.button != 1:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return

        ax      = self._ax
        x_scale = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1
        y_scale = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1

        best_d, best_r = float('inf'), None
        for xs, ys, sub in self._scatter_data.values():
            for i, r in enumerate(sub):
                dx = (xs[i] - cx) / x_scale
                dy = (ys[i] - cy) / y_scale
                d  = (dx**2 + dy**2) ** 0.5
                if d < best_d:
                    best_d, best_r = d, r

        if best_r is None or best_d > PICK_TOLERANCE:
            return
        self._select_pair(int(best_r['atom1_idx']),
                          int(best_r['atom2_idx']), source='scatter')

    def _on_table_select(self):
        rows = self.table.selectedItems()
        if not rows:
            return
        pair = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        if pair is None:
            return
        self._select_pair(pair[0], pair[1], source='table')

    def _select_pair(self, a1: int, a2: int, source: str = 'click'):
        self._selected_pair = (a1, a2)

        rec = next(
            (r for r in self._results
             if int(r['atom1_idx']) == a1 and int(r['atom2_idx']) == a2),
            None,
        )
        if rec:
            factor = EV_ANG2_TO_N_M if self._unit == 'N/m' else 1.0
            self._selection_bar.setText(
                f'atom {a1} ({rec["species1"]}) — atom {a2} ({rec["species2"]})'
                f'   ref dist = {rec["atom1_ref_dist"]:.3f} Å'
                f'   pFC = {rec["mean_pfc"] * factor:+.5f} {UNIT_LABEL[self._unit]}'
            )

        self._refresh_plot()
        if source != 'table':
            self._sync_table_selection()

        if self._structure_view is not None:
            self._structure_view.highlight_bond(a1, a2)

        self.pair_selected.emit(a1, a2)

class MultiRefsitePFCWidget(QWidget):
    """Tabbed refsite pFC viewer — one tab per site, falls back to placeholder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._structure_view = None
        self._unit           = UNIT_EV
        self._tab_viewers    = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)
        self._show_placeholder()

    def _show_placeholder(self):
        ph = QLabel('No data — run refsite analysis or load CSV.')
        ph.setAlignment(Qt.AlignCenter)
        self._tabs.clear()
        self._tab_viewers = []
        self._tabs.addTab(ph, 'Results')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_structure_view(self, sv):
        self._structure_view = sv
        for w in self._tab_viewers:
            w.set_structure_view(sv)

    def _on_colours_changed(self):
        """Propagate colours_changed from StructureView to all site tabs."""
        for w in self._tab_viewers:
            w._refresh_plot()

    def set_unit(self, unit):
        self._unit = unit
        for w in self._tab_viewers:
            w.set_unit(unit)

    def load_multi_data(self, site_results):
        """Load results for one or more sites and rebuild all tabs.

        Parameters
        ----------
        site_results : list of (label, frac_pos, offsite_results, onsite_results)
        """
        self._tabs.clear()
        self._tab_viewers = []

        for label, _frac_pos, offsite, _onsite in site_results:
            viewer = RefsitePFCWidget()
            viewer.set_unit(self._unit)
            if self._structure_view is not None:
                viewer.set_structure_view(self._structure_view)
                self._structure_view.colours_changed.connect(viewer._refresh_plot)
            viewer.load_data(offsite)
            self._tabs.addTab(viewer, label)
            self._tab_viewers.append(viewer)

    def load_from_csv(self, path=None):
        """Load a single-site refsite_pFCs CSV into a fresh tab."""
        self._tabs.clear()
        self._tab_viewers = []

        viewer = RefsitePFCWidget()
        viewer.set_unit(self._unit)
        if self._structure_view is not None:
            viewer.set_structure_view(self._structure_view)
            self._structure_view.colours_changed.connect(viewer._refresh_plot)

        from pathlib import Path as _Path
        tab_label = _Path(path).stem if path else 'Results'
        self._tabs.addTab(viewer, tab_label)
        self._tab_viewers.append(viewer)
        viewer.load_from_csv(path)
