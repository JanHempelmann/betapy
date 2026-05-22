"""
Refsite pFC viewer widget.

Scatter plot (x = atom1–refsite distance, y = pFC value) on the left,
sortable pair table on the right. Clicking either panel highlights the
corresponding atom pair in the linked StructureView.
"""

import math
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QGroupBox,
    QScrollArea, QFrame, QCheckBox,
    QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


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
        toolbar.addWidget(btn_load)
        toolbar.addStretch()
        toolbar.addWidget(self._status_label)
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

    def load_data(self, results: list, supercell=None):
        """Load from a fresh refsite analysis (list of offsite result dicts)."""
        self._results       = results
        self._selected_pair = None
        self._rebuild_checkboxes()
        self._refresh_plot()
        self._refresh_table()
        self._status_label.setText(f'{len(results)} off-site pairs.')

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
                'Atom-Atom Distance (Angstr.)': 'atom_distance',
                'Mean pFC value':              'mean_pfc',
                'RMS pFC value':               'rms_pfc',
            }
            df = df.rename(columns=col_map)
            required = {'atom1_idx', 'species1', 'atom2_idx', 'species2',
                        'atom1_ref_dist', 'atom_distance', 'mean_pfc'}
            if not required.issubset(df.columns):
                raise ValueError(
                    f'Missing columns. Found: {list(df.columns)}'
                )
            self._results       = df.to_dict('records')
            self._selected_pair = None
            self._rebuild_checkboxes()
            self._refresh_plot()
            self._refresh_table()
            self._status_label.setText(
                f'{len(self._results)} pairs loaded from {path}'
            )
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

        for pt in self._pair_types:
            if pt not in active_pairs:
                continue
            sub = [r for r in self._results
                   if (r['species1'], r['species2']) == pt]
            if not sub:
                continue

            xs = np.array([r['atom1_ref_dist'] for r in sub])
            ys = np.array([r['mean_pfc']        for r in sub])
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
        ax.set_ylabel('Projected force constant (eV/Å²)', fontsize=11)
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
                r['atom_distance'],
                r['mean_pfc'],
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
            self._selection_bar.setText(
                f'atom {a1} ({rec["species1"]}) — atom {a2} ({rec["species2"]})'
                f'   ref dist = {rec["atom1_ref_dist"]:.3f} Å'
                f'   pFC = {rec["mean_pfc"]:+.5f} eV/Å²'
            )

        self._refresh_plot()
        if source != 'table':
            self._sync_table_selection()

        if self._structure_view is not None:
            self._structure_view.highlight_bond(a1, a2)

        self.pair_selected.emit(a1, a2)
