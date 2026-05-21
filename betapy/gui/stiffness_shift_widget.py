"""
Stiffness Shift tab — side-by-side ΔpFC comparison of two structures.

Layout
------
Top bar   : directory paths for A and B, REFPOS, analysis settings
3D views  : side-by-side StructureView with refsite cube + connection toggles
Bottom    : overlay scatter (A ○, B △, matched connected, unmatched grey)
            + sortable matched-pair table
"""

import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QDoubleSpinBox, QLineEdit,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFrame, QScrollArea, QGridLayout,
)
from PyQt5.QtCore import Qt
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
    match_atoms_across_structures,
    match_fc_pairs,
    stiffness_shift_from_pairs,
)
from betapy.gui.structure_view import StructureView


PICK_TOLERANCE = 0.025


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
        self._sc_a          = None
        self._sc_b          = None
        self._refsite_frac  = None   # first refsite (for 3D view)
        self._offsite_a     = []
        self._offsite_b     = []
        self._matched       = []
        self._unmatched_a   = []
        self._unmatched_b   = []
        self._selected_midx = None   # index into _matched, or None
        self._pair_types    = []
        self._checkboxes    = {}
        self._scatter_pts   = {}     # (sp1,sp2) -> [{xa,ya,xb,yb,midx}, ...]
        self._ax            = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self._build_load_bar())

        v = QSplitter(Qt.Vertical)
        outer.addWidget(v)
        v.addWidget(self._build_views_panel())
        v.addWidget(self._build_bottom_panel())
        v.setSizes([480, 380])

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
        btn_a.clicked.connect(lambda: self._browse_dir(self._edit_a))
        grid.addWidget(btn_a, 0, 2)

        grid.addWidget(QLabel('Structure B (intercalated):'), 1, 0)
        self._edit_b = QLineEdit()
        self._edit_b.setPlaceholderText('directory containing SPOSCAR, FORCE_CONSTANTS')
        grid.addWidget(self._edit_b, 1, 1)
        btn_b = QPushButton('Browse…')
        btn_b.setFixedWidth(80)
        btn_b.clicked.connect(lambda: self._browse_dir(self._edit_b))
        grid.addWidget(btn_b, 1, 2)

        grid.addWidget(QLabel('REFPOS:'), 2, 0)
        self._edit_refpos = QLineEdit()
        self._edit_refpos.setPlaceholderText('auto-detected from Dir A, or browse')
        grid.addWidget(self._edit_refpos, 2, 1)
        btn_rp = QPushButton('Browse…')
        btn_rp.setFixedWidth(80)
        btn_rp.clicked.connect(lambda: self._browse_file(
            self._edit_refpos, 'REFPOS files (REFPOS);;All files (*)'
        ))
        grid.addWidget(btn_rp, 2, 2)
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

        srow.addWidget(QLabel('Match tol (Å):'))
        self._spin_tol = QDoubleSpinBox()
        self._spin_tol.setRange(0.01, 2.0)
        self._spin_tol.setSingleStep(0.05)
        self._spin_tol.setDecimals(2)
        self._spin_tol.setValue(0.3)
        self._spin_tol.setFixedWidth(72)
        srow.addWidget(self._spin_tol)

        self._chk_excl = QCheckBox('Exclude refsite species')
        self._chk_excl.setChecked(True)
        self._chk_excl.setToolTip(
            'Exclude off-site pairs where either atom is the species\n'
            'occupying the reference site in structure B.'
        )
        srow.addWidget(self._chk_excl)

        btn_run = QPushButton('Run analysis')
        btn_run.setFixedWidth(110)
        btn_run.clicked.connect(self._run_analysis)
        srow.addWidget(btn_run)

        self._result_lbl = QLabel('')
        self._result_lbl.setWordWrap(True)
        srow.addWidget(self._result_lbl, stretch=1)

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
            self._view_a       = view
            self._conn_spin_a  = spin
            self._btn_conn_a   = btn
        else:
            self._view_b       = view
            self._conn_spin_b  = spin
            self._btn_conn_b   = btn

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

        # Right: matched-pair table
        table_w = QWidget()
        tl = QVBoxLayout(table_w)
        tl.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel('Matched pairs — sorted by |ΔpFC|')
        hdr.setStyleSheet('font-weight: bold; padding: 3px 0;')
        tl.addWidget(hdr)

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
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_select)
        tl.addWidget(self._table)

        self._sel_bar = QLabel('')
        self._sel_bar.setFrameStyle(QFrame.StyledPanel)
        self._sel_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._sel_bar.setFixedHeight(24)
        tl.addWidget(self._sel_bar)

        h.addWidget(table_w)
        h.setSizes([560, 380])
        return h

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if path:
            edit.setText(path)

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
        try:
            self._do_analysis()
        except Exception as e:
            QMessageBox.critical(self, 'Analysis error', str(e))

    def _do_analysis(self):
        dir_a_str  = self._edit_a.text().strip()
        dir_b_str  = self._edit_b.text().strip()
        refpos_str = self._edit_refpos.text().strip()

        if not dir_a_str or not dir_b_str:
            raise ValueError('Please specify both structure directories.')

        dir_a = Path(dir_a_str)
        dir_b = Path(dir_b_str)

        sc_a = Supercell(read_SPOSCAR(dir_a / 'SPOSCAR'))
        sc_b = Supercell(read_SPOSCAR(dir_b / 'SPOSCAR'))
        fc_a = read_FORCE_CONSTANTS(dir_a / 'FORCE_CONSTANTS')
        fc_b = read_FORCE_CONSTANTS(dir_b / 'FORCE_CONSTANTS')

        # REFPOS resolution
        if refpos_str:
            refpos_path = Path(refpos_str)
        else:
            refpos_path = dir_a / 'REFPOS'
            if not refpos_path.exists():
                refpos_path = dir_b / 'REFPOS'
        refpos_data  = read_refpos(refpos_path)
        all_refsites = refpos_data['positions']
        if not all_refsites:
            raise ValueError(f'No positions found in {refpos_path}')

        cutoff = self._spin_cutoff.value()
        msd    = self._spin_msd.value()
        tol    = self._spin_tol.value()
        excl   = self._chk_excl.isChecked()

        # Determine species to exclude (look for occupying atom in B, first site)
        excl_sp = None
        if excl:
            frac_0 = np.asarray(all_refsites[0])
            dists  = [sc_b.distance_to_point(k + 1, frac_0) for k in range(sc_b.n_atoms)]
            ni     = min(range(sc_b.n_atoms), key=lambda k: dists[k])
            if dists[ni] < msd:
                excl_sp = {sc_b.species(ni + 1)}

        # Refsite projections (all sites)
        offsite_a = []
        for frac in all_refsites:
            res, _ = find_refsite_pairs(
                sc_a, fc_a['atomic_pairs'], fc_a['force_matrices'],
                frac, cutoff=cutoff, min_distance=0.0,
                exclude_species=excl_sp, show_progress=False,
            )
            offsite_a.extend(res)

        offsite_b = []
        for frac in all_refsites:
            res, _ = find_refsite_pairs(
                sc_b, fc_b['atomic_pairs'], fc_b['force_matrices'],
                frac, cutoff=cutoff, min_distance=msd,
                exclude_species=excl_sp, show_progress=False,
            )
            offsite_b.extend(res)

        # Atom position matching
        all_species = sorted(set(sc_a.chem_symbols) & set(sc_b.chem_symbols))
        atom_matches = {}
        for sp in all_species:
            m, _ = match_atoms_across_structures(sc_a, sc_b, sp, tolerance=tol)
            atom_matches.update(m)

        matched, unmatched_a, unmatched_b = match_fc_pairs(
            offsite_a, offsite_b, atom_matches, sc_a
        )

        # Commit state
        self._sc_a          = sc_a
        self._sc_b          = sc_b
        self._refsite_frac  = np.asarray(all_refsites[0])
        self._offsite_a     = offsite_a
        self._offsite_b     = offsite_b
        self._matched       = matched
        self._unmatched_a   = unmatched_a
        self._unmatched_b   = unmatched_b
        self._selected_midx = None

        # Update 3D views
        self._view_a.load_supercell(sc_a)
        self._view_a.set_ref_site(self._refsite_frac)
        self._view_b.load_supercell(sc_b)
        self._view_b.set_ref_site(self._refsite_frac)

        # Reset connection buttons
        for btn, view in [(self._btn_conn_a, self._view_a),
                          (self._btn_conn_b, self._view_b)]:
            btn.setChecked(False)
            btn.setText('Show connections')
            view.set_refsite_bonds(None)

        # Scatter + table
        self._rebuild_checkboxes()
        self._refresh_plot()
        self._refresh_table()

        # Result summary
        _, total = stiffness_shift_from_pairs(matched)
        excl_note = f'  excl. {next(iter(excl_sp))} pairs\n' if excl_sp else ''
        self._result_lbl.setText(
            f'Matched: {len(matched)}   '
            f'Unmatched A: {len(unmatched_a)}   '
            f'Unmatched B: {len(unmatched_b)}\n'
            f'{excl_note}'
            f'Σ ΔpFC (B−A): {total:+.5f} eV/Å²'
        )

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

        seen_species = set()   # track which species pairs appear in matched

        for pt in self._pair_types:
            if pt not in active:
                continue
            c1, _ = self._view_a.pair_colours_hex(pt[0], pt[1])

            sub = [(i, m) for i, m in enumerate(self._matched)
                   if (m['species1'], m['species2']) == pt]
            if sub:
                midxs = [i for i, _ in sub]
                ms    = [m for _, m in sub]
                xa = np.array([m['atom1_ref_dist_a'] for m in ms])
                ya = np.array([m['mean_pfc_a']       for m in ms])
                xb = np.array([m['atom1_ref_dist_b'] for m in ms])
                yb = np.array([m['mean_pfc_b']       for m in ms])

                # Connecting lines (behind points)
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

        # Unmatched A (grey x)
        um_a = [r for r in self._unmatched_a
                if (r['species1'], r['species2']) in active]
        if um_a:
            ax.scatter(
                [r.get('atom1_ref_dist', 0.0) for r in um_a],
                [r['mean_pfc'] for r in um_a],
                s=28, color='#aaaaaa', marker='x',
                linewidths=1.2, zorder=2,
            )

        # Unmatched B (grey squares)
        um_b = [r for r in self._unmatched_b
                if (r['species1'], r['species2']) in active]
        if um_b:
            ax.scatter(
                [r.get('atom1_ref_dist', 0.0) for r in um_b],
                [r['mean_pfc'] for r in um_b],
                s=22, color='#aaaaaa', marker='s',
                edgecolors='none', zorder=2,
            )

        # Gold ring on selected pair (both A and B points)
        if self._selected_midx is not None and self._selected_midx < len(self._matched):
            m  = self._matched[self._selected_midx]
            pt = (m['species1'], m['species2'])
            if pt in active:
                for px, py in [(m['atom1_ref_dist_a'], m['mean_pfc_a']),
                               (m['atom1_ref_dist_b'], m['mean_pfc_b'])]:
                    ax.scatter([px], [py], s=140,
                               facecolors='none', edgecolors='#c8a000',
                               linewidths=2.5, zorder=5)

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
        if um_a:
            handles.append(Line2D([0], [0], linestyle='', marker='x',
                                  color='#aaaaaa', markersize=7,
                                  markeredgewidth=1.4, label='Unmatched A'))
        if um_b:
            handles.append(Line2D([0], [0], linestyle='', marker='s',
                                  color='#aaaaaa', markersize=6,
                                  label='Unmatched B'))
        if handles:
            ax.legend(handles=handles, loc='upper right',
                      framealpha=0.9, fontsize=8)

        ax.set_xlabel('Atom 1 – refsite distance (Å)', fontsize=11)
        ax.set_ylabel('Projected force constant (eV/Å²)', fontsize=11)
        ax.set_title('Stiffness shift: A (○) vs B (△)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _refresh_table(self):
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        sorted_pairs = sorted(
            enumerate(self._matched),
            key=lambda p: abs(p[1]['delta_pfc']),
            reverse=True,
        )

        for midx, m in sorted_pairs:
            row = self._table.rowCount()
            self._table.insertRow(row)

            sp_item = QTableWidgetItem(f"{m['species1']}–{m['species2']}")
            sp_item.setFlags(sp_item.flags() & ~Qt.ItemIsEditable)
            sp_item.setData(Qt.UserRole, midx)
            self._table.setItem(row, 0, sp_item)

            for col, val in enumerate([
                m.get('atom1_ref_dist_a', 0.0),
                m.get('atom1_ref_dist_b', 0.0),
                m['mean_pfc_a'],
                m['mean_pfc_b'],
                m['delta_pfc'],
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
                self._table.scrollToItem(
                    item, QAbstractItemView.PositionAtCenter
                )
                self._table.blockSignals(False)
                return

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_scatter_click(self, event):
        if event.inaxes is None or not self._scatter_pts or event.button != 1:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return

        ax      = self._ax
        x_range = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1
        y_range = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1

        best_d, best_midx = float('inf'), None
        for pts_list in self._scatter_pts.values():
            for pt_info in pts_list:
                for px, py in [(pt_info['xa'], pt_info['ya']),
                               (pt_info['xb'], pt_info['yb'])]:
                    dx = (px - cx) / x_range
                    dy = (py - cy) / y_range
                    d  = (dx**2 + dy**2) ** 0.5
                    if d < best_d:
                        best_d, best_midx = d, pt_info['midx']

        if best_midx is None or best_d > PICK_TOLERANCE:
            return
        self._select_pair(best_midx, source='scatter')

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

    def _select_pair(self, midx: int, source: str = 'click'):
        self._selected_midx = midx
        if midx >= len(self._matched):
            return
        m = self._matched[midx]

        self._sel_bar.setText(
            f'  {m["species1"]}–{m["species2"]}'
            f'   A: {m["atom1_idx_a"]}→{m["atom2_idx_a"]}'
            f'  pFC = {m["mean_pfc_a"]:+.5f}'
            f'   B: {m["atom1_idx_b"]}→{m["atom2_idx_b"]}'
            f'  pFC = {m["mean_pfc_b"]:+.5f}'
            f'   ΔpFC = {m["delta_pfc"]:+.5f} eV/Å²'
        )

        if source != 'scatter':
            self._refresh_plot()
        if source != 'table':
            self._sync_table_selection()

        self._view_a.highlight_bond(m['atom1_idx_a'], m['atom2_idx_a'])
        self._view_b.highlight_bond(m['atom1_idx_b'], m['atom2_idx_b'])
