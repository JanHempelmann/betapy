"""
Standalone LOBSTER multicenter COBI browser.

Independent of the phonon-based multicenter bonding detector
(betapy.core.multicenter / betapy.gui.multicenter_viewer) — this widget only
needs a LOBSTER directory (POSCAR + COBICAR.lobster [+ NcICOBILIST.lobster])
to list every multicenter (Nc) chain the file contains and browse its
orbitalwise breakdown (px/py/pz, individual d orbitals, etc.), independent
of whether any phonon force-constant anomaly detection has been run.

Usage
-----
    widget = LobsterCobiWidget()
    widget.set_lobster_dir(lobster_dir)   # also reachable via the widget's
                                           # own 'Open LOBSTER directory…' button
"""

import csv
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QLabel, QDoubleSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from betapy.core.lobster import (
    parse_poscar_lobster, parse_car_header, parse_ncicobi_list,
    lookup_ncicobi, entry_to_directive,
    load_nc_entry_orbital_curves, group_orbital_curves_by_type,
    filter_orbital_curves,
)
from betapy.gui.plot_utils import symmetric_xlim, exact_energy_ylim

_TOTAL_COLOUR = '#111111'
# Okabe-Ito colorblind-safe qualitative palette, cycled for orbital curves.
_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
            '#E69F00', '#56B4E9', '#F0E442', '#999999']


class LobsterCobiWidget(QWidget):
    """
    Standalone browser for LOBSTER multicenter (Nc) COBI data, orbitalwise.

    Lists every multicenter chain found in COBICAR.lobster — independent of
    any phonon force-constant analysis. Selecting a chain shows its total
    energy-resolved curve plus a groupable/filterable list of orbital-
    combination curves, overlaid on a shared plot, exportable as a vector
    graphic.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lobster_dir        = None
        self._lob_poscar         = None
        self._nc_icobi_records   = []
        self._cobicar_path       = None
        self._header             = None
        self._entries            = []    # header['nc_pairs'], in list order
        self._current_entry      = None
        self._current_result     = None  # load_nc_entry_orbital_curves() output
        self._displayed_orbitals = []     # rows currently in the orbital list, in order
        self._build_ui()

    # ------------------------------------------------------------------ public API

    def set_lobster_dir(self, lobster_dir, silent: bool = False) -> bool:
        """
        Point the browser at a LOBSTER directory. Returns True if a usable
        POSCAR + COBICAR.lobster were found and parsed, False otherwise.

        With silent=False (the default, used by the widget's own 'Open
        LOBSTER directory…' button) a warning dialog explains what's
        missing. Pass silent=True for a best-effort push from elsewhere in
        the app (e.g. auto-discovery alongside an SPOSCAR load) — most
        LOBSTER runs have no COBICAR.lobster at all, so that path must not
        pop up a dialog every time.
        """
        return self._load_dir(Path(lobster_dir), silent=silent)

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        self._btn_open = QPushButton('Open LOBSTER directory…')
        self._btn_open.clicked.connect(self._browse_lobster_dir)
        self._lbl_dir = QLabel('No LOBSTER directory loaded.')
        self._lbl_dir.setStyleSheet('color: #666;')
        top.addWidget(self._btn_open)
        top.addWidget(self._lbl_dir, stretch=1)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        # ── Left panel — chain list ────────────────────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(2, 2, 2, 2)
        self._lbl_chains = QLabel('Chains: —')
        self._lbl_chains.setStyleSheet('font-size: 10px; color: #444;')
        lv.addWidget(self._lbl_chains)
        self._chain_list = QListWidget()
        self._chain_list.setFont(mono)
        self._chain_list.setToolTip(
            'Every multicenter (Nc) chain found in COBICAR.lobster.\n'
            'Select one to browse its orbitalwise breakdown.')
        self._chain_list.itemSelectionChanged.connect(self._on_chain_selected)
        lv.addWidget(self._chain_list, stretch=1)
        splitter.addWidget(left)

        # ── Middle panel — orbital list + controls ────────────────────
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(2, 2, 2, 2)

        self._chk_group = QCheckBox('Group by orbital type (s/p/d/f)')
        self._chk_group.setChecked(True)
        self._chk_group.setToolTip(
            'Collapse individual m-resolved rows (5p_x, 5p_y, 5p_z, …)\n'
            'into coarse s/p/d/f-type sums.')
        self._chk_group.stateChanged.connect(self._rebuild_orbital_list)
        mv.addWidget(self._chk_group)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel('Hide |ICOBI| <'))
        self._spin_threshold = QDoubleSpinBox()
        self._spin_threshold.setDecimals(5)
        self._spin_threshold.setRange(0.0, 10.0)
        self._spin_threshold.setSingleStep(0.001)
        self._spin_threshold.setValue(0.0)
        self._spin_threshold.setToolTip(
            'Unlist orbital contributions with nothing going on\n'
            '(|ICOBI(N) at EF| below this value). 0 = show everything.')
        self._spin_threshold.valueChanged.connect(self._rebuild_orbital_list)
        thr_row.addWidget(self._spin_threshold)
        thr_row.addStretch()
        mv.addLayout(thr_row)

        self._lbl_orbitals = QLabel('—')
        self._lbl_orbitals.setStyleSheet('font-size: 10px; color: #444;')
        mv.addWidget(self._lbl_orbitals)

        self._orbital_list = QListWidget()
        self._orbital_list.setFont(mono)
        self._orbital_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._orbital_list.setToolTip(
            'Orbital-combination rows, sorted by |ICOBI(N) at EF|.\n'
            'Select one or more to overlay their curves on the plot.')
        self._orbital_list.itemSelectionChanged.connect(self._draw_plot)
        mv.addWidget(self._orbital_list, stretch=1)

        sel_row = QHBoxLayout()
        btn_select_all = QPushButton('Select shown')
        btn_select_all.clicked.connect(self._orbital_list.selectAll)
        btn_clear = QPushButton('Clear')
        btn_clear.clicked.connect(self._orbital_list.clearSelection)
        sel_row.addWidget(btn_select_all)
        sel_row.addWidget(btn_clear)
        sel_row.addStretch()
        mv.addLayout(sel_row)

        splitter.addWidget(mid)

        # ── Right panel — plot ─────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self._lbl_total = QLabel('')
        self._lbl_total.setAlignment(Qt.AlignCenter)
        rv.addWidget(self._lbl_total)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        rv.addWidget(self._toolbar)
        rv.addWidget(self._canvas, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_export_csv = QPushButton('Export CSV…')
        self._btn_export_csv.setToolTip(
            'Save the currently shown orbital rows (total + whatever grouping/\n'
            'threshold is applied) as a CSV table of ICOBI(N) at EF values')
        self._btn_export_csv.clicked.connect(self._export_csv)
        self._btn_export_csv.setEnabled(False)
        self._btn_export = QPushButton('Export plot…')
        self._btn_export.setToolTip('Save the current plot as SVG/PDF/PNG')
        self._btn_export.clicked.connect(self._export_plot)
        self._btn_export.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_export_csv)
        btn_row.addWidget(self._btn_export)
        rv.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([300, 320, 480])

        self._draw_placeholder()

    # ------------------------------------------------------------------ data loading

    def _browse_lobster_dir(self):
        start = str(self._lobster_dir) if self._lobster_dir else str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, 'Select LOBSTER calculation directory', start)
        if path:
            self._load_dir(Path(path), silent=False)

    def _load_dir(self, ldir: Path, silent: bool = False) -> bool:
        poscar_path = next(
            (ldir / n for n in ('POSCAR.lobster', 'POSCAR.lobster.vasp', 'POSCAR')
             if (ldir / n).exists()), None)
        if poscar_path is None:
            if not silent:
                QMessageBox.warning(
                    self, 'No POSCAR found',
                    f'No POSCAR file found in:\n{ldir}\n\n'
                    'A POSCAR (or POSCAR.lobster[.vasp]) is required for atom mapping.')
            return False

        cobicar_path = ldir / 'COBICAR.lobster'
        if not cobicar_path.exists():
            if not silent:
                QMessageBox.warning(
                    self, 'No COBICAR.lobster found',
                    f'No COBICAR.lobster found in:\n{ldir}\n\n'
                    'Multicenter COBI browsing requires COBICAR.lobster, produced '
                    'by one or more cobiBetween directives in lobsterin.')
            return False

        try:
            lob_poscar = parse_poscar_lobster(poscar_path)
            header     = parse_car_header(cobicar_path)
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, 'Failed to parse LOBSTER files', str(exc))
            return False

        self._lobster_dir  = ldir
        self._lob_poscar   = lob_poscar
        self._cobicar_path = cobicar_path
        self._header       = header
        self._entries      = header['nc_pairs']

        self._nc_icobi_records = []
        ni_path = ldir / 'NcICOBILIST.lobster'
        if ni_path.exists():
            try:
                self._nc_icobi_records = parse_ncicobi_list(ni_path)
            except Exception:
                pass

        self._lbl_dir.setText(str(ldir))
        self._populate_chain_list()
        return True

    def _populate_chain_list(self):
        self._chain_list.clear()
        for entry in self._entries:
            n = len(entry['atoms'])
            chain_str = ' → '.join(lbl for lbl, _ in entry['atoms'])
            directive = entry_to_directive(entry)

            icobi_ref = (lookup_ncicobi(self._nc_icobi_records, directive, self._lob_poscar)
                         if self._nc_icobi_records else None)
            icobi_tag = f'   ICOBI(N)={icobi_ref:+.5f}' if icobi_ref is not None else ''
            orb_tag   = (f'  ·  {len(entry["orbital_rows"])} orbital rows'
                         if entry['orbital_rows'] else '')

            item = QListWidgetItem(f'{n}-center: {chain_str}{icobi_tag}{orb_tag}')
            item.setData(Qt.UserRole, entry)
            self._chain_list.addItem(item)

        self._lbl_chains.setText(f'Chains: {len(self._entries)}')

    # ------------------------------------------------------------------ selection / display

    def _on_chain_selected(self):
        items = self._chain_list.selectedItems()
        if not items:
            self._current_entry  = None
            self._current_result = None
            self._orbital_list.clear()
            self._draw_placeholder()
            return

        entry = items[0].data(Qt.UserRole)
        self._current_entry = entry
        try:
            self._current_result = load_nc_entry_orbital_curves(
                self._cobicar_path, self._header, entry)
        except Exception as exc:
            QMessageBox.warning(self, 'Failed to load curves', str(exc))
            self._current_result = None
        self._rebuild_orbital_list()

    def _rebuild_orbital_list(self):
        self._orbital_list.clear()
        self._displayed_orbitals = []

        if self._current_result is None:
            self._lbl_total.setText('')
            self._lbl_orbitals.setText('—')
            self._draw_plot()
            return

        rows = self._current_result['orbitals']
        if self._chk_group.isChecked():
            rows = group_orbital_curves_by_type(rows)

        threshold = self._spin_threshold.value()
        shown = filter_orbital_curves(rows, threshold) if threshold > 0 else rows
        self._displayed_orbitals = shown

        total = self._current_result['total']
        self._lbl_total.setText(
            f"ICOBI(N) @ EF = {total['ival_ef']:+.5f}" if total is not None
            else 'no total row written for this chain (requested orbitalwise only)')

        hidden = len(rows) - len(shown)
        note = f'  ({hidden} hidden, |ICOBI| < {threshold:g})' if hidden else ''
        self._lbl_orbitals.setText(f'{len(shown)} shown{note}')

        for row in shown:
            key   = row.get('orbital_types', row.get('orbitals'))
            label = ' '.join(key)
            n_rows = row.get('n_rows')
            suffix = f'  (Σ{n_rows})' if n_rows and n_rows > 1 else ''
            self._orbital_list.addItem(
                QListWidgetItem(f'{label:<28s}{row["ival_ef"]:+.5f}{suffix}'))

        self._draw_plot()

    def _draw_plot(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._btn_export.setEnabled(self._current_result is not None)
        self._btn_export_csv.setEnabled(self._current_result is not None)

        if self._current_result is None:
            ax.text(0.5, 0.5, 'Select a chain', transform=ax.transAxes,
                    ha='center', va='center', color='grey', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            self._canvas.draw_idle()
            return

        any_line   = False
        energy_ref = None
        plotted_curves = []
        total = self._current_result['total']
        if total is not None:
            ax.plot(total['curve'], total['energy'], color=_TOTAL_COLOUR,
                    linewidth=1.6, label=f"total ({total['ival_ef']:+.4f})", zorder=3)
            any_line = True
            energy_ref = total['energy']
            plotted_curves.append(total['curve'])

        selected_idx = sorted(self._orbital_list.row(it)
                              for it in self._orbital_list.selectedItems())
        for n, idx in enumerate(selected_idx):
            row    = self._displayed_orbitals[idx]
            key    = row.get('orbital_types', row.get('orbitals'))
            label  = ' '.join(key)
            colour = _PALETTE[n % len(_PALETTE)]
            ax.plot(row['curve'], row['energy'], color=colour, linewidth=1.1,
                    label=f"{label} ({row['ival_ef']:+.4f})", zorder=2)
            any_line = True
            energy_ref = row['energy']
            plotted_curves.append(row['curve'])

        ax.axhline(0, color='#777', linestyle='--', linewidth=0.9, zorder=1)
        ax.axvline(0, color='#999', linestyle='-',  linewidth=0.5, zorder=1)
        ax.grid(True, linestyle=':', alpha=0.35)
        ax.set_xlabel('COBI', fontsize=10)
        ax.set_ylabel('Energy (eV)', fontsize=10)
        if any_line:
            ax.legend(fontsize=7, loc='best')

        xlim = symmetric_xlim(plotted_curves)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ylim = exact_energy_ylim(energy_ref)
        if ylim is not None:
            ax.set_ylim(*ylim)

        self._canvas.draw_idle()

    def _draw_placeholder(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Open a LOBSTER directory\nwith COBICAR.lobster',
                transform=ax.transAxes, ha='center', va='center',
                color='grey', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ export

    def _export_plot(self):
        if self._current_entry is None:
            return
        chain_str = '-'.join(lbl for lbl, _ in self._current_entry['atoms'])
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export plot', f'cobi_{chain_str}.svg',
            'SVG (*.svg);;PDF (*.pdf);;PNG (*.png);;All files (*)')
        if path:
            self._figure.savefig(path)

    def _export_csv(self):
        if self._current_entry is None or self._current_result is None:
            return
        chain_str = '-'.join(lbl for lbl, _ in self._current_entry['atoms'])
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export CSV', f'cobi_{chain_str}.csv',
            'CSV (*.csv);;All files (*)')
        if not path:
            return

        directive = entry_to_directive(self._current_entry)
        grouped   = self._chk_group.isChecked()
        threshold = self._spin_threshold.value()

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# chain', chain_str])
            writer.writerow(['# directive', directive])
            writer.writerow(['# grouped_by_orbital_type', grouped])
            writer.writerow(['# hide_below_threshold', threshold])
            writer.writerow([])
            writer.writerow(['orbitals', 'n_rows', 'ICOBI(N)_at_EF'])

            total = self._current_result['total']
            if total is not None:
                writer.writerow(['TOTAL', '', f"{total['ival_ef']:.6f}"])

            for row in self._displayed_orbitals:
                key = row.get('orbital_types', row.get('orbitals'))
                writer.writerow(
                    [' '.join(key), row.get('n_rows', 1), f"{row['ival_ef']:.6f}"])
