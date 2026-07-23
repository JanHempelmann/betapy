"""
Standalone LOBSTER multicenter COBI browser.

Independent of the phonon-based multicenter bonding detector
(betapy.core.multicenter / betapy.gui.multicenter_viewer) — this widget only
needs a LOBSTER directory (POSCAR + NcICOBILIST.lobster) to list every
multicenter (Nc, 3+ atom) chain and browse its orbitalwise breakdown
(px/py/pz, individual d orbitals, etc.).

Data source note
-----------------
By default this widget reads only NcICOBILIST.lobster's own integrated
"Nc-ICOBI (at) eF" values, not COBICAR.lobster's energy-resolved curves.
In our own testing (not traced to a betapy parsing issue — checked against
the raw file text directly), COBICAR.lobster/COHPCAR.lobster/COOPCAR.lobster
energy-resolved values for an interaction spanning a periodic-cell
translation did not match the corresponding integrated value, while
interactions confined to the origin cell (e.g. a chain fully internalised
in a large enough supercell, needing no periodic-image translation) agreed
well. We can't say how general this is across LOBSTER versions or settings,
so this is presented as something worth checking per calculation, not as a
blanket claim about LOBSTER.

NcICOBILIST is not assumed exempt from the same kind of divergence just
because it's a different file — whether it holds up for a given
calculation isn't something we can determine in general. So rather than
assume, this widget runs an empirical self-consistency check
(check_ncicobi_consistency) on every directory it loads and reports the
result in a banner at the top: LOBSTER writes an automatic Nc-ICOBI entry
for every ordinary nearest-neighbour pair regardless of any cobiBetween
directive, and most bonds appear at more than one periodic translation, so
the same physical bond can almost always be cross-checked against itself.
Treat the banner's verdict as the actual basis for trust, not the mere
presence of this file.

The 'Use COBICAR energy-resolved curves' checkbox (off by default) lets you
opt back into COBICAR-sourced curves, e.g. for chains you already know (or
have arranged, via a supercell) to be translation-free. A warning is shown
whenever the selected chain actually involves a translation, flagging that
this is the situation where we've observed COBICAR and NcICOBILIST to
disagree.

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
    parse_poscar_lobster, parse_ncicobi_list, parse_car_header,
    entry_to_directive, is_translation_free, check_ncicobi_consistency,
    load_nccobicar_orbital_curves,
    group_orbital_curves_by_type, filter_orbital_curves,
)
from betapy.gui.plot_utils import symmetric_xlim, exact_energy_ylim, export_figure

# Above this relative spread (fraction of the mean |ICOBI|), a bond that
# should agree across periodic-cell translations is flagged as suspicious —
# ordinary DFT/numerical noise for a consistent computation is typically
# well under 1%; the divergence we've observed for translated interactions
# runs to 100%+, sometimes with a sign flip.
_CONSISTENCY_WARN_THRESHOLD = 0.05

_TOTAL_COLOUR = '#111111'
_COLOUR_BOND  = '#4d94ff'   # positive ICOBI — bonding, matches cohp_viewer.py
_COLOUR_ANTI  = '#ff6666'   # negative ICOBI — antibonding
# Okabe-Ito colorblind-safe qualitative palette, cycled for overlaid curves
# (COBICAR curve mode only — the bar chart colours by sign instead).
_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
            '#E69F00', '#56B4E9', '#F0E442', '#999999']


class LobsterCobiWidget(QWidget):
    """
    Standalone browser for LOBSTER multicenter (Nc) COBI data, orbitalwise.

    Lists every multicenter (3+ atom) chain found in NcICOBILIST.lobster —
    independent of any phonon force-constant analysis. Selecting a chain
    shows a bar chart of its total ICOBI(N) plus a groupable/filterable
    breakdown by orbital combination, exportable as a vector graphic or CSV.
    Optionally (off by default) sources energy-resolved curves from
    COBICAR.lobster instead — see the module docstring for the reliability
    caveat that motivates the default being off.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lobster_dir          = None
        self._lob_poscar           = None
        self._entries              = []    # NcICOBILIST records with n_atoms >= 3
        self._current_entry        = None
        self._displayed_orbitals   = []     # rows currently in the orbital list, in order
        self._current_total_value  = None   # whichever source is currently active
        self._current_curve_result = None   # load_nccobicar_orbital_curves() output, or None
        self._cobicar_path         = None
        self._cobicar_header       = None
        self._cobicar_load_failed  = False
        self._build_ui()

    # ------------------------------------------------------------------ public API

    def set_lobster_dir(self, lobster_dir, silent: bool = False) -> bool:
        """
        Point the browser at a LOBSTER directory. Returns True if a usable
        POSCAR + NcICOBILIST.lobster were found and parsed, False otherwise.

        With silent=False (the default, used by the widget's own 'Open
        LOBSTER directory…' button) a warning dialog explains what's
        missing. Pass silent=True for a best-effort push from elsewhere in
        the app (e.g. auto-discovery alongside an SPOSCAR load) — most
        LOBSTER runs have no multicenter directives at all, so that path
        must not pop up a dialog every time.
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

        self._lbl_reliability = QLabel('')
        self._lbl_reliability.setWordWrap(True)
        self._lbl_reliability.setStyleSheet('font-size: 10px;')
        root.addWidget(self._lbl_reliability)

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
            'Every multicenter (3+ atom) chain found in NcICOBILIST.lobster.\n'
            '[cell-free] chains involve no periodic-cell translation — the\n'
            'only case where COBICAR energy-resolved curves are reliable.\n'
            'Select a chain to browse its orbitalwise breakdown.')
        self._chain_list.itemSelectionChanged.connect(self._on_chain_selected)
        lv.addWidget(self._chain_list, stretch=1)
        splitter.addWidget(left)

        # ── Middle panel — orbital list + controls ────────────────────
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(2, 2, 2, 2)

        self._chk_use_cobicar = QCheckBox('Use COBICAR energy-resolved curves')
        self._chk_use_cobicar.setChecked(False)
        self._chk_use_cobicar.setToolTip(
            'Off by default: in our testing, COBICAR.lobster (and COHPCAR/\n'
            'COOPCAR) energy-resolved values did not match the integrated\n'
            'value for interactions spanning a periodic-cell translation.\n'
            'Only trust this for chains marked [cell-free], or once you\'ve\n'
            'confirmed your own LOBSTER output agrees for the chain you\'re\n'
            'looking at. See docs/lobster-cobi.md.')
        self._chk_use_cobicar.stateChanged.connect(self._rebuild_orbital_list)
        mv.addWidget(self._chk_use_cobicar)

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
            'Bar-chart mode: click to highlight the matching bar.\n'
            'COBICAR curve mode: select one or more to overlay their curves.')
        self._orbital_list.itemSelectionChanged.connect(self._on_orbital_selection_changed)
        mv.addWidget(self._orbital_list, stretch=1)

        splitter.addWidget(mid)

        # ── Right panel — plot ─────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self._lbl_total = QLabel('')
        self._lbl_total.setAlignment(Qt.AlignCenter)
        rv.addWidget(self._lbl_total)

        self._lbl_warning = QLabel('')
        self._lbl_warning.setAlignment(Qt.AlignCenter)
        self._lbl_warning.setWordWrap(True)
        self._lbl_warning.setStyleSheet('color: #b35900; font-weight: bold; font-size: 10px;')
        rv.addWidget(self._lbl_warning)

        self._chk_gridlines = QCheckBox('Show gridlines')
        self._chk_gridlines.setChecked(True)
        self._chk_gridlines.stateChanged.connect(self._draw)
        rv.addWidget(self._chk_gridlines)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        rv.addWidget(self._toolbar)
        rv.addWidget(self._canvas, stretch=1)

        self._chk_lock_aspect = QCheckBox('Lock aspect ratio for export (narrow, golden ratio)')
        self._chk_lock_aspect.setToolTip(
            'Exported plots use a fixed narrow portrait aspect ratio\n'
            '(width : height = 1 : 1.618, the golden ratio) instead of\n'
            'whatever shape this panel happens to be on screen.\n'
            'Only affects the saved file — this preview is unaffected.')
        rv.addWidget(self._chk_lock_aspect)

        btn_row = QHBoxLayout()
        self._btn_export_csv = QPushButton('Export CSV…')
        self._btn_export_csv.setToolTip(
            'Save the currently shown orbital rows (total + whatever grouping/\n'
            'threshold is applied) as a CSV table of ICOBI(N) at EF values')
        self._btn_export_csv.clicked.connect(self._export_csv)
        self._btn_export_csv.setEnabled(False)
        self._btn_export = QPushButton('Export plot…')
        self._btn_export.setToolTip('Save the current chart as SVG/PDF/PNG')
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

        ni_path = ldir / 'NcICOBILIST.lobster'
        if not ni_path.exists():
            if not silent:
                QMessageBox.warning(
                    self, 'No NcICOBILIST.lobster found',
                    f'No NcICOBILIST.lobster found in:\n{ldir}\n\n'
                    'Multicenter COBI browsing requires NcICOBILIST.lobster, produced '
                    'by one or more cobiBetween directives in lobsterin.')
            return False

        try:
            lob_poscar = parse_poscar_lobster(poscar_path)
            records    = parse_ncicobi_list(ni_path)
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, 'Failed to parse LOBSTER files', str(exc))
            return False

        self._lobster_dir = ldir
        self._lob_poscar   = lob_poscar
        # Plain 2-atom entries duplicate ordinary nearest-neighbour ICOBI
        # (already covered by the pairwise LOBSTER integration elsewhere);
        # this browser is specifically for genuine multicenter chains.
        self._entries = [r for r in records if r['n_atoms'] >= 3]

        # COBICAR is loaded lazily (only if/when the curve-mode checkbox is
        # used), since most sessions will never need it.
        self._cobicar_path        = None
        self._cobicar_header      = None
        self._cobicar_load_failed = False

        self._lbl_dir.setText(str(ldir))
        self._update_reliability_label(records, lob_poscar)
        self._populate_chain_list()
        return True

    def _update_reliability_label(self, records, lob_poscar):
        """
        We don't assume NcICOBILIST is reliable for translated interactions
        just because it's a different file from COBICAR — that has to be
        checked, not assumed, for each calculation. The same physical bond,
        reached via different periodic-cell translations, should agree;
        LOBSTER writes such duplicates automatically for ordinary
        nearest-neighbour pairs, so this is usually possible to check
        directly against the loaded file itself.
        """
        try:
            consistency = check_ncicobi_consistency(records, lob_poscar)
        except Exception:
            self._lbl_reliability.setText('')
            return

        if not consistency:
            self._lbl_reliability.setText(
                'Reliability check: no bond was found at more than one translation '
                'in this file, so NcICOBILIST could not be cross-checked here.')
            self._lbl_reliability.setStyleSheet('font-size: 10px; color: #666;')
            return

        worst = consistency[0]
        n_bad = sum(1 for c in consistency if c['relative_spread'] >= _CONSISTENCY_WARN_THRESHOLD)
        if n_bad:
            self._lbl_reliability.setText(
                f'⚠ Reliability check: {n_bad}/{len(consistency)} cross-checked bonds disagree by '
                f'{_CONSISTENCY_WARN_THRESHOLD:.0%}+ across translations (worst: '
                f"{worst['sp1']}-{worst['sp2']} @ {worst['distance']:.3f} Å, "
                f"{worst['relative_spread']:.0%}). This directory's NcICOBILIST shows a "
                'similar disagreement to what we\'ve seen affect COBICAR for translated '
                'interactions — treat multicenter values here with the same caution.')
            self._lbl_reliability.setStyleSheet('font-size: 10px; color: #b35900; font-weight: bold;')
        else:
            self._lbl_reliability.setText(
                f'✓ Reliability check: {len(consistency)} bond(s) cross-checked across translations, '
                f"largest disagreement {worst['relative_spread']:.2%} — looks internally consistent.")
            self._lbl_reliability.setStyleSheet('font-size: 10px; color: #2a7a2a;')

    def _populate_chain_list(self):
        self._chain_list.clear()
        for entry in self._entries:
            n = len(entry['atoms'])
            chain_str = ' → '.join(lbl for lbl, _ in entry['atoms'])
            orb_tag = (f'  ·  {len(entry["orbital_rows"])} orbital rows'
                       if entry['orbital_rows'] else '  ·  no orbitalwise data')
            free_tag = '  [cell-free]' if is_translation_free(entry) else ''

            item = QListWidgetItem(
                f'{n}-center: {chain_str}   ICOBI(N)={entry["icobi"]:+.5f}{orb_tag}{free_tag}')
            item.setData(Qt.UserRole, entry)
            self._chain_list.addItem(item)

        self._lbl_chains.setText(f'Chains: {len(self._entries)}')

    # ------------------------------------------------------------------ COBICAR (opt-in)

    def _load_curve_result(self, entry):
        """
        Load COBICAR.lobster energy-resolved data for *entry*, lazily
        parsing its header on first use. Returns None (after a one-time
        warning dialog) if COBICAR.lobster is missing, fails to parse, or
        doesn't contain a matching chain.
        """
        if self._cobicar_header is None and not self._cobicar_load_failed:
            cobicar_path = self._lobster_dir / 'COBICAR.lobster'
            if not cobicar_path.exists():
                self._cobicar_load_failed = True
                QMessageBox.warning(
                    self, 'No COBICAR.lobster found',
                    f'No COBICAR.lobster found in:\n{self._lobster_dir}\n\n'
                    'Falling back to NcICOBILIST integrated values.')
                return None
            try:
                self._cobicar_header = parse_car_header(cobicar_path)
                self._cobicar_path   = cobicar_path
            except Exception as exc:
                self._cobicar_load_failed = True
                QMessageBox.warning(self, 'Failed to parse COBICAR.lobster', str(exc))
                return None

        if self._cobicar_header is None:
            return None

        directive = entry_to_directive(entry)
        try:
            return load_nccobicar_orbital_curves(
                self._cobicar_path, self._cobicar_header, directive, self._lob_poscar)
        except Exception:
            return None

    # ------------------------------------------------------------------ selection / display

    def _on_chain_selected(self):
        items = self._chain_list.selectedItems()
        self._current_entry = items[0].data(Qt.UserRole) if items else None
        self._rebuild_orbital_list()

    def _rebuild_orbital_list(self):
        self._orbital_list.clear()
        self._displayed_orbitals   = []
        self._current_curve_result = None

        if self._current_entry is None:
            self._lbl_total.setText('')
            self._lbl_orbitals.setText('—')
            self._lbl_warning.setText('')
            self._draw()
            return

        entry = self._current_entry
        translation_free = is_translation_free(entry)
        warning = ''

        if self._chk_use_cobicar.isChecked():
            curve_result = self._load_curve_result(entry)
            if curve_result is not None:
                self._current_curve_result = curve_result
                rows = curve_result['orbitals']
                total = curve_result['total']
                self._current_total_value = (
                    total['ival_ef'] if total is not None else entry['icobi'])
                if not translation_free:
                    warning = ('⚠ this chain involves a periodic-cell translation — '
                              'this is the situation where we\'ve observed COBICAR values '
                              'to disagree with the integrated ones (see docs/lobster-cobi.md)')
            else:
                rows = entry['orbital_rows']
                self._current_total_value = entry['icobi']
                warning = 'COBICAR.lobster has no matching data for this chain — showing NcICOBILIST instead'
        else:
            rows = entry['orbital_rows']
            self._current_total_value = entry['icobi']

        if self._chk_group.isChecked():
            rows = group_orbital_curves_by_type(rows)

        threshold = self._spin_threshold.value()
        shown = filter_orbital_curves(rows, threshold) if threshold > 0 else rows
        self._displayed_orbitals = shown

        source_tag = '  (COBICAR curve)' if self._current_curve_result is not None else ''
        self._lbl_total.setText(f'ICOBI(N) @ EF = {self._current_total_value:+.5f}{source_tag}')
        self._lbl_warning.setText(warning)

        hidden = len(rows) - len(shown)
        note = f'  ({hidden} hidden, |ICOBI| < {threshold:g})' if hidden else ''
        self._lbl_orbitals.setText(f'{len(shown)} shown{note}')

        for row in shown:
            key    = row.get('orbital_types', row.get('orbitals'))
            label  = ' '.join(key)
            n_rows = row.get('n_rows')
            suffix = f'  (Σ{n_rows})' if n_rows and n_rows > 1 else ''
            self._orbital_list.addItem(
                QListWidgetItem(f'{label:<28s}{row["ival_ef"]:+.5f}{suffix}'))

        self._draw()

    def _on_orbital_selection_changed(self):
        if self._current_curve_result is not None:
            self._draw_curves()
        else:
            self._highlight_selected_bars()

    def _draw(self):
        if self._current_curve_result is not None:
            self._draw_curves()
        else:
            self._draw_bars()

    def _draw_bars(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        has_data = self._current_entry is not None
        self._btn_export.setEnabled(has_data)
        self._btn_export_csv.setEnabled(has_data)

        if not has_data:
            ax.text(0.5, 0.5, 'Select a chain', transform=ax.transAxes,
                    ha='center', va='center', color='grey', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            self._canvas.draw_idle()
            return

        rows   = self._displayed_orbitals
        labels = ['total'] + [' '.join(r.get('orbital_types', r.get('orbitals')))
                              for r in rows]
        values = [self._current_total_value] + [r['ival_ef'] for r in rows]
        colours = [_TOTAL_COLOUR] + [
            _COLOUR_BOND if v >= 0 else _COLOUR_ANTI for v in values[1:]]

        y_pos = list(range(len(values)))
        self._bar_container = ax.barh(y_pos, values, color=colours, height=0.7, zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()   # total on top, then descending |value|

        ax.axvline(0, color='#999', linestyle='-', linewidth=0.7, zorder=1)
        if self._chk_gridlines.isChecked():
            ax.grid(True, axis='x', linestyle=':', alpha=0.35)
        else:
            ax.grid(False)
        ax.set_xlabel('ICOBI(N) @ EF', fontsize=10)

        xlim = symmetric_xlim([values])
        if xlim is not None:
            ax.set_xlim(*xlim)

        self._canvas.draw_idle()

    def _highlight_selected_bars(self):
        if not hasattr(self, '_bar_container') or self._bar_container is None:
            return
        selected_rows = {self._orbital_list.row(it)
                         for it in self._orbital_list.selectedItems()}
        for i, bar in enumerate(self._bar_container):
            # bar index 0 is 'total'; orbital rows start at index i-1
            is_selected = (i - 1) in selected_rows
            bar.set_edgecolor('black' if is_selected else 'none')
            bar.set_linewidth(1.4 if is_selected else 0.0)
        self._canvas.draw_idle()

    def _draw_curves(self):
        """COBICAR curve mode: energy-resolved overlay, selection-gated."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._btn_export.setEnabled(True)
        self._btn_export_csv.setEnabled(True)

        result = self._current_curve_result
        any_line = False
        energy_ref = None
        plotted_curves = []

        total = result['total']
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
        if self._chk_gridlines.isChecked():
            ax.grid(True, linestyle=':', alpha=0.35)
        else:
            ax.grid(False)
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
        ax.text(0.5, 0.5, 'Open a LOBSTER directory\nwith NcICOBILIST.lobster',
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
            export_figure(self._figure, path,
                          lock_aspect=self._chk_lock_aspect.isChecked())
            self._canvas.draw_idle()

    def _export_csv(self):
        if self._current_entry is None:
            return
        chain_str = '-'.join(lbl for lbl, _ in self._current_entry['atoms'])
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export CSV', f'cobi_{chain_str}.csv',
            'CSV (*.csv);;All files (*)')
        if not path:
            return

        entry     = self._current_entry
        directive = entry_to_directive(entry)
        grouped   = self._chk_group.isChecked()
        threshold = self._spin_threshold.value()
        source    = 'COBICAR' if self._current_curve_result is not None else 'NcICOBILIST'

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# chain', chain_str])
            writer.writerow(['# directive', directive])
            writer.writerow(['# data_source', source])
            writer.writerow(['# grouped_by_orbital_type', grouped])
            writer.writerow(['# hide_below_threshold', threshold])
            writer.writerow([])
            writer.writerow(['orbitals', 'n_rows', 'ICOBI(N)_at_EF'])
            writer.writerow(['TOTAL', '', f'{self._current_total_value:.6f}'])

            for row in self._displayed_orbitals:
                key = row.get('orbital_types', row.get('orbitals'))
                writer.writerow(
                    [' '.join(key), row.get('n_rows', 1), f"{row['ival_ef']:.6f}"])
