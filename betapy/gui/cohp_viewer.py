"""
Energy-resolved LOBSTER curve viewer (COHP / COOP / COBI).

Opens as a persistent, non-modal window.  Call show_pair() whenever the
user selects a bond in the pFC viewer — the plot updates in-place so the
window can stay open across multiple selections.

Plot convention (standard LOBSTER style)
-----------------------------------------
- Energy (eV, Fermi = 0) on the vertical axis
- Curve value on the horizontal axis
- Blue fill: bonding region  (COHP < 0  |  COOP > 0  |  COBI > 0)
- Red fill : antibonding region
- Dashed horizontal line at E = 0 (Fermi level)
"""

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from betapy.core.lobster import (
    parse_car_header, enrich_cobicar_distances, load_car_curves,
)

_CAR_FILES = {
    'cohp': 'COHPCAR.lobster',
    'coop': 'COOPCAR.lobster',
    'cobi': 'COBICAR.lobster',
}

_LABELS = {
    'cohp': 'COHP (eV)',
    'coop': 'COOP',
    'cobi': 'COBI',
}

_BONDING_NEGATIVE = {'cohp'}   # negative values = bonding (COOP/COBI: positive = bonding)
_COLOUR_BOND  = '#4d94ff'
_COLOUR_ANTI  = '#ff6666'


class COHPViewerWidget(QDialog):
    """
    Non-modal dialog showing energy-resolved LOBSTER curves for a selected pair.

    Usage
    -----
        viewer = COHPViewerWidget(parent)
        viewer.set_lobster_dir(lobster_dir)
        viewer.show_pair('Sc', 'F', 2.012)   # opens / updates
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('LOBSTER — Energy-resolved bonding')
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)   # keep alive on close

        self._lobster_dir    = None
        self._headers        = {}   # car_type → header dict
        self._cache          = {}   # (car_type, sp1, sp2, distance) → list[dict]
        self._loaded_groups  = {}   # car_type → list[dict] for current pair
        self._current_pair   = None
        self._selected_group = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lobster_dir(self, lobster_dir) -> None:
        """
        Point the viewer at a LOBSTER output directory.

        Parses all available *CAR.lobster headers immediately (fast).
        Curve data is loaded lazily on first show_pair() call.
        """
        lobster_dir = Path(lobster_dir)
        if lobster_dir == self._lobster_dir:
            return
        self._lobster_dir = lobster_dir
        self._headers.clear()
        self._cache.clear()

        poscar_lob = next(
            (lobster_dir / n for n in ('POSCAR.lobster', 'POSCAR.lobster.vasp', 'POSCAR')
             if (lobster_dir / n).exists()),
            None,
        )

        for car_type, fname in _CAR_FILES.items():
            fpath = lobster_dir / fname
            if not fpath.exists():
                continue
            try:
                hdr = parse_car_header(fpath)
                if car_type == 'cobi' and poscar_lob is not None:
                    enrich_cobicar_distances(hdr, poscar_lob)
                self._headers[car_type] = hdr
            except Exception:
                pass

    def show_pair(self, sp1: str, sp2: str, distance: float) -> None:
        """Update the plot for (sp1, sp2, distance) and bring the window forward."""
        if (sp1, sp2, distance) != self._current_pair:
            self._selected_group = 0
        self._current_pair = (sp1, sp2, distance)
        self._update_plot()
        if not self.isVisible():
            self._position_beside_parent()
        self.show()
        self.raise_()
        self.activateWindow()

    def _position_beside_parent(self):
        """On first show, place to the right of the parent window (screen-safe)."""
        top = self.parent().window() if self.parent() else None
        if top is None:
            return
        pg = top.frameGeometry()
        x, y = pg.right() + 10, pg.top()
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.screenAt(pg.center())
        if screen:
            sr = screen.availableGeometry()
            if x + self.width() > sr.right():
                x = max(sr.left(), pg.left() - self.width() - 10)
            y = max(sr.top(), min(y, sr.bottom() - self.height()))
        self.move(x, y)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._title_label = QLabel('')
        self._title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title_label)

        # Group selector row — hidden when only one group exists
        self._group_row = QWidget()
        gr = QHBoxLayout(self._group_row)
        gr.setContentsMargins(0, 0, 0, 0)
        self._group_combo = QComboBox()
        self._group_combo.currentIndexChanged.connect(self._on_group_changed)
        self._group_warn = QLabel('⚠ Divergent values in distance shell')
        self._group_warn.setStyleSheet('color: #cc7700;')
        gr.addWidget(QLabel('Group:'))
        gr.addWidget(self._group_combo)
        gr.addStretch()
        gr.addWidget(self._group_warn)
        self._group_row.setVisible(False)
        layout.addWidget(self._group_row)

        self.figure = Figure(figsize=(5, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        self._info_label = QLabel('')
        self._info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._info_label)

        btn_row = QHBoxLayout()
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.hide)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.resize(480, 600)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _update_plot(self):
        if self._current_pair is None or not self._headers:
            return
        sp1, sp2, distance = self._current_pair

        available = list(self._headers.keys())
        if not available:
            return

        # Load groups for each car_type
        self._loaded_groups = {}
        for car_type in available:
            self._loaded_groups[car_type] = self._get_curves(car_type, sp1, sp2, distance)

        max_groups = max((len(g) for g in self._loaded_groups.values()), default=0)

        # Rebuild combo box without triggering redraws mid-update
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        if max_groups > 1:
            # Use the first type with multiple groups (prefer cobi) for item labels
            label_type = next(
                (t for t in ('cobi', 'cohp', 'coop')
                 if t in self._loaded_groups and len(self._loaded_groups[t]) > 1),
                available[0],
            )
            label_groups = self._loaded_groups[label_type]
            qty = _LABELS.get(label_type, label_type.upper()).split()[0]
            for i, grp in enumerate(label_groups):
                self._group_combo.addItem(
                    f'Group {i + 1} — I{qty} = {grp["ival_ef"]:.4f}  (n={grp["n"]})'
                )
            for i in range(len(label_groups), max_groups):
                self._group_combo.addItem(f'Group {i + 1}')
            self._group_combo.setCurrentIndex(
                min(self._selected_group, self._group_combo.count() - 1)
            )
            self._group_row.setVisible(True)
        else:
            self._group_row.setVisible(False)
        self._group_combo.blockSignals(False)

        cs1, cs2 = sorted([sp1, sp2])
        self._title_label.setText(f'{cs1}–{cs2}   d = {distance:.4f} Å')

        self._draw_plots()

    def _draw_plots(self):
        if not self._loaded_groups:
            return

        available = list(self._loaded_groups.keys())
        n = len(available)

        self.figure.clear()
        _raw = self.figure.subplots(1, n, sharey=True)
        axes = [_raw] if n == 1 else list(_raw)

        icohp_summary = []

        for ax, car_type in zip(axes, available):
            groups = self._loaded_groups[car_type]

            ax.set_xlabel(_LABELS.get(car_type, car_type.upper()), fontsize=10)
            ax.axhline(0, color='#777', linestyle='--', linewidth=0.9, zorder=1)
            ax.axvline(0, color='#999', linestyle='-',  linewidth=0.5, zorder=1)
            ax.grid(True, linestyle=':', alpha=0.35)

            if not groups:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='grey', fontsize=9)
                continue

            grp_idx = min(self._selected_group, len(groups) - 1)
            result  = groups[grp_idx]
            energy  = result['energy']
            curve   = result['curve']
            icurve  = result['icurve']
            bonding_negative = car_type in _BONDING_NEGATIVE

            if bonding_negative:
                ax.fill_betweenx(energy, 0, curve,
                                 where=(curve <= 0),
                                 color=_COLOUR_BOND, alpha=0.35, linewidth=0)
                ax.fill_betweenx(energy, 0, curve,
                                 where=(curve >= 0),
                                 color=_COLOUR_ANTI, alpha=0.35, linewidth=0)
            else:
                ax.fill_betweenx(energy, 0, curve,
                                 where=(curve >= 0),
                                 color=_COLOUR_BOND, alpha=0.35, linewidth=0)
                ax.fill_betweenx(energy, 0, curve,
                                 where=(curve <= 0),
                                 color=_COLOUR_ANTI, alpha=0.35, linewidth=0)

            ax.plot(curve, energy, color='#111', linewidth=1.0, zorder=2)

            ef_idx = int(np.argmin(np.abs(energy)))
            ival = icurve[ef_idx]
            icohp_summary.append(f'I{car_type.upper()}(eF) = {ival:.4f}')

        if axes:
            axes[0].set_ylabel('Energy (eV)', fontsize=10)

        self._info_label.setText('   '.join(icohp_summary) if icohp_summary else '')
        self.canvas.draw_idle()

    def _on_group_changed(self, idx: int) -> None:
        self._selected_group = idx
        self._draw_plots()

    def _get_curves(self, car_type, sp1, sp2, distance) -> list:
        """Load curves with a per-pair cache. Returns list[dict]."""
        key = (car_type, sp1, sp2, round(distance, 4))
        if key in self._cache:
            return self._cache[key]

        hdr   = self._headers.get(car_type)
        fpath = self._lobster_dir / _CAR_FILES[car_type]
        if hdr is None or not fpath.exists():
            self._cache[key] = []
            return []

        result = load_car_curves(fpath, hdr, sp1, sp2, distance)
        self._cache[key] = result
        return result
