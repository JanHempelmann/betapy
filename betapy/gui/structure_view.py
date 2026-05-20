"""
Shared 3D structure renderer for betapy.

Features:
  - CPK/Jmol atom colours with per-species colour picker
  - Automatic bond drawing with per-species-pair enable/disable toggles
  - On pair selection: centers atom1 at cell centre (minimum-image convention)
    and recomputes bond endpoints from shifted positions so no bonds cross cell
  - Non-selected atoms dimmed when pair highlighted
  - Reference site marker sphere
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QColorDialog,
    QGroupBox, QScrollArea, QCheckBox,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

from betapy.data.elements import covalent_radius, element_colour, display_radius

# -----------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------

BOND_FACTOR           = 1.1
BOND_RADIUS           = 0.06
BOND_COLOUR           = (0.65, 0.65, 0.65)
BOND_OPACITY          = 0.5

HIGHLIGHT_COLOUR      = (1.0, 0.75, 0.0)
HIGHLIGHT_ATOM_FACTOR = 1.4
HIGHLIGHT_BOND_RADIUS = 0.10

REFSITE_COLOUR        = (1.0, 0.2, 0.2)
REFSITE_RADIUS        = 0.30

DIM_OPACITY           = 0.12
FULL_OPACITY          = 1.0


class StructureView(QWidget):
    """
    Embeddable 3D structure viewer backed by PyVista/VTK.

    Parameters
    ----------
    parent            : QWidget or None
    show_color_picker : bool — show per-species colour picker panel
    """

    def __init__(self, parent=None, show_color_picker=True):
        super().__init__(parent)
        self.supercell       = None
        self._highlight_pair = None
        # bond_pairs: list of (i_1based, j_1based, sp_i, sp_j)
        self._bond_pairs     = []
        self._colours        = {}
        self._display_frac   = None
        # enabled bond types: set of frozenset({sp_i, sp_j})
        self._enabled_bond_types = set()
        self._bond_checkboxes    = {}   # frozenset -> QCheckBox

        self._build_ui(show_color_picker)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, show_color_picker):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        outer.addWidget(self.plotter, stretch=1)

        if show_color_picker:
            right_panel = QVBoxLayout()
            right_widget = QWidget()
            right_widget.setFixedWidth(155)
            right_widget.setLayout(right_panel)

            self._colour_group  = self._build_colour_panel()
            self._bond_group    = self._build_bond_toggle_panel()
            right_panel.addWidget(self._colour_group)
            right_panel.addWidget(self._bond_group)
            right_panel.addStretch()
            outer.addWidget(right_widget)
        else:
            self._colour_layout = None
            self._bond_layout   = None

    def _build_colour_panel(self):
        group = QGroupBox('Atom colours')
        self._colour_layout = QVBoxLayout()
        self._colour_layout.setAlignment(Qt.AlignTop)
        group.setLayout(self._colour_layout)
        scroll = QScrollArea()
        scroll.setWidget(group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        return scroll

    def _build_bond_toggle_panel(self):
        group = QGroupBox('Bonds')
        self._bond_layout = QVBoxLayout()
        self._bond_layout.setAlignment(Qt.AlignTop)
        group.setLayout(self._bond_layout)
        return group

    def _rebuild_colour_buttons(self):
        if self._colour_layout is None:
            return
        while self._colour_layout.count():
            item = self._colour_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for sp in sorted(self._colours.keys()):
            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(sp)
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            self._update_swatch(btn, self._colours[sp])
            btn.clicked.connect(
                lambda checked, s=sp, b=btn: self._pick_colour(s, b)
            )
            rl.addWidget(lbl)
            rl.addWidget(btn)
            self._colour_layout.addWidget(row)

    def _rebuild_bond_toggles(self):
        if self._bond_layout is None:
            return
        while self._bond_layout.count():
            item = self._bond_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._bond_checkboxes = {}

        bond_types = sorted(
            set(frozenset({sp_i, sp_j})
                for _, _, sp_i, sp_j in self._bond_pairs),
            key=lambda s: '-'.join(sorted(s))
        )
        for bt in bond_types:
            label = '-'.join(sorted(bt))
            cb = QCheckBox(label)
            cb.setChecked(bt in self._enabled_bond_types)
            cb.stateChanged.connect(
                lambda state, b=bt: self._on_bond_toggle(b, state)
            )
            self._bond_layout.addWidget(cb)
            self._bond_checkboxes[bt] = cb

    def _on_bond_toggle(self, bond_type, state):
        if state == Qt.Checked:
            self._enabled_bond_types.add(bond_type)
        else:
            self._enabled_bond_types.discard(bond_type)
        self._redraw()

    def _update_swatch(self, btn, rgb_float):
        r, g, b = [int(c * 255) for c in rgb_float]
        btn.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 1px solid #888;'
        )

    def _pick_colour(self, species, btn):
        current = self._colours.get(species, (0.5, 0.5, 0.5))
        r, g, b = [int(c * 255) for c in current]
        colour  = QColorDialog.getColor(
            QColor(r, g, b), self, f'{species} colour'
        )
        if colour.isValid():
            new_rgb = (colour.redF(), colour.greenF(), colour.blueF())
            self._colours[species] = new_rgb
            self._update_swatch(btn, new_rgb)
            self._redraw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_supercell(self, supercell):
        """Render all atoms and compute bonds. Call once per structure load."""
        self.supercell       = supercell
        self._highlight_pair = None
        self._display_frac   = supercell.positions.copy()
        self._colours        = {
            sp: element_colour(sp) for sp in supercell.chem_symbols
        }

        # Compute bond connectivity (indices + species)
        self._bond_pairs = self._compute_bonds(supercell)

        # Default: all bond types enabled except same-species metal pairs
        # (user can override via checkboxes)
        all_types = set(
            frozenset({sp_i, sp_j})
            for _, _, sp_i, sp_j in self._bond_pairs
        )
        metals = {'V', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                  'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'W', 'Re', 'Os',
                  'Ir', 'Pt', 'Au'}
        self._enabled_bond_types = {
            bt for bt in all_types
            if not (len(bt) == 1 and next(iter(bt)) in metals)
        }

        self._rebuild_colour_buttons()
        self._rebuild_bond_toggles()
        self._redraw()
        self.plotter.reset_camera()
        self.plotter.render()

    def highlight_bond(self, atom1_idx_1based, atom2_idx_1based):
        """Select a pair: center atom1, dim background, highlight bond."""
        self._highlight_pair = (atom1_idx_1based, atom2_idx_1based)
        self._update_display_frac(atom1_idx_1based)
        self._redraw()

    def clear_highlight(self):
        """Remove highlight and restore original view."""
        self._highlight_pair = None
        if self.supercell is not None:
            self._display_frac = self.supercell.positions.copy()
        self._redraw()

    def set_ref_site(self, frac_coords):
        """Place or move the reference site marker."""
        self.plotter.remove_actor('refsite_marker', render=False)
        if self.supercell is None:
            return
        cart   = np.asarray(frac_coords) @ self.supercell.lattice
        sphere = pv.Sphere(radius=REFSITE_RADIUS, center=cart,
                           theta_resolution=20, phi_resolution=20)
        self.plotter.add_mesh(
            sphere, color=REFSITE_COLOUR,
            name='refsite_marker', opacity=0.85, render=True,
        )

    # ------------------------------------------------------------------
    # Bond computation (once on load)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bonds(supercell):
        """
        Find all bonded pairs using a KDTree.
        Returns list of (i_1based, j_1based, species_i, species_j).
        """
        sc       = supercell
        cart_pos = sc.positions @ sc.lattice
        max_r    = max(covalent_radius(sp) for sp in sc.chem_symbols)
        q_radius = 2 * max_r * BOND_FACTOR * 1.05
        tree     = cKDTree(cart_pos)
        bonds    = []

        for i in range(sc.n_atoms):
            sp_i = sc.species(i + 1)
            r_i  = covalent_radius(sp_i)
            for j in tree.query_ball_point(cart_pos[i], q_radius):
                if j <= i:
                    continue
                sp_j = sc.species(j + 1)
                d    = sc.atom_distance(i + 1, j + 1)
                if d <= (r_i + covalent_radius(sp_j)) * BOND_FACTOR:
                    bonds.append((i + 1, j + 1, sp_i, sp_j))

        return bonds

    # ------------------------------------------------------------------
    # Display coordinate management
    # ------------------------------------------------------------------

    def _update_display_frac(self, center_idx_1based):
        """
        Shift all fractional coords so center_idx is at (0.5, 0.5, 0.5)
        using the minimum-image convention.
        Bond endpoints are recomputed from these coords in _redraw,
        so no bond crosses the cell after centering.
        """
        sc         = self.supercell
        pos_center = sc.positions[center_idx_1based - 1]
        display    = np.empty_like(sc.positions)
        for k in range(sc.n_atoms):
            diff       = sc.positions[k] - pos_center
            diff      -= np.floor(diff + 0.5)
            display[k] = 0.5 + diff
        self._display_frac = display

    # ------------------------------------------------------------------
    # Scene drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        """Full scene redraw from current display state."""
        if self.supercell is None:
            return

        self.plotter.clear_actors()
        sc            = self.supercell
        highlight     = self._highlight_pair
        selected_idxs = set(highlight) if highlight else set()

        # Cartesian positions from current display fractional coords
        cart = self._display_frac @ sc.lattice

        # --- Background bonds (batched, per-bond minimum-image endpoints) ---
        # Bond endpoints are computed from display coords of atom i, then
        # the minimum-image vector from i to j is added to get atom j's
        # display position. This ensures no bond crosses the cell regardless
        # of where atoms were shifted by the centering transform.
        active_bonds = [
            (i_1, j_1) for i_1, j_1, sp_i, sp_j in self._bond_pairs
            if frozenset({sp_i, sp_j}) in self._enabled_bond_types
        ]
        if active_bonds:
            bond_points, bond_lines, pt_idx = [], [], 0
            for i_1, j_1 in active_bonds:
                p1       = cart[i_1 - 1]
                # Minimum-image vector from display position of i to j
                frac_i   = self._display_frac[i_1 - 1]
                frac_j   = self._display_frac[j_1 - 1]
                diff     = frac_j - frac_i
                diff    -= np.floor(diff + 0.5)   # minimum image
                p2       = p1 + diff @ sc.lattice
                bond_points.extend([p1, p2])
                bond_lines.extend([2, pt_idx, pt_idx + 1])
                pt_idx += 2
            bond_mesh        = pv.PolyData()
            bond_mesh.points = np.array(bond_points)
            bond_mesh.lines  = np.array(bond_lines)
            bond_tubed       = bond_mesh.tube(radius=BOND_RADIUS, n_sides=6)
            self.plotter.add_mesh(
                bond_tubed,
                color=BOND_COLOUR,
                name='all_bonds',
                opacity=DIM_OPACITY if selected_idxs else BOND_OPACITY,
                render=False,
            )

        # --- Background atoms ---
        bg_groups  = {}
        opacity_bg = DIM_OPACITY if selected_idxs else FULL_OPACITY
        for i in range(sc.n_atoms):
            if (i + 1) in selected_idxs:
                continue
            sp = sc.species(i + 1)
            bg_groups.setdefault(sp, []).append(cart[i])

        for sp, positions in bg_groups.items():
            colour = self._colours.get(sp, (0.5, 0.5, 0.5))
            radius = display_radius(sp)
            cloud  = pv.PolyData(np.array(positions))
            proto  = pv.Sphere(radius=radius,
                               theta_resolution=8, phi_resolution=8)
            glyphs = cloud.glyph(geom=proto, scale=False, orient=False)
            self.plotter.add_mesh(
                glyphs, color=colour,
                name=f'atoms_{sp}',
                opacity=opacity_bg, render=False,
            )

        # --- Selected atoms (gold, larger) ---
        for idx_1 in selected_idxs:
            sp     = sc.species(idx_1)
            r      = display_radius(sp) * HIGHLIGHT_ATOM_FACTOR
            sphere = pv.Sphere(radius=r, center=cart[idx_1 - 1],
                               theta_resolution=18, phi_resolution=18)
            self.plotter.add_mesh(
                sphere, color=HIGHLIGHT_COLOUR,
                name=f'selected_{idx_1}',
                opacity=FULL_OPACITY, render=False,
            )

        # --- Highlighted bond ---
        if highlight:
            i_1, j_1 = highlight
            p1, p2   = cart[i_1 - 1], cart[j_1 - 1]
            tube = pv.Line(p1, p2).tube(
                radius=HIGHLIGHT_BOND_RADIUS, n_sides=16
            )
            self.plotter.add_mesh(
                tube, color=HIGHLIGHT_COLOUR,
                name='highlight_bond',
                opacity=FULL_OPACITY, render=False,
            )

        self.plotter.render()
