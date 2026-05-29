# pFC decomposition: longitudinal and transverse components

## Notation and symbols

### Phonopy force constant matrices (tensors)

$$\boldsymbol{\Phi}_{\kappa\kappa'} \in \mathbb{R}^{3\times3}$$

The interatomic force constant (IFC) matrix from phonopy for atom pair $(\kappa, \kappa')$
in the supercell. Its elements are defined as the second derivative of the potential energy:

$$\Phi^{\alpha\beta}_{\kappa\kappa'} = \frac{\partial^2 U}{\partial u^\alpha_\kappa \, \partial u^\beta_{\kappa'}}$$

where $u^\alpha_\kappa$ is the displacement of atom $\kappa$ in Cartesian direction $\alpha \in \{x,y,z\}$.
This is the raw 3×3 matrix that phonopy writes to `FORCE_CONSTANTS`.

The Newton/symmetry relation (finite supercell limit) is:

$$\Phi^{\alpha\beta}_{\kappa\kappa'} = \Phi^{\beta\alpha}_{\kappa'\kappa}
\quad\Longleftrightarrow\quad
\boldsymbol{\Phi}_{\kappa'\kappa} = \boldsymbol{\Phi}_{\kappa\kappa'}^{\top}$$

### Geometric quantities (scalars and vectors)

| Symbol | Type | Definition |
|--------|------|------------|
| $\mathbf{r}(\kappa)$ | vector (Å) | Cartesian position of atom $\kappa$ in the supercell |
| $\mathbf{r}_{\kappa\kappa'} = \mathbf{r}(\kappa') - \mathbf{r}(\kappa)$ | vector (Å) | interatomic displacement vector |
| $d_{\kappa\kappa'} = \|\mathbf{r}_{\kappa\kappa'}\|$ | scalar (Å) | interatomic distance |
| $\hat{e}_{\kappa\kappa'} = \mathbf{r}_{\kappa\kappa'} / d_{\kappa\kappa'}$ | unit vector | bond direction (from $\kappa$ toward $\kappa'$) |

---

## 1. Current betapy pFC

Displace atom $\kappa'$ by one unit along $\hat{e}$. The resulting force 3-vector on atom $\kappa$ is:

$$\mathbf{f} = \boldsymbol{\Phi}_{\kappa\kappa'} \, \hat{e}_{\kappa\kappa'} \;\in\; \mathbb{R}^3$$

This vector has components both *along* and *perpendicular to* the bond, because
$\boldsymbol{\Phi}_{\kappa\kappa'}$ is a general (not necessarily symmetric or diagonal)
3×3 matrix. Taking its Euclidean norm gives the scalar force magnitude:

$$\Phi_\mathrm{p}(\kappa\kappa') = \|\boldsymbol{\Phi}_{\kappa\kappa'} \, \hat{e}\| \;=\; \sqrt{\hat{e}^\top \boldsymbol{\Phi}_{\kappa\kappa'}^\top \boldsymbol{\Phi}_{\kappa\kappa'} \hat{e}}$$

The symmetrized (averaged) value used in betapy is:

$$\Phi_\mathrm{p}^\mathrm{sym}(\kappa\kappa') = \frac{1}{2}\left(
    \|\boldsymbol{\Phi}_{\kappa\kappa'} \, \hat{e}\| +
    \|\boldsymbol{\Phi}_{\kappa\kappa'}^\top \hat{e}\|
\right)$$

where $\boldsymbol{\Phi}_{\kappa\kappa'}^\top \hat{e} = \boldsymbol{\Phi}_{\kappa'\kappa} \hat{e}$
is the force on atom $\kappa'$ when $\kappa$ is displaced along the same bond direction.

**Important:** $\|\boldsymbol{\Phi} \hat{e}\|$ is *not* the same as the scalar quadratic form
$\hat{e}^\top \boldsymbol{\Phi} \hat{e}$ (see Section 2 below). The norm $\|\boldsymbol{\Phi}\hat{e}\|$
includes the transverse force response — force generated *perpendicular* to the bond
when displaced *along* it — but does *not* include the response to perpendicular displacements.

---

## 2. Longitudinal (stretching) projection

Project the force vector $\mathbf{f}$ onto the bond direction $\hat{e}$:

$$\varphi_\mathrm{L}(\kappa\kappa') = \hat{e}^\top \boldsymbol{\Phi}_{\kappa\kappa'} \hat{e} \;\in\; \mathbb{R}$$

This is the classical **bond stretching force constant**: the component of the restoring
force along the bond axis when atom $\kappa'$ is displaced along that axis.
It is a scalar quadratic form — the analogue of a spring constant.

**Sign convention:** $\varphi_\mathrm{L} < 0$ means the interaction is bonding/restoring
(displacement of $\kappa'$ away from $\kappa$ exerts a force pulling $\kappa$ toward $\kappa'$);
$\varphi_\mathrm{L} > 0$ means antibonding. (Note: betapy's current $\|\boldsymbol{\Phi}\hat{e}\|$
loses this sign information.)

**Symmetry under index swap:** Since the quadratic form is invariant under transposition,

$$\hat{e}^\top \boldsymbol{\Phi}_{\kappa\kappa'} \hat{e}
= \hat{e}^\top \boldsymbol{\Phi}_{\kappa\kappa'}^\top \hat{e}
= \hat{e}^\top \boldsymbol{\Phi}_{\kappa'\kappa} \hat{e}$$

so $\varphi_\mathrm{L}(\kappa\kappa') = \varphi_\mathrm{L}(\kappa'\kappa)$ **identically**.
No symmetrization step is needed; both displacement directions yield the same value.

The relationship to the current pFC quantity:

$$|\Phi_\mathrm{p}(\kappa\kappa')|^2 = \varphi_\mathrm{L}^2 + |\mathbf{f}_\perp|^2$$

where $\mathbf{f}_\perp = \boldsymbol{\Phi}\hat{e} - \varphi_\mathrm{L}\hat{e}$
is the component of the force response perpendicular to the bond.
The current pFC is therefore always $\geq |\varphi_\mathrm{L}|$.

---

## 3. Average transverse (bending) projection via the trace

Choose any orthonormal basis $\{\hat{e}, \hat{e}_{\perp 1}, \hat{e}_{\perp 2}\}$ aligned with the bond.
The trace of $\boldsymbol{\Phi}_{\kappa\kappa'}$ decomposes as:

$$\mathrm{Tr}(\boldsymbol{\Phi}_{\kappa\kappa'})
= \hat{e}^\top \boldsymbol{\Phi} \hat{e}
+ \hat{e}_{\perp 1}^\top \boldsymbol{\Phi} \hat{e}_{\perp 1}
+ \hat{e}_{\perp 2}^\top \boldsymbol{\Phi} \hat{e}_{\perp 2}
= \varphi_\mathrm{L} + \varphi_{\perp 1} + \varphi_{\perp 2}$$

Because $\mathrm{Tr}$ is rotationally invariant, we can define the average transverse
projection **without choosing $\hat{e}_{\perp 1}, \hat{e}_{\perp 2}$ explicitly**:

$$\boxed{\varphi_\mathrm{T}(\kappa\kappa')
= \frac{\mathrm{Tr}(\boldsymbol{\Phi}_{\kappa\kappa'}) - \varphi_\mathrm{L}(\kappa\kappa')}{2}}$$

Physical meaning: $\varphi_\mathrm{T}$ is the average force constant for displacing
atom $\kappa'$ **perpendicular** to the bond — the average bending stiffness.

**Symmetry under index swap:** Both $\mathrm{Tr}$ and $\varphi_\mathrm{L}$ are invariant
under $\boldsymbol{\Phi} \to \boldsymbol{\Phi}^\top$, so $\varphi_\mathrm{T}(\kappa\kappa') = \varphi_\mathrm{T}(\kappa'\kappa)$
**identically**. Again, no symmetrization step is needed.

**Contrast with the 2014 Lee et al. descriptor** (doi: 10.1038/ncomms4525): that paper
uses $\mathrm{Tr}(\boldsymbol{\Phi}_{\kappa\kappa'}) / \mathrm{Tr}(\boldsymbol{\Phi}_{\kappa\kappa})$
as a single scalar, normalizing by the on-site self-interaction trace. This captures
overall interaction magnitude but discards directional information entirely.
The present decomposition retains the directional split into stretching vs. bending.

---

## 4. Information content summary

| Quantity | Symbol | Formula | Includes |
|----------|--------|---------|----------|
| Current betapy pFC | $\Phi_\mathrm{p}^\mathrm{sym}$ | $\tfrac{1}{2}(\|\boldsymbol{\Phi}\hat{e}\| + \|\boldsymbol{\Phi}^\top\hat{e}\|)$ | stretching + shear coupling (mixed) |
| Longitudinal | $\varphi_\mathrm{L}$ | $\hat{e}^\top \boldsymbol{\Phi} \hat{e}$ | pure stretching only; retains sign |
| Shear coupling | $\|\mathbf{f}_\perp\|$ | $\sqrt{\|\boldsymbol{\Phi}\hat{e}\|^2 - \varphi_\mathrm{L}^2}$ | force ⊥ bond when displaced ∥ bond |
| Average bending | $\varphi_\mathrm{T}$ | $(\mathrm{Tr}(\boldsymbol{\Phi}) - \varphi_\mathrm{L})/2$ | pure bending; force ∥ ⊥ when displaced ⊥ |

All four are rotationally invariant and require no arbitrary choice of coordinate axes.

---

## 5. Bond-character anisotropy

The ratio

$$A(\kappa\kappa') = \frac{\varphi_\mathrm{T}(\kappa\kappa')}{\varphi_\mathrm{L}(\kappa\kappa')}$$

characterizes the angular character of the interaction:

- **Ionic limit** (isotropic Coulombic): from the acoustic sum rule, the self-term
  $\boldsymbol{\Phi}_{\kappa\kappa} = -\sum_{\kappa'\neq\kappa}\boldsymbol{\Phi}_{\kappa\kappa'}$.
  For a high-symmetry ionic pair along $\hat{x}$, one expects
  $\Phi^{yy} = \Phi^{zz} \approx -\tfrac{1}{2}\Phi^{xx}$, giving $A \approx -\tfrac{1}{2}$.

- **Covalent limit** (directed bond): stretching dominates over bending,
  so $|\varphi_\mathrm{L}| \gg |\varphi_\mathrm{T}|$ and $|A| \ll \tfrac{1}{2}$.

This ratio could serve as a bond-type fingerprint in a Badger-rule style plot,
complementary to the existing pFC vs. distance analysis.
