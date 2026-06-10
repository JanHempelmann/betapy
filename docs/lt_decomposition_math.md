# LT Decomposition — Linear Algebra Primer

## Symbols

| Symbol | Meaning |
|--------|---------|
| **Φ** | 3×3 force-constant matrix for an atom pair (i, j) |
| **ê** | Unit vector pointing from atom i to atom j (the bond direction) |
| **êᵀ** | The transpose of **ê** — turns it from a column into a row vector |
| φ_L | Longitudinal (bond-stretching) force constant |
| φ_T | Average transverse (bending) force constant |
| Tr(**Φ**) | Trace — sum of the diagonal elements Φ₁₁ + Φ₂₂ + Φ₃₃ |

---

## What does êᵀ Φ ê actually compute?

**ê** is a 3-element column vector, e.g. `[0.6, 0.8, 0.0]` for a bond at 53° in the xy-plane.

The product `Φ ê` multiplies the 3×3 matrix by the column vector, giving a new 3-element vector — this is the force response of the pair *in the direction of the bond*.

Then `êᵀ (Φ ê)` takes the dot product of that response with **ê** again, projecting it *back onto the bond axis*. The result is a single number: how much of the force acts purely along the bond.

In index notation:

```
φ_L = Σᵢⱼ  êᵢ Φᵢⱼ êⱼ
```

This is called a **bilinear form** — ê appears on both sides, sandwiching Φ.

---

## Why this equals "rotate into bond frame, read off Φ₁₁"

If you built a rotation matrix **R** that aligns the bond with the x-axis, then in the rotated frame:

```
Φ' = R Φ Rᵀ
```

the (1,1) entry of Φ' is exactly φ_L. But `êᵀ Φ ê` gives you that same number directly — **R** never needs to be constructed. The bilinear form *is* the rotation, implicitly.

---

## Why the transverse part uses the Trace

The trace `Tr(Φ) = Φ₁₁ + Φ₂₂ + Φ₃₃` is **rotationally invariant** — it has the same value no matter how you orient your coordinate axes. This is a standard result from linear algebra (the trace equals the sum of eigenvalues).

In the bond-aligned frame, the three diagonal entries of Φ' are:

```
Φ'₁₁ = φ_L      (along bond)
Φ'₂₂ = φ_T1     (transverse direction 1)
Φ'₃₃ = φ_T2     (transverse direction 2)
```

So:

```
Tr(Φ) = φ_L + φ_T1 + φ_T2
```

Assuming the two transverse directions behave equally (isotropic transverse response):

```
φ_T = φ_T1 ≈ φ_T2 = (Tr(Φ) − φ_L) / 2
```

No rotation needed — the invariance of the trace does the work.

---

## Summary

```
φ_L = êᵀ Φ ê                   ← project onto bond axis (bilinear form)
φ_T = (Tr(Φ) − φ_L) / 2        ← remainder, split over 2 transverse DOF
```

Both quantities are independent of which Cartesian coordinate system you use.
The bond vector **ê** carries all the geometric information about the pair orientation.
