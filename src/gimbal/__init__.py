"""Modello dinamico rigido a 2 GdL di un telescopio su montatura alt-azimutale.

Il pacchetto implementa, modulo per modulo, la specifica
``SPEC_modello_gimbal_2dof.md``:

===================  ====================================================
modulo               contenuto
===================  ====================================================
:mod:`cad_data`      dati grezzi Onshape (sezione 3), immutabili
:mod:`params`        lettura di ``config.yaml`` e conversione in SI (sez. 4)
:mod:`inertia`       preprocessing inerziale (sezione 5)
:mod:`kinematics`    cinematica simbolica, formula di Poisson (sezione 6)
:mod:`lagrange`      equazioni di Lagrange e forma matriciale (sezione 6)
:mod:`linearize`     linearizzazione, spazio di stato, FdT (sezione 7)
:mod:`analysis`      poli, zeri, analisi parametriche (sezione 8)
:mod:`simulate`      scenari S1..S5 (sezione 8.4)
:mod:`plotting`      figure (sezione 8.2, 8.3)
:mod:`report`        generazione di ``results/report.md`` (sezione 11)
:mod:`flex4dof`      estensione flessibile a 4 GdL (sezione 10)
===================  ====================================================

Convenzioni vincolanti (sezione 2 della specifica): unita' SI internamente,
angoli in rad, tensori d'inerzia in convenzione tensoriale
``I_ij = -int(x_i x_j dm)`` per ``i != j``.
"""

from __future__ import annotations

__all__ = [
    "cad_data",
    "params",
    "inertia",
    "kinematics",
    "lagrange",
    "linearize",
    "analysis",
    "simulate",
    "plotting",
    "report",
    "flex4dof",
]

__version__ = "1.0.0"
