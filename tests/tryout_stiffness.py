import numpy as np

# Materiale: acciaio
G = 7.7e10  # Pa, modulo di taglio

# Geometria albero (sezione circolare piena)
d_1 = 10e-3   # m, diametro
L = 0.2     # m, lunghezza

J_1 = np.pi * d_1**4 / 32  # m^4, momento d'inerzia polare

k_1 = G * J_1 / L  # Nm/rad, rigidezza torsionale

print(f"J_1 = {J_1:.3e} m^4")
print(f"k_1 = {k_1:.3f} Nm/rad")



# Geometria albero (sezione ad anello circolare)
d_i = 8e-3   # m, diametro interno
d_e = 10e-3  # m, diametro esterno

J_anello = np.pi * (d_e**4 - d_i**4) / 32  # m^4, momento d'inerzia polare
J_int = np.pi * d_i**4 / 32

k_anello = G * J_anello / L  # Nm/rad, rigidezza torsionale
k_int = G * J_int / L

k_tot = k_anello + k_int
print(f"J_anello = {J_anello:.3e} m^4")
print(f"k_anello = {k_anello:.3f} Nm/rad")
print(f"J_int = {J_int:.3e} m^4")
print(f"k_int = {k_int:.3f} Nm/rad")
print(f"k_tot = {k_tot:.3f} Nm/rad")
