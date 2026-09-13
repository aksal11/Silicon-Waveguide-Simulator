
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags



# ============================================================
# fn: CREATE WAVEGUIDE INDEX PROFILE
# ============================================================
def create_waveguide_profile(x, width, n_core, n_clad):

    # validate
    if width <= 0:
        raise ValueError("Width must be positive")

    if n_core <= n_clad:
        raise ValueError("n_core must be greater than n_clad")

    if not isinstance(x, np.ndarray):
        raise TypeError("x must be Numpy array")

    if x.ndim != 1:
        raise ValueError("X must be one-dimensional array")

    n = np.where(
    np.abs(x) <= width/2, #if this point is inside the wavelength
    n_core,
    n_clad
)
    return n


# ============================================================
# fn: BUILD WAVEGUIDE MATRIX
# ============================================================
def build_waveguide_matrix(x, n_profile, wavelength):

    # validate
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array")

    if not isinstance(n_profile, np.ndarray):
        raise TypeError("n_profile must be a NumPy array")

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")

    if n_profile.ndim != 1:
        raise ValueError("n_profile must be one-dimensional")

    if len(x) != len(n_profile):
        raise ValueError("x and n_profile must have the same length")

    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")

    # grid spacing
    dx = x[1] - x[0]

    if not np.allclose(np.diff(x), dx):
        raise ValueError("x must be uniformly spaced")

    # free-space wavenumber
    k0 = 2 * np.pi / wavelength

    # number of interior points
    N_interior = len(x) - 2

    # second derivative matrix
    main_diagonal = -2 * np.ones(N_interior)
    off_diagonal = np.ones(N_interior - 1)

    D2 = diags(
        [off_diagonal, main_diagonal, off_diagonal],
        [-1, 0, 1]
    ) / dx**2

    # refractive-index term
    n_interior = n_profile[1:-1]

    material_term = k0**2 * n_interior**2

    material_matrix = diags(material_term, 0)

    # complete wave equation matrix
    A = D2 + material_matrix

    return A

# ============================================================
# fn: SOLVE MODES
# ============================================================

def solve_modes(A, wavelength, num_modes=4):
    
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    # solve eigenvalue problem
    eigenvalues, eigenvectors = eigsh(
        A,
        k=num_modes,
        which="LA"
    )

    # sort eigenvalues from largest to smallest
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # beta^2 = eigenvalue
    beta_squared = eigenvalues

    # beta = sqrt(beta^2)
    beta = np.sqrt(beta_squared)

    # free-space wavenumber
    k0 = 2 * np.pi / wavelength

    # effective refractive index
    n_eff = beta / k0

    return beta_squared, beta, n_eff, eigenvectors

# ============================================================
# MATERIAL PROPERTIES
# ============================================================
n_core = 3.48
n_clad = 1.44

wavelength = 1550e-9

width = 450e-9

# ============================================================
# SIMULATION GRID
# ============================================================
N = 1000

x_min = -2e-6
x_max = 2e-6

x = np.linspace(
    x_min,
    x_max,
    N
)

# ============================================================
# REFRACTIVE INDEX PROFILE
# ============================================================
n = create_waveguide_profile(
    x,
    width,
    n_core,
    n_clad
)

# ============================================================
# WAVEGUIDE MATRIX
# ============================================================
A = build_waveguide_matrix(
    x,
    n,
    wavelength
)

# ============================================================
# solver
# ============================================================
beta_squared, beta, n_eff, modes = solve_modes(
    A,
    wavelength,
    num_modes=4
)


# ============================================================
# OUTPUT
# ============================================================
print("========================================")
print("SILICON WAVEGUIDE SIMULATOR")
print("========================================")
print("Wavelength =", wavelength * 1e9, "nm")
print("Core index =", n_core)
print("Cladding index =", n_clad)
print("Waveguide width =", width * 1e9, "nm")
print("Grid points =", N)

print("\nWaveguide matrix:")
print(A)

print("\nCalculated modes:")

for i in range(len(n_eff)):

    print(f"Mode {i}")
    print(f"  beta^2 = {beta_squared[i]:.6e}")
    print(f"  beta   = {beta[i]:.6e} 1/m")
    print(f"  n_eff  = {n_eff[i]:.6f}")


# ============================================================
# PLOT REFRACTIVE INDEX
# ============================================================
plt.figure(figsize=(10,6))

plt.plot(
    x * 1e6,
    n
)

plt.xlabel("x (µm)")
plt.ylabel("Refractive index")
plt.title("Silicon Waveguide Refractive-Index Profile")

plt.grid()
plt.show()

