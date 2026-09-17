
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh



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
# fn: IDENTIFY GUIDED MODES
# ============================================================

def identify_guided_modes(n_eff, n_core, n_clad):

    # guided modes must satisfy:
    # n_clad < n_eff < n_core

    guided = (
        (n_eff > n_clad) &
        (n_eff < n_core)
    )

    return guided

# ============================================================
# fn: VALIDATE MODE RESULTS
# ============================================================
def validate_modes(beta_squared, n_eff, n_core, n_clad):
    # checl for invalid beta^2 values
    if np.any(beta_squared <=0):
        raise ValueError(
            "One or more beta^2 values are not positive."
        )
    # check effective index range
    invalid_neff=(
        (n_eff < 0) |
        (n_eff > n_core)
    )

    if np.any(invalid_neff):
        print(
            "Warning: One or more effective-index values "
            "are outside the physical range."
        )

    # guided mode condition
    guided =(
        (n_eff > n_clad) &
        (n_eff < n_core)
    )

    return guided

# to extract the guided modes
# ============================================================
# fn: EXTRACT GUIDED MODES
# ============================================================
def extract_guided_modes(beta_squared, beta, n_eff, modes, guided):

    guided_beta_squared = beta_squared[guided]
    guided_beta = beta[guided]
    guided_n_eff = n_eff[guided]
    guided_modes = modes[:, guided]

    return (
        guided_beta_squared,
        guided_beta,
        guided_n_eff,
        guided_modes
    )


# ============================================================
# fn: RECONSTRUCT FULL MODE FIELDS
# ============================================================
def reconstruct_modes(modes, N):
    # array for full grid
    full_modes = np.zeros(
        (N, modes.shape[1])
    )

    # insert interior field values
    full_modes[1:-1, :] = modes

    return full_modes

# ============================================================
# fn: NORMALIZE THE MODE FIELDS
# ============================================================
def normalize_modes(modes):
    """
    normalize each mode independently
    so that its maximum absolute value is 1
    """


    normalized = np.zeros_like(modes)

    for i in range(modes.shape[1]):

        max_value = np.max(
            np.abs(modes[:,i])
        )

        if max_value > 0:
            normalized[:,i] =(
                modes[:,i] / max_value
            )

    return normalized

# fn to visualize and calculate the guided mode properties
# ============================================================
# fn: PLOT GUIDED MODES
# ============================================================
def plot_guided_modes(x, guided_modes, guided_n_eff):
    """
    plot the electric-field distribution of all guided modes

    parameters
    ----------
    x : np.ndarray
        Spatial grid in meters.

    guided_modes : np.ndarray
        Matrix containing the guided mode field distributions
        Shape: (number_of_grid_points, number_of_guided_modes)

    guided_n_eff : np.ndarray
        Effective refractive index of each guided mode
    """
    # number of guided modes
    num_modes = guided_modes.shape[1]

    # one plot for each guided mode
    for mode_number in range(num_modes):

        # extract one mode from the matrix
        mode_field = guided_modes[:, mode_number]

        # plot the field
        plt.figure(figsize=(8,4))

        plt.plot(
            x[1:-1] * 1e6,
            mode_field,
            label = f"Mode {mode_number}"
        )

        # add the waveguide center line
        plt.axvline(0, linestyle="--", alpha=0.5)

        # labels
        plt.xlabel("Positon x (um)")
        plt.ylabel("Normalized field")

        plt.title(
            f"Guided mode {mode_number}"
            f"(n_eff = {guided_n_eff[mode_number]:.4f})"
        )

        plt.grid(True)
        plt.legend()

        plt.show()

# ============================================================
# fn: PLOT MODE WITH INDEX
# ============================================================
def plot_mode_with_index(x, n_profile, guided_modes, guided_n_eff):
     """
    plot guided optical modes together with the refractive index profile

    parameters
    ----------
    x : np.ndarray
        spatial grid in meters

    n_profile : np.ndarray
        refractive-index profile

    guided_modes : np.ndarray
        matrix containing guided modes

    guided_n_eff : np.ndarray
        effective refractive index of each guided mode
    """
     x_interior = x[1:-1]

     for mode_number in range(guided_modes.shape[1]):

        #  extract one mode
        mode_field = guided_modes[:, mode_number]

        # create figure
        fig, ax1 = plt.subplots(figsize=(9,5))

        # ============================================================
        # left axis : refractive index
        # ============================================================
        ax1.plot(
            x_interior *1e6,
            n_profile[1:-1],
            label = "Refractive index"
        )

        ax1.set_xlabel("Position x (um)")
        ax1.set_ylabel("Refractive index")

        ax1.grid(True)

        # ============================================================
        # right axis : optical field
        # ============================================================
        ax2 = ax1.twinx()
        ax2.plot(
            x_interior * 1e6,
            mode_field,
            linestyle = "--",
            label = f"Model {mode_number}"
        )

        ax2.set_ylabel("Normalized field")

        # ============================================================
        # Title
        # ============================================================
        ax1.set_title(
            f"Mode {mode_number} and refractive index profile"
            f"(n_eff = {guided_n_eff[mode_number]: .4f})"
        )

        plt.show()


# ============================================================
# fn: INTENSITY FUNCTION
# ============================================================
def plot_mode_intensity(x, guided_modes, guided_n_eff):
    """
    plot the optical intensity of all guided modes

    intensity is calculated as |E(x)|^2
    """

    x_interior = x[1:-1]

    for mode_number in range(guided_modes.shape[1]):

        mode_field = guided_modes[:, mode_number]

        intensity = np.abs(mode_field)**2

        # normalize intensity
        intensity = intensity / np.max(intensity)

        plt.figure(figsize=(9,5))

        plt.plot(
            x_interior * 1e6,
            intensity,
            label = f"Mode {mode_number}"
        )

        plt.xlabel("Position x (um)")
        plt.ylabel("Normalized intensity")

        plt.title(
            f"Optical intensity of guided mode {mode_number}"
            f"(n_eff = {guided_n_eff[mode_number]:.4f})"
        )

        plt.grid(True)
        plt.legend()

        plt.show()






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

# 1.solve modes
beta_squared, beta, n_eff, modes = solve_modes(
    A,
    wavelength,
    num_modes=4
)

# 2.validate results and identify guided modes
guided = validate_modes(
    beta_squared,
    n_eff,
    n_core,
    n_clad
)

# 3.extract guided modes
guided_beta_squared, guided_beta, guided_n_eff, guided_modes = (
    extract_guided_modes(
        beta_squared,
        beta,
        n_eff,
        modes,
        guided
    )
)

# 4. reconstruct full mode fields
full_modes = reconstruct_modes(
    modes,
    N
)

# 5. normalize mode fields
full_modes = normalize_modes(
    full_modes
)


# reconstruct fields on the full simulation grid
full_modes = reconstruct_modes(
    modes,
    N
)



beta_squared, beta, n_eff, modes = solve_modes(
    A,
    wavelength,
    num_modes=4
)

guided = validate_modes(
    beta_squared,
    n_eff,
    n_core,
    n_clad
)


# ============================================================
# VISUALIZATION
# ============================================================


# plot guided mode fields
plot_guided_modes(
    x,
    guided_modes, 
    guided_n_eff
)

# plot guided modes together with refractive-index profile
plot_mode_with_index(
    x,
    n,
    guided_modes,
    guided_n_eff
)


# plot optical intensity
plot_mode_intensity(
    x,
    guided_modes,
    guided_n_eff
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

print(
    f"\nNumber of guided modes: "
    f"{len(guided_n_eff)}"
)

# print guided mode results
print("\nGuided modes:")

for i in range(len(guided_n_eff)):

    print(f"\nGuided Mode {i}")
    print(
        f"  beta^2 = "
        f"{guided_beta_squared[i]:.6e}"
    )
    print(
        f"  beta   = "
        f"{guided_beta[i]:.6e} 1/m"
    )
    print(
        f"  neff   = "
        f"{guided_n_eff[i]:.6f}"
    )

print("\nWaveguide matrix:")
print(A)


print("\nCalculated modes:")

for i in range(len(n_eff)):

    if guided[i]:
        status = "GUIDED"
    else:
        status = "NOT GUIDED"

    print(f"\nMode {i}")
    print(f"  beta^2 = {beta_squared[i]:.6e}")
    print(f"  beta   = {beta[i]:.6e} 1/m")
    print(f"  neff   = {n_eff[i]:.6f}")
    print(f"  status = {status}")




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


# ============================================================
# PLOT FUNDAMENTAL MODE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x * 1e6,
    full_modes[:, 0]
)

plt.xlabel("x (µm)")
plt.ylabel("Normalized electric field")
plt.title(
    f"Fundamental Mode | neff = {n_eff[0]:.6f}"
)

plt.grid()
plt.show()
