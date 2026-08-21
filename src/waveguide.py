
import numpy as np
import matplotlib.pyplot as plt



# function : create waveguide refractive index
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

# material properties
n_core = 3.48  
n_clad = 1.44

# Number of grid points
N_values = [50, 100, 500, 1000, 5000]


# simulation region
# x = np.linspace(-2e-6, 2e-6,1000 )
x_min = -2e-6
x_max = 2e-6

# different waveguide widths
# widths =[
#     450e-9,
#     800e-9,
#     1200e-9
# ]

# single width
width = 450e-9

# generate and plot each profile
for N in N_values:

    # grid with N points
    x = np.linspace(x_min, x_max, N)

    # refractive index profile
    n = create_waveguide_profile(
        x,
        width,
        n_core,
        n_clad
    )

    # plot
    plt.plot(
        x * 1e6,
        n,
        label=f"N = {N}"
    )



# visualization
plt.xlabel("Positive x (um)")
plt.ylabel("Refractive index")
plt.title("Waveguide refractive-index profile")
plt.legend()
plt.grid()
plt.show()

