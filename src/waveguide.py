
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

# wavelength
wavelength = 1550e-9

# free space number
# ko free space wavenumber ko = 2 * 3.14 / wavelength
k0 =  2* np.pi / wavelength

# material wavenumbers
k_core = k0 * n_core
k_clad = k0 * n_clad

# propagation constant range
beta_clad = k_clad
beta_core = k_core


# print values
print("Wavelength =",wavelength * 1e9, "nm")
print("k0 =",k0, "1/m")

print()
print("Core wavenumber =", k_core, "1/m")
print("Cladding wavenumber =", k_clad, "1/m")

print()
print("Guided beta range:")
print(beta_clad, "< beta <", beta_core)

# effective index
n_eff_test = 2.5

beta_test = k0 * n_eff_test

print()
print("Test n_eff =", n_eff_test)
print("Corresponding beta =", beta_test, "1/m")

print("Recovered n_eff =", beta_test/ k0)

# Physical check
def check_effective_index(n_eff, n_core, n_clad):

    if not ( n_clad < n_eff < n_core):
        raise ValueError("Effective index is outside the guided-mode range")
    return True

check_effective_index(
    n_eff_test,
    n_core,
    n_clad
)

# Number of grid points
# N_values = [50, 100, 500, 1000, 5000]

# grid
N = 1000


# simulation region
# x = np.linspace(-2e-6, 2e-6,1000 )
x_min = -2e-6
x_max = 2e-6

x = np.linspace(x_min, x_max, N)

# different waveguide widths
# widths =[
#     450e-9,
#     800e-9,
#     1200e-9
# ]

# single width
width = 450e-9

n = create_waveguide_profile(
        x,
        width,
        n_core,
        n_clad
    )

# generate and plot each profile
# for N in N_values:

#     # grid with N points
#     # x = np.linspace(x_min, x_max, N)

#     # refractive index profile
#     n = create_waveguide_profile(
#         x,
#         width,
#         n_core,
#         n_clad
#     )

# material-dependent wavenumber
k = k0 * n

# k**2
k_squared = k0**2 * n**2


# quantities
print("Wavelength: ", wavelength, "m")
print("Wavelength: ", wavelength * 1e9 , "nm")


print("k0: ", k0,"1/m" )

print("Core wavenumber :", k0* n_core, "1/m")
print("Cladding wavenumber: ", k0* n_clad,"1/m")

print()
print("Sanity check:")
print("n_core =", n_core)
print("n_clad =", n_clad)
print("k_core / k0 =", (k0 * n_core) / k0)
print("k_clad / k0 =", (k0 * n_clad) / k0)


# visualize k**2(x)

plt.figure(figsize=(10,6))

plt.plot(
    x * 1e6,
    k_squared,
)

plt.xlabel("x (µm)")
plt.ylabel(r"$k_0^2 n^2(x)$")
plt.title(r"Optical potential term $k_0^2 n^2(x)$")
plt.grid()
plt.show()

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

