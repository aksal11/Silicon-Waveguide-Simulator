
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

beta_min = k_clad
beta_max = k_core


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
# N = 1000
N = 5


# simulation region
# x = np.linspace(-2e-6, 2e-6,1000 )
x_min = -2e-6
x_max = 2e-6

x = np.linspace(x_min, x_max, N)


# dx = spatial grid spacing (step size) between adjacent x-points.
# it determines the numerical resolution of the simulation.
# smaller dx -> finer spatial resolution -> generally lower discretization error.
# dx is determined by the simulation domain and number of grid points:
# dx = (x_max - x_min) / (N - 1)
dx = x[1] - x[0]

print("Grid points N = ",N)
print("Grid spacing dx = ",dx,"m")
print("Grid spacing dx =", dx * 1e9, "nm")

print("x =", x)
print("dx =", dx)

# number of interior grid points
N_interior = N-2

# create the second-derivative matrix
D2 = np.zeros((N_interior, N_interior)) 

# fill the matrix
for i in range(N_interior):

    DS[i,i] = -2 #creates diagonal

    if i > 0:
        D2[i,i-1] = 1 #creates the lower diagonal

    if i < N_interior - 1:
        D2[I,I+1] = 1 #creates the upper diagonal

# divide by dx^2
D2 = D2 / dx**2

print("\nSecond derivative matrix D2:")
print(D2)


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
# k = k0 * n

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

plt.figure(figsize=(10,6))

plt.plot(
    x * 1e6,
    n,
    label = f"N = {N}"
)


# visualization
plt.xlabel("Positive x (um)")
plt.ylabel("Refractive index")
plt.title("Waveguide refractive-index profile")
plt.legend()
plt.grid()
plt.show()

