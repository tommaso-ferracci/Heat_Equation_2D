import numpy as np
import matplotlib.pyplot as plt

from heat_diffusion_class import heat_diffusion
from heat_diffusion_class import unstable_heat_diffusion

plt.style.use("default")
plt.rc("text", usetex=True)
plt.rc("font", family="cm")
plt.rcParams["grid.color"] = (0.5, 0.5, 0.5, 0.2)
  
hd = heat_diffusion(L=1, R=0.3, D=1, dt=6e-5, N=100, n=1601, T1=20, T2=30)
hd.implicit_euler()
hd.plot_2D(t=hd.n-1, id=1)
hd.plot_3D(t=hd.n-1, id=1)
hd.animation_2D(id=1)
hd.animation_3D(id=1)

hd = heat_diffusion(L=1, R=0.3, D=1, dt=6e-5, N=100, n=1601, T1=30, T2=20)
hd.implicit_euler()
hd.plot_3D(t=hd.n-1, id=2)
hd.animation_3D(id=2)

err = []
for dt in np.linspace(2e-5, 6e-5, 40):
    hd = unstable_heat_diffusion(L=1, R=0.3, D=1, dt=dt, N=100, n=10, T1=20, T2=30)
    hd.explicit_euler()
    rho_exp = hd.rho[:, :, -1]
    hd = unstable_heat_diffusion(L=1, R=0.3, D=1, dt=dt, N=100, n=10, T1=20, T2=30)
    hd.implicit_euler()
    rho_imp = hd.rho[:, :, -1]
    err.append(np.sum(np.abs(rho_exp - rho_imp))/100**2) # append average absolute error

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
ax.set_title('explicit method instability')
ax.set_xlabel('$\\frac{D dt}{h^2}$', fontsize=12)
ax.set_ylabel('average absolute error', fontsize=12)
ax.scatter(np.linspace(2e-5, 6e-5, 40)/(1/99)**2, np.array(err), s=8)
ax.grid()
fig.savefig('images/error.png', dpi=300, bbox_inches="tight", pad_inches=0.2)

hd = unstable_heat_diffusion(L=1, R=0.3, D=1, dt=6e-5, N=100, n=50, T1=20, T2=30)
hd.explicit_euler()
hd.plot_2D(t=hd.n-1, id=3)

hd = heat_diffusion(L=1, R=0.3, D=1, dt=1e-5, N=100, n=100, T1=20, T2=30)
hd.explicit_euler()
rho_exp = hd.rho[30, 30, :]
hd = heat_diffusion(L=1, R=0.3, D=1, dt=1e-5, N=100, n=100, T1=20, T2=30)
hd.implicit_euler()
rho_imp = hd.rho[30, 30, :]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
ax.set_title('explicit vs implicit - $(x, y) = (0.3, 0.3)$')
ax.set_xlabel('n')
ax.set_ylabel('T', rotation=0)
ax.plot(np.arange(100), rho_exp, label='explicit')
ax.plot(np.arange(100), rho_imp, label='implicit')
ax.legend()
ax.grid()
fig.savefig('images/comparison.png', dpi=300, bbox_inches="tight", pad_inches=0.2)