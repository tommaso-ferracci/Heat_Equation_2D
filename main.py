import numpy as np
import matplotlib.pyplot as plt

from heat_diffusion_class import heat_diffusion
from heat_diffusion_class import unstable_heat_diffusion

plt.style.use('science')

if __name__ == '__main__':
    
    diff1 = heat_diffusion(L=1, R=0.2, D=1, dt=8e-5, N=100, n=1200, T1=20, T2=30)
    diff1.eulero_implicito()
    diff1.plot_2D(t=diff1.n-1, id=1)
    diff1.plot_3D(t=diff1.n-1, id=1)
    diff1.animation_2D(id=1)
    diff1.animation_3D(id=1)
    
    diff2 = heat_diffusion(L=1, R=0.2, D=1, dt=8e-5, N=100, n=1200, T1=30, T2=20)
    diff2.eulero_implicito()
    diff2.plot_2D(t=diff2.n-1, id=2)
    diff2.plot_3D(t=diff2.n-1, id=2)
    diff2.animation_2D(id=2)
    diff2.animation_3D(id=2)
    
    err = []
    for dt in np.linspace(2e-5, 5e-5, 30):
        diff = unstable_heat_diffusion(L=1, R=0.2, D=1, dt=dt, N=100, n=10, T1=20, T2=30)
        diff.eulero_esplicito()
        rho_exp = diff.rho[:, :, -1]
        diff = unstable_heat_diffusion(L=1, R=0.2, D=1, dt=dt, N=100, n=10, T1=20, T2=30)
        diff.eulero_implicito()
        rho_imp = diff.rho[:, :, -1]
        err.append(np.sum(np.abs(rho_exp - rho_imp))/100**2)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    ax.set_title('explicit method instability')
    ax.set_xlabel('$\\frac{D dt}{h^2}$', fontsize=12)
    ax.set_ylabel('average absolute error', fontsize=12)
    ax.scatter(np.linspace(2e-5, 5e-5, 30)/(1/99)**2, np.array(err), s=8)
    ax.grid()
    fig.savefig('error.png')
    
    diff3 = unstable_heat_diffusion(L=1, R=0.2, D=1, dt=5e-5, N=100, n=50, T1=20, T2=30)
    diff3.eulero_esplicito()
    diff3.plot_2D(t=diff3.n-1, id=3)
    
    diff = heat_diffusion(L=1, R=0.2, D=1, dt=1e-5, N=100, n=100, T1=20, T2=30)
    diff.eulero_esplicito()
    rho_exp = diff.rho[30, 30, :]
    diff = heat_diffusion(L=1, R=0.2, D=1, dt=1e-5, N=100, n=100, T1=20, T2=30)
    diff.eulero_implicito()
    rho_imp = diff.rho[30, 30, :]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    ax.set_title('explicit vs implicit - $(x, y) = (0.3, 0.3)$')
    ax.set_xlabel('n')
    ax.set_ylabel('T', rotation=0)
    ax.plot(np.arange(100), rho_exp, label='explicit')
    ax.plot(np.arange(100), rho_imp, label='implicit')
    ax.legend()
    ax.grid()
    fig.savefig('comparison.png')
    
       
