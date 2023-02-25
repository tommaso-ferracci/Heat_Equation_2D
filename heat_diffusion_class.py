import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

plt.style.use('science')

class heat_diffusion:
    def __init__(self, L, R, D, dt, N, n, T1, T2):
        self.L = L # lato cella di simulazione
        self.R = R # raggio cerchio interno
        self.D = D # costante di diffusione
        self.dt = dt # passo temporale
        self.N = N # n. di passi della griglia spaziale
        self.n = n # n. di passi della griglia temporale
        self.T1 = T1 # temperatura sul bordo e a t=0
        self.T2 = T2 # temperatura nel cerchio
        self.h = L/(N-1) # passo spaziale
        self.rho = self.inizializza() 


    def inizializza(self): # metodo responsabile di condizioni iniziali
        rho = np.zeros((self.N, self.N, self.n))    
        for i in range(self.N):
            for j in range(self.N):
                if ((0.5*self.L - i*self.h)**2 + (0.5*self.L - j*self.h)**2 <= self.R**2):
                    rho[i, j, 0] = self.T2
                else:
                    rho[i, j, 0] = self.T1 
        return rho


    def eulero_esplicito(self):
        if self.D*self.dt/((self.L - 2*self.R)/(self.N-1))**2 < 0.5:
            self.rho[0, :, 1:] = self.T1
            self.rho[self.N-1, :, 1:] = self.T1
            self.rho[:, 0, 1:] = self.T1
            self.rho[:, self.N-1, 1:] = self.T1
            for i in range(self.N):
                for j in range(self.N):
                    if ((0.5*self.L - i*self.h)**2 + (0.5*self.L - j*self.h)**2 <= self.R**2):
                        self.rho[i, j, :] = self.T2
            for t in range(self.n-1):
                for i in range(1, self.N-1):
                    for j in range(1, self.N-1):
                        if ((0.5*self.L - i*self.h)**2 + (0.5*self.L - j*self.h)**2 > self.R**2):
                            self.rho[i, j, t+1] = self.rho[i, j, t] + self.D/self.h**2 * (self.rho[i+1, j, t] + self.rho[i-1, j, t] 
                                                + self.rho[i, j+1, t] + self.rho[i, j-1, t] - 4*self.rho[i, j, t]) * self.dt
        else:
            raise ValueError('Algoritmo instabile!')


    def eulero_implicito(self):
        M = np.zeros((self.N**2, self.N**2))
        for i in range(self.N):
            for j in range(self.N):
                r = j*self.N + i   
                if (i != 0) and (i != self.N-1) and (j != 0) and (j != self.N-1) and ((0.5*self.L - i*self.h)**2 + (0.5*self.L - j*self.h)**2 > self.R**2):
                    for n in range(self.N):
                        for m in range(self.N):
                            s = m*self.N + n
                            if (n == i+1) and (m == j):
                                M[r, s] = -self.D*self.dt/self.h**2
                            elif (n == i-1) and (m == j):
                                M[r, s] = -self.D*self.dt/self.h**2
                            elif (n == i) and (m == j+1):
                                M[r, s] = -self.D*self.dt/self.h**2
                            elif (n == i) and (m == j-1):
                                M[r, s] = -self.D*self.dt/self.h**2
                            elif (n == i) and (m == j):
                                M[r, s] = 1 + 4*self.D*self.dt/self.h**2
                else:
                    M[r, r] = 1

        sM = csc_matrix(M)
        invM = linalg.inv(sM)
        for t in range(self.n-1):
            self.rho[:, :, t+1] = invM.dot(self.rho[:, :, t].flatten()).reshape(self.N, self.N)


    def plot_2D(self, id, t=0):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax.set_title('Heat Diffusion - 2D Plot')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12, rotation=0)
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        contour = ax.contourf(X, Y, self.rho[:, :, t], cmap=cm.coolwarm)
        cbar = fig.colorbar(contour, ax=ax, shrink=1.0, aspect=5)
        cbar.set_label('T', rotation=0, labelpad=12)
        fig.savefig(f'heat_diffusion_2D_{id}.png')


    def plot_3D(self, id, t=0):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,9), subplot_kw={"projection": "3d"})
        ax.set_title('Heat Diffusion - 3D Plot')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('T', fontsize=12, rotation=0)
        ax.view_init(elev=12, azim=30)
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        surface = ax.plot_surface(X, Y, self.rho[:, :, t], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('T', rotation=0, labelpad=12)
        fig.savefig(f'heat_diffusion_3D_{id}.png')


    def animation_2D(self, id):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
        ax.set_title('Heat Diffusion - 2D Animation')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12, rotation=0)
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))

        def animate(t):
            ax.contourf(X, Y, self.rho[:, :, 10*t], cmap=cm.coolwarm)

        anim = FuncAnimation(fig, animate, int(self.n/10), blit=False)
        anim.save(f'heat_diffusion_2D_{id}.gif', writer=PillowWriter(fps=20))


    def animation_3D(self, id):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,9), subplot_kw={"projection": "3d"})
        ax.set_title('Heat Diffusion - 3D Animation')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('T', fontsize=12, rotation=0)
        ax.view_init(elev=12, azim=30)
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))

        def animate(t):
            ax.plot_surface(X, Y, self.rho[:, :, 10*t], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        anim = FuncAnimation(fig, animate, int(self.n/10), blit=False)
        anim.save(f'heat_diffusion_3D_{id}.gif', writer=PillowWriter(fps=20))


class unstable_heat_diffusion(heat_diffusion):  
    def eulero_esplicito(self): # override senza check stabilità
        self.rho[0, :, 1:] = self.T1
        self.rho[self.N-1, :, 1:] = self.T1
        self.rho[:, 0, 1:] = self.T1
        self.rho[:, self.N-1, 1:] = self.T1
        for i in range(self.N):
            for j in range(self.N):
                if ((0.5*self.L - i*self.h)**2 + (0.5*self.L - j*self.h)**2 <= self.R**2):
                        self.rho[i, j, :] = self.T2
        for t in range(self.n-1):
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    if ((0.5*self.L - i*self.h)**2 + (0.5*self.L - j*self.h)**2 > self.R**2):
                        self.rho[i, j, t+1] = self.rho[i, j, t] + self.D/self.h**2 * (self.rho[i+1, j, t] + self.rho[i-1, j, t] 
                                            + self.rho[i, j+1, t] + self.rho[i, j-1, t] - 4*self.rho[i, j, t]) * self.dt
 