import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

plt.style.use('science')

class heat_diffusion:
    def __init__(self, L, R, D, dt, N, n, T1, T2):
        self.L = L # simulation cell length
        self.R = R # inside cell length
        self.D = D # diffusion constant
        self.dt = dt # timestep size
        self.N = N # n. of gridsteps
        self.n = n # n. of timesteps
        self.T1 = T1 # temperature at boundary and t=0
        self.T2 = T2 # temperature in the inside circle
        self.h = L/(N-1) # gridstep size
        self.rho = self.initialize() 


    def initialize(self): # method to create initial configuration
        rho = np.zeros((self.N, self.N, self.n))    
        for i in range(self.N):
            for j in range(self.N):
                if (i*self.h >= self.L/2 - self.R/2) and (i*self.h <= self.L/2 + self.R/2) and (j*self.h >= self.L/2 - self.R/2) and (j*self.h <= self.L/2 + self.R/2):
                    rho[i, j, 0] = self.T2
                else:
                    rho[i, j, 0] = self.T1 
        return rho


    def explicit_euler(self):
        if self.D*self.dt/((self.L - self.R)/(self.N-1))**2 < 0.5: # instability check
            self.rho[0, :, 1:] = self.T1
            self.rho[self.N-1, :, 1:] = self.T1
            self.rho[:, 0, 1:] = self.T1
            self.rho[:, self.N-1, 1:] = self.T1
            for i in range(self.N):
                for j in range(self.N):
                    if (i*self.h >= self.L/2 - self.R/2) and (i*self.h <= self.L/2 + self.R/2) and (j*self.h >= self.L/2 - self.R/2) and (j*self.h <= self.L/2 + self.R/2):
                        self.rho[i, j, :] = self.T2
            for t in range(self.n-1):
                for i in range(1, self.N-1):
                    for j in range(1, self.N-1):
                        if (i*self.h < self.L/2 - self.R/2) or (i*self.h > self.L/2 + self.R/2) or (j*self.h < self.L/2 - self.R/2) or (j*self.h > self.L/2 + self.R/2):
                            self.rho[i, j, t+1] = self.rho[i, j, t] + self.D/self.h**2 * (self.rho[i+1, j, t] + self.rho[i-1, j, t] 
                                                + self.rho[i, j+1, t] + self.rho[i, j-1, t] - 4*self.rho[i, j, t]) * self.dt
        else:
            raise ValueError('Unstable algorithm!')


    def implicit_euler(self):
        M = np.zeros((self.N**2, self.N**2))
        for i in range(self.N):
            for j in range(self.N):
                r = j*self.N + i   
                if (i != 0) and (i != self.N-1) and (j != 0) and (j != self.N-1) and ((i*self.h < self.L/2 - self.R/2) or (i*self.h > self.L/2 + self.R/2) or (j*self.h < self.L/2 - self.R/2) or (j*self.h > self.L/2 + self.R/2)):
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

        sM = csc_matrix(M) # cast to scipy sparse matrix for better efficiency
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
        fig.savefig(f'images/heat_diffusion_2D_{id}.png')


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
        fig.savefig(f'images/heat_diffusion_3D_{id}.png')


    def animation_2D(self, id):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))

        def animate(t):
            ax.clear()
            ax.set_title('Heat Diffusion - 2D Animation')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12, rotation=0)
            ax.contourf(X, Y, self.rho[:, :, 10*t], cmap=cm.coolwarm)

        anim = FuncAnimation(fig, animate, int(self.n/10), blit=False)
        anim.save(f'images/heat_diffusion_2D_{id}.gif', writer=PillowWriter(fps=20))


    def animation_3D(self, id):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,9), subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))

        def animate(t):
            ax.clear()
            ax.set_title('Heat Diffusion - 3D Animation')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_zlabel('T', fontsize=12, rotation=0)
            ax.view_init(elev=12, azim=30)
            ax.plot_surface(X, Y, self.rho[:, :, 10*t+1], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        anim = FuncAnimation(fig, animate, int(self.n/10), blit=False)
        anim.save(f'images/heat_diffusion_3D_{id}.gif', writer=PillowWriter(fps=20))


# this class is used only to show what happens to an unstable explicit propagator
class unstable_heat_diffusion(heat_diffusion):  
    def explicit_euler(self): # override without instability check
        self.rho[0, :, 1:] = self.T1
        self.rho[self.N-1, :, 1:] = self.T1
        self.rho[:, 0, 1:] = self.T1
        self.rho[:, self.N-1, 1:] = self.T1
        for i in range(self.N):
            for j in range(self.N):
                if (i*self.h >= self.L/2 - self.R/2) and (i*self.h <= self.L/2 + self.R/2) and (j*self.h >= self.L/2 - self.R/2) and (j*self.h <= self.L/2 + self.R/2):
                        self.rho[i, j, :] = self.T2
        for t in range(self.n-1):
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    if (i*self.h < self.L/2 - self.R/2) or (i*self.h > self.L/2 + self.R/2) or (j*self.h < self.L/2 - self.R/2) or (j*self.h > self.L/2 + self.R/2):
                        self.rho[i, j, t+1] = self.rho[i, j, t] + self.D/self.h**2 * (self.rho[i+1, j, t] + self.rho[i-1, j, t] 
                                            + self.rho[i, j+1, t] + self.rho[i, j-1, t] - 4*self.rho[i, j, t]) * self.dt