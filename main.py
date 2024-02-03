import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry symulacji
NN = 400
hbar = 1.054e-34
m0 = 9.1e-31
meff = 1.0
melec = meff * m0
ecoul = 1.6e-19
epsz = 8.85e-9
eV2J = 1.6e-19
J2eV = 1/eV2J
del_x = 0.1e-9
dt = 2e-17
ra = (0.5 * hbar / melec) * (dt / del_x**2)
DX = del_x * 1e9
XX = np.arange(DX, DX * NN + DX, DX)

# Inicjalizacja funkcji falowej dla potencjału 0.1 eV na lewej stronie
lambda_val = 50
sigma = 50
nc = 150

V_index = 1
prl = np.zeros(NN)
pim = np.zeros(NN)
ptot = 0.

for n in range(1, NN-1):
    prl[n] = float(np.exp(-1.0 * ((n - nc) / sigma)**2) * np.cos(2 * np.pi * (n - nc) / lambda_val))
    pim[n] = float(np.exp(-1.0 * ((n - nc) / sigma)**2) * np.sin(2 * np.pi * (n - nc) / lambda_val))
    ptot += prl[n]**2 + pim[n]**2

pnorm = np.sqrt(ptot)
prl /= pnorm
pim /= pnorm

fig, ax = plt.subplots(figsize=(10, 4))

# Initialize the lines to be animated
line_prl, = ax.plot(XX, prl, label='Real Part')
line_pim, = ax.plot(XX, pim, '--', label='Imaginary Part')
line_potential, = ax.plot(XX, J2eV * np.zeros(NN), '-.', label='Potential')

def update(frame):
    V = np.zeros(NN)
    V[0:NN//2] = 0.1 * eV2J

    T = frame
    global prl, pim
    for n in range(1, NN-1):
        prl[n] = prl[n] - ra * (pim[n-1] - 2*pim[n] + pim[n+1]) + (dt/hbar) * V[n] * pim[n]
    for n in range(1, NN-1):
        pim[n] = pim[n] + ra * (prl[n-1] - 2*prl[n] + prl[n+1]) - (dt/hbar) * V[n] * prl[n]

    # Sprawdzenie, czy fale wychodzą poza zakres
    if np.max(np.abs(prl)) > 1.0 or np.max(np.abs(pim)) > 1.0:
        prl /= 2.0
        pim /= 2.0

    # Update the data for the animated lines
    line_prl.set_ydata(prl)
    line_pim.set_ydata(pim)
    line_potential.set_ydata(J2eV * V)
    ax.set_ylim(-0.25, 0.25)

    return line_prl, line_pim, line_potential

# Set up the animation
num_frames = 10000
ani = FuncAnimation(fig, update, frames=num_frames, repeat=False, interval=1, blit=True)

plt.tight_layout()
plt.show()