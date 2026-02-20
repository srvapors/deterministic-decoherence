# deterministic-decoherence
repository for python code for paper



import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# DETERMINISTIC DECOHERENCE SIMULATOR V4 (1D Double Slit)
# Includes Coherence Decay (Purity) Tracking
# =====================================================================

N = 256                 
L = 20.0                
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
dt = 0.001              
steps = 2000            
hbar = 1.0              
m = 1.0                 

x_left = -3.0
x_right = 3.0
sigma = 0.8             

H = np.zeros((N, N), dtype=complex)
for i in range(N):
    H[i, i] = 2.0 / (dx**2)
    if i > 0: H[i, i-1] = -1.0 / (dx**2)
    if i < N-1: H[i, i+1] = -1.0 / (dx**2)
H = (hbar**2 / (2 * m)) * H

detector_mask = np.exp(-((x - x_right)**2) / (2 * (1.5)**2))
P_R = np.diag(detector_mask)  

def run_simulation(noise_rate_gamma):
    psi_0 = np.exp(-((x - x_left)**2) / (4 * sigma**2)) + \
            np.exp(-((x - x_right)**2) / (4 * sigma**2))
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    
    rho = np.outer(psi_0, np.conj(psi_0))
    
    purity_history = []
    
    for step in range(steps):
        commutator = np.dot(H, rho) - np.dot(rho, H)
        d_rho_H = -(1j / hbar) * commutator
        
        if noise_rate_gamma > 0:
            term1 = np.dot(P_R, np.dot(rho, P_R))
            term2 = np.dot(P_R, np.dot(P_R, rho))
            term3 = np.dot(rho, np.dot(P_R, P_R))
            d_rho_noise = noise_rate_gamma * (term1 - 0.5 * (term2 + term3))
        else:
            d_rho_noise = np.zeros_like(rho)
            
        rho = rho + (d_rho_H + d_rho_noise) * dt
        
        if step % 20 == 0:
            trace = np.real(np.trace(rho))
            if trace > 0:
                rho /= trace
                
        # Track State Purity Tr(rho^2)
        purity = np.real(np.trace(np.dot(rho, rho)))
        purity_history.append(purity)
                
    prob = np.real(np.diag(rho))
    prob = np.clip(prob, 0, None)  
    prob /= (np.sum(prob) * dx)    
    
    return prob, purity_history

print("Running pure quantum state (No Measurement)...")
prob_0, pur_0 = run_simulation(0.0)

print("Running Weak Measurement (Low Noise)...")
prob_5, pur_5 = run_simulation(5.0)

print("Running Hard Measurement (High Noise)...")
prob_50, pur_50 = run_simulation(50.0)

# =====================================================================
# PLOT THE RESULTS (2 PANELS)
# =====================================================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
time_array = np.linspace(0, steps*dt, steps)

# --- PANEL 1: Probability Density ---
ax1.plot(x, prob_0, color='cyan', linewidth=2, label='1. No Detector (γ = 0): Full Coherence')
ax1.plot(x, prob_5, color='yellow', linewidth=2, linestyle='--', label='2. Weak Detector (γ = 5): Partial Decoherence')
ax1.plot(x, prob_50, color='magenta', linewidth=2, label='3. Hard Detector (γ = 50): Classical Limit')
ax1.set_title('Final Probability Density P(x)', fontsize=14, pad=15)
ax1.set_xlabel('Position on Back Wall (x)', fontsize=12)
ax1.set_ylabel('Probability Density P(x) [1/units of x]', fontsize=12)
ax1.set_ylim(0, max(prob_0)*1.1)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- PANEL 2: Coherence Decay ---
ax2.plot(time_array, pur_0, color='cyan', linewidth=2)
ax2.plot(time_array, pur_5, color='yellow', linewidth=2, linestyle='--')
ax2.plot(time_array, pur_50, color='magenta', linewidth=2)
ax2.set_title('Coherence Decay over Time', fontsize=14, pad=15)
ax2.set_xlabel('Simulation Time (t)', fontsize=12)
ax2.set_ylabel('State Purity $Tr(ρ^2)$', fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

plt.suptitle('Measurement as Mechanical Noise (Lindblad Decoherence)', fontsize=18, y=1.05)
plt.tight_layout()
plt.show()
