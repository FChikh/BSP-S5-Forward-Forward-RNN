import numpy as np
import matplotlib.pyplot as plt

# plot distributions of G for pos and neg samples; also plot the threshold; with fixed bin width

mu_pos = np.mean(G[labels == 1].mean(dim=-1).cpu().numpy())
sigma2_pos = np.var(G[labels == 1].mean(
    dim=-1).cpu().numpy())  # дисперсия
sigma_pos = np.sqrt(sigma2_pos)

mu_neg = np.mean(G[labels == 0].mean(dim=-1).cpu().numpy())
sigma2_neg = np.var(G[labels == 0].mean(
    dim=-1).cpu().numpy())
sigma_neg = np.sqrt(sigma2_neg)

width = 0.1
x_vals_pos = np.linspace(np.min(G[labels == 1].mean(
    dim=-1).cpu().numpy()), np.max(G[labels == 1].mean(dim=-1).cpu().numpy()) + 1, 200)
pdf_vals_pos = (1.0 / (sigma_pos * np.sqrt(2*np.pi))) * \
    np.exp(-0.5 * ((x_vals_pos - mu_pos)/sigma_pos)**2)

plt.plot(x_vals_pos, pdf_vals_pos, lw=2,
         label='pos approximated normal distribution', color='tab:blue')

x_vals_neg = np.linspace(np.min(G[labels == 0].mean(
    dim=-1).cpu().numpy()), np.max(G[labels == 0].mean(dim=-1).cpu().numpy()) + 1, 200)
pdf_vals_neg = (1.0 / (sigma_neg * np.sqrt(2*np.pi))) * \
    np.exp(-0.5 * ((x_vals_neg - mu_neg)/sigma_neg)**2)

# plt.plot(x_vals_neg, pdf_vals_neg, lw=2,
#          label='neg approximated normal distribution', color='tab:orange')

est_lam = 1.0 / np.mean(G[labels == 0].mean(dim=-1).cpu().numpy())

x_vals = np.linspace(
    0, np.max(G[labels == 0].mean(dim=-1).cpu().numpy()), 200)
pdf_vals = est_lam * np.exp(-est_lam * x_vals)
plt.plot(x_vals, pdf_vals, lw=2,
         label='neg exponential distribution', color='tab:red')

plt.hist(G[labels == 1].mean(dim=-1).cpu().numpy(),
         bins=np.arange(np.min(G[labels == 1].mean(dim=-1).cpu().numpy()), np.max(G[labels == 1].mean(dim=-1).cpu().numpy()) + 1, width), alpha=0.5, label='pos', density=True)
plt.hist(G[labels == 0].mean(dim=-1).cpu().numpy(),
         bins=np.arange(np.min(G[labels == 0].mean(dim=-1).cpu().numpy()), np.max(G[labels == 0].mean(dim=-1).cpu().numpy()) + 1, width), alpha=0.5, label='neg', density=True)
plt.axvline(x=threshold, color='b', linestyle='--', label='Threshold')
plt.title(f'Goodness distribution, layer={layer_idx}, threshold={threshold}')
plt.legend()
plt.show()
