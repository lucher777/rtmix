# rtmix Overview

`rtmix` implements **Dual-Tail Process-Data Mixture IRT (`DT-MixIRT`)** for modern digital assessments where examinees may exhibit either **left-tail rapid guessing** or **right-tail AI-assisted copying**, while honest **slow careful examinees** must remain strictly protected against false forensic accusations.

## Model Summary

For examinee $i$ and item $j$, `DT-MixIRT` models the trivariate observation $(Y_{ij}, \log T_{ij}, H_{ij})$ as a three-regime mixture:

- **Regime 0 (Genuine Solution Behavior, weight $1 - \pi_i^{\mathrm{RG}} - \pi_i^{\mathrm{AI}}$)**:
  3PL accuracy $P_{0,ij} = c_j + (1 - c_j)\sigma(a_j(\theta_i - b_j))$, log-normal response time $\log T_{ij} \sim \mathcal{N}(\beta_j - \tau_i, \sigma_{T,j}^2)$, and baseline action/edit entropy $H_{ij} \sim \mathcal{N}(\gamma_j + \nu_i, \sigma_{H,j}^2)$.
- **Regime 1 (Left-Tail Rapid Guessing, weight $\pi_i^{\mathrm{RG}}$)**:
  Chance-level accuracy $c_j$, compressed response time $\mathcal{N}(\mu_{T,\mathrm{RG}}, \sigma_{T,\mathrm{RG}}^2)$, and minimal action entropy $\mathcal{N}(\mu_{H,\mathrm{RG}}, \sigma_{H,\mathrm{RG}}^2)$.
- **Regime 2 (Right-Tail AI-Assisted Copying, weight $\pi_i^{\mathrm{AI}}$)**:
  Elevated accuracy floor $p_{\mathrm{AI}}$, positive prompting-pause shift $\mathcal{N}(\beta_j - \tau_i + \Delta_T^{\mathrm{AI}}, \sigma_{T,\mathrm{AI}}^2)$, and collapsed edit entropy $\mathcal{N}(\mu_{H,\mathrm{AI}}, \sigma_{H,\mathrm{AI}}^2)$.
