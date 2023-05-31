# curved_inflating_universe_solveb
The aim of this project is to solve the b variable, which is important for setting quantum initial condition of curved inflating universe.

It contains 8 folders: 
1. BG_evolution: calfulate evolution of background variables (N, phi, b) based on 
      Lukas [2205.07374] and Mary [2211.17248]'s paper.
2. Solve_b_analytical: solve variable b analytically in KD and SR by mathematica
3. Solve_b_numerical: solve variable b analytically in KD and deep inflation. Use the solution to set IC in both end to get the numerical solutions, and try to find the unknown  constants in analytical solution by matching the two numerical solutions in the middle. 
4. modifies_primpy: The original primpy is based on [2205.07374], which apply flat RSET. We would like to modify it for curved RSET based on equations in [2211.17248] and my own work on solving new variable b.
5. PPS&CMB: generate PPS and CMB of the new model (with b). 
6. Find_best-fit_Optimize: calculate likelihood by comparing with Planck 2018 data and CMB prediction. Derive chi_eff. And find best-fit parameter by minimizing chi_eff.
7. Find_best-fit_Polychord: find best-fit parameter set by the Baysian nested sampling code (Polychord)
8. Result: result of BG evolution, PPS&CMB, and chi_eff