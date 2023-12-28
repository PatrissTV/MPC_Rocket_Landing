# 16.32 Project: MPC Powered Rocket Landing

Two examples provided for lossless convexification. Use anaconda to install ipopt and necessary packages. There is a conda.yml file with all the versions of the packages I used. The most important ones are:

      - python==3.11.3
      - casadi==3.6.1
      - ply==3.11
      - pyomo==6.5.0
      - scipy==1.10.1
      - ipopt=3.11.1=2

You can generate 3d trajectories with lossless convexification in the folder lossless3d. It is based on pyomo. Simply open the notebook and run all cells.

You can also test the lossless convexification on a 2d trajectory in the folder lossless2d. This one uses casadi with RK45 for translational motion and Euler Forward for attitude. Simply open the notebook and run all cells.

Used repository from Igor Shvab to generate animation: https://github.com/Igor-Shvab/Rocket_soft_landing

You can read the paper here: https://github.com/PatrissTV/MPC_Rocket_Landing/blob/main/Optimal_Control_Project.pdf
