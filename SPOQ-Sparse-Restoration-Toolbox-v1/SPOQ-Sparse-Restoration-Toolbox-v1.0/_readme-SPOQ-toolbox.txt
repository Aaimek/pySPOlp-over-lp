%%% SPOQ Sparse restoration Matlab toolbox
%%% Version 1.0, 15/03/2021
%%% 
%%% This code implements the SPOQ regularization function presented in
%%% "SPOQ $\ell_p$-Over-$\ell_q$ Regularization for Sparse Signal: Recovery applied to Mass Spectrometry"
%%% IEEE Transactions on Signal Processing, 2020, Volume 68, pages 6070--6084
%%% Afef Cherni, IEEE member, 
%%% Emilie Chouzenoux, IEEE Member,
%%% Laurent Duval, IEEE Member,
%%% Jean-Christophe Pesquet, IEEE Fellow
%%% https://arxiv.org/abs/2001.08496
%%% https://doi.org/10.1109/TSP.2020.3025731

**** Main functions/scripts:
** Display_SPOQ_Penalty_2D.m
** Load_SPOQ_Data_Paper.m
** Load_SPOQ_Data_Simulated.m
** Load_SPOQ_Data_User.m
** Present_SPOQ_Data_Information
** Run_SPOQ_Recovery.m

Scripts "Load_***.m" load (or create) signals, displays them, set SPOQ parameters, present data information from "Present_SPOQ_Data_Information.m" and run the recovery in "Run_SPOQ_Recovery.m"
- a sparse 'xtrue' vector, 
- an array 'K' coding the (non-stationary) measurement/degradation operator/matrix,
- a 'noise' vector,
- the 'y' observation vector (y = K*x + [someFactor]*noise).

Use either:
- 'Load_SPOQ_Data_Paper.m': to reproduce paper results
- 'Load_SPOQ_Data_Simulated.m': to perform recovery from simulated peaks
- 'Load_SPOQ_Data_User': to perform recovery on user signals (and check their dimensions)

Function
-Display_SPOQ_Penalty_2D.m
compares the \ell_0 count measure and smooth/non-convex SPOQ penalties