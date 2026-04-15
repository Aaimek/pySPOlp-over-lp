%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SPOQ: Smooth lp-Over-lq ratio
%%% This code implements the SPOQ regularization function presented in
%%% "SPOQ $\ell_p$-Over-$\ell_q$ Regularization for Sparse Signal: Recovery 
%%% applied to Mass Spectrometry"
%%% IEEE Transactions on Signal Processing, 2020, Volume 68, pages 6070--6084
%%% Afef Cherni, IEEE member, 
%%% Emilie Chouzenoux, IEEE Member,
%%% Laurent Duval, IEEE Member,
%%% Jean-Christophe Pesquet, IEEE Fellow
%%% https://arxiv.org/abs/2001.08496
%%% https://doi.org/10.1109/TSP.2020.3025731
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Verify data dimensions and display observed data components
%dimensions
disp(['The size of your original data is ', num2str(size(xtrue))]);
disp(['The size of your observation operator is ', num2str(size(K))]);
disp(['The size of your measurement data is ', num2str(size(y))]);
%plots
figure()
% xtrue
subplot(2,2,1);
plot(xtrue, 'b-', 'linewidth', 1);
ylim([-0.05*max(xtrue) 1.1*max(xtrue)]);
axis tight; grid on
title("Original signal x");
xlabel("Samples", 'linewidth', 2);
% K
subplot(2,2,2);
contourf(K);
 set(gca,'YDir','reverse')
title("Measurement matrix K");
% noise
subplot(2,2,3)
plot(noise, 'k-', 'linewidth', 1)
axis tight; grid on
title("Noise");
xlabel("Samples", 'linewidth', 2);
% y
subplot(2,2,4);
plot(y, 'r-', 'linewidth', 1);
title("Observation y", 'linewidth', 2);
ylim([-0.05*max(y) 1.1*max(y)]);
axis tight; grid on

xlabel("Samples", 'linewidth', 2);