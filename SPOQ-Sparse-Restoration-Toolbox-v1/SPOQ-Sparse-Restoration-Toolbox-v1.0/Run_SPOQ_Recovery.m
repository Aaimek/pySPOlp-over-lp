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

%% SPOQ recovery
disp(['Running TR-VMFB algorithm on SPOQ penalty with p = ', num2str(p), ' and q = ', num2str(q)]);
tic;
[xrec,fcost,Bwhile,time,mysnr]=FB_PPXALpLq(K,y,p,q,2,alpha,beta,eta,xi,nbiter,xtrue);
tf= toc;
text=['Reconstruction in ', num2str(length(time)), ' iterations'];
disp(text);
text=['SNR = ', num2str(-10*log10(sum((xtrue-xrec).^2)/sum(xtrue.^2)))];
disp(text);
text=['Reconstruction time is ', num2str(sum(time)), 'seconds'];
disp(text);
disp('_____________________________________________________')

%% Results
figure();
plot(xtrue, 'r-o'); hold on; plot(xrec, 'b--'); hold off;
axis tight; grid on
legend("Original signal", "Estimated signal");
title("Reconstruction results")
figure();
plot([0;cumsum(time(:))],mysnr,'-k','linewidth',2);
axis tight; grid on
title("Algorithm convergence");
xlabel("Time (s)");
ylabel("SNR (dB)");
legend('TR-VMFB')