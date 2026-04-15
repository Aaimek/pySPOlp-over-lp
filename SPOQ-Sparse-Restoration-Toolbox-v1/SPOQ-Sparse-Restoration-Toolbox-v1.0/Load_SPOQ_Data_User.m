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
% Created by Afef Cherni 08/03/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load and check custom user data
clc
clear all
close all
addpath(genpath('Tools'));

disp('_____________________________________________________');
disp('.................... loading data ...................');
disp('_____________________________________________________');
disp(' Warning! Do not forget quotes in the input of your data path');
original_signal = 'Provide the path for the original sparse signal ';
xtrue = input(original_signal);
xtrue = load(xtrue);
if size(xtrue,2)~= 1
    disp("Warning, the size of your data should be (N,1)");
    original_signal = 'Try to upload your data correctly ';
    xtrue = input(original_signal);
    xtrue = load(xtrue);
end

Kmeasure = 'Provide the path for the measurement operator ';
K = input(Kmeasure);
K = load(K);
if size(xtrue,1)~= size(K,1)
    disp("Attention, the size of your measurement matrix is not adequate");
    Kmeasure = 'Try to upload your data correctly ';
    K = input(Kmeasure);
    K = load(K);
end

Noise = 'Provide the path for the noise ';
noise = input(Noise);
noise = load(noise);
if size(xtrue,2)~= size(Noise,1)
    disp("Warning, the size of your noise should be (N,1)");
        Noise = 'Try to upload your data correctly ';
    noise = input(Noise);
    noise = load(noise);
end

% add the gaussian noise with a standard deviation sigma
ytrue = K*xtrue;
sigma = 0.1*max(ytrue)/100;
%sigma=1;
y = ytrue + sigma*noise;

% choose SPOQ parameters
N = length(xtrue);
xi = 1.1*sqrt(N)*sigma;
eta = 2E-6;
alpha = 7E-7;
beta = 3E-2;
p = 0.75;
q = 2;
nbiter=5000;

%% Verifiy/dispaly data information
Present_SPOQ_Data_Information

%% Run SPOQ Recovery
Run_SPOQ_Recovery