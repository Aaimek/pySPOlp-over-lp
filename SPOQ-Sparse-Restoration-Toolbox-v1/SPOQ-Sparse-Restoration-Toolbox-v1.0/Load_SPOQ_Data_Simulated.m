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
%%% Created by Laurent Duval 16/03/2021

%% Random location and amplitude peaks convolved with a finite support
%% "Gaussians" from binomial coefficients
clc
clear all
close all
addpath(genpath('Data'));
addpath(genpath('Tools'));

%% Initialization
disp('_____________________________________________________')
disp('...loading data ...');
nSample = 500;
nPeak = 20 ;
peakWidth = 5;
xtrue = zeros(nSample,1);
xtrueLocation = randperm(nSample,nPeak);
xtrueAmplitude = rand(nPeak,1);
xtrue(xtrueLocation) = xtrueAmplitude;

peakMatrix = pascal(peakWidth); peakShape = diag(fliplr(peakMatrix)); 
peakShape = peakShape/sum(peakShape); 
peakShapeFilled = [peakShape;zeros(nSample-peakWidth,1)]';
K = toeplitz([peakShapeFilled(1) fliplr(peakShapeFilled(2:end))], peakShapeFilled);
y = K*xtrue;
% add the gaussian noise with a standard deviation sigma
noise = randn(nSample,1);
sigma = 0.5*max(y)/100;
y = y + sigma*noise;

% choose SPOQ parameters
xi = 1.1*sqrt(nSample)*sigma;
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