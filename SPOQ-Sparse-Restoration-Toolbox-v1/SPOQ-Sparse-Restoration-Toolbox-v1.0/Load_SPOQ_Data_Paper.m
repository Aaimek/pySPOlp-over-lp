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
%%% Created by Afef Cherni 04/03/2021

%% Reproduce results given in the paper
clc
clear all
close all
addpath(genpath('Data'));
addpath(genpath('Tools'));

%% Initialization
disp('_____________________________________________________')
disp('...loading data ...');
% load xtrue (sparse signal)
xtrue = load('x'); 
% load K (measurement matrix)
K = load('K');  
% build y = K*x
y = K*xtrue;
% add the gaussian noise with a standard deviation sigma
sigma = 0.1*max(y)/100;
noise = load('noise');
y = y + sigma*noise;
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