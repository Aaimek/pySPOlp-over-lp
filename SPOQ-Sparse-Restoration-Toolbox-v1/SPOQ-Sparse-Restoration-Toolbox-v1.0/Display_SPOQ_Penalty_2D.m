function Display_SPOQ_Penalty_2D;
clear all
close all
clc

x = (-1:0.005:1)';
y = (-1:0.005:1)';
[X,Y] = meshgrid(x,y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------- l0 count measure  -------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xl0 = ones(size(X));
Yl0 = ones(size(Y));
Xl0(X == 0 ) = 0;
Yl0(Y == 0 ) = 0;
Zl0 = Xl0 + Yl0;
Zl0 = Zl0/max(Zl0(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------- l0.75/l2 smoothed quasi/norm ratio  -------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 0.75;
q = 2;
alpha = 7e-7;
beta = 3e-3;
eta = 0.1;
Zlplq =  lplqnorm(X,alpha,beta,eta,p,q) + lplqnorm(Y,alpha,beta,eta,p,q);
Zlplq = Zlplq./max(Zlplq(:));
ZlplqSPOQ = log(Zlplq);
ZlplqSPOQ = (ZlplqSPOQ-min(ZlplqSPOQ(:)))/(max(ZlplqSPOQ(:))-min(ZlplqSPOQ(:)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------- Figure  -------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,2,1);
surf(X,Y,Zl0, 'EdgeColor','none')
grid on;axis tight
h1 = title('$\ell_0$ count measure');

subplot(1,2,2);
surf(X,Y,Zlplq, 'EdgeColor','none')
grid on;axis tight
h2 = title('Smoothed $\ell_{3/4}$-over-$\ell_2$ quasinorm ratio');

set([h1,h2],'Interpreter','latex');
colormap jet

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------- Functions  -------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%>>>> lplq
function [lq] = lqnorm(X,Y,q)
    lq = ((X.^2 + Y.^2 ).^(1/q));
end

function [lp] = lpnorm(X,Y,p)
    lp = (abs(X).^p+abs(Y).^p);
end

function [normxx] = lplqnorm(x,alpha,beta,eta,p,q)
    normx = zeros(size(x));
    for i=1:size(x,1)
        for j=1:size(x,2)
            lp = ((x(i,j)^2 + alpha^2)^(p/2) - alpha^p)^(1/p);
            lpalpha = (lp^p + beta^p)^(1/p);
            lq = (eta^q + abs(x(i,j))^q )^(1/q);
            normx(i,j) = lpalpha/lq;
        end
    end
    normxx=normx;
end

function [y] = Ind3(x,w)
    if x <= w
        y = 1;
    else
        y = 0;
    end
end

        