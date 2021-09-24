%_________________________________________________________________________%
%  
% Hybrid Whale Optimization Algorithm 
% with Simulated Annealing for Feature Selection 
%           By: Majdi Mafarja and Seyedali Mirjalili   
%           email: mmafarjeh@gmail.com
% 
% Main paper: M. Mafarja and S. Mirjalili                                 %
%               Hybrid Whale Optimization Algorithm                       %
%               with Simulated Annealing for Feature Selection            %
%               Neurocomputing , in press,                                %
%               DOI: https://doi.org/10.1016/j.neucom.2017.04.053         %
%                                                                         %
%  Developed in MATLAB R2014a                                             %
%                                                                         %
%  the original code of WOA is availble on                                %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                e-Mail: ali.mirjalili@gmail.com                          %
%                      
%_________________________________________________________________________%

function out=CrossOverU(RA,RE)
r=rand(size(RA))<0.5;
out(r)=RA(r);
out(~r)=RE(~r);