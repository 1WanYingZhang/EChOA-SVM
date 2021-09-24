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

function RA=MutationU(dim,Max_iter, inp,Current_iter)
r=rand(1,dim);
r=r>=(Current_iter/Max_iter);
RA=inp;
%RA(r)=1-RA(r);%set it to inverted value
RA(r)=rand(sum(r(:)),1)>0.5;%set it to new values