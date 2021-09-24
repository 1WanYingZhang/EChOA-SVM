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

function iSelected = TournamentSelect(fitness,tournamentSelectionParameter);
populationSize = size(fitness,1);
iTmp1 = 1 + fix(rand*populationSize);
iTmp2 = 1 + fix(rand*populationSize);
r = rand;
if (r < tournamentSelectionParameter)
    if (fitness(iTmp1) > fitness(iTmp2))
        iSelected = iTmp1;
    else
        iSelected = iTmp2;
    end
else
    if (fitness(iTmp1) > fitness(iTmp2))
        iSelected = iTmp2;
    else
        iSelected = iTmp1;
    end
end