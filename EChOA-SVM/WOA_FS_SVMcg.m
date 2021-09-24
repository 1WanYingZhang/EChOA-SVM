function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = WOA_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
WOA_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
WOAPositions=initialization(WOA_option.sizepop,dim,1,0);
pop=zeros(WOA_option.sizepop,dim+2);
X_new=zeros(WOA_option.sizepop,dim+2);
pop(:,3:end)=WOAPositions; 
WOAFitness= zeros(1,WOA_option.sizepop);
fit_gen=zeros(1,WOA_option.maxgen);
Leader_pos=zeros(1,dim+2);
Leader_score=zeros(1,WOA_option.maxgen);
MM=zeros(1,dim);
M=zeros(WOA_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:WOA_option.sizepop
     % 随机产生c,g
     pop(i,1) = (WOA_option.popcmax-WOA_option.popcmin)*rand+WOA_option.popcmin;
     pop(i,2) = (WOA_option.popgmax-WOA_option.popgmin)*rand+WOA_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(WOA_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     WOAFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_fitness,sorted_indexes]=sort(WOAFitness);
for newindex=1:WOA_option.sizepop
    Sorted_WOA(newindex,:)=pop(sorted_indexes(newindex),:);
end
pop=Sorted_WOA;
WOAFitness=sorted_fitness;
TargetPosition=Sorted_WOA(1,:);
TargetFitness=sorted_fitness(1);
fit_gen(1)=TargetFitness;
% worstFitness = sorted_fitness(WOA_option.sizepop);
% bestFitness = sorted_fitness(1);

t=0;% Loop counter

% Main loop
for t=1:WOA_option.maxgen
%     for i=1:WOA_option.sizepop
%         for j=1:dim+2
%                 r = rand();
%                 rand_index_A = tournamentSelection(1./WOAFitness,0.5);
%                 rand_index_B = tournamentSelection(1./WOAFitness,0.5);
%         end     
%     end
    
    a=2-t*((2)/WOA_option.maxgen); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/WOA_option.maxgen);
    
    % Update the Position of search agents 
    for i=1:WOA_option.sizepop
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        
        p = rand();        % p in Eq. (2.6)
        
        for j=1:dim+2
            
            if p<0.5   
                if abs(A)>=1
                    rand_leader_index = floor(WOA_option.sizepop*rand()+1);
                    X_rand = pop(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-pop(i,j)); % Eq. (2.7)
                    pop(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)
                    
                elseif abs(A)<1
                    D_Leader=abs(C*Leader_pos(j)-pop(i,j)); % Eq. (2.1)
                    pop(i,j)=Leader_pos(j)-A*D_Leader;      % Eq. (2.2)
                end
                
            elseif p>=0.5
              
                distance2Leader=abs(Leader_pos(j)-pop(i,j));
                % Eq. (2.5)
                pop(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
                
            end
            
        end
    end
     for i=1:WOA_option.sizepop
 
        FU=pop(i,1)>WOA_option.popcmax;
        FL=pop(i,1)<WOA_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+WOA_option.popcmax.*FU+WOA_option.popcmin.*FL;
        
        FU=pop(i,2)>WOA_option.popgmax;
        FL=pop(i,2)<WOA_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+WOA_option.popgmax.*FU+WOA_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
        D_Leader = MutationU(dim,WOA_option.maxgen,pop(i,3:end),t);
        pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        WOAPositions=logical(real(pop(i,3:end)));
        
        cmd=['-v ',num2str(WOA_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        WOAFitness(i)=svmtrain(train_label, train(:,WOAPositions), cmd);
        [sorted_fitness,sorted_indexes]=sort(WOAFitness);
    
        for newindex=1:WOA_option.sizepop
            Sorted_WOA(newindex,:)=pop(sorted_indexes(newindex),:);
        end

        pop=Sorted_WOA(1:WOA_option.sizepop,:);
        WOAFitness = sorted_fitness(1:WOA_option.sizepop);       
       
        if WOAFitness(i)>TargetFitness
           TargetFitness=WOAFitness(i);
           TargetPosition=pop(i,:);
        end
         MM=TargetPosition(1,3:end);
    end  
    fit_gen(t)=TargetFitness;
    M(t,:)=MM;  
  
    acct=(TargetFitness/100);%换算成百分数
    SzW=0.01;
    score(t)=real((1-SzW)*(1-acct)+SzW*sum(MM)/(dim));
end 
bestM= M(WOA_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(WOA_option.maxgen);
end



