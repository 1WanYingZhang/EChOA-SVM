function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = EChimp_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
EChimp_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
EChimpPositions=initialization(EChimp_option.sizepop,dim,1,0);
pop=zeros(EChimp_option.sizepop,dim+2);
pop(:,3:end)=EChimpPositions; 
EChimpFitness= zeros(1,EChimp_option.sizepop);
fit_gen=zeros(1,EChimp_option.maxgen);
score=zeros(1,EChimp_option.maxgen);
MM=zeros(1,dim);
M=zeros(EChimp_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:EChimp_option.sizepop
     % 随机产生c,g
     pop(i,1) = (EChimp_option.popcmax-EChimp_option.popcmin)*rand+EChimp_option.popcmin;
     pop(i,2) = (EChimp_option.popgmax-EChimp_option.popgmin)*rand+EChimp_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(EChimp_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     EChimpFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_fitness,sorted_indexes]=sort(EChimpFitness);
for newindex=1:EChimp_option.sizepop
    Sorted_EChimp(newindex,:)=pop(sorted_indexes(newindex),:);
end
pop=Sorted_EChimp;
EChimpFitness=sorted_fitness;
TargetPosition=Sorted_EChimp(1,:);
TargetFitness=sorted_fitness(1);
fit_gen(1)=TargetFitness;
% worstFitness = sorted_fitness(EChimp_option.sizepop);
% bestFitness = sorted_fitness(1);

l=0;% Loop counter
%%
% Main loop
for l=1:EChimp_option.maxgen
    for i=1:EChimp_option.sizepop 
        
%          for j=1:dim+2
%                 rand_index_A = tournamentSelection(1./EChimpFitness,0.5);
%                 rand_index_B = tournamentSelection(1./EChimpFitness,0.5);
%          end
        
       % HDPM      
        r=rand;
        Pm=0.5;
        if r<Pm
           w1=(EChimpPositions-lb)/(ub-lb);
           w2=(ub-EChimpPositions)/(ub-lb);
           n=1;
           if r<0.5
              delta=(((2*r)+(1-2*r)*(1-w1).^(n+1)).^(1/(n+1)))-1;
           else
              delta=1-(2*(1-r)+2*(r-0.5)*(1-w2).^(n+1)).^(1/(n+1));
              EChimpPositions=EChimpPositions+delta*(ub-lb);
           end 
        end
       % HDPM
    f=2-l*((2)/EChimp_option.maxgen); % a decreases linearly fron 2 to 0    
     % Oppose the least fitness elements
    threshold=f;
    EChimpPositions=corBAS(EChimpPositions,EChimpFitness,threshold);  
    % Update the Position of search agents including omegas
    for i=1:EChimp_option.sizepop
        for j=1:dim+2      
% initialize Attacker, Barrier, Chaser, and Driver
Attacker_pos=zeros(1,dim+2);
Attacker_score=zeros(1,EChimp_option.maxgen); %change this to -inf for maximization problems
Barrier_pos=zeros(1,dim+2);
Barrier_score=zeros(1,EChimp_option.maxgen); %change this to -inf for maximization problemsv
%% Please note that to choose a other groups you should use the related group strategies

            r11=rand(); % r1 is a random number in [0,1]
            r12=rand(); % r2 is a random number in [0,1]
            
            r21=rand(); % r1 is a random number in [0,1]
            r22=rand(); % r2 is a random number in [0,1]

%% % Please note that to choose various Chaotic maps you should use the related Chaotic maps strategies
            m=chaos(3,1,1); % Equation (5)
                      
            A1=2*f*r11-f; % Equation (3)
            C1=2*r12; % Equation (4)
            D_Attacker=abs(C1*Attacker_pos(j)-m*pop(i,j)); % Equation (6)
            X1=Attacker_pos(j)-A1*D_Attacker; % Equation (7)
                       
            A2=2*f*r21-f; % Equation (3)
            C2=2*r22; % Equation (4)                          
            D_Barrier=abs(C2*Barrier_pos(j)-m*pop(i,j)); % Equation (6)
            X2=Barrier_pos(j)-A2*D_Barrier; % Equation (7)           
     
            EChimpPositions(i,j)=(X1+X2)/2;% Equation (8)          
        end
    end 
    
    for i=1:EChimp_option.sizepop
 
        FU=pop(i,1)>EChimp_option.popcmax;
        FL=pop(i,1)<EChimp_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+EChimp_option.popcmax.*FU+EChimp_option.popcmin.*FL;
        
        FU=pop(i,2)>EChimp_option.popgmax;
        FL=pop(i,2)<EChimp_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+EChimp_option.popgmax.*FU+EChimp_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
         D_Leader = MutationU(dim,EChimp_option.maxgen,pop(i,3:end),l);
         pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        EChimpPositions=logical(real(pop(i,3:end)));
        
        cmd=['-v ',num2str(EChimp_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        EChimpFitness(i)=svmtrain(train_label, train(:,EChimpPositions), cmd); 
        [sorted_fitness,sorted_indexes]=sort(EChimpFitness);
        
         for newindex=1:EChimp_option.sizepop
            Sorted_EChimp(newindex,:)=pop(sorted_indexes(newindex),:);
         end
        
         pop=Sorted_EChimp(1:EChimp_option.sizepop,:);
        EChimpFitness = sorted_fitness(1:EChimp_option.sizepop);       
       
        
        if EChimpFitness(i)>TargetFitness
           TargetFitness=EChimpFitness(i);
           TargetPosition=pop(i,:);
        end
        MM=TargetPosition(1,3:end);
    end  
    fit_gen(l)=TargetFitness;
    M(l,:)=MM;  
  
    acct=(TargetFitness/100);%换算成百分数
    SzW=0.01;
    score(l)=real((1-SzW)*(1-acct)+SzW*sum(MM)/(dim));
end 
bestM= M(EChimp_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(EChimp_option.maxgen);
end