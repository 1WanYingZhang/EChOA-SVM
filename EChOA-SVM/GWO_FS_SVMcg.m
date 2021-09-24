function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = GWO_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
GWO_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
GWOPositions=initialization(GWO_option.sizepop,dim,1,0);
pop=zeros(GWO_option.sizepop,dim+2);
X_new=zeros(GWO_option.sizepop,dim+2);
pop(:,3:end)=GWOPositions; 
GWOFitness= zeros(1,GWO_option.sizepop);
fit_gen=zeros(1,GWO_option.maxgen);
score=zeros(1,GWO_option.maxgen);
MM=zeros(1,dim);
M=zeros(GWO_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:GWO_option.sizepop
     % 随机产生c,g
     pop(i,1) = (GWO_option.popcmax-GWO_option.popcmin)*rand+GWO_option.popcmin;
     pop(i,2) = (GWO_option.popgmax-GWO_option.popgmin)*rand+GWO_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(GWO_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     GWOFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_fitness,sorted_indexes]=sort(GWOFitness);
for newindex=1:GWO_option.sizepop
    Sorted_GWO(newindex,:)=pop(sorted_indexes(newindex),:);
end
pop=Sorted_GWO;
GWOFitness=sorted_fitness;
TargetPosition=Sorted_GWO(1,:);
TargetFitness=sorted_fitness(1);
fit_gen(1)=TargetFitness;
worstFitness = sorted_fitness(GWO_option.sizepop);
bestFitness = sorted_fitness(1);

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim+2);
Alpha_score=zeros(1,GWO_option.maxgen); %change this to -inf for maximization problems

Beta_pos=zeros(1,dim+2);
Beta_score=zeros(1,GWO_option.maxgen); %change this to -inf for maximization problems

Delta_pos=zeros(1,dim+2);
Delta_score=zeros(1,GWO_option.maxgen); %change this to -inf for maximization problems

l=0;% Loop counter

% Main loop
for l=1:GWO_option.maxgen
    for i=1:GWO_option.sizepop
        
        for j=1:dim+2
                rand_index_A = tournamentSelection(1./GWOFitness,0.5);
                rand_index_B = tournamentSelection(1./GWOFitness,0.5);
        end
         
    a=2-l*((2)/GWO_option.maxgen); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:GWO_option.sizepop
        for j=1:dim+2   
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-pop(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-pop(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-pop(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            pop(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    
     for i=1:GWO_option.sizepop
 
        FU=pop(i,1)>GWO_option.popcmax;
        FL=pop(i,1)<GWO_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+GWO_option.popcmax.*FU+GWO_option.popcmin.*FL;
        
        FU=pop(i,2)>GWO_option.popgmax;
        FL=pop(i,2)<GWO_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+GWO_option.popgmax.*FU+GWO_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
        D_Leader = MutationU(dim,GWO_option.maxgen,pop(i,3:end),l);
        pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        GWOPositions=logical(real(pop(i,3:end)));
        
        cmd=['-v ',num2str(GWO_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        GWOFitness(i)=svmtrain(train_label, train(:,GWOPositions), cmd);
        [sorted_fitness,sorted_indexes]=sort(GWOFitness);
    
        for newindex=1:GWO_option.sizepop
            Sorted_GWO(newindex,:)=pop(sorted_indexes(newindex),:);
        end

        pop=Sorted_GWO(1:GWO_option.sizepop,:);
        GWOFitness = sorted_fitness(1:GWO_option.sizepop);       
       
        if GWOFitness(i)>TargetFitness
           TargetFitness=GWOFitness(i);
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
bestM= M(GWO_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(GWO_option.maxgen);
end



