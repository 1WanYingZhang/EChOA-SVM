function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = ALO_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
antlions_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ants_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
Elite_antlion_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
antlionsPositions=initialization(antlions_option.sizepop,dim,1,0);
popantlions=zeros(antlions_option.sizepop,dim+2);
X_new=zeros(antlions_option.sizepop,dim+2);
popantlions(:,3:end)=antlionsPositions; 
antlionsFitness= zeros(1,antlions_option.sizepop);
antsPositions=initialization(antlions_option.sizepop,dim,1,0);
popants=zeros(ants_option.sizepop,dim+2);
X_ant_new=zeros(ants_option.sizepop,dim+2);
popants(:,3:end)=antsPositions; 
ants_Fitness=zeros(1,ants_option.sizepop);
fit_gen=zeros(1,antlions_option.maxgen);
score=zeros(1,antlions_option.maxgen);
Sorted_antlions=zeros(antlions_option.sizepop,dim+2);
Elite_antlion_Position=zeros(1,dim+2);
Elite_antlion_Fitness=zeros(1,Elite_antlion_option.maxgen);
MM=zeros(1,dim);
M=zeros(antlions_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:antlions_option.sizepop
     % 随机产生c,g
     popantlions(i,1) = (antlions_option.popcmax-antlions_option.popcmin)*rand+antlions_option.popcmin;
     popantlions(i,2) = (antlions_option.popgmax-antlions_option.popgmin)*rand+antlions_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(antlions_option.v),' -c ',num2str( popantlions(i,1) ),' -g ',num2str( popantlions(i,2) )];
     antlionsFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_antlion_fitness,sorted_indexes]=sort(antlionsFitness);
for newindex=1:antlions_option.sizepop
    Sorted_antlions(newindex,:)=popantlions(sorted_indexes(newindex),:);
end
popantlions=Sorted_antlions;
antlionsFitness=sorted_antlion_fitness;
TargetPosition=Sorted_antlions(1,:);
TargetFitness=sorted_antlion_fitness(1);
fit_gen(1)=TargetFitness; 
Elite_antlion_Position=Sorted_antlions(1,:);
Elite_antlion_Fitness=sorted_antlion_fitness(1);

% Main loop start from the second iteration since the first iteration 
% was dedicated to calculating the fitness of antlions
l=2; 
for l=1:antlions_option.maxgen+1
    
    % This for loop simulate random walks
    for i=1:ants_option.sizepop
        % Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
        Rolette_index=RouletteWheelSelection(1./sorted_antlion_fitness);
        if Rolette_index==-1  
            Rolette_index=1;
        end
      
        % RA is the random walk around the selected antlion by rolette wheel
        RA=Random_walk_around_antlion(dim,antlions_option.maxgen,lb,ub, Sorted_antlions(Rolette_index,3:end),l);
        
        % RA is the random walk around the elite (best antlion so far)
        [RE]=Random_walk_around_antlion(dim,antlions_option.maxgen,lb,ub, Elite_antlion_Position(1,3:end),l);

        popants(i,3:end)= (RA(l,:)+RE(l,:))/2; % Equation (2.13) in the paper
    end
    
    for i=1:ants_option.maxgen+1
    
    % Update antlion positions and fitnesses based of the ants (if an ant 
    % becomes fitter than an antlion we assume it was cought by the antlion  
    % and the antlion update goes to its position to build the trap)
    double_population=[Sorted_antlions;popants];
    double_fitness=[sorted_antlion_fitness ants_Fitness];
        
    [double_fitness_sorted I]=sort(double_fitness);
    double_sorted_population=double_population(I,:);
        
    antlionsFitness=double_fitness_sorted(1:antlions_option.sizepop);
    Sorted_antlions=double_sorted_population(1:antlions_option.sizepop,:);
        
    % Update the position of elite if any antlinons becomes fitter than it
    if antlionsFitness(1)<Elite_antlion_Fitness 
        Elite_antlion_Position=Sorted_antlions(1,:);
        Elite_antlion_Fitness=antlionsFitness(1);
    end
      
    % Keep the elite in the population
    Sorted_antlions(1,:)=Elite_antlion_Position;
    antlionsFitness(1)=Elite_antlion_Fitness;
    end

for i=1:antlions_option.sizepop
 
        FU=popantlions(i,1)>antlions_option.popcmax;
        FL=popantlions(i,1)<antlions_option.popcmin;
        popantlions(i,1)=(popantlions(i,1).*(~(FU+FL)))+antlions_option.popcmax.*FU+antlions_option.popcmin.*FL;
        
        FU=popantlions(i,2)>antlions_option.popgmax;
        FL=popantlions(i,2)<antlions_option.popgmin;
        popantlions(i,2)=(popantlions(i,2).*(~(FU+FL)))+antlions_option.popgmax.*FU+antlions_option.popgmin.*FL;
        
        FU=popantlions(i,3:end)>1;
        FL=popantlions(i,3:end)<0;
        popantlions(i,3:end)=(popantlions(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
        D_Leader = MutationU(dim,antlions_option.maxgen,popantlions(i,3:end),l);
        popantlions(i,3:end)=CrossOverU(popantlions(i,3:end),D_Leader);
        antlionsPositions=logical(real(popantlions(i,3:end)));
        
        cmd=['-v ',num2str(antlions_option.v),' -c ',num2str( popantlions(i,1) ),' -g ',num2str( popantlions(i,2) )];
        antlionsFitness(i)=svmtrain(train_label, train(:,antlionsPositions), cmd);
        [sorted_fitness,sorted_indexes]=sort(antlionsFitness);
    
        for newindex=1:antlions_option.sizepop
            Sorted_antlions(newindex,:)=popantlions(sorted_indexes(newindex),:);
        end

        popantlions=Sorted_antlions(1:antlions_option.sizepop,:);
        antlionsFitness = sorted_fitness(1:antlions_option.sizepop);       
       
        if antlionsFitness(i)>TargetFitness
           TargetFitness=antlionsFitness(i);
           TargetPosition=popantlions(i,:);
        end
         MM=TargetPosition(1,3:end);
    end  
    fit_gen(l)=TargetFitness;
    M(l,:)=MM;  
  
    acct=(TargetFitness/100);%换算成百分数
    SzW=0.01;
    score(l)=real((1-SzW)*(1-acct)+SzW*sum(MM)/(dim));
end 
bestM= M(antlions_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(antlions_option.maxgen);
end





