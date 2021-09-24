function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = SSA_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
SSA_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
SSAPositions=initialization(SSA_option.sizepop,dim,1,0);
pop=zeros(SSA_option.sizepop,dim+2);
pop(:,3:end)=SSAPositions; 
SSAFitness= zeros(1,SSA_option.sizepop);
fit_gen=zeros(1,SSA_option.maxgen);
score=zeros(1,SSA_option.maxgen);
MM=zeros(1,dim);
M=zeros(SSA_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:SSA_option.sizepop
     % 随机产生c,g
     pop(i,1) = (SSA_option.popcmax-SSA_option.popcmin)*rand+SSA_option.popcmin;
     pop(i,2) = (SSA_option.popgmax-SSA_option.popgmin)*rand+SSA_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(SSA_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     SSAFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_fitness,sorted_indexes]=sort(SSAFitness);
for newindex=1:SSA_option.sizepop
    Sorted_barnacle(newindex,:)=pop(sorted_indexes(newindex),:);
end
pop=Sorted_barnacle;
SSAFitness=sorted_fitness;
TargetPosition=Sorted_barnacle(1,:);
TargetFitness=sorted_fitness(1);
fit_gen(1)=TargetFitness;
%%% 迭代寻优
for t=1:SSA_option.maxgen
    c1 = 2*exp(-(4*t/SSA_option.maxgen)^2); % Eq. (3.2) in the paper
    
    for i=1:size(SSAPositions,1)
        
        SSAPositions= SSAPositions';
        
        if i<=SSA_option.sizepop/2
            for j=1:1:dim+2
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5 
                    SSAPositions(j,i)=TargetPosition(j)+c1*((ub-lb)*c2+lb);
                else
                    SSAPositions(j,i)=TargetPosition(j)-c1*((ub-lb)*c2+lb);
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        elseif i>SSA_option.sizepop/2 && i<SSA_option.sizepop+1
            point1=SSAPositions(:,i-1);
            point2=SSAPositions(:,i);
            SSAPositions(:,i)=(point2+point1)/2; % % Eq. (3.4) in the paper
        end
        
        SSAPositions= SSAPositions';
    end  
    % GrassHopperPositions
    SSAPositions=SSAPositions;
    Barnaclescolony= [pop;SSAPositions];% 原先位置与后代位置合并成一个新的位置矩阵
    pop= Barnaclescolony;
    
    for i=1:SSA_option.sizepop
 
        FU=pop(i,1)>SSA_option.popcmax;
        FL=pop(i,1)<SSA_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+SSA_option.popcmax.*FU+SSA_option.popcmin.*FL;
        
        FU=pop(i,2)>SSA_option.popgmax;
        FL=pop(i,2)<SSA_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+SSA_option.popgmax.*FU+SSA_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
        D_Leader = MutationU(dim,SSA_option.maxgen,pop(i,3:end),t);
        pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        SSAPositions=logical(pop(i,3:end));
        
        cmd=['-v ',num2str(SSA_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        SSAFitness(i)=svmtrain(train_label, train(:,SSAPositions), cmd);
        [sorted_fitness,sorted_indexes]=sort(SSAFitness);
    
        for newindex=1:SSA_option.sizepop   %%%%%%%%%%%%%%%%%%%%%%这里原来是2N？%%%%%%%%%%%%%%%%%%%%%%%
            Sorted_barnacle(newindex,:)=pop(sorted_indexes(newindex),:);
        end

        pop=Sorted_barnacle(1:SSA_option.sizepop,:);
        SSAFitness = sorted_fitness(1:SSA_option.sizepop);       
       
        if SSAFitness(i)>TargetFitness
           TargetFitness=SSAFitness(i);
           TargetPosition=pop(i,:);
        end
        MM=TargetPosition(1,3:end);
    end  
    fit_gen(t)=TargetFitness;
    M(t,:)=MM;   
    
    acct=(TargetFitness/100);%换算成百分数
    SzW=0.01;
    score(t)=(1-SzW)*(1-acct)+SzW*sum(MM)/(dim);
    
end 
bestM= M(SSA_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(SSA_option.maxgen);
end        