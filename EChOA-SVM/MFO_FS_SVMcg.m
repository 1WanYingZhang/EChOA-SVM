function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = MFO_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
Moth_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
MothPositions=initialization(Moth_option.sizepop,dim,1,0);
pop=zeros(Moth_option.sizepop,dim+2);
pop(:,3:end)=MothPositions; 
MothFitness= zeros(1,Moth_option.sizepop);
fit_gen=zeros(1,Moth_option.maxgen);
score=zeros(1,Moth_option.maxgen);
MM=zeros(1,dim);
M=zeros(Moth_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:Moth_option.sizepop
     % 随机产生c,g
     pop(i,1) = (Moth_option.popcmax-Moth_option.popcmin)*rand+Moth_option.popcmin;
     pop(i,2) = (Moth_option.popgmax-Moth_option.popgmin)*rand+Moth_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(Moth_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     MothFitness(i) = svmtrain(train_label, train, cmd);
end
% [sorted_fitness,sorted_indexes]=sort(MothFitness);
% for newindex=1:Moth_option.sizepop
%     Sorted_Moth(newindex,:)=pop(sorted_indexes(newindex),:);
% end
% pop=Sorted_Moth;
% MothFitness=sorted_fitness;
% TargetPosition=Sorted_Moth(1,:);
% TargetFitness=sorted_fitness(1);
% fit_gen(1)=TargetFitness;

l=1;
% Main loop
for l=1:Moth_option.maxgen
    
    % Number of flames Eq. (3.14) in the paper
    Flame_no=round(Moth_option.sizepop-l*((Moth_option.sizepop-1)/Moth_option.maxgen));
       
    if l==1
        % Sort the first population of moths
        [sorted_fitness I]=sort(MothFitness);
        Sorted_Moth=pop(I,:);
        
        % Update the flames
        best_flames=Sorted_Moth;
        best_flame_fitness=sorted_fitness;
    else
        
        % Sort the moths
        double_population=[previous_population;best_flames];
        double_fitness=[previous_fitness best_flame_fitness];
        
        [double_fitness_sorted I]=sort(double_fitness);
        double_sorted_population=double_population(I,:);
        
        sorted_fitness=double_fitness_sorted(1:Moth_option.sizepop);
        Sorted_Moth=double_sorted_population(1:Moth_option.sizepop,:);
        
        % Update the flames
        best_flames=Sorted_Moth;
        best_flame_fitness=sorted_fitness;
    end
    % Update the position best flame obtained so far
    Best_flame_score=sorted_fitness(1);
    Best_flame_pos=Sorted_Moth(1,:);
    fit_gen(1)=Best_flame_score;
    
    previous_population=pop;
    previous_fitness=MothFitness;
    
    % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a=-1+l*((-1)/Moth_option.maxgen);
    
    for i=1:Moth_option.sizepop
        
        for j=1:dim+2
            if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame
                
                % D in Eq. (3.13)
                distance_to_flame=abs(Sorted_Moth(i,j)-pop(i,j));
                b=1;
                t=(a-1)*rand+1;
                
                % Eq. (3.12)
                MothPositions(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+Sorted_Moth(i,j);
            end
            
            if i>Flame_no % Upaate the position of the moth with respct to one flame
                
                % Eq. (3.13)
                distance_to_flame=abs(Sorted_Moth(i,j)-pop(i,j));
                b=1;
                t=(a-1)*rand+1;
                
                % Eq. (3.12)
                MothPositions(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+Sorted_Moth(Flame_no,j);
            end
            
        end
        
    end
    
for i=1:Moth_option.sizepop
 
        FU=pop(i,1)>Moth_option.popcmax;
        FL=pop(i,1)<Moth_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+Moth_option.popcmax.*FU+Moth_option.popcmin.*FL;
        
        FU=pop(i,2)>Moth_option.popgmax;
        FL=pop(i,2)<Moth_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+Moth_option.popgmax.*FU+Moth_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
        D_Leader = MutationU(dim,Moth_option.maxgen,pop(i,3:end),l);
        pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        MothPositions=logical(real(pop(i,3:end)));
        
        cmd=['-v ',num2str(Moth_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        MothFitness(i)=svmtrain(train_label, train(:,MothPositions), cmd);
        [sorted_fitness,sorted_indexes]=sort(MothFitness);
    
        for newindex=1:Moth_option.sizepop
            Sorted_Moth(newindex,:)=pop(sorted_indexes(newindex),:);
        end

        pop=Sorted_Moth(1:Moth_option.sizepop,:);
        MothFitness = sorted_fitness(1:Moth_option.sizepop);       
       
        if MothFitness(i)>Best_flame_score
           Best_flame_score=MothFitness(i);
           Best_flame_pos=pop(i,:);
        end
        MM=Best_flame_pos(1,3:end);
    end  
    fit_gen(l)=Best_flame_score;
    M(l,:)=MM;  
  
    acct=(Best_flame_score/100);%换算成百分数
    SzW=0.01;
    score(l)=real((1-SzW)*(1-acct)+SzW*sum(MM)/(dim));
end 
bestM= M(Moth_option.maxgen,:);
bestc = Best_flame_pos(1);
bestg = Best_flame_pos(2);
bestCVaccuarcy = fit_gen(Moth_option.maxgen);
end
