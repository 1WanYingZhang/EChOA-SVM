function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = GOA_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
GOA_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
GOAPositions=initialization(GOA_option.sizepop,dim,1,0);
pop=zeros(GOA_option.sizepop,dim+2);
pop(:,3:end)=GOAPositions; 
GOAFitness= zeros(1,GOA_option.sizepop);
fit_gen=zeros(1,GOA_option.maxgen);
score=zeros(1,GOA_option.maxgen);
MM=zeros(1,dim);
M=zeros(GOA_option.maxgen,dim);
cMax=1;
cMin=0.00004;
%%% 产生初代粒子并计算初代粒子适应度
for i=1:GOA_option.sizepop
     % 随机产生c,g
     pop(i,1) = (GOA_option.popcmax-GOA_option.popcmin)*rand+GOA_option.popcmin;
     pop(i,2) = (GOA_option.popgmax-GOA_option.popgmin)*rand+GOA_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(GOA_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     GOAFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_fitness,sorted_indexes]=sort(GOAFitness);
for newindex=1:GOA_option.sizepop
    Sorted_GOA(newindex,:)=pop(sorted_indexes(newindex),:);
end
pop=Sorted_GOA;
GOAFitness=sorted_fitness;
TargetPosition=Sorted_GOA(1,:);
TargetFitness=sorted_fitness(1);
fit_gen(1)=TargetFitness;
%%% 迭代寻优
for t=1:GOA_option.maxgen
    c=cMax-t*((cMax-cMin)/GOA_option.maxgen); % Eq. (2.8) in the paper
   for i=1:size(GOAPositions,1)
        temp= pop';
        for k=1:2:dim+2
            S_i=zeros(2,1);
            for j=1:GOA_option.sizepop
                if i~=j
                    Dist=distance(temp(k:k+1,j), temp(k:k+1,i)); % Calculate the distance between two grasshoppers
                    
                    r_ij_vec=(temp(k:k+1,j)-temp(k:k+1,i))/(Dist+eps); % xj-xi/dij in Eq. (2.7)
                    xj_xi=2+rem(Dist,2); % |xjd - xid| in Eq. (2.7)                     
                    s_ij=((ub-lb)*c/2)*S_func(xj_xi).*r_ij_vec; % The first part inside the big bracket in Eq. (2.7)
                    S_i=S_i+s_ij;
                end
            end
            S_i_total(k:k+1, :) = S_i;
            
        end
        
        X_new = c * S_i_total'+ (TargetPosition); % Eq. (2.7) in the paper      
        GOAPositions_temp(i,:)=X_new'; 
    end
    % GrassHopperPositions
    GOAPositions=GOAPositions_temp;
    HHH= [pop;GOAPositions];% 原先位置与后代位置合并成一个新的位置矩阵
    pop= HHH;
    
    for i=1:GOA_option.sizepop
 
        FU=pop(i,1)>GOA_option.popcmax;
        FL=pop(i,1)<GOA_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+GOA_option.popcmax.*FU+GOA_option.popcmin.*FL;
        
        FU=pop(i,2)>GOA_option.popgmax;
        FL=pop(i,2)<GOA_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+GOA_option.popgmax.*FU+GOA_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
        D_Leader = MutationU(dim,GOA_option.maxgen,pop(i,3:end),t);
        pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        GOAPositions=logical(pop(i,3:end));
        
        cmd=['-v ',num2str(GOA_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        GOAFitness(i)=svmtrain(train_label, train(:,GOAPositions), cmd);
        [sorted_fitness,sorted_indexes]=sort(GOAFitness);
    
        for newindex=1:GOA_option.sizepop   %%%%%%%%%%%%%%%%%%%%%%这里原来是2N？%%%%%%%%%%%%%%%%%%%%%%%
            Sorted_GOA(newindex,:)=pop(sorted_indexes(newindex),:);
        end

        pop=Sorted_GOA(1:GOA_option.sizepop,:);
        GOAFitness = sorted_fitness(1:GOA_option.sizepop);       
       
        if GOAFitness(i)>TargetFitness
           TargetFitness=GOAFitness(i);
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
bestM= M(GOA_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(GOA_option.maxgen);
end        