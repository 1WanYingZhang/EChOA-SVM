function [bestM,fit_gen,score,bestCVaccuarcy,bestc,bestg] = Chimp_FS_SVMcg(train_label,train,dim)
%%% 参数初始化
Chimp_option = struct('maxgen',100,'sizepop',30,'v',5,'popcmax',10^2,'popcmin',10^(-3),'popgmax',10^3,'popgmin',10^(-2));
ub=1;
lb=0;
ChimpPositions=initialization(Chimp_option.sizepop,dim,1,0);
pop=zeros(Chimp_option.sizepop,dim+2);
pop(:,3:end)=ChimpPositions; 
ChimpFitness= zeros(1,Chimp_option.sizepop);
fit_gen=zeros(1,Chimp_option.maxgen);
score=zeros(1,Chimp_option.maxgen);
Attacker_pos=zeros(1,dim+2);
Attacker_score=zeros(1,Chimp_option.maxgen); 
Barrier_pos=zeros(1,dim+2);
Barrier_score=zeros(1,Chimp_option.maxgen);
Cttacker_pos=zeros(1,dim+2);
Cttacker_score=zeros(1,Chimp_option.maxgen); 
Darrier_pos=zeros(1,dim+2);
Darrier_score=zeros(1,Chimp_option.maxgen); 
MM=zeros(1,dim);
M=zeros(Chimp_option.maxgen,dim);
%%% 产生初代粒子并计算初代粒子适应度
for i=1:Chimp_option.sizepop
     % 随机产生c,g
     pop(i,1) = (Chimp_option.popcmax-Chimp_option.popcmin)*rand+Chimp_option.popcmin;
     pop(i,2) = (Chimp_option.popgmax-Chimp_option.popgmin)*rand+Chimp_option.popgmin;
     % 计算初始适应度
     cmd = ['-v ',num2str(Chimp_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
     ChimpFitness(i) = svmtrain(train_label, train, cmd);
end
[sorted_fitness,sorted_indexes]=sort(ChimpFitness);
for newindex=1:Chimp_option.sizepop
    Sorted_Chimp(newindex,:)=pop(sorted_indexes(newindex),:);
end
pop=Sorted_Chimp;
ChimpFitness=sorted_fitness;
TargetPosition=Sorted_Chimp(1,:);
TargetFitness=sorted_fitness(1);
fit_gen(1)=TargetFitness;

l=0;% Loop counter
%%
% Main loop
for l=1:Chimp_option.maxgen
    for i=1:Chimp_option.sizepop 
        
    f=2-l*((2)/Chimp_option.maxgen); % a decreases linearly fron 2 to 0   
      
    %Group 1
    C1G1=1.95-((2*l^(1/3))/(Chimp_option.maxgen^(1/3)));
    C2G1=(2*l^(1/3))/(Chimp_option.maxgen^(1/3))+0.5;
        
    %Group 2
    C1G2= 1.95-((2*l^(1/3))/(Chimp_option.maxgen^(1/3)));
    C2G2=(2*(l^3)/(Chimp_option.maxgen^3))+0.5;
    
    %Group 3
    C1G3=(-2*(l^3)/(Chimp_option.maxgen^3))+2.5;
    C2G3=(2*l^(1/3))/(Chimp_option.maxgen^(1/3))+0.5;
    
    %Group 4
    C1G4=(-2*(l^3)/(Chimp_option.maxgen^3))+2.5;
    C2G4=(2*(l^3)/(Chimp_option.maxgen^3))+0.5;
    
     for i=1:Chimp_option.sizepop
        for j=1:dim+2      
%% Please note that to choose a other groups you should use the related group strategies

           r11=C1G1*rand(); % r1 is a random number in [0,1]
            r12=C2G1*rand(); % r2 is a random number in [0,1]
            
            r21=C1G2*rand(); % r1 is a random number in [0,1]
            r22=C2G2*rand(); % r2 is a random number in [0,1]
            
            r31=C1G3*rand(); % r1 is a random number in [0,1]
            r32=C2G3*rand(); % r2 is a random number in [0,1]
            
            r41=C1G4*rand(); % r1 is a random number in [0,1]
            r42=C2G4*rand(); % r2 is a random number in [0,1]
            
            A1=2*f*r11-f; % Equation (3)
            C1=2*r12; % Equation (4)

%% % Please note that to choose various Chaotic maps you should use the related Chaotic maps strategies
            m=chaos(3,1,1); % Equation (5)
            
             D_Attacker=abs(C1*Attacker_pos(j)-m*pop(i,j)); % Equation (6)
            X1=Attacker_pos(j)-A1*D_Attacker; % Equation (7)
                       
            A2=2*f*r21-f; % Equation (3)
            C2=2*r22; % Equation (4)
            
                   
            D_Barrier=abs(C2*Barrier_pos(j)-m*pop(i,j)); % Equation (6)
            X2=Barrier_pos(j)-A2*D_Barrier; % Equation (7)     
            
        
            
            A3=2*f*r31-f; % Equation (3)
            C3=2*r32; % Equation (4)
            
            D_Cttacker=abs(C3*Cttacker_pos(j)-m*pop(i,j)); % Equation (6)
            X3=Cttacker_pos(j)-A3*D_Cttacker; % Equation (7)      
            
            A4=2*f*r41-f; % Equation (3)
            C4=2*r42; % Equation (4)
            
            D_Driver=abs(C4*Darrier_pos(j)-m*pop(i,j)); % Equation (6)
            X4=Darrier_pos(j)-A4*D_Driver; % Equation (7)       
            
            ChimpPositions(i,j)=(X1+X2+X3+X4)/4;% Equation (8)   
       
        end
    end 
    
    for i=1:Chimp_option.sizepop
 
        FU=pop(i,1)>Chimp_option.popcmax;
        FL=pop(i,1)<Chimp_option.popcmin;
        pop(i,1)=(pop(i,1).*(~(FU+FL)))+Chimp_option.popcmax.*FU+Chimp_option.popcmin.*FL;
        
        FU=pop(i,2)>Chimp_option.popgmax;
        FL=pop(i,2)<Chimp_option.popgmin;
        pop(i,2)=(pop(i,2).*(~(FU+FL)))+Chimp_option.popgmax.*FU+Chimp_option.popgmin.*FL;
        
        FU=pop(i,3:end)>1;
        FL=pop(i,3:end)<0;
        pop(i,3:end)=(pop(i,3:end).*(~(FU+FL)))+1.*FU+0.*FL;
               
         D_Leader = MutationU(dim,Chimp_option.maxgen,pop(i,3:end),l);
         pop(i,3:end)=CrossOverU(pop(i,3:end),D_Leader);
        ChimpPositions=logical(real(pop(i,3:end)));
        
        cmd=['-v ',num2str(Chimp_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
        ChimpFitness(i)=svmtrain(train_label, train(:,ChimpPositions), cmd); 
        [sorted_fitness,sorted_indexes]=sort(ChimpFitness);
        
         for newindex=1:Chimp_option.sizepop
            Sorted_Chimp(newindex,:)=pop(sorted_indexes(newindex),:);
         end
        
         pop=Sorted_Chimp(1:Chimp_option.sizepop,:);
        ChimpFitness = sorted_fitness(1:Chimp_option.sizepop);       
       
        
        if ChimpFitness(i)>TargetFitness
           TargetFitness=ChimpFitness(i);
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
bestM= M(Chimp_option.maxgen,:);
bestc = TargetPosition(1);
bestg = TargetPosition(2);
bestCVaccuarcy = fit_gen(Chimp_option.maxgen);
end