clear all
clc
tic

[aaa_label,aaa_inst]= libsvmread('Zoo.txt');
data = aaa_inst;
label = aaa_label;
A=full(aaa_inst);
train_matrix = data(:,:);
train_label = label(:,1);
[M,fit_gen,score,bestacc,bestc,bestg] = EChimp_FS_SVMcg(train_label,train_matrix,size(A,2));

M=M>0.5;
M=logical(M);
sum(M)
c=bestacc;
acc=(c(1,:)/100);%换算成百分数
SzW=0.01;
Best_score=(1-SzW)*(1-acc)+SzW*sum(M)/length(A(1,:));

time=toc;
% plot(fit_gen);
plot(score);