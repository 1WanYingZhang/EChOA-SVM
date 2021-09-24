function y=AccSz(x,cmd)
%we used this function to calculate the fitness value as in the paper 计算适应度
global A label train_matrix train_label test_matrix test_label aaa_label trn data aaa_inst vald a;%定义全局变量
 SzW=0.01;
  x=x>0.5;
 x=cat(2,x,zeros(size(x,1),1));
 x=logical(x);

if sum(x)==0
    y=inf;
    return;
    
end


% train_matrix = data(1:i,:);
% train_label = label(1:i,:);
% 
% test_matrix = data((i+1:end),:);
% test_label = label((i+1:end),:);
%  model = svmtrain(train_label(),train_matrix,cmd);

% [ptrain,acctrain,a] = svmpredict(train_label,train_matrix,model);
% model = svmtrain(aaa_inst(trn,end),aaa_inst(trn,x),cmd);
% [ptest,c,b] = svmpredict(aaa_inst(vald,end),aaa_inst(vald,x),model)


 model = svmtrain(aaa_label(1:80,1),aaa_inst(1:80,x),cmd);

[ptest,c,b] = svmpredict(aaa_label(81:end,1),aaa_inst(81:end,x),model);

% [ptest,c,b] = svmpredict(test_label,test_matrix,model);

 c=(c(1,:)/100);
 
% c = knnclassify(A(vald,x),A(trn,x),A(trn,end));
% cp = classperf(A(vald,end),c);%classperf提供了一个接口来跟踪分类器验证期间的性能。classperf创建并可选地更新一个分类器性能对象CP，它会累积分类器的结果。
% s=length(test_label);
%  cp = classperf(test_label);
%  y=(1-SzW)*(1-cp.CorrectRate)+SzW*sum(x)/(length(x)-1);
% y=c;
y=(1-SzW)*(1-c)+SzW*sum(x)/(length(x)-1);

