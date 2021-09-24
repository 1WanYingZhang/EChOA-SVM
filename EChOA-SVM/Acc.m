function acc=Acc(x,cmd)
%we used this function to calculate the accuracy 计算精度
global A trn aaa_inst aaa_label vald;%定义全局变量
%  SzW=0.01;
  x=x>0.5;%判断x是否大于0.5  输出1或0
 x=cat(2,x,zeros(size(x,1),1));%cat(2, A, B)相当于[A, B] B=zeros(m,n)：生成m×n全零阵
 x=logical(x);%将数据类型转换成逻辑类型（true 或false）！任何非零的数据都转换成true，而0被转换成false;

% if sum(x)==0
%     y=inf;%inf无穷大
%     return;
% end
% format compact;
% % 利用建立的模型看其在训练集合上的分类效果
% [ptrain,acctrain,a] = svmpredict(trainlabel,traindata,model);
% model = svmtrain(aaa_label(trn,1),aaa_inst(trn,x),cmd);
% [ptest,c,b] = svmpredict(aaa_label(vald,1),aaa_inst(vald,x),model);
 model = svmtrain(aaa_label(1:80,1),aaa_inst(1:80,x),cmd);

[ptest,c,b] = svmpredict(aaa_label(81:end,1),aaa_inst(81:end,x),model);

% c = knnclassify(A(vald,x),A(trn,x),A(trn,end));%利用最近邻进行分类的分类器；（A，B，C） A：测试数据 B：训练样本特征 C：样本标号
% cp = classperf(A(vald,end),c);%classperf提供了一个接口来跟踪分类器验证期间的性能。classperf创建并可选地更新一个分类器性能对象CP，它会累积分类器的结果。
% y=(1-SzW)*(1-cp.CorrectRate)+SzW*sum(x)/(length(x)-1);
% acc = cp.CorrectRate;
acc=(c(1,:)/100);
