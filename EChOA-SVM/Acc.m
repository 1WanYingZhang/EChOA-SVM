function acc=Acc(x,cmd)
%we used this function to calculate the accuracy ���㾫��
global A trn aaa_inst aaa_label vald;%����ȫ�ֱ���
%  SzW=0.01;
  x=x>0.5;%�ж�x�Ƿ����0.5  ���1��0
 x=cat(2,x,zeros(size(x,1),1));%cat(2, A, B)�൱��[A, B] B=zeros(m,n)������m��nȫ����
 x=logical(x);%����������ת�����߼����ͣ�true ��false�����κη�������ݶ�ת����true����0��ת����false;

% if sum(x)==0
%     y=inf;%inf�����
%     return;
% end
% format compact;
% % ���ý�����ģ�Ϳ�����ѵ�������ϵķ���Ч��
% [ptrain,acctrain,a] = svmpredict(trainlabel,traindata,model);
% model = svmtrain(aaa_label(trn,1),aaa_inst(trn,x),cmd);
% [ptest,c,b] = svmpredict(aaa_label(vald,1),aaa_inst(vald,x),model);
 model = svmtrain(aaa_label(1:80,1),aaa_inst(1:80,x),cmd);

[ptest,c,b] = svmpredict(aaa_label(81:end,1),aaa_inst(81:end,x),model);

% c = knnclassify(A(vald,x),A(trn,x),A(trn,end));%��������ڽ��з���ķ���������A��B��C�� A���������� B��ѵ���������� C���������
% cp = classperf(A(vald,end),c);%classperf�ṩ��һ���ӿ������ٷ�������֤�ڼ�����ܡ�classperf��������ѡ�ظ���һ�����������ܶ���CP�������ۻ��������Ľ����
% y=(1-SzW)*(1-cp.CorrectRate)+SzW*sum(x)/(length(x)-1);
% acc = cp.CorrectRate;
acc=(c(1,:)/100);
