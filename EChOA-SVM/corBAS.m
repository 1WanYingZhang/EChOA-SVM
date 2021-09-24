function [Positions]=corBAS(Positions,fitness,threshold)
n=size(fitness);
for i=4:n(1)
    sum=0;
    greater=[];
    less=[];
    x=1;z=1;y=1;
    
    for j=1:size(Positions(1,:),2)
        
        d(x)=abs(Positions(1,j)-Positions(i,j));
        if d(x)<threshold
            greater(y)=j;
            y=y+1;
        else
            less(z)=j;
            z=z+1;
        end
        sum=sum+d(x)*d(x);
        x=x+1;
    end

    src=1-(double(6*sum))/(double(n(1)*(n(1)*n(1)-1)));

    if src<=0
        if size(greater)<size(less)
        else
            for j=1:size(greater)
                dim=greater(j);
                
                step=0.8;
                eta_step=0.95;            
             
                b=rands(dim,1);
                b=b/(eps+norm(b));
                step=eta_step*step+0.01;   
                d=step/2;
                
                xleft=Positions(i,dim)+d*b;
                fleft=fobj(xleft);
                xright=Positions(i,dim)-d*b;
                fright=fobj(xright);              
               
                Positions(i,dim)=Positions(i,dim)-step*b*sign(fleft-fright);
                
            end
        end
    end
end
end

