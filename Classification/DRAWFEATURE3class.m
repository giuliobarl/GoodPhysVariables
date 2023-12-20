clear all
clc
close all

load GNIELnor
load GNIELTESTnor
load Gniel3class

Nsample = max(size(TESTDATAnor));

thresh1 = 400;         % [-]
thresh2 = 900;         % [-]

Nbins = 70;

flag = 0;
if ObjectiveValue(any(isinf(ObjectiveValue(:,2)),2),:)
    ObjectiveValue(any(isfinite(ObjectiveValue(:,2)),2),:) = [];
    flag = 1;
end

m1 = min(ObjectiveValue(:,1));
mm1 = max(ObjectiveValue(:,1));

m2 = min(ObjectiveValue(:,2));
mm2 = max(ObjectiveValue(:,2));

[m3,pic2] = min(ObjectiveValue(:,3));

if m3>0
ObjectiveValue(:,3) = ObjectiveValue(:,3) - abs(m3);
else
ObjectiveValue(:,3) = ObjectiveValue(:,3) + abs(m3);
end

m3 = min(ObjectiveValue(:,3));
mm3 = max(ObjectiveValue(:,3));

ObjectiveValue(:,1) = (ObjectiveValue(:,1)-m1)/(mm1-m1);
ObjectiveValue(:,2) = (ObjectiveValue(:,2)-m2)/(mm2-m2);
ObjectiveValue(:,3) = (ObjectiveValue(:,3)-m3)/(mm3-m3);

mx = 10000;
pic = -1;

if flag == 1
    for i = 1:max(size(ObjectiveValue))
    
        if norm([ObjectiveValue(i,1),ObjectiveValue(i,3)])
        
            mx = norm([ObjectiveValue(i,1),ObjectiveValue(i,3)]);
    
            pic = i;
        end
    
    end
else
    for i = 1:max(size(ObjectiveValue))
    
        if norm([ObjectiveValue(i,1),ObjectiveValue(i,2),...
            ObjectiveValue(i,3)])
        
            mx = norm([ObjectiveValue(i,1),ObjectiveValue(i,2),...
            ObjectiveValue(i,3)]);
    
            pic = i;
        end
    
    end
end

x=solution(pic,:)/norm(solution(pic,:));

Nx = max(size(x));

Nsample=max(size(TESTDATAnor));

set0=find(Nu_test<=thresh1);
nset0=find(Nu_test>thresh1);
set2=find(Nu_test>=thresh2);
set1=setdiff(nset0,set2);

N0=max(size(set0));
N1=max(size(set1));
N2=max(size(set2));

FEAT0=zeros(N0,Nx);
FEAT1=zeros(N1,Nx);
FEAT2=zeros(N2,Nx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEAT0(:,i)=TESTDATAnor(set0,i);    % Class 0-thresh1
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEAT1(:,i)=TESTDATAnor(set1,i);    % Class thresh1-thresh2
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEAT2(:,i)=TESTDATAnor(set2,i);    % Class thresh2 - Tmax
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% construction of the first feature
ff1=ones(Nsample,1);

f10=ones(N0,1);
f11=ones(N1,1);
f12=ones(N2,1);


for i=1:Nx

    ff1=ff1.*TESTDATAnor(:,i).^x(i);
    
    f10=f10.*FEAT0(:,i).^x(i);
    f11=f11.*FEAT1(:,i).^x(i);
    f12=f12.*FEAT2(:,i).^x(i);

end

b1min=min(ff1);
b1max=max(ff1);

fig=figure;

f10nor=(f10-b1min)/(b1max-b1min);

f11nor=(f11-b1min)/(b1max-b1min);

f12nor=(f12-b1min)/(b1max-b1min);

h0=histogram(f10nor,0:1/Nbins:1,'Normalization','probability');

delta=h0.BinWidth;
W0=h0.Values;
f0=h0.BinEdges;
F0=f0(1:end-1)+delta/2;

hold on

h1=histogram(f11nor,0:1/Nbins:1,'Normalization','probability');

W1=h1.Values;
f1=h1.BinEdges;
F1=f1(1:end-1)+delta/2;

h2=histogram(f12nor,0:1/Nbins:1,'Normalization','probability');

W2=h2.Values;
f2=h2.BinEdges;
F2=f2(1:end-1)+delta/2;

legend('Class 0', 'Class 1', 'Class 2')

