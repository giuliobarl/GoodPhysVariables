clear all
clc
close all

load GNIELnor
load GNIELTESTnor

Nsample = max(size(TRAINDATAnor));

thresh = 500;         % [-]

Nbins = 70;

utopian = 0;

load Gniel1feature.mat

%%%%%%%%%%%% UTOPIA POINT FROM PARETO FRONT %%%%%%%%%%%%%%%%%%%
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

for i = 1:max(size(ObjectiveValue))

    if norm([ObjectiveValue(i,1),ObjectiveValue(i,2),...
        ObjectiveValue(i,3)]) < mx
    
        mx = norm([ObjectiveValue(i,1),ObjectiveValue(i,2),...
        ObjectiveValue(i,3)]);

        pic = i;
    end

end
%%%%%%%%%%%% Least POINT FROM PARETO FRONT %%%%%%%%%%%%%%%%%%%

if utopian == 0 
    pic = pic2;
end

x=solution(pic,:)/norm(solution(pic,:));

Nx = max(size(x));

indexSI = find(Nu>=thresh);
indexNO = find(Nu<thresh);

Nsi = max(size(indexSI));
Nno = max(size(indexNO));

FEATSI = zeros(Nsi,Nx);
FEATNO = zeros(Nno,Nx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:Nx
    FEATSI(:,i) = TRAINDATAnor(indexSI,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEATNO(:,i) = TRAINDATAnor(indexNO,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%CONSTRUCTION FIRST FEATURE%%%%%%%%

ff1=ones(Nsample,1);
f1si=ones(Nsi,1);
f1no=ones(Nno,1);

%%% CONSTRUCTION FIRST FEATURE

for i=1:Nx

    ff1=ff1.*TRAINDATAnor(:,i).^x(i);
    f1si=f1si.*FEATSI(:,i).^x(i);
    f1no=f1no.*FEATNO(:,i).^x(i);

end
%%%%%%%%%%%%%%%CONSTRUCTION FIRST FEATURE%%%%%%%%

b1min = min(ff1);
b1max = max(ff1);

f = figure(1)

%%%%%
f1sinor = (f1si-b1min)/(b1max-b1min);

f1nonor = (f1no-b1min)/(b1max-b1min);

hsi = histogram(f1sinor,0:1/Nbins:1,'Normalization','probability', ...
    'EdgeColor', 'none');
%%%%%

H1=hsi.Values;

W1=hsi.Values;
f1=hsi.BinEdges;
F1=f1(1:end-1);

delta=hsi.BinWidth;

hold on
hno=histogram(f1nonor,0:1/Nbins:1,'Normalization','probability', ...
    'EdgeColor', 'none');

legend('Class 2','Class 1')
xlabel('y_{1}','fontsize',22)
ylabel('PDF','fontsize',22)
title('Training set','fontsize',22)

H2=hno.Values;
W2=hno.Values;
f2=hno.BinEdges;
F2=f2(1:end-1);

[N1,N2]=size(H1);

Nsi=max(size(FEATSI));
Nno=max(size(FEATNO));

%%%%%%%%%%%%%%%%%%%%%%5

FSI = FEATSI(:,1);
FNO = FEATNO(:,1);

figure(2)
hsi = histogram(FSI,1:1/Nbins:2,'Normalization','probability');
hold on
hno = histogram(FNO,1:1/Nbins:2,'Normalization','probability');

legend('Class 2','Class 1')
xlabel('u_{norm}')
ylabel('PDF')
title('Training set')


musi=F1*W1';
muno=F2*W2';

stdsi=0;
stdno=0;

for i=1:Nbins

    stdsi=stdsi+W1(i)*(F1(i)-musi)^2;
    stdno=stdno+W2(i)*(F2(i)-muno)^2;

end 

disp(sqrt(stdsi))
disp(sqrt(stdno))