clear all
clc
close all

load GNIELnor
load GNIELnor

Nbins=35;

Nsample=max(size(TESTDATAnor));

thresh=500;          % [k]

utopian=0;  % Do you want to select the Utopian point (1) or the least distance (0)?

load Gniel2features.mat

%%%%%%%%%%%% UTOPIA POINT FROM PARETO FRONT %%%%%%%%%%%%%%%%%%%
m1=min(ObjectiveValue(:,1));
mm1=max(ObjectiveValue(:,1));

m2=min(ObjectiveValue(:,2));
mm2=max(ObjectiveValue(:,2));

[m3,pic2]=min(ObjectiveValue(:,3));

if m3>0
    ObjectiveValue(:,3)=ObjectiveValue(:,3)-abs(m3);
else
    ObjectiveValue(:,3)=ObjectiveValue(:,3)+abs(m3);
end

m3=min(ObjectiveValue(:,3));
mm3=max(ObjectiveValue(:,3));

ObjectiveValue(:,1)=(ObjectiveValue(:,1)-m1)/(mm1-m1);
ObjectiveValue(:,2)=(ObjectiveValue(:,2)-m2)/(mm2-m2);
ObjectiveValue(:,3)=(ObjectiveValue(:,3)-m3)/(mm3-m3);

mx=10000;
pic=-1;

for i=1:max(size(ObjectiveValue))

    if norm([ObjectiveValue(i,1),ObjectiveValue(i,2),...
        ObjectiveValue(i,3)])<mx
    
    mx=norm([ObjectiveValue(i,1),ObjectiveValue(i,2),...
        ObjectiveValue(i,3)]);

    pic=i;
    end

end
%%%%%%%%%%%% UTOPIA POINT FROM PARETO FRONT %%%%%%%%%%%%%%%%%%%

if utopian==0 
    pic=pic2;
end

x=solution(pic,:);

figure(1)
plot3(ObjectiveValue(:,1),ObjectiveValue(:,2),...
    ObjectiveValue(:,3),'m*')
grid on
box on

hold on
plot3(ObjectiveValue(pic,1),ObjectiveValue(pic,2),...
    ObjectiveValue(pic,3),'sg','MarkerSize',13,'LineWidth',3)
legend('Pareto Front','Selection')

figure(2)

Nx=max(size(x));

x1=x(1:Nx/2);
x2=x(Nx/2+1:Nx);

x1=x1/norm(x1);
x2=x2/norm(x2);

x(1:Nx/2)=x1(:);
x(Nx/2+1:Nx)=x2(:);

indexSI=find(Nu_test>=thresh);
indexNO=find(Nu_test<thresh);

Nsi=max(size(indexSI));
Nno=max(size(indexNO));

FEATSI=zeros(Nsi,Nx/2);
FEATNO=zeros(Nno,Nx/2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx/2
    FEATSI(:,i)=TESTDATAnor(indexSI,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx/2
    FEATNO(:,i)=TESTDATAnor(indexNO,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% costruzione prima feature mista

ff1=ones(Nsample,1);
f1si=ones(Nsi,1);
f1no=ones(Nno,1);

for i=1:Nx/2

    ff1=ff1.*TESTDATAnor(:,i).^x(i);
    f1si=f1si.*FEATSI(:,i).^x(i);
    f1no=f1no.*FEATNO(:,i).^x(i);

end

%%% costruzione SECONDA feature mista


ff2=ones(Nsample,1);
f2si=ones(Nsi,1);
f2no=ones(Nno,1);


for i=1:Nx/2

    ff2=ff2.*TESTDATAnor(:,i).^x(i+Nx/2);
    f2si=f2si.*FEATSI(:,i).^x(i+Nx/2);
    f2no=f2no.*FEATNO(:,i).^x(i+Nx/2);

end

b1min=min(ff1);
b1max=max(ff1);

b2min=min(ff2);
b2max=max(ff2);

fig = figure(2)

f1sinor=(f1si-b1min)/(b1max-b1min);
f2sinor=(f2si-b2min)/(b2max-b2min);

f1nonor=(f1no-b1min)/(b1max-b1min);
f2nonor=(f2no-b2min)/(b2max-b2min);


hno=histogram2(f1nonor,f2nonor,0:1/Nbins:1,0:1/Nbins:1,...
    'Normalization','probability');

H2=hno.Values;

zz1=0:1/(Nbins-1):1;
zz2=0:1/(Nbins-1):1;


[XX2,YY2] = meshgrid(zz1(1:end),zz2(1:end));

hold on

hsi=histogram2(f1sinor,f2sinor,0:1/Nbins:1,0:1/Nbins:1,...
    'Normalization','probability');

H1=hsi.Values;

yy1=0:1/(Nbins-1):1;
yy2=0:1/(Nbins-1):1;

[XX1,YY1] = meshgrid(yy1(1:end),yy2(1:end));

title('Testing set','FontWeight','Normal')
xlabel('y_{1}')
ylabel('y_{2}')
zlabel('PDF')
legend('Class 1','Class 2')

saveas(fig,'2features.svg')

[F2,W2]=converhist(H2);
[F1,W1]=converhist(H1);

Nsi=max(size(FEATSI));
Nno=max(size(FEATNO));

