function difff=FeatureRoutine1d3class(x)

load GNIELnor

%%%%%%%%%%%%%%%%%%%%%%%PARAMETERS%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thresh1=400;
thresh2=900;                                          
%

Nbins=70;   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display=0;

difff=zeros(1,3);

Nx=max(size(x));

x = x/norm(x);

Nsample=max(size(TRAINDATAnor));

set0=find(Nu<=thresh1);
nset0=find(Nu>thresh1);
set2=find(Nu>=thresh2);
set1=setdiff(nset0,set2);

N0=max(size(set0));
N1=max(size(set1));
N2=max(size(set2));

FEAT0=zeros(N0,Nx);
FEAT1=zeros(N1,Nx);
FEAT2=zeros(N2,Nx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEAT0(:,i)=TRAINDATAnor(set0,i);    % Classe 0-thresh1
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEAT1(:,i)=TRAINDATAnor(set1,i);    % Classe thresh1-thresh2
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEAT2(:,i)=TRAINDATAnor(set2,i);    % Classe thresh2 - Tmax
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% costruzione prima feature mista
ff1=ones(Nsample,1);

f10=ones(N0,1);
f11=ones(N1,1);
f12=ones(N2,1);


for i=1:Nx

    ff1=ff1.*TRAINDATAnor(:,i).^x(i);
    
    f10=f10.*FEAT0(:,i).^x(i);
    f11=f11.*FEAT1(:,i).^x(i);
    f12=f12.*FEAT2(:,i).^x(i);

end

b1min=min(ff1);
b1max=max(ff1);

fig=figure; 

if display==0
    set(fig,'visible','off');
end

f10nor=(f10-b1min)/(b1max-b1min);

f11nor=(f11-b1min)/(b1max-b1min);

f12nor=(f12-b1min)/(b1max-b1min);

h0=histogram(f10nor,0:1/Nbins:1,'Normalization','probability');

delta=h0.BinWidth;
W0=h0.Values;
f0=h0.BinEdges;
F0=f0(1:end-1)+delta/2;

if display==0
    close(fig);
else
    hold on
end

if display==0
    fig=figure; 
    set(fig,'visible','off');
else
    hold on
end

h1=histogram(f11nor,0:1/Nbins:1,'Normalization','probability');

W1=h1.Values;
f1=h1.BinEdges;
F1=f1(1:end-1)+delta/2;

if display==0
    close(fig);
end

%%%%
if display==0
    fig=figure; 
    set(fig,'visible','off');
else
    hold on
end

h2=histogram(f12nor,0:1/Nbins:1,'Normalization','probability');

W2=h2.Values;
f2=h2.BinEdges;
F2=f2(1:end-1)+delta/2;

if display==0
    close(fig);
else
    legend('Class 0', 'Class 1', 'Class 2')
end


N1=max(size(W0));

measure=zeros(1,N1);

for i=1:N1
    measure(i)=sqrt(W0(i)*W1(i)); % Bhattacharyya distance
end

difff(1)=log(sum(sum(measure)));

measure=zeros(1,N1);

for i=1:N1
    measure(i)=sqrt(W0(i)*W2(i)); % Bhattacharyya distance
end

difff(2)=log(sum(sum(measure)));

measure=zeros(1,N1);

for i=1:N1
    measure(i)=sqrt(W1(i)*W2(i)); % Bhattacharyya distance
end

difff(3)=log(sum(sum(measure)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%