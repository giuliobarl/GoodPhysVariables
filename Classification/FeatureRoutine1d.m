function difff=FeatureRoutine1d(x)

load GNIELnor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thresh = 500;        % [-]
Nbins = 70;          % sqrt(Nno);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

difff = zeros(1,3);

x = x/norm(x);

Nsample = max(size(TRAINDATAnor));

Nx = max(size(x));

indexSI = find(Fg >= thresh);
indexNO = find(Fg < thresh);

Nsi = max(size(indexSI));
Nno = max(size(indexNO));

FEATSI = zeros(Nsi,Nx);
FEATNO = zeros(Nno,Nx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:Nx
    FEATSI(:,i) = TRAINDATAnor(indexSI,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:Nx
    FEATNO(:,i) = TRAINDATAnor(indexNO,i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ff1=ones(Nsample,1);
f1si=ones(Nsi,1);
f1no=ones(Nno,1);


%%% first mixed feature construction

for i = 1:Nx

    ff1 = ff1.*TRAINDATAnor(:,i).^x(i);
    f1si=f1si.*FEATSI(:,i).^x(i);
    f1no=f1no.*FEATNO(:,i).^x(i);

end

b1min=min(ff1);
b1max=max(ff1);

fig=figure; 
set(fig,'visible','off');

f1sinor=(f1si-b1min)/(b1max-b1min);
f1nonor=(f1no-b1min)/(b1max-b1min);

hsi=histogram(f1sinor,0:1/Nbins:1,'Normalization','probability');

delta=hsi.BinWidth;
W1=hsi.Values;
f1=hsi.BinEdges;
F1=f1(1:end-1)+delta/2;

close(fig);

fig=figure; 
set(fig,'visible','off');
 
hno=histogram(f1nonor,0:1/Nbins:1,'Normalization','probability');

W2=hno.Values;
f2=hno.BinEdges;
F2=f2(1:end-1)+delta/2;

close(fig);


%%% Bhattacharyya distance

N1=max(size(W1));

measure=zeros(1,N1);

for i=1:N1

    measure(i)=sqrt(W1(i)*W2(i)); % Bhattacharyya distance

end

difff(3)=sum(sum(measure));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

musi=F1*W1';
muno=F2*W2';

stdsi=0;
stdno=0;

for i=1:Nbins

stdsi=stdsi+W1(i)*(F1(i)-musi)^2;
stdno=stdno+W2(i)*(F2(i)-muno)^2;

end 

difff(1)=sqrt(stdsi);
difff(2)=sqrt(stdno);