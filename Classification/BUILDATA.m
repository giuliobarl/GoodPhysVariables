clear all
close all
clc

DATA = table2array(readtable('Gniel_noise.xlsx'));

% train: 85%, test: 15%
cv = cvpartition(size(DATA,1),'HoldOut',0.15);
idx = cv.test;

% Separate to training and test data
TRAINDATA = DATA(~idx,:);
TESTDATA  = DATA(idx,:);

salva=1;

[Nsample,NF] = size(TRAINDATA);

Nimp = 5; % number of important features

BOUND = zeros(Nimp,2);

for i = 1:Nimp
    BOUND(i,1) = min(TRAINDATA(:,i));
    BOUND(i,2) = max(TRAINDATA(:,i));
end

TRAINDATAnor = zeros(Nsample,Nimp);
TESTDATAnor = zeros(max(size(TESTDATA)),Nimp);
Nu = zeros(1,Nsample);

%%%%%%%%%%FEATURE NORMALIZATION into [1-2]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

for i=1:Nimp
    TRAINDATAnor(:,i) = (TRAINDATA(:,i) - BOUND(i,1)) /...
        (BOUND(i,2) - BOUND(i,1)) + 1;
    TESTDATAnor(:,i) = (TESTDATA(:,i) - BOUND(i,1))/...
        (BOUND(i,2) - BOUND(i,1)) + 1;
end
%%%%%%%%%%FEATURE NORMALIZATION into [1-2]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

Nu=TRAINDATA(:,9);
Nu_test=TESTDATA(:,9);
thresh=0.0015;
%
if salva==1

    save GNIELnor.mat TRAINDATA TRAINDATAnor Nu

    save GNIELTESTnor.mat TESTDATAnor Nu_test

end

disp(100*sum(Nu>=thresh)/max(size(Nu)))

disp(100*sum(Nu<thresh)/max(size(Nu)))

