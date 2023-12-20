clear all
close all
clc
set(0,'defaultlinelinewidth',2);
set(0,'defaulttextfontsize',20);

Nimp=5; %number of features to be mixed

fun1=@FeatureRoutine1d; %construct 1 mixed feature
fun2=@FeatureRoutine2d; %construct 2 mixed feature

lb = 0;
ub = 1;

maxminu=2*60; % Max duration time [minutes]

optionspa = optimoptions('gamultiobj','PopulationSize',120,...
                         'ParetoFraction',0.7,'PlotFcn',...
                         @gaplotpareto,'MaxTime',maxminu*60);

[solution,ObjectiveValue] = gamultiobj(fun1,Nimp,...
                         [],[],[],[],[],[],optionspa);

save Gniel1feature solution ObjectiveValue