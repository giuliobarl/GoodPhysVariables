clear all
close all
clc

Nimp=5; % number of features to be mixed

fun1d3class=@FeatureRoutine1d3class;

maxminu=2*60; % Max duration time in minutes

optionspa = optimoptions('gamultiobj','PopulationSize',120,...
          'ParetoFraction',0.7,'PlotFcn',@gaplotpareto,'MaxTime',maxminu*60, ...
          'FunctionTolerance',1e-8,'MaxGenerations',1000);

[solution,ObjectiveValue] = gamultiobj(fun1d3class,Nimp,[],[],[],[],[],[],optionspa);

save Gniel3class solution ObjectiveValue
