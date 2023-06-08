addpath('../../..')
%% 100,42,100
load('accuracies_thresholds100,42,100.mat');
load('accuracies_thresholds100,42,100_T1000.mat');
Figure()
accuracies_thresholds = [accuracies_thresholds;accuracies_thresholds_biggerT]
imagesc(accuracies_thresholds)
set(gca,'YDir','normal');
colorbar()
xlabel('s_0')
ylabel('T')
set(gca,'YTick',[1,2,3,6,11,13,16,19,22,26,30,33])
set(gca,'YTickLabel',{'1', '2', '4', '10', '20', '30', '55', '103', '192', '438', '1000','10000'})
set(gca,'XTick',[1,2,3,6,11,13,16,19,22,26,30,33])
set(gca,'XTickLabel',{'1', '2', '4', '10', '20', '30', '55', '103', '192', '438', '1000'})
title('coincidence detector accuracy Ns=100,Ne=42,K=100')
[M,IM] = max(accuracies_thresholds,[],2);
hold on; 
plot(IM,1:33)
saveas(gcf,'avs_plot100,42,100','epsc')

%% 100,50,100
load('accuracies_thresholds100,50,100.mat');
load('accuracies_thresholds100,50,100_T1000.mat');
Figure()
accuracies_thresholds = [accuracies_thresholds;accuracies_thresholds_biggerT]
imagesc(accuracies_thresholds)
set(gca,'YDir','normal');
colorbar()
xlabel('s_0')
ylabel('T')
set(gca,'YTick',[1,2,3,6,11,13,16,19,22,26,30,33])
set(gca,'YTickLabel',{'1', '2', '4', '10', '20', '30', '55', '103', '192', '438', '1000','10000'})
set(gca,'XTick',[1,2,3,6,11,13,16,19,22,26,30,33])
set(gca,'XTickLabel',{'1', '2', '4', '10', '20', '30', '55', '103', '192', '438', '1000'})
title('coincidence detector accuracy Ns=100,Ne=50,K=100')
[M,IM] = max(accuracies_thresholds,[],2);
hold on; 
plot(IM,1:33)
%%
saveas(gcf,'avs_plot100,50,100','epsc')


%% 100,54,100
load('accuracies_thresholds100,54,100.mat');
load('accuracies_thresholds100,54,100_T1000.mat');
Figure()
accuracies_thresholds = [accuracies_thresholds;accuracies_thresholds_biggerT]
imagesc(accuracies_thresholds)
set(gca,'YDir','normal');
colorbar()
xlabel('s_0')
ylabel('T')
set(gca,'YTick',[1,2,3,6,11,13,16,19,22,26,30,33])
set(gca,'YTickLabel',{'1', '2', '4', '10', '20', '30', '55', '103', '192', '438', '1000','10000'})
set(gca,'XTick',[1,2,3,6,11,13,16,19,22,26,30,33])
set(gca,'XTickLabel',{'1', '2', '4', '10', '20', '30', '55', '103', '192', '438', '1000'})
title('coincidence detector accuracy Ns=100,Ne=54,K=100')
[M,IM] = max(accuracies_thresholds,[],2);
hold on; 
plot(IM,1:33)
saveas(gcf,'avs_plot100,54,100','epsc')
