addpath('../..')

%% 100,44
load('avalanche_plot100,44,100.mat')
Figure();
loglog(avs_figure(:,1),avs_figure(:,2),'DisplayName','figure avs distribution')
hold on;
loglog(avs_background(:,1),avs_background(:,2),'DisplayName','background avs distribution')
loglog(avs_ref(:,1),avs_ref(:,2),'DisplayName','reference avs distribution')
legend('show')
xlim([0,100])
title('2AFC avalanche distributions Ns=100,Ne=44,K=100')
saveas(gcf,'avs_plot100,44,100','epsc')
%% 100,54
load('avalanche_plot100,54,100.mat')
Figure();
loglog(avs_figure(:,1),avs_figure(:,2),'DisplayName','figure avs distribution')
hold on;
loglog(avs_background(:,1),avs_background(:,2),'DisplayName','background avs distribution')
loglog(avs_ref(:,1),avs_ref(:,2),'DisplayName','reference avs distribution')
legend('show')
xlim([0,100])
title('2AFC avalanche distributions Ns=100,Ne=54,K=100')
saveas(gcf,'avs_plot100,54,100','epsc')
%% 100,58
load('avalanche_plot100,58,100.mat')
Figure();
loglog(avs_figure(:,1),avs_figure(:,2),'DisplayName','figure avs distribution')
hold on;
loglog(avs_background(:,1),avs_background(:,2),'DisplayName','background avs distribution')
loglog(avs_ref(:,1),avs_ref(:,2),'DisplayName','reference avs distribution')
legend('show')
xlim([0,100])
title('2AFC avalanche distributions Ns=100,Ne=58,K=100')
saveas(gcf,'avs_plot100,58,100','epsc')
