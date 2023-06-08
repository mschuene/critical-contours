addpath('../../../../../')
load('thresholds_dist_1000,30.mat')
figure_dist = [zeros(thds_figure(1),1);ones(thds_figure(2),1);ones(thds_figure(3),1)*2;ones(thds_figure(4),1)*3];
background_dist = [zeros(thds_background(1),1);ones(thds_background(2),1);ones(thds_background(3),1)*2;ones(thds_background(4),1)*3];
Figure()
histogram(figure_dist,'DisplayStyle','stairs')
hold on;
histogram(background_dist,'DisplayStyle','stairs')
legend('Target','Distractor')
xlabel('observed avs bigger than s_0')
title('coincidence distribution T=1000,s_0=30,accuracy=0.82,bc=4.56')
saveas(gcf,'coincidence_distrib','epsc')