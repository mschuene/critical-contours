addpath('../../..')

%% best accuracy
load('best_accuracy_k100.mat')
Figure()
imagesc(psm)
set(gca,'YDir','normal');
colorbar()
set(gca,'YTick',1:10)
set(gca,'YTickLabel',{'80','84','88','92','96','100','104','108','112','116','120'})
set(gca,'XTick',1:16)
set(gca,'XTickLabel',{'40','42','44','46','48','50','52','54','56','58','60','62','64','66','68','70'})
xlabel('N_s')
ylabel('N_e')
title('best accuracies for K=100 and T=1000')
%saveas(gca,'best_accuracies_k100_T1000','epsc')

%% best threshold
load('best_threshold_k100.mat')
Figure()
imagesc(psm)
set(gca,'YDir','normal');
colorbar()
set(gca,'YTick',1:10)
set(gca,'YTickLabel',{'80','84','88','92','96','100','104','108','112','116','120'})
set(gca,'XTick',1:16)
set(gca,'XTickLabel',{'40','42','44','46','48','50','52','54','56','58','60','62','64','66','68','70'})
xlabel('N_s')
ylabel('N_e')
title('best thresholds s_0 for K=100 and T=1000')
%saveas(gca,'best_tresholds_k100_T1000','epsc')
%load('best_threshold_k100.mat')


%% best accuracy
load('best_accuracy_k100_T10000.mat')
load('best_accuracy_more_params.mat')
figure()
Ne_range = [70:2:94];
empty_rest = 0.4*ones(6,length(Ne_range)-1);
whole_psm = [psm',[psm_more_params(2:end,:)';empty_rest]];
imagesc(whole_psm)
load('w_approx_small')
load('wb_small')
hold on;
Ne_s = [40:2:70];
Ns_s = [80:4:116];
%contour(1:16, 1:10, wb_small, [0 0], 'k');
% contour(Ne_s, Ns_s, wb_small, [-100 100], 'r');
%contour(Ne_s, Ns_s, w_approx', [0.5 0.5], 'w--');
set(gca,'YDir','normal');
colorbar()
% set(gca,'YTick',1:10)
% set(gca,'YTickLabel',{'80','84','88','92','96','100','104','108','112','116','120'})
% set(gca,'XTick',1:16)
% set(gca,'XTickLabel',{'40','42','44','46','48','50','52','54','56','58','60','62','64','66','68','70'})
xlabel('N_s')
ylabel('N_e')
title('best accuracies for K=100 and T=1000')
hold off;
%saveas(gca,'best_accuracies_k100_T10000','epsc')