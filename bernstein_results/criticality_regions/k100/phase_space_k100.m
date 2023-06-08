addpath('../..')
load('phase_space_k100.mat')
load('extra_params.mat')
empty_rest = zeros(6,length(Ne_range)-1);
whole_psm = [psm,[extra_psm(2:end,:)';empty_rest]];
Figure()
imagesc(whole_psm)
set(gca,'YDir','normal');
colorbar()
set(gca,'YTick',1:10)
set(gca,'YTickLabel',{'80','84','88','92','96','100','104','108','112','116','120'})
set(gca,'XTick',1:28)
set(gca,'XTickLabel',{'40','42','44','46','48','50','52','54','56','58','60','62','64','66','68','70','72',...
'74','76','78','80','82','84','86','88','90','92','94'})
xlabel('N_e')
ylabel('N_s')
title('recorded avalanches K=100')
saveas(gcf,'phase_space_k100_more_params','epsc')

%%
load('exp_ovl.mat')
Figure()
exp_ovl = exp_ovl'
imagesc(exp_ovl)
ovl_thr_pos = zeros(1,size(exp_ovl,1))
ovl_thr = 5
for row = 1:size(exp_ovl,1)
   pos = find(exp_ovl(row,:) > ovl_thr,1,'first') 
   if(pos)
       ovl_thr_pos(row) = pos
   else
       ovl_thr_pos(row) = Inf
   end
end
hold on; 
plot(ovl_thr_pos,1:size(exp_ovl,1))
set(gca,'YDir','normal');
colorbar()
set(gca,'YTick',1:10)
set(gca,'YTickLabel',{'80','84','88','92','96','100','104','108','112','116','120'})
set(gca,'XTick',1:16)
set(gca,'XTickLabel',{'40','42','44','46','48','50','52','54','56','58','60','62','64','66','68','70'})
xlabel('N_s')
ylabel('N_e')
title('recorded avalanches K=100')
%saveas(gcf,'phase_space_k100','epsc')