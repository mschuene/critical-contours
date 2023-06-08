addpath('../..')
load('pop_act100,50,100.mat')
Figure()
subplot(2,1,1)
bc = 1;
Tend = min(length(pop_act_3_4_a)/bc,length(pop_act_3_4_b));
T = 1:Tend;
plot(T*bc,pop_act_3_4_a(T))
hold on; 
h = refline(0,30)
h.Color='r'
xlim([326000,366000])
xlabel('T')
ylabel('avalanche size')
title('Population activity figure activation')
subplot(2,1,2)
plot(T,pop_act_3_4_b(T))
h = refline(0,30)
h.Color='r'
xlim([326000,366000])
xlabel('T')
ylabel('avalanche size')
title('Population activity background activation')
suptitle('Population activity N_s=100,N_e=50,K=100')
saveas(gcf,'pop_act.eps','epsc')