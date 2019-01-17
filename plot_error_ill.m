clear;
%close all

figure('color','w');
%%
load('error_n_10.mat')
t = [50:50:600];
y = error_lose;
subplot 141;
plot(t,y,'--o','linewidth', 3)
xlabel('Number of Neurons', 'fontsize', 18, 'fontweight', 'bold');
ylabel('MSE Error', 'fontsize', 20, 'fontweight', 'bold');

%%
load('error_alpha_10.mat')
t = [0.5:0.5:6];
y = error_lose([1:12]);
subplot 142;
plot(t,y,'--o','linewidth', 3)
xlabel('\alpha', 'fontsize', 18, 'fontweight', 'bold');
ylabel('MSE Error', 'fontsize', 20, 'fontweight', 'bold');
%%
load('error_gate_10.mat')
t =[0.75:0.05:1.65,1.66:0.02:1.7];
y = error_lose([1:19,20:2:end]);
subplot 143;
plot(t,y,'--o','linewidth', 3)
xlabel('Gate Rate', 'fontsize', 18, 'fontweight', 'bold');
ylabel('MSE Error', 'fontsize', 20, 'fontweight', 'bold');
%%
load('error_kill_10.mat')
t = [0:2:25,50:50:500]/5;
y = error_lose([1:2:26,27:2:45]);
subplot 144;
plot(t,y,'--o','linewidth', 3)
xlabel('Percentage of Injured Neurons (%)', 'fontsize', 18, 'fontweight', 'bold');
ylabel('MSE Error', 'fontsize', 20, 'fontweight', 'bold');

