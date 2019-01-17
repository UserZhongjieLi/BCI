clear;
close all
linewidth = 3;
fontsize = 16;
fontweight = 'bold';


p = 0.3;
g = 1.5;				% g greater than 1 leads to chaotic networks.
%% SECTION TITLE
% DESCRIPTIVE TEXT
alpha = 5;
train_nsecs =120*20;
test_nsecs =120*3;
dt = 0.1;

nRec2Out = 500;
nRec2Control = 100;
N = nRec2Out+nRec2Control;

% rand('seed',0);
% Mc = sprandn(nRec2Control,nRec2Control,p);
% M = sprandn(N,N,p)*g/sqrt(p*N);
% M = full(M);
% M_add=find(M~=0);
% randn('seed',0);
% M(M_add)=randn(1,length(M_add))*g/sqrt(p*N);
for loop_rate=0
    loop_rate
rand('seed',0);
Mo = sprandn(nRec2Out,nRec2Out,p);
Mo = full(Mo);
M_add=find(Mo~=0);
randn('seed',0);
Mo(M_add)=randn(1,length(M_add))*g/sqrt(p*nRec2Out);

rand('seed',0);
Mc = sprandn(nRec2Control,nRec2Control,p);
Mc = full(Mc);
M_add=find(Mc~=0);
randn('seed',0);
Mc(M_add)=randn(1,length(M_add))*g/sqrt(p*nRec2Control);

M =blkdiag(Mo,Mc);



% disp(['   nRec2Out: ', num2str(nRec2Out)]);
% disp(['   nRec2Control: ', num2str(nRec2Control)]);

simtime_train = 0:dt:train_nsecs-dt;
simtime_train_len = length(simtime_train);

simtime_test=0:dt:test_nsecs-dt;
simtime_test_len = length(simtime_test);


load('Trajectory');
mydata=repmat(Trajectory_data,(train_nsecs+test_nsecs)/dt/size(Trajectory_data,1),1);
Num_trail=size(Trajectory_data,2);
ft_train = mydata(1:simtime_train_len ,1:Num_trail)' ;%3Xsimtime_len_train
ft_test = mydata(1+simtime_train_len :simtime_test_len+simtime_train_len,1:Num_trail)';%3Xsimtime_len_test


wo = zeros(nRec2Out,Num_trail);
rand('seed',0)
wo_f= 2*(rand(N,Num_trail)-0.5);
wo_len = zeros(Num_trail,simtime_train_len);


%wc = beta*randn(nRec2Control, 1)/sqrt(nRec2Control);		
wc = zeros(nRec2Control, 1); % synaptic strengths from internal pool to control unit

%rand('seed',loop_rate)
wc_f = [rand(nRec2Out,1);zeros(nRec2Control,1)];% the feedback now comes from the control unit as opposed to the out
wc_len = zeros(1,simtime_train_len);


rand('seed',0)
x = 0.4*rand(N,1);
y = zeros(1,1);
z = zeros(Num_trail,1);
r = tanh(x);
zt = zeros(Num_trail,simtime_train_len);

% Deliberatley set the pre-synaptic neurons to nonoverlapping between the output and control units.
out_idxs = 1:nRec2Out;			
con_idxs = nRec2Out+1:N;			 
rate_o = 0.3;
%figure;
flag = 0;
P = eye(nRec2Out);
Pc = eye(nRec2Control);
for ti = 1:length(simtime_train)
%      if mod(ti, round(length(simtime_train)/2)) == 0
%        % disp(['time: ' num2str(t,3) '.']);
%         subplot 211;
%         plot(simtime_train, ft_train(6,:), 'linewidth', linewidth, 'color', 'green');
%         hold on;
%         plot(simtime_train, zt(6,:), 'linewidth', linewidth, 'color', 'red');
%         title('training', 'fontsize', fontsize, 'fontweight', fontweight);
%         legend('f', 'z');	
%         xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
%         ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
%         hold off;
% 
%         subplot 212;
%         plot(simtime_train, wo_len(6,:), 'linewidth', linewidth);
%         hold on;
%         plot(simtime_train, wc_len, 'linewidth', linewidth,'color','r');
%         xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
%         ylabel('|w|,|wc|', 'fontsize', fontsize, 'fontweight', fontweight);
%         legend('|w|','|wc|');
%          hold off;
%         pause(0.01);	
%      end
    
    if flag == 0 
        x = (1.0-dt)*x + M*(r*dt) + (wo_f*(z*dt)+ wc_f*(y*dt)).*[ones(nRec2Out,1);zeros(nRec2Control,1)];
    else
        x = (1.0-dt)*x + M*(r*dt) + wo_f*(z*dt) + wc_f*(y*dt);
    end

    r = tanh(x);
    rx = r(out_idxs);			% the neurons that project to the output
    ry = r(con_idxs);			% the neurons that project to the control unit
    z = wo'*rx;
    y = wc'*ry*flag;
% update inverse correlation matrix
    k = P*rx;
    %rPrx = rx'*k;
    c = 1.0/(alpha + rx'*k);
    P = P - k*(k'*c);
    % update the error for the linear readout
     
    e =z - ft_train(:,ti);
    % update the output weights
    dwo = -k*e';
    wo = wo + dwo*rate_o;       
%%update inverse correlation matrix for the control unit
%NOTE WE USE THE OUTPUT'S ERROR %%% update the output weights
if ti> (1200*10)
    M(1:nRec2Out,1:100)=0;
    %break
    rate_o = 0.003;
    flag = 0;
    rate_c = 0.017;
   
    kc = Pc*ry;
    %rPry = ry'*kc;
    cy = 1.0/(1.0 + ry'*kc);
    Pc = Pc - kc*(kc'*cy);  
    dwc = -kc*e';
    wc = wc + mean(dwc,2)*rate_c*flag;       
    
%     %update the internal weight matrix using the output's error
%     dM_t=[dwo;dwc]';
%     M = M + repmat( sum(dM_t,1), N, 1);

end

    % Store the output of the system.
    zt(:,ti) = z;
    wo_len(:,ti) = sqrt(sum(wo.^2,1))';	
	wc_len(ti) = sqrt(wc'*wc);	
    
end

% error_avg_train = (mean( ((zt-ft_train)).^2,2)).^0.5; 
% error_all_train = mean(error_avg_train)
 
% save state M x z y;
% 
% %%
% %%%%test
% pred_data_function;

 
zpred_t = zeros(Num_trail,simtime_test_len);
for ti = 1:simtime_test_len	% don't want to subtract time in indices 

    if flag == 0 
        x = (1.0-dt)*x + M*(r*dt) + (wo_f*(z*dt)+ wc_f*(y*dt)).*[ones(nRec2Out,1);zeros(nRec2Control,1)];
    else
        x = (1.0-dt)*x + M*(r*dt) + wo_f*(z*dt) + wc_f*(y*dt);
    end
    r = tanh(x);
    rx = r(out_idxs);			% the neurons that project to the output
    ry = r(con_idxs);			% the neurons that project to the control unit
    z = wo'*rx;
    y = wc'*ry;
    
    zpred_t(:,ti) = z;          
end
error_avg_test = (mean( ((zpred_t-ft_test)).^2,2)).^0.5;
error_all_test = mean(error_avg_test)
disp(['End testing... please wait.']);  
end
save pred_data_owm_ill zpred_t;

%%
figure;
subplot 311;
hold on;
plot(simtime_test, ft_test(6,:), 'linewidth', linewidth, 'color', 'k'); 
axis tight;
plot(simtime_test, zpred_t(6,:), 'linewidth', linewidth, 'color', 'r');
axis tight;
title('Trajectory 1', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f1 and z1', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f1', 'z1');

subplot 312;
hold on;
plot(simtime_test, ft_test(7,:), 'linewidth', linewidth, 'color', 'k'); 
axis tight;
plot(simtime_test, zpred_t(7,:), 'linewidth', linewidth, 'color', 'r');
axis tight;
title('Trajectory 2', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f2 and z2', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f2', 'z2');
	
subplot 313;
hold on;
plot(simtime_test, ft_test(8,:), 'linewidth', linewidth, 'color', 'k'); 
axis tight;
plot(simtime_test, zpred_t(8,:), 'linewidth', linewidth, 'color', 'r');
axis tight;
title('Trajectory 3', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f3 and z3', 'fontsize', fontsize, 'fontweight', fontweight);
legend('f3', 'z3');
