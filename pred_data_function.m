load('state');
disp(['Now testing... please wait.']);   
zpred_t = zeros(Num_trail,simtime_test_len);
for ti = 1:simtime_test_len	% don't want to subtract time in indices 
    if flag == 0 % No Control uints
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


%save test_state M x z y ;
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




