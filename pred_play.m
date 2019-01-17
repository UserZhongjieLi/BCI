clear
%close all
load('Trajectory.mat')
load('pred_data_owm.mat')
spine_label = [8 9  16 12 19];

left_arm_label  = [1 3 10 2 5 ];
right_arm_label = [4 18  25 30 26];

left_leg_label  = [19 27 37 32 34 ];
right_leg_label = [28  35 38 41 40];

key_label=[left_leg_label];

pred_data=zpred_t';
tra_max=repmat(tra_max,size(pred_data,1)/120,1);
tra_mean=repmat(tra_mean,size(pred_data,1)/120,1);
pred_data=pred_data/0.35.*tra_max+tra_mean;
pred_data=pred_data(1:10:end,:);
pred_data=log(pred_data)*1000;

for i=1:size(pred_data,2)
    each_y=pred_data(:,i);
    each_y=smooth(each_y,5);
    pred_data(:,i)=each_y;
end

my_marker_x=repmat(my_marker_x,size(pred_data,1)/120,1);
my_marker_y=repmat(my_marker_y,size(pred_data,1)/120,1);
my_marker_x(:,key_label)=pred_data(:,1:5);
my_marker_y(:,key_label)=pred_data(:,6:end);
my_marker_x=repmat(my_marker_x,2,1);
my_marker_y=repmat(my_marker_y,2,1);

figure('color','w');
linewidth = 2.5;
markersize =15;
 for i=1:12:120
    %clf reset
    %move_length = 0;
    move_length = -600+i*20;
	offset=50+zeros(1,length(spine_label));
    offset(3)=offset(3)+40;
	offset(4)=offset(4)+50;
	offset(length(spine_label))=offset(length(spine_label))-100;  
    
    plot(move_length + my_marker_x(i,spine_label)+offset,my_marker_y(i,spine_label),'.-','linewidth',linewidth,'color',[0.67  0   1],'markersize',markersize);
    hold on;
    plot(move_length + my_marker_x(i,right_arm_label),my_marker_y(i,right_arm_label),'.-','linewidth',linewidth,'color',[0.67  0   1],'markersize',markersize);
    hold on;
    plot(move_length + my_marker_x(i,left_arm_label),my_marker_y(i,left_arm_label),'.-','linewidth',linewidth,'color',[0.67  0   1],'markersize',markersize);
    hold on;
    plot(move_length + my_marker_x(i,right_leg_label),my_marker_y(i,right_leg_label),'.-','linewidth',linewidth,'color',[0.67  0   1],'markersize',markersize);
    hold on;
    plot(move_length + my_marker_x(i,left_leg_label),my_marker_y(i,left_leg_label),'.-','linewidth',linewidth,'color',[0.5  0.5  0.5],'markersize',markersize);
    
    hold off;
    axis([-850 2050 -5 1600]);
    axis off;
    pause(0.0005);
    picname=[pwd,'/image3/',num2str(i), '.jpg'];
    hold on 
    %saveas(gcf,picname)
    
 end
 