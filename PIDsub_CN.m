function z=PIDsub_CN(x) %这个方程要与matlab.m文件名称一样
try
    assignin('base','Kp',x(1)); %将工作区的变量赋值到simulink的Kp，以下同理
    assignin('base','Ki',x(2));
    assignin('base','Kd',x(3));
    [t_time,x_state,y_out]=sim('SimulinkModel',[0,1]); %引号内为需要仿真的PIDsimulink模型
    z=y_out(end,1);
catch e
    disp('发生错误:');
    disp(e.message);
    z = NaN; % 返回一个NaN，表示仿真失败
end