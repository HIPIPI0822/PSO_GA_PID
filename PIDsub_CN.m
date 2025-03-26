function z=PIDsub_CN(x) %�������Ҫ��matlab.m�ļ�����һ��
try
    assignin('base','Kp',x(1)); %���������ı�����ֵ��simulink��Kp������ͬ��
    assignin('base','Ki',x(2));
    assignin('base','Kd',x(3));
    [t_time,x_state,y_out]=sim('SimulinkModel',[0,1]); %������Ϊ��Ҫ�����PIDsimulinkģ��
    z=y_out(end,1);
catch e
    disp('��������:');
    disp(e.message);
    z = NaN; % ����һ��NaN����ʾ����ʧ��
end