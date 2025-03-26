%% ��ջ���
clear 
clc
%% �������� 
tic % ��ʼ��ʱ
Dim = 3;                       % ����ά����Kp��Ki��Kd��������Ҳ������ά��
SwarmSize = 50;          % PSO������
popSize = 20;               % GA��Ⱥ����������Ϊż����
ObjFun = @PIDsub_CN;      % ���õ�PID��ֵ�ӳ�������������õ�simulinkģ�ͣ�
MaxIter = 50;              % PSO����������
MinFit = 0.00001;       % ��С��Ӧ��ֵ�����õ�����Ӧ��С�����ֵʱ�������� 
Vmax = 10;                 % ����ٶȱ߽�
Vmin = -10;                % ��С�ٶȱ߽�
Ub = [10 50 10];         % ���λ�ñ߽�
Lb = [0 0 0];                % ��Сλ�ñ߽�
nIterGA = 20;             % GA��������
% crossoverProb = 0.8;    %GA������ʣ���һ��Ҫ�õģ����þ���ȫ�����Ӷ����н��棬������һ����GA���ӽ��н��棩
mutationProb = 0.2;         %GA�������

%% ִ�з���PSO_GA
[zbest, fzbest, K_p, K_i, K_d] = PSO_GA(ObjFun, Dim, SwarmSize, MaxIter, MinFit, Vmax, Vmin, Ub, Lb, popSize, nIterGA, mutationProb);

toc % ������ʱ
disp(['Total optimization time: ', num2str(toc)]);   %��ִ�п�����ʾ����ʱ�䣬��λ��

%% ���������Ϊ.mat�ļ�
save('result.mat') 

%% PSO_GA����
function [zbest, fzbest, K_p, K_i, K_d] = PSO_GA(ObjFun, Dim, SwarmSize, MaxIter, MinFit, Vmax, Vmin, Ub, Lb, popSize, nIterGA,mutationProb)
    % There is a 'crossoverProb' before mutationProb
    %��ʼ������Ⱥ
    %PSO�����������ͨ������Ȩ�صݼ�PSO�Ż�PID�㷨��һ��
    ws=0.9;       % ��ʼȨ��
    we=0.4;       % ����Ȩ��
    c1 = 2;        % ����ѧϰ���� 1
    c2 = 2;        % ���ѧϰ���� 2
    Range = ones(SwarmSize,1)*(Ub-Lb);
    Swarm = rand(SwarmSize,Dim).*Range + ones(SwarmSize,1)*Lb;
    
    VStep = rand(SwarmSize,Dim).*(Vmax-Vmin) + Vmin; % Added dot operator for element-wise multiplication
    fSwarm = zeros(SwarmSize,1);
    for i=1:SwarmSize
        fSwarm(i,:) = feval(ObjFun,Swarm(i,:));
    end

    [bestf, bestindex]=min(fSwarm); %
    zbest=Swarm(bestindex,:);    
    gbest=Swarm;                 
    fgbest=fSwarm;               
    fzbest=bestf;                
    
    % PSO
    iter = 0;
    K_p = zeros(1, MaxIter);
    K_i = zeros(1, MaxIter);
    K_d = zeros(1, MaxIter);
    fzbest_history = zeros(1, MaxIter);
    
    while (iter < MaxIter && (fzbest > MinFit))
        % PSO��������
        w=ws-(ws-we)*iter/MaxIter; % Weight self-adjustment PSO algorithm
          for j=1:SwarmSize
            % �ٶȸ���
            VStep(j,:) = w*VStep(j,:) + c1*rand*(gbest(j,:) - Swarm(j,:)) + c2*rand*(zbest - Swarm(j,:));
            VStep(j,:) = min(max(VStep(j,:), Vmin), Vmax); % Ensure velocity is within bounds
            % λ�ø���
            Swarm(j,:) = Swarm(j,:) + VStep(j,:);
            Swarm(j,:) = min(max(Swarm(j,:), Lb), Ub); % Ensure position is within bounds
            % ������Ӧ��ֵ
            fSwarm(j,:) = feval(ObjFun,Swarm(j,:));
            % �������Ÿ���     
            if fSwarm(j) < fgbest(j)
                gbest(j,:) = Swarm(j,:);
                fgbest(j) = fSwarm(j);
            end
            % ȫ�����Ÿ���
            if fSwarm(j) < fzbest
                zbest = Swarm(j,:);
                fzbest = fSwarm(j);
            end
          end
        % ��¼ÿһ������PIDֵ��Ϊ��ͼ��׼��
        K_p(iter+1) = zbest(1);
        K_i(iter+1) = zbest(2);
        K_d(iter+1) = zbest(3);

        % �����PSO���ӳ�ʼ��GA��Ⱥ
        GAPopulation = repmat(zbest, popSize, 1);
        
        % �����Ž⴫�ݸ�GA
        for gaIter = 1:nIterGA
            % ����GA��Ⱥ����Ӧ��
            fitnessGA = zeros(popSize, 1);
            for i = 1:popSize
                fitnessGA(i) = feval(ObjFun, GAPopulation(i, :));
            end

            % ������Ӧ�ȶ�GA��Ⱥ���д�С��������ѡ�񣨷ֳ���ǰ��õ�һ��ͺ��滵��һ�룩
            [~, sortedIndices] = sort(fitnessGA);
            sortedGAPopulation = GAPopulation(sortedIndices, :);

            % ���е���ʵֵ�����Բ����µĺ����û���õ�������ʣ�ȫ�����н��棩
            crossoverPoint = round(Dim / 2); % ���е���н���
            offspring = zeros(popSize, Dim);
            for i = 1:2:popSize-1
                parent1 = sortedGAPopulation(i, :); %���������Ŵ��㷨��Ⱥ��ѡ���i��������Ϊ��һ������parent1
                parent2 = sortedGAPopulation(i+1, :); %���������Ŵ��㷨��Ⱥ��ѡ���i+1��������Ϊ�ڶ�������parent2
                % ���ȴ�parent1��ȡǰcrossoverPoint������Ȼ���parent2��ȡ��crossoverPoint+1�����Ļ��򣬽��������ֺϲ���һ���µĸ���
                offspring(i, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)]; 
                offspring(i+1, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)]; %�Ӵ�2ͬ��
            end

            % ���е���ʵֵ�����Բ����µĺ�����õ�������ʣ����ֽ��н��棩
            % crossoverPoint = round(Dim / 2); % ���е���н���
            % offspring = zeros(popSize, Dim);
            % for i = 1:2:popSize
            %     parent1 = sortedGAPopulation(i, :);
            %     parent2 = sortedGAPopulation(i+1, :);
            %     if rand < crossoverProb % rand��������ӽ�����ʼ��
            %         offspring(i, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)];
            %         offspring(i+1, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)];
            %     else
            %         offspring(i, :) = parent1; % ���û�з������棬����븸ĸ��ȫ��ͬ
            %         offspring(i+1, :) = parent2;
            %     end
            % end

            % ���½������ɵ�ǰһ���Ӵ�����֮ǰ��Ӧ�Ȳ��ǰһ��GA����
            % �����һ�����ʼ���� ��GAPopulation�ĺ�һ�뱻offspring��ǰһ��ȡ��
            startIndex = ceil(popSize / 2);            
            GAPopulation(startIndex:end, :) = offspring(1:floor(popSize/2), :);
            % GAPopulation(1:floor(popSize/2), :) = offspring(1:floor(popSize/2), :); ֮ǰ�����

            % �±��滻��GA���ӽ��еĵ���������
            mutationIndices = rand(ceil(popSize/2), Dim) < mutationProb;
            mutatedPopulation = GAPopulation(1:popSize/2, :) + (rand(popSize/2, Dim) - 0.5) .* mutationIndices;
            GAPopulation(1:popSize/2, :) = max(min(mutatedPopulation, Ub), Lb);

            % �������º��GA��Ⱥ����Ӧ��
            for i = 1:popSize
                fitnessGA(i) = feval(ObjFun, GAPopulation(i, :));
            end

            % GA��ȺѰ��
            [bestf, bestindex] = min(fitnessGA);
            zbest = GAPopulation(bestindex, :);
            fzbest = bestf;
        end
        fzbest_history(iter+1) = fzbest;
        iter = iter + 1;
    end
    % ��ͼ
   figure(1)
    plot( fzbest_history, 'g-', 'LineWidth', 3); % fzbest
    title('Global Best Fitness', 'fontsize', 18);
    xlabel('Iteration Number', 'fontsize', 18);
    ylabel('Fitness Value', 'fontsize', 18);
    set(gca, 'Fontsize', 18);
    hold off;
    figure(2)
    plot( K_p ,'k-', 'LineWidth', 3); % Kp, Ki, Kd
    hold on;
    plot( K_i ,'r-','LineWidth', 3);
    plot( K_d ,'b-','LineWidth', 3);
    title(' PID Parameters Optimization', 'fontsize', 18);
    xlabel('Iteration Number', 'fontsize', 18);
    ylabel('Parameter Value', 'fontsize', 18);
    legend({'K_p', 'K_i', 'K_d'}, 'fontsize', 18);
    set(gca, 'Fontsize', 18);
    hold off;
end