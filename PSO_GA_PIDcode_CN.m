%% 清空环境
clear 
clc
%% 参数设置 
tic % 开始计时
Dim = 3;                       % 粒子维数（Kp、Ki、Kd三个粒子也就是三维）
SwarmSize = 50;          % PSO粒子数
popSize = 20;               % GA种群个数（必须为偶数）
ObjFun = @PIDsub_CN;      % 调用的PID赋值子程序（里面包含调用的simulink模型）
MaxIter = 50;              % PSO最大迭代次数
MinFit = 0.00001;       % 最小适应度值，当得到的适应度小于这个值时结束迭代 
Vmax = 10;                 % 最大速度边界
Vmin = -10;                % 最小速度边界
Ub = [10 50 10];         % 最大位置边界
Lb = [0 0 0];                % 最小位置边界
nIterGA = 20;             % GA迭代次数
% crossoverProb = 0.8;    %GA交叉概率（不一定要用的，不用就是全部粒子都进行交叉，而不是一部分GA粒子进行交叉）
mutationProb = 0.2;         %GA变异概率

%% 执行方程PSO_GA
[zbest, fzbest, K_p, K_i, K_d] = PSO_GA(ObjFun, Dim, SwarmSize, MaxIter, MinFit, Vmax, Vmin, Ub, Lb, popSize, nIterGA, mutationProb);

toc % 结束计时
disp(['Total optimization time: ', num2str(toc)]);   %在执行框里显示仿真时间，单位秒

%% 保存仿真结果为.mat文件
save('result.mat') 

%% PSO_GA方程
function [zbest, fzbest, K_p, K_i, K_d] = PSO_GA(ObjFun, Dim, SwarmSize, MaxIter, MinFit, Vmax, Vmin, Ub, Lb, popSize, nIterGA,mutationProb)
    % There is a 'crossoverProb' before mutationProb
    %初始化粒子群
    %PSO迭代程序和普通的线性权重递减PSO优化PID算法的一样
    ws=0.9;       % 初始权重
    we=0.4;       % 结束权重
    c1 = 2;        % 个人学习因子 1
    c2 = 2;        % 社会学习因子 2
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
        % PSO迭代程序
        w=ws-(ws-we)*iter/MaxIter; % Weight self-adjustment PSO algorithm
          for j=1:SwarmSize
            % 速度更新
            VStep(j,:) = w*VStep(j,:) + c1*rand*(gbest(j,:) - Swarm(j,:)) + c2*rand*(zbest - Swarm(j,:));
            VStep(j,:) = min(max(VStep(j,:), Vmin), Vmax); % Ensure velocity is within bounds
            % 位置更新
            Swarm(j,:) = Swarm(j,:) + VStep(j,:);
            Swarm(j,:) = min(max(Swarm(j,:), Lb), Ub); % Ensure position is within bounds
            % 计算适应度值
            fSwarm(j,:) = feval(ObjFun,Swarm(j,:));
            % 个体最优更新     
            if fSwarm(j) < fgbest(j)
                gbest(j,:) = Swarm(j,:);
                fgbest(j) = fSwarm(j);
            end
            % 全体最优更新
            if fSwarm(j) < fzbest
                zbest = Swarm(j,:);
                fzbest = fSwarm(j);
            end
          end
        % 记录每一代最优PID值，为绘图做准备
        K_p(iter+1) = zbest(1);
        K_i(iter+1) = zbest(2);
        K_d(iter+1) = zbest(3);

        % 用最佳PSO粒子初始化GA种群
        GAPopulation = repmat(zbest, popSize, 1);
        
        % 将最优解传递给GA
        for gaIter = 1:nIterGA
            % 评估GA种群的适应度
            fitnessGA = zeros(popSize, 1);
            for i = 1:popSize
                fitnessGA(i) = feval(ObjFun, GAPopulation(i, :));
            end

            % 根据适应度对GA种群进行从小到大排序选择（分成了前面好的一半和后面坏的一半）
            [~, sortedIndices] = sort(fitnessGA);
            sortedGAPopulation = GAPopulation(sortedIndices, :);

            % 进行单点实值交叉以产生新的后代（没有用到交叉概率，全部进行交叉）
            crossoverPoint = round(Dim / 2); % 在中点进行交叉
            offspring = zeros(popSize, Dim);
            for i = 1:2:popSize-1
                parent1 = sortedGAPopulation(i, :); %在排序后的遗传算法种群中选择第i个个体作为第一个父代parent1
                parent2 = sortedGAPopulation(i+1, :); %在排序后的遗传算法种群中选择第i+1个个体作为第二个父代parent2
                % 首先从parent1中取前crossoverPoint个基因，然后从parent2中取从crossoverPoint+1到最后的基因，将这两部分合并成一个新的个体
                offspring(i, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)]; 
                offspring(i+1, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)]; %子代2同理
            end

            % 进行单点实值交叉以产生新的后代（用到交叉概率，部分进行交叉）
            % crossoverPoint = round(Dim / 2); % 在中点进行交叉
            % offspring = zeros(popSize, Dim);
            % for i = 1:2:popSize
            %     parent1 = sortedGAPopulation(i, :);
            %     parent2 = sortedGAPopulation(i+1, :);
            %     if rand < crossoverProb % rand随机数增加交叉概率检查
            %         offspring(i, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)];
            %         offspring(i+1, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)];
            %     else
            %         offspring(i, :) = parent1; % 如果没有发生交叉，后代与父母完全相同
            %         offspring(i+1, :) = parent2;
            %     end
            % end

            % 用新交叉生成的前一半子代代替之前适应度差的前一半GA粒子
            % 计算后一半的起始索引 将GAPopulation的后一半被offspring的前一半取代
            startIndex = ceil(popSize / 2);            
            GAPopulation(startIndex:end, :) = offspring(1:floor(popSize/2), :);
            % GAPopulation(1:floor(popSize/2), :) = offspring(1:floor(popSize/2), :); 之前错误的

            % 新被替换的GA粒子进行的单点变异操作
            mutationIndices = rand(ceil(popSize/2), Dim) < mutationProb;
            mutatedPopulation = GAPopulation(1:popSize/2, :) + (rand(popSize/2, Dim) - 0.5) .* mutationIndices;
            GAPopulation(1:popSize/2, :) = max(min(mutatedPopulation, Ub), Lb);

            % 评估更新后的GA种群的适应度
            for i = 1:popSize
                fitnessGA(i) = feval(ObjFun, GAPopulation(i, :));
            end

            % GA种群寻优
            [bestf, bestindex] = min(fitnessGA);
            zbest = GAPopulation(bestindex, :);
            fzbest = bestf;
        end
        fzbest_history(iter+1) = fzbest;
        iter = iter + 1;
    end
    % 绘图
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