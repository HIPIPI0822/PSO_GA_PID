%% Clear workspace
clear 
clc
%% Parameter setting 
tic % Start timing
Dim = 3;                % Dimension 
SwarmSize = 2;          % PSO Swarm size
popSize = 2; % GA Population size (has to be a Even number)
ObjFun = @PSO_PIDdiscrete1;      % Objective function
MaxIter = 2;            % Maximum iterations  
MinFit = 0.00001;       % Minimum fitness 
Vmax = 3;       % Maximum velocity
Vmin = -3;
Ub = [10 50 10]; % Upper bound
Lb = [0 0 0];    % Lower bound
nIterGA = 2; % GA
% crossoverProb = 0.8;
mutationProb = 0.2;

%% Call  function
[zbest, fzbest, K_p, K_i, K_d] = PSO_GA(ObjFun, Dim, SwarmSize, MaxIter, MinFit, Vmax, Vmin, Ub, Lb, popSize, nIterGA, mutationProb);

toc % Stop timing
disp(['Total optimization time: ', num2str(toc)]);

%% Save function results
save('result') % Save as result.mat file, you can view the PID parameter values of each iteratio

%% PSO_GA function
function [zbest, fzbest, K_p, K_i, K_d] = PSO_GA(ObjFun, Dim, SwarmSize, MaxIter, MinFit, Vmax, Vmin, Ub, Lb, popSize, nIterGA,mutationProb)
    % There is a 'crossoverProb' before mutationProb
    % Initialize swarm
    % (PSO initialization code remains the same)
    ws=0.9;       % Initial weight
    we=0.4;       % Final weight
    c1 = 2;       % Learning factor 1
    c2 = 2;       % Learning factor 2
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
    
    % PSO main loop
    iter = 0;
    K_p = zeros(1, MaxIter);
    K_i = zeros(1, MaxIter);
    K_d = zeros(1, MaxIter);
    fzbest_history = zeros(1, MaxIter);
    
    while (iter < MaxIter && (fzbest > MinFit))
        % PSO iteration code
        w=ws-(ws-we)*iter/MaxIter; % Weight self-adjustment PSO algorithm
          for j=1:SwarmSize
            % Velocity update
            VStep(j,:) = w*VStep(j,:) + c1*rand*(gbest(j,:) - Swarm(j,:)) + c2*rand*(zbest - Swarm(j,:));
            VStep(j,:) = min(max(VStep(j,:), Vmin), Vmax); % Ensure velocity is within bounds
            % Position update
            Swarm(j,:) = Swarm(j,:) + VStep(j,:);
            Swarm(j,:) = min(max(Swarm(j,:), Lb), Ub); % Ensure position is within bounds
            % Fitness
            fSwarm(j,:) = feval(ObjFun,Swarm(j,:));
            % Personal best update     
            if fSwarm(j) < fgbest(j)
                gbest(j,:) = Swarm(j,:);
                fgbest(j) = fSwarm(j);
            end
            % Global best update
            if fSwarm(j) < fzbest
                zbest = Swarm(j,:);
                fzbest = fSwarm(j);
            end
          end
        % Record the best PID parameters at each iteration
        K_p(iter+1) = zbest(1);
        K_i(iter+1) = zbest(2);
        K_d(iter+1) = zbest(3);

        % Initialize GA population with the best PSO solution
        GAPopulation = repmat(zbest, popSize, 1);
        
        % Pass the best solution to GA
        for gaIter = 1:nIterGA
            % Evaluate the fitness of the GA population
            fitnessGA = zeros(popSize, 1);
            for i = 1:popSize
                fitnessGA(i) = feval(ObjFun, GAPopulation(i, :));
            end

            % Sort the GA population based on fitness
            [~, sortedIndices] = sort(fitnessGA);
            sortedGAPopulation = GAPopulation(sortedIndices, :);

            % Perform crossover to generate new offspring
            crossoverPoint = round(Dim / 2); % Crossover at the midpoint
            offspring = zeros(popSize, Dim);
            for i = 1:2:popSize-1
                parent1 = sortedGAPopulation(i, :);
                parent2 = sortedGAPopulation(i+1, :);
                offspring(i, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)];
                offspring(i+1, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)];
            end

            % Perform crossover to generate new offspring
            % crossoverPoint = round(Dim / 2); % Crossover at the midpoint
            % offspring = zeros(popSize, Dim);
            % for i = 1:2:popSize
            %     parent1 = sortedGAPopulation(i, :);
            %     parent2 = sortedGAPopulation(i+1, :);
            %     if rand < crossoverProb % Add crossover probability check
            %         offspring(i, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)];
            %         offspring(i+1, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)];
            %     else
            %         offspring(i, :) = parent1; % If crossover doesn't occur, offspring is identical to parents
            %         offspring(i+1, :) = parent2;
            %     end
            % end

            % Replace the lower half of the population with the new offspring
            startIndex = ceil(popSize / 2);            
            GAPopulation(startIndex:end, :) = offspring(1:floor(popSize/2), :);

            % Perform mutation on the lower half of the population
            mutationIndices = rand(ceil(popSize/2), Dim) < mutationProb;
            mutatedPopulation = GAPopulation(1:popSize/2, :) + (rand(popSize/2, Dim) - 0.5) .* mutationIndices;
            GAPopulation(1:popSize/2, :) = max(min(mutatedPopulation, Ub), Lb);

            % Evaluate the fitness of the updated GA population
            for i = 1:popSize
                fitnessGA(i) = feval(ObjFun, GAPopulation(i, :));
            end

            % Find the best solution from GA
            [bestf, bestindex] = min(fitnessGA);
            zbest = GAPopulation(bestindex, :);
            fzbest = bestf;
        end
        fzbest_history(iter+1) = fzbest;
        iter = iter + 1;
    end
    % Plot
    figure(1)
    plot(1:MaxIter , fzbest_history, 'g-', 'LineWidth', 3); % fzbest
    title('Global Best Fitness', 'fontsize', 18);
    xlabel('Iteration Number', 'fontsize', 18);
    ylabel('Fitness Value', 'fontsize', 18);
    set(gca, 'Fontsize', 18);
    hold off;
    
    figure(2)
    plot(1:iter, K_p(1:iter), 'k-', 1:iter, K_i(1:iter), 'r-', 1:iter, K_d(1:iter) , 'b-', 'LineWidth', 3); % Kp, Ki, Kd
    title(' PID Parameters Optimization', 'fontsize', 18);
    xlabel('Iteration Number', 'fontsize', 18);
    ylabel('Parameter Value', 'fontsize', 18);
    legend({'K_p', 'K_i', 'K_d'}, 'fontsize', 18);
    set(gca, 'Fontsize', 18);
    hold off;

    disp(['fzbest history: ', num2str(fzbest_history)]);
end