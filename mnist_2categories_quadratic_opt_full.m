close all;
fsz = 20;
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;

%% Find training and testing data for digits 1 and 7
ind1 = find(double(labels_train)==1);
ind2 = find(double(labels_train)==7);
n1train = length(ind1);
n2train = length(ind2);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);

itest1 = find(double(labels_test)==1);
itest2 = find(double(labels_test)==7);
n1test = length(itest1);
n2test = length(itest2);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);

%% Prepare data and perform PCA to get the first 20 principal components
train_data = [reshape(train1, [], n1train)'; reshape(train2, [], n2train)'];
test_data = [reshape(test1, [], n1test)'; reshape(test2, [], n2test)'];
train_labels = [ones(n1train, 1); -ones(n2train, 1)];
test_labels = [ones(n1test, 1); -ones(n2test, 1)];

[coeff, score, ~] = pca(train_data);
nPCA = 20;
Xtrain = score(:, 1:nPCA);
Xtest = (test_data - mean(train_data)) * coeff(:, 1:nPCA);

%% Hyperparameters
num_epochs = 500;           % Total number of epochs
batch_sizes = [1, 32, 64, 128]; % Different batch sizes to experiment with
initial_step_size = 0.01;   % Initial learning rate
lambda = 0.01;              % Regularization coefficient

% Initialize parameters
d = nPCA;
Ntrain = size(Xtrain, 1);
Ntest = size(Xtest, 1);

% Run comparisons for different optimizers and batch sizes
optimizers = {'nesterov', 'adam', 'sgd'};
results = struct();

for optimizer = optimizers
    for batch_size = batch_sizes
        fprintf('Running %s optimizer with batch size %d\n', optimizer{1}, batch_size);
        [train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, grad_norm_history] = run_optimizer(optimizer{1}, batch_size, num_epochs, initial_step_size, lambda, Xtrain, train_labels, Xtest, test_labels, d, Ntrain, Ntest, false);
        results.(optimizer{1}).(['batch_' num2str(batch_size)]).train_loss_history = train_loss_history;
        results.(optimizer{1}).(['batch_' num2str(batch_size)]).test_loss_history = test_loss_history;
        results.(optimizer{1}).(['batch_' num2str(batch_size)]).train_accuracy_history = train_accuracy_history;
        results.(optimizer{1}).(['batch_' num2str(batch_size)]).test_accuracy_history = test_accuracy_history;
        results.(optimizer{1}).(['batch_' num2str(batch_size)]).grad_norm_history = grad_norm_history;
    end
end

% Run deterministic versions
for optimizer = optimizers
    fprintf('Running deterministic %s optimizer\n', optimizer{1});
    [train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, grad_norm_history] = run_optimizer(optimizer{1}, Ntrain, num_epochs, initial_step_size, lambda, Xtrain, train_labels, Xtest, test_labels, d, Ntrain, Ntest, true);
    results.(optimizer{1}).deterministic.train_loss_history = train_loss_history;
    results.(optimizer{1}).deterministic.test_loss_history = test_loss_history;
    results.(optimizer{1}).deterministic.train_accuracy_history = train_accuracy_history;
    results.(optimizer{1}).deterministic.test_accuracy_history = test_accuracy_history;
    results.(optimizer{1}).deterministic.grad_norm_history = grad_norm_history;
end

% Plot results
figure('Position', [100, 100, 1200, 800]); % Increase figure size
for optimizer = optimizers
    for batch_size = batch_sizes
        subplot(length(optimizers), length(batch_sizes) + 1, (find(strcmp(optimizers, optimizer{1})) - 1) * (length(batch_sizes) + 1) + find(batch_sizes == batch_size));
        plot(1:num_epochs, results.(optimizer{1}).(['batch_' num2str(batch_size)]).train_loss_history, '-o', 'MarkerSize', 1, 'LineWidth', 1);
        hold on;
        plot(1:num_epochs, results.(optimizer{1}).(['batch_' num2str(batch_size)]).test_loss_history, '-s', 'MarkerSize', 1, 'LineWidth', 1);
        xlabel('Epoch', 'FontSize', fsz);
        ylabel('Loss', 'FontSize', fsz);
        legend('Train Loss', 'Test Loss');
        title(sprintf('%s, Batch %d', optimizer{1}, batch_size), 'FontSize', fsz);
    end
    subplot(length(optimizers), length(batch_sizes) + 1, find(strcmp(optimizers, optimizer{1})) * (length(batch_sizes) + 1));
    plot(1:num_epochs, results.(optimizer{1}).deterministic.train_loss_history, '-o', 'MarkerSize', 1, 'LineWidth', 1);
    hold on;
    plot(1:num_epochs, results.(optimizer{1}).deterministic.test_loss_history, '-s', 'MarkerSize', 1, 'LineWidth', 1);
    xlabel('Epoch', 'FontSize', fsz);
    ylabel('Loss', 'FontSize', fsz);
    legend('Train Loss', 'Test Loss');
    title(sprintf('Deterministic %s', optimizer{1}), 'FontSize', fsz);
end
% tight_layout(); % Adjust layout to minimize whitespace

figure('Position', [100, 100, 1200, 800]); % Increase figure size
for optimizer = optimizers
    for batch_size = batch_sizes
        subplot(length(optimizers), length(batch_sizes) + 1, (find(strcmp(optimizers, optimizer{1})) - 1) * (length(batch_sizes) + 1) + find(batch_sizes == batch_size));
        plot(1:num_epochs, results.(optimizer{1}).(['batch_' num2str(batch_size)]).train_accuracy_history * 100, '-o', 'MarkerSize', 1, 'LineWidth', 1);
        hold on;
        plot(1:num_epochs, results.(optimizer{1}).(['batch_' num2str(batch_size)]).test_accuracy_history * 100, '-s', 'MarkerSize', 1, 'LineWidth', 1);
        xlabel('Epoch', 'FontSize', fsz);
        ylabel('Accuracy (%)', 'FontSize', fsz);
        legend('Train Accuracy', 'Test Accuracy');
        title(sprintf('%s, Batch %d', optimizer{1}, batch_size), 'FontSize', fsz);
    end
    subplot(length(optimizers), length(batch_sizes) + 1, find(strcmp(optimizers, optimizer{1})) * (length(batch_sizes) + 1));
    plot(1:num_epochs, results.(optimizer{1}).deterministic.train_accuracy_history * 100, '-o', 'MarkerSize', 1, 'LineWidth', 1);
    hold on;
    plot(1:num_epochs, results.(optimizer{1}).deterministic.test_accuracy_history * 100, '-s', 'MarkerSize', 1, 'LineWidth', 1);
    xlabel('Epoch', 'FontSize', fsz);
    ylabel('Accuracy (%)', 'FontSize', fsz);
    legend('Train Accuracy', 'Test Accuracy');
    title(sprintf('Deterministic %s', optimizer{1}), 'FontSize', fsz);
end
% tight_layout(); % Adjust layout to minimize whitespace

figure('Position', [100, 100, 1200, 800]); % Increase figure size
for optimizer = optimizers
    for batch_size = batch_sizes
        subplot(length(optimizers), length(batch_sizes) + 1, (find(strcmp(optimizers, optimizer{1})) - 1) * (length(batch_sizes) + 1) + find(batch_sizes == batch_size));
        plot(1:num_epochs, results.(optimizer{1}).(['batch_' num2str(batch_size)]).grad_norm_history, '-o', 'MarkerSize', 1, 'LineWidth', 1);
        xlabel('Epoch', 'FontSize', fsz);
        ylabel('Gradient Norm', 'FontSize', fsz);
        title(sprintf('%s, Batch %d', optimizer{1}, batch_size), 'FontSize', fsz);
    end
    subplot(length(optimizers), length(batch_sizes) + 1, find(strcmp(optimizers, optimizer{1})) * (length(batch_sizes) + 1));
    plot(1:num_epochs, results.(optimizer{1}).deterministic.grad_norm_history, '-o', 'MarkerSize', 1, 'LineWidth', 1);
    xlabel('Epoch', 'FontSize', fsz);
    ylabel('Gradient Norm', 'FontSize', fsz);
    title(sprintf('Deterministic %s', optimizer{1}), 'FontSize', fsz);
end
tight_layout(); % Adjust layout to minimize whitespace

% Function definitions
% Function to run optimization
function [train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, grad_norm_history] = run_optimizer(optimizer, batch_size, num_epochs, initial_step_size, lambda, Xtrain, train_labels, Xtest, test_labels, d, Ntrain, Ntest, deterministic)
    w = randn(d^2 + d + 1, 1); % Random initialization of parameter vector
    train_loss_history = zeros(num_epochs, 1);
    test_loss_history = zeros(num_epochs, 1);
    train_accuracy_history = zeros(num_epochs, 1);
    test_accuracy_history = zeros(num_epochs, 1);
    grad_norm_history = zeros(num_epochs, 1);
    
    % Nesterov parameters
    mu = 0.9;
    v = zeros(size(w));
    
    % Adam parameters
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    m = zeros(size(w));
    v_adam = zeros(size(w));
    
    for epoch = 1:num_epochs
        if ~deterministic
            % Shuffle training data for stochastic versions
            perm = randperm(Ntrain);
            Xtrain = Xtrain(perm, :);
            train_labels = train_labels(perm);
        end
        
        % Number of batches
        num_batches = ceil(Ntrain / batch_size);
        for batch = 1:num_batches
            if deterministic
                % Use the entire dataset for deterministic versions
                X_batch = Xtrain;
                y_batch = train_labels;
            else
                % Get current batch for stochastic versions
                start_idx = (batch - 1) * batch_size + 1;
                end_idx = min(batch * batch_size, Ntrain);
                X_batch = Xtrain(start_idx:end_idx, :);
                y_batch = train_labels(start_idx:end_idx);
            end
            
            % Compute loss and gradient
            [loss, grad] = compute_loss_and_gradient(w, X_batch, y_batch, lambda);
            
            % Update parameters based on optimizer
            switch optimizer
                case 'nesterov'
                    v_prev = v;
                    v = mu * v - initial_step_size * grad;
                    w = w + mu * (v - v_prev) - initial_step_size * grad;
                case 'adam'
                    m = beta1 * m + (1 - beta1) * grad;
                    v_adam = beta2 * v_adam + (1 - beta2) * (grad .^ 2);
                    m_hat = m / (1 - beta1^epoch);
                    v_hat = v_adam / (1 - beta2^epoch);
                    w = w - initial_step_size * m_hat ./ (sqrt(v_hat) + epsilon);
                case 'sgd'
                    w = w - initial_step_size * grad;
            end
        end
        
        % Evaluate model on training and testing data
        [train_loss, train_accuracy] = evaluate_model(w, Xtrain, train_labels, lambda);
        [test_loss, test_accuracy] = evaluate_model(w, Xtest, test_labels, lambda);
        
        % Record loss, accuracy, and gradient norm
        train_loss_history(epoch) = train_loss;
        test_loss_history(epoch) = test_loss;
        train_accuracy_history(epoch) = train_accuracy;
        test_accuracy_history(epoch) = test_accuracy;
        grad_norm_history(epoch) = norm(grad);
        
        % Print current epoch results
        fprintf('Epoch %d/%d: Train Loss=%.4f, Train Accuracy=%.2f%%, Test Loss=%.4f, Test Accuracy=%.2f%%\n', ...
            epoch, num_epochs, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100);
    end
end

function [loss, grad] = compute_loss_and_gradient(w, X, y, lambda)
    n = size(X, 1);
    d = size(X, 2);
    W = reshape(w(1:d^2), [d, d]);
    v = w(d^2+1:d^2+d);
    b = w(end);
    q = zeros(n, 1);
    for i = 1:n
        x_i = X(i, :)';
        q(i) = y(i) * (x_i' * W * x_i + v' * x_i + b);
    end
    loss = (1/n) * sum(log(1 + exp(-q))) + (lambda/2) * (w' * w);
    exp_neg_q = exp(-q);
    factors = - y .* exp_neg_q ./ (1 + exp_neg_q);
    grad_W = zeros(d, d);
    grad_v = zeros(d, 1);
    grad_b = 0;
    for i = 1:n
        x_i = X(i, :)';
        grad_W = grad_W + factors(i) * (x_i * x_i');
        grad_v = grad_v + factors(i) * x_i;
        grad_b = grad_b + factors(i);
    end
    grad_W = grad_W / n + lambda * W;
    grad_v = grad_v / n + lambda * v;
    grad_b = grad_b / n + lambda * b;
    grad = [grad_W(:); grad_v; grad_b];
end

function [loss, accuracy] = evaluate_model(w, X, y, lambda)
    test = myquadratic(X, y, w);
    hits = find(test > 0);
    misses = find(test < 0);
    nhits = length(hits);
    nmisses = length(misses);
    accuracy = nhits / (nhits + nmisses);
    [loss, ~] = compute_loss_and_gradient(w, X, y, lambda);
end

function q = myquadratic(X, y, w)
    n = size(X, 1);
    d = size(X, 2);
    W = reshape(w(1:d^2), [d, d]);
    v = w(d^2+1:d^2+d);
    b = w(end);
    q = zeros(n, 1);
    for i = 1:n
        x_i = X(i, :)';
        q(i) = y(i) * (x_i' * W * x_i + v' * x_i + b);
    end
end