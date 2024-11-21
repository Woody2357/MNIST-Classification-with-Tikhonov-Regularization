function mnist_2categories_quadratic_SGD()
close all;
fsz = 20;
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;

%% 找到训练数据中的数字1和7
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);

%% 找到测试数据中的数字1和7
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);

%% 准备数据进行PCA并取前20个主成分
train_data = [reshape(train1, [], n1train)'; reshape(train2, [], n2train)'];
test_data = [reshape(test1, [], n1test)'; reshape(test2, [], n2test)'];
train_labels = [ones(n1train, 1); -ones(n2train, 1)];
test_labels = [ones(n1test, 1); -ones(n2test, 1)];

[coeff, score, ~] = pca(train_data);
nPCA = 20;
Xtrain = score(:, 1:nPCA);
Xtest = (test_data - mean(train_data)) * coeff(:, 1:nPCA);

%% 设置随机梯度下降参数
d = nPCA;
w = zeros(d^2 + d + 1, 1);
lambda = 0.01;
max_iters = 1000;
batch_sizes = [32, 64, 128];
initial_step_size = 0.1;
step_size_decay = @(t) initial_step_size / (1 + 0.001 * t);

loss_history = [];
grad_norm_history = [];
nmisses_history = [];
batch_size = 128;

for iter = 1:max_iters
    idx = randperm(size(Xtrain, 1), batch_size);
    X_batch = Xtrain(idx, :);
    y_batch = train_labels(idx);
    [loss, grad] = compute_loss_and_gradient(w, X_batch, y_batch, lambda);
    loss_history = [loss_history; loss];
    grad_norm_history = [grad_norm_history; norm(grad)];
    
    step_size = initial_step_size / (1 + 0.001 * iter);
%     step_size = initial_step_size;
%     step_size = initial_step_size * exp(-0.001 * iter);
%     step_size = initial_step_size / sqrt(iter);

    w = w - step_size * grad;
    test = myquadratic(Xtest, test_labels, w);
    misses = find(test < 0);
    nmisses = length(misses);
    nmisses_history = [nmisses_history; nmisses];
end

%% 绘制损失和梯度范数曲线
figure;
plot(loss_history, 'LineWidth', 2);
xlabel('iter', 'FontSize', fsz);
ylabel('loss function f', 'FontSize', fsz);
set(gca, 'FontSize', fsz, 'YScale', 'log');
title('Loss functions versus iterations');

figure;
plot(grad_norm_history, 'LineWidth', 2);
xlabel('iter', 'FontSize', fsz);
ylabel('gradient norm ||g||', 'FontSize', fsz);
set(gca, 'FontSize', fsz, 'YScale', 'log');
title('Gradient norms versus iterations');

figure;
plot(nmisses_history, 'LineWidth', 2);
xlabel('iter', 'FontSize', fsz);
ylabel('nmisses', 'FontSize', fsz);
set(gca, 'FontSize', fsz, 'YScale', 'log');
title('Nmisses versus iterations');

%% 在测试集上评估模型
test = myquadratic(Xtest, test_labels, w);
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
accuracy = nhits / (nhits + nmisses) * 100;
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n', nhits, nmisses, accuracy);

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