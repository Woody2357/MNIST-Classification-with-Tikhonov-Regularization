function mnist_2categories_quadratic_NLLS_GN()
close all
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
%% 准备数据进行PCA
train_data = [reshape(train1, [], n1train)'; reshape(train2, [], n2train)'];
test_data = [reshape(test1, [], n1test)'; reshape(test2, [], n2test)'];
train_labels = [ones(n1train, 1); -ones(n2train, 1)];
test_labels = [ones(n1test, 1); -ones(n2test, 1)];
%% 进行PCA并只取前20个主成分
[coeff, score, ~] = pca(train_data);
nPCA = 20;
Xtrain = score(:, 1:nPCA);
Xtest = (test_data - mean(train_data)) * coeff(:, 1:nPCA);
%% 设置高斯-牛顿算法参数
d = nPCA;
r_and_J = @(w)Res_and_Jac(Xtrain, train_labels, w);
w = ones(d^2 + d + 1, 1);
kmax = 600;
tol = 1e-3;
%% 调用高斯-牛顿算法
[w_all, f_all, gnorm_all] = GaussNewton(r_and_J, w, kmax, tol);
%% 绘制损失函数和梯度范数
figure;
plot(f_all, 'LineWidth', 2);
xlabel('iter', 'FontSize', fsz);
ylabel('loss function f', 'FontSize', fsz);
set(gca, 'FontSize', fsz, 'YScale', 'log');
figure;
plot(gnorm_all, 'LineWidth', 2);
xlabel('iter', 'FontSize', fsz);
ylabel('gradient norm ||g||', 'FontSize', fsz);
set(gca, 'FontSize', fsz, 'YScale', 'log');
%% 在测试集上应用结果
test = myquadratic(Xtest, test_labels, w_all);
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n', nhits, nmisses, nhits / (nhits + nmisses) * 100);
end

function [r, J] = Res_and_Jac(X, y, w)
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
r = log(1 + exp(-q));
J = zeros(n, length(w));
for i = 1:n
    x_i = X(i, :)';
    exp_neg_q = exp(-q(i));
    factor = - y(i) * exp_neg_q / (1 + exp_neg_q);
    dW = x_i * x_i';
    J(i, 1:d^2) = factor * dW(:)';
    J(i, d^2+1:d^2+d) = factor * x_i';
    J(i, end) = factor;
end
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