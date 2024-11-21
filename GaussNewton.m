function [w_all, f_all, gnorm_all] = GaussNewton(r_and_J, w0, kmax, tol)
w = w0;
f_all = [];
gnorm_all = [];
for k = 1:kmax
    [r, J] = r_and_J(w);
    f = 0.5 * sum(r.^2);
    g = J' * r;
    gnorm = norm(g);
    f_all = [f_all; f];
    gnorm_all = [gnorm_all; gnorm];
    if gnorm < tol
        break;
    end
    delta_w = - (J' * J+eye(size(J',1))*1e-6) \ (J' * r);
    w = w + delta_w;
end
w_all = w;
end