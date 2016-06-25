function average_err = my_grad_check(fun, theta, input, repeat)

    epsilon = 1e-4;
    total_error = 0;

    [f,g] = fun(theta, input{:});

    fprintf(' Iter       k             err');
    fprintf('           g_est               g               f\n')

    for k=1:repeat
        i = randsample(numel(theta), 1);
        t_plus = theta(:);
        t_minus = theta(:);

        t_plus(i) = t_plus(i) + epsilon;
        t_minus(i) = t_minus(i) - epsilon;

        [f_plus,] = fun(t_plus, input{:});
        [f_minus,] = fun(t_minus, input{:});

        g_est = (f_plus - f_minus)/(2*epsilon);
        err = g(i) - g_est;

        fprintf('% 5d  % 6d % 15g % 15f % 15f % 15f\n', k,i,err,g(i),g_est,f);

        total_error = total_error + err;
    endfor

    average_err = total_error/repeat;

