function average_err = my_grad_check(fun, theta, input)
    epsilon = 1e-4;
    total_error = 0;

    [f,g] = fun(theta, input{:});

    n = size(g,1);
    for i=1:n
        t_plus = theta(:);
        t_minus = theta(:);

        t_plus(i) = t_plus(i) + epsilon;
        t_minus(i) = t_minus(i) - epsilon;

        [f_plus,] = fun(t_plus, input{:});
        [f_minus,] = fun(t_minus, input{:});

        g_est = (f_plus - f_minus)/(2*epsilon);
        err = g(i) - g_est;
        total_error = total_error + err;
    endfor

    average_err = total_error/n;

