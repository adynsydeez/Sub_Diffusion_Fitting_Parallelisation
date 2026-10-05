function [D, beta, err] = fit_voxel_sub_ml(svec, bvec, D_guess, beta_guess, Dmin, Dmax, betamin, betamax)
    if svec(1) <= 0
        D = 0; beta = 0; err = 0; return;
    end

    options = optimoptions('lsqcurvefit', 'Display', 'off', 'FunctionTolerance', 1e-5);
    
    model = @(p, b) svec(1) * ml(-b * p(1), p(2));
    
    [p, resnorm] = lsqcurvefit(model, [D_guess, beta_guess], bvec, svec, ...
                              [Dmin, betamin], [Dmax, betamax], options);
    
    D = p(1);
    beta = p(2);
    
    err = sqrt(resnorm / length(bvec));
end