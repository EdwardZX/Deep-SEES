function [result] = msd_fit_alpha_Dt(msd)
msd = msd(:);
lag = length(msd);
t = (1:lag);
% timeInterv = (1:lag)';
% randWalkModel = @(b,x)4*b(1)*x.^b(2);
% beta = nlinfit(timeInterv,msd,randWalkModel,[10,1]);
% result.Dt = beta(1);
% result.alpha = beta(2);
% result.y_fit = randWalkModel(beta, (0:lag)');
% result.y_error = norm(msd- result.y_fit(2:end));

    [p,s] = polyfit(log10(t)',log10(msd),1);
    
    [result.y_fit,~] = polyval(p,log10(t)',s);
            %plot(log((1:obj.lag)*tjObj.dt)',log(temp))
            alpha = p(1);
            Dt = 10^p(2)/4;
            result.alpha = alpha;
            result.Dt = Dt;
            result.msd = msd';
            result.y_error =norm(log10(msd)-result.y_fit);

end