function  result  = msd( vec,lag,varargin )
    %function [ result ] = msd( vec,lag,varargin )
    if nargin == 2
        p = 2;
    else if nargin == 3
            p = varargin{1};
        end
    end
    [nr,~] = size(vec);
    if nr <= lag
        warning('vec length should higher than lag!');
    end
    result = zeros(lag,1);
    gIndexM = @(calLength,tau)bsxfun(@plus,(1:1:calLength)',[0,tau]);
    for m = 1:1:lag
        calLength = nr - m; %calculate length with tau = m;
        vecIndex = gIndexM(calLength,m);
        tmpM_L = vec(vecIndex(:,1),:);
        tmpM_H = vec(vecIndex(:,2),:);
        result(m) = mean(sum((abs(tmpM_H - tmpM_L)).^p,2))';
    end   
    %%and error
%     [p,s] = polyfit(log10((1:lag))',log10(temp),1);
%     
%     [result.y_fit,~] = polyval(p,log10((1:lag))',s);
%             %plot(log((1:obj.lag)*tjObj.dt)',log(temp))
%             alpha = p(1);
%             Dt = 10^p(2)/4;
%             result.alpha = alpha;
%             result.Dt = Dt;
%             result.msd = temp';
%             result.y_error =norm(temp- 10.^(result.y_fit));
end

