function [sigma,varargout] = cal_std(x,r,varargin)
x_std = (x-mean(x));
varargout{1} = std(x_std);
if nargin == 3
   x_std = sqrt(sum(std(x_std(:,1:varargin{1})).^2));
   sz = varargin{1};
else
   x_std = sqrt(sum(std(x_std).^2)); 
   sz = size(x,2);
end
% x_std = sqrt(sum(std(x_std).^2));%/sqrt(size(x,2));
sigma = r*x_std/sqrt(sz);
% y = sqrt(mean(y(:)));
end