classdef ParamsFactory < handle
    %PARAMSFACTORY Summary of this class goes here
    %   Detailed explanation goes here   
    methods
        function factory = get_params_factory(obj,varargin)
            %PARAMSFACTORY Construct an instance of this class
            %   Detailed explanation goes here
            varargin = varargin{1};
            str = varargin{1};
            varargin(1) = [];  
            % other parameters
            switch str
                case 'msd'
                %if nargin == 2
                   factory = Msd(varargin{:}); 
               % else
                   %factory = Msd();
                %end             
                case 'Rg'
                   factory = Rg();
                case 'autoc'
                   basic =  Basic(varargin{:});                  
                   factory = Autoc(basic,varargin{:}); 
                   %factory = Autoc(basic);
                            %
                otherwise
                   factory = Basic(varargin{:}); 
                   %disp('disp velocity and omega');
            end
        end 
    end
end

