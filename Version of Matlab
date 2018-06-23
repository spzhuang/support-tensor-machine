classdef STM3
    %STM is a algrithm can classificate samples
    %  STM is short for the whole name "support tensor machine"
    %  You need to start it with some data(a set consist of matrix) and
    %  corresponding labels
    %  STM has the following methods:
    %  1:predict(z): output the classification of z, z is a sample.
    %  2:score(test,test_label): get the accuary of test data, you need to
    %    input test and corresponding labels.
    %  3:pretty_predict(test,test_label): get the table consist of number
    %    of samples, the predicted labels and the real labels.
    properties(SetAccess = 'protected',GetAccess = 'public')
        data 
        label
        tol 
        max_iter  
        Boxconstrn
        data_L
        ns1
        ns2
        u
        v
        Error 
        coef_tensor
        SupportTensor
        SupportTensorIndex
        bias    
    end
    methods
        function obj = STM3(data,label,varargin)          % initial some value by inputing data
            % data = the data we inputing ,data should be cell and the samples should be matrix 
            % label = the data's label we inputing
            % varargin are ordly tol(the tolerance of algrithm), max_iter(maximum iteration), Boxconstrn (the punishment coefficient of svm models) 
            if nargin < 2
                error('You need to input data and labels at least ! ')
            end
            if nargin < 3; obj.tol = 0.001; end
            if nargin < 4; obj.max_iter = 200; end
            if nargin < 5; obj.Boxconstrn = 1; end
            if nargin >= 6 ; error('Numbers of parameters can''t more than 5! '); end
            obj.data = data;
            obj.label = label;
            obj.ns1 = size(data{1},1);
            obj.ns2 = size(data{1},2);
            obj.data_L = size(label,2);             % Over obj.data obj.label obj.tol obj.max_iter obj.Boxconstrn obj.ns1 obj.ns2 obj.data_L
            obj.u = ones(obj.ns1,1);
            obj.v = ones(obj.ns2,1);
            obj.Error = 10;
            inde = 1;
            obj.Error = 10;
            while obj.Error>obj.tol && inde < obj.max_iter
                    old_norm = norm(obj.u*obj.v');
                    for i = 1:2
                        if i == 1
                            tem_data = zeros(obj.data_L,obj.ns2);
                            for j = 1:obj.data_L
                                tem_data(j,:) = obj.u' * obj.data{j};
                            end
                            norm_u = norm(obj.u);
                            svm = fitcsvm(tem_data,obj.label,'BoxConstrain',obj.Boxconstrn/norm_u,'KernelFunction','linear');
                            s_vector = svm.SupportVectors;
                            s_alpha = svm.Alpha;
                            s_label = svm.SupportVectorLabels;
                            w = sum(s_alpha.*s_label.*s_vector);       % get the coefficient of support tensors
                            obj.v = w';
                        elseif  i==2 
                            tem_data = zeros(obj.data_L,obj.ns1);
                            for j = 1:obj.data_L
                                tem_data(j,:) = (obj.data{j}*obj.v)';
                            end
                            norm_v = norm(obj.v);
                            svm = fitcsvm(tem_data,obj.label,'BoxConstrain',obj.Boxconstrn/norm_v,'KernelFunction','linear');
                            s_vector = svm.SupportVectors;
                            s_alpha = svm.Alpha;
                            s_label = svm.SupportVectorLabels;
                            w = sum(s_alpha.*s_label.*s_vector);
                            obj.u = w';
                            [s_long,~] = size(s_label);
                            index_address = zeros(1,s_long);
                            for k = 1:s_long
                                for m = 1:obj.data_L
                                    if all(s_vector(k,:) == tem_data(m,:))
                                        index_address(k) = m;
                                        break
                                    else
                                        continue
                                    end
                                end
                            end
                            Tensor = cell(1,s_long);
                            for k = 1:s_long
                                Tensor{k} = obj.data{index_address(k)};
                            end
                        end
                    end
                    obj.coef_tensor = obj.u*obj.v';            % val{4} = obj.coef_tensor
                    obj.SupportTensor = Tensor;                    % val{5} = obj.SupportTensor
                    obj.SupportTensorIndex = index_address;         % val{6} = obj.SupportTensorIndex
                    obj.Error = abs(old_norm - norm(obj.coef_tensor));
                    inde = inde + 1;
            end   
            bbias = zeros(1,s_long);
            for i = 1:s_long
                to = obj.SupportTensorIndex(i);
                bbias(i) = obj.label(to) - (obj.u' * obj.SupportTensor{i} *obj.v );
            end
            obj.bias = mean(bbias);
        end
    end
    methods
        function pred=predict(obj,z)
            pred = sum(sum(z.*obj.coef_tensor)) + obj.bias;
        end
        function getscore = score(obj,test,test_label)
            [~,L] = size(test_label);
            flag = 0;
            sto = zeros(1,L);
            for i = 1:L
                sto(i) = sign(obj.predict(test{i}));
                if sto(i) == test_label(i)
                    flag = flag + 1;
                end
            end
           getscore= flag / L;
            fprintf('accuracy = %.2f',getscore);
        end
        function [Tab,p] = pretty_predict(obj,test,test_label)
            if size(test_label,1) ~= 1
                error('the size of test_label must be 1*N! please transpose your label !')
            end
            L = size(test_label,2);
            flag = 0;
            pre_accuary = zeros(L,1);
            for i = 1:L
                pre_accuary(i) = sign(obj.predict(test{i}));
                if pre_accuary(i) == test_label(i)
                    flag = flag + 1;
                end
            end
            p = flag/L;
            times = [1:L]';
            Label = test_label';
            Tab = table(times,pre_accuary,Label);
        end
    end
end
