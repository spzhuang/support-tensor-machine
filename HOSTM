classdef HOSTM
    %HOSTM is a algrithm which can classificate samples
    %  HOSTM is short for the whole name "high order support tensor machine"
    %  You need to start it with some data(a set consist of tensor) and
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
        Boxconstrain
        data_L
        Size
        number_Size
        w
        Error 
        coef_tensor
        SupportTensor
        SupportTensorIndex
        bias    
    end
    methods
        function obj = HOSTM(data,label,varargin)          % initial some value by inputing data
            % data = the data we inputing ,data should be cell and the kind of samples should be tensor 
            % label = the data's label we inputing
            % varargin are ordly tol(the tolerance of algrithm), max_iter(maximum iteration), Boxconstrn (the punishment coefficient of svm models) 
            if nargin < 2
                error('You need to input data and labels at least ! ')
            end
            if nargin < 3; obj.tol = 0.001; end
            if nargin < 4; obj.max_iter = 1000; end
            if nargin < 5; obj.Boxconstrain = 1; end
            if nargin >= 6 ; error('Numbers of parameters can''t more than 5! '); end
            obj.data = data;
            obj.label = label;
            obj.Size = size(data{1});
            obj.number_Size = size(obj.Size,2);
            obj.data_L = size(label,2);           
            obj.w = cell(1,obj.number_Size);               % w是系数元胞，w的每一个项是对应的k-mode系数
            for i = 1:obj.number_Size
                obj.w{i} = ones(1,obj.Size(i));
            end
            inde = 1;
            obj.Error = 10;
            obj.coef_tensor = ones(obj.Size);
            while obj.Error>obj.tol && inde < obj.max_iter
                    old_norm = frob(obj.coef_tensor);
                    for i = 1:obj.number_Size
                            tem_data_size = [obj.data_L,obj.Size(i)];
                            tem_data = zeros(tem_data_size);
                            k_mode_coef = obj.w;
                            k_mode_coef(i) = [];
                            j_iter = 1:obj.number_Size;
                            j_iter(i) = [];
                            for k = 1:obj.data_L
                                need_data = data{k};         
                                 for j = j_iter
                                     if j < i
                                         need_data = tmprod(need_data,k_mode_coef{j},j);
                                     else
                                         need_data = tmprod(need_data,k_mode_coef{j-1},j);
                                     end
                                 end
                                 need_data = reduce_dimension(need_data);
                                 tem_data(k,:) = need_data;
                            end
                            kmode_coef_tensor = outprod(k_mode_coef);
                            norm_fix = frob(kmode_coef_tensor);
                            svm = fitcsvm(tem_data,obj.label,'BoxConstrain',obj.Boxconstrain/norm_fix,'KernelFunction','linear');
                            s_vector = svm.SupportVectors;
                            s_alpha = svm.Alpha;
                            s_label = svm.SupportVectors;
                            w_new = sum(s_alpha.*s_label.*s_vector);
                            obj.w{i} = w_new;      
                    end
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
                    obj.coef_tensor = outprod(obj.w);           
                    obj.SupportTensor = Tensor;               
                    obj.SupportTensorIndex = index_address;       
                    obj.Error = abs(old_norm - frob(obj.coef_tensor));
                    inde = inde + 1;
            end   
            bbias = zeros(1,s_long);
            for i = 1:s_long
                to = obj.SupportTensorIndex(i);
                ker = obj.SupportTensor{i};
                for j = 1:obj.number_Size
                    ker = tmprod(ker,obj.w{j},j);
                end
                bbias(i) = obj.label(to) - ker;
            end
            obj.bias = mean(bbias);
        end
    end
    methods
        function pred=predict(obj,z)
            pred = inprod(obj.coef_tensor,z) + obj.bias;
        end
        function getscore = score(obj,test,test_label)
            if size(test_label,1) ~= 1
                error('the size of test_label must be 1-by-N ! Please transpose your label !')
            end
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
                error('the size of test_label must be 1-by-N ! Please transpose your label !')
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
function ten = reduce_dimension(Tensor)
    type = size(Tensor);
    flag = 0;
    if any(type == 1)
        type2 = type(type~=1);
        if size(type2,2) == 1
            type2 = [1,type2];
        end
        if size(type2,2) == 0
            flag = 1;
        end
    else
        type2 = type;
    end
    if flag == 0
        ten = reshape(Tensor,type2);
    else
        ten = sum(Tensor);
    end
end

% ---------------------------------ATTENTION--------------------------------------------------------------------------
% -----------------------------The Following context is a addition function for generating random data set.-----------
function [data,label] = generate_data(n1,n2,Size,low_n1,high_n1,low_n2,high_n2)
%   函数生成基于均匀分布的随机数，
%   输入参数：n1,n2分别是负类样本和正类样本的数量,Size是样本数据的size，这三个是必须要输入的
%   low_n1,high_n1,是负类均匀分布的下和上区间；low_n2,high_n2是正类均匀分布的下和上区间
%   输出参数：data 是生成的随机数据，是一个 1-by-(n1+n2)的cell,而label 是对应的标签 是
%   1-by-(n1+n2)的向量
if nargin < 4; low_n1 = 10;end
if nargin < 5; high_n1 = 20;end
if nargin < 6; low_n2 = 25; end
if nargin < 7; high_n2 = 30;end
data = cell(1,n1+n2);
negative_label = ones(1,n1)*-1;
positive_label = ones(1,n2);
label = [negative_label,positive_label];
if sum(Size) == Size          % detect the array Size is a number or not; it that ,let Size be 1-Size array!
    Size = [1,Size];
end
for i = 1:n1
    data{i} = rand(Size)*(high_n1-low_n1)+low_n1;
end
for i = n1+1:n1+n2
    data{i} = rand(Size)*(high_n2-low_n2)+low_n2;
end
end
