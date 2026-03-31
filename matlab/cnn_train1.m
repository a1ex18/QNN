%% TRAIN
function [w1, w2,b1,b2,net] = cnn_train1(x_train, y_train, layers, options)%w1, w2, b1,b2, N)
%     
    net = trainNetwork(x_train, y_train, layers, options);
    w1 = net.Layers(2).Weights;
    w2 = net.Layers(5).Weights;
    b1 = net.Layers(2).Bias;
    b2 = net.Layers(5).Bias;

end