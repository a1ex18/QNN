clear all;
clc

load HumanActivityTrain
load HumanActivityTest
a1=XTrain{1}';
a2=XTrain{2}';
a3=XTrain{3}';
a4=XTrain{4}';
a5=XTrain{5}';
a6=XTrain{6}';
by1=YTrain{1}';
by2=YTrain{2}';
by3=YTrain{3}';
by4=YTrain{4}';
by5=YTrain{5}';
by6=YTrain{6}';
classNames = categories(by1);
numFeatures = size(a1,2) ;
numClasses = numel(classNames);
w1=[];
w2=[];
b1=[];%
b2=[];
neurons=20;
numHiddenUnits=50;
layers = [
    featureInputLayer(numFeatures,'Normalization', 'none','Name','input1')
    fullyConnectedLayer(neurons,'Name','connect','Weights', w1, 'Bias',b1)%(50)
    batchNormalizationLayer('Name','batch')
    reluLayer('Name','relu')
    fullyConnectedLayer(numClasses,'Name','fc','Weights', w2,'Bias',b2)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
miniBatchSize =500;
numRounds = 50;
maxEpochs=1;
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs,...
            'MiniBatchSize',miniBatchSize,...
            'InitialLearnRate',0.001,...
            'Shuffle','every-epoch', ...
            'Verbose',0);%'Shuffle','every-epoch',... %, 'GradientThreshold',2,...
[w1_1, w1_2,b1_1,b1_2,net1] = cnn_train1(XTrain{1}',YTrain{1}',layers,options);
[w2_1, w2_2,b2_1,b2_2,net2] = cnn_train1(XTrain{2}',YTrain{2}',layers,options);
[w3_1, w3_2,b3_1,b3_2,net3] = cnn_train1(XTrain{3}',YTrain{3}',layers,options);
[w4_1, w4_2,b4_1,b4_2,net4] = cnn_train1(XTrain{4}',YTrain{4}',layers,options);
% [w5_1, w5_2,b5_1,b5_2,net5] = cnn_train1(XTrain{5}',YTrain{5}',layers,options);
% [w6_1, w6_2,b6_1,b6_2,net6] = cnn_train1(XTrain{6}',YTrain{6}',layers,options);
YPred1=classify(net1, XTest{1}');
acc_1 = sum(YPred1 == YTest{1}')./numel(YTest{1})
YPred2=classify(net2, XTest{1}');
acc_2 = sum(YPred2 == YTest{1}')./numel(YTest{1})
YPred3=classify(net3, XTest{1}');
acc_3 = sum(YPred3 == YTest{1}')./numel(YTest{1})
 YPred4=classify(net4, XTest{1}');
 acc_4 = sum(YPred4 == YTest{1}')./numel(YTest{1})
% YPred5=classify(net5, XTest{1}');
% acc_5 = sum(YPred5 == YTest{1}')./numel(YTest{1})
totaldataset=sum(size(a1,1)+size(a2,1)+size(a3,1)+size(a4,1));
norm1=size(a1,1)/totaldataset;
norm2=size(a2,1)/totaldataset;
norm3=size(a3,1)/totaldataset;
norm4=size(a4,1)/totaldataset;
SNRdB=-4;
dist=1;
Nt=4;Nr=32;
tStart=tic;
dev1=zeros(1,numRounds);
dev2=zeros(1,numRounds);
dev3=zeros(1,numRounds);
dev4=zeros(1,numRounds);
repw=5;repb=5;
for i=1:numRounds
    i
w1=(norm1.*w1_1+norm2.*w2_1+norm3.*w3_1+norm4.*w4_1);
b1=(norm1.*b1_1+norm2.*b2_1+norm3.*b3_1+norm4.*b4_1);
w2=(norm1.*w1_2+norm2.*w2_2+norm3.*w3_2+norm4.*w4_2);
b2=(norm1.*b1_2+norm2.*b2_2+norm3.*b3_2+norm4.*b4_2);
net_g=net1;
tmp_net = net_g.saveobj;
tmp_net.Layers(2).Weights = w1;
tmp_net.Layers(5).Weights = w2;
tmp_net.Layers(2).Bias = b1;
tmp_net.Layers(5).Bias = b2;
net_g = net_g.loadobj(tmp_net);
[wg_1, wg_2,bg_1,bg_2,net_g] = cnn_train1(XTest{1}',YTest{1}',net_g.Layers,options);
YPred=classify(net_g, XTest{1}');
accg(i) = sum(YPred == YTest{1}')./numel(YTest{1});
[w1_1, w1_2,b1_1,b1_2,net1] = cnn_train1(XTrain{1}',YTrain{1}',net_g.Layers,options);
[w2_1, w2_2,b2_1,b2_2,net2] = cnn_train1(XTrain{2}',YTrain{2}',net_g.Layers,options);
[w3_1, w3_2,b3_1,b3_2,net3] = cnn_train1(XTrain{3}',YTrain{3}',net_g.Layers,options);
[w4_1, w4_2,b4_1,b4_2,net4] = cnn_train1(XTrain{4}',YTrain{4}',net_g.Layers,options);
% [w5_1, w5_2,b5_1,b5_2,net5] = cnn_train1(XTrain{5}',YTrain{5}',net_g.Layers,options);
% [w6_1, w6_2,b6_1,b6_2,net6] = cnn_train1(XTrain{6}',YTrain{6}',net_g.Layers,options);
 w_sign1=horzcat(build_data_to_signal(w1_1), build_data_to_signal(w1_2));
 b_sign1=horzcat(build_data_to_signal(b1_1), build_data_to_signal(b1_2));
  [decodedwt1,  BERoptwt1(i)]=spat_mod_wt(Nt,Nr,SNRdB,w_sign1,dist,repw);
 [wBeam1_1,wBeam1_2]=signals_wt(decodedwt1,w1_1,w1_2,neurons);
  [decodedbias1, BERoptb1(i)]=spat_mod_wt(Nt,Nr,SNRdB,b_sign1,dist,repb);
   [bBeam1_1,bBeam1_2]=beaming_bias(decodedbias1,b1_1,b1_2,neurons);
  w_sign2=horzcat(build_data_to_signal(w2_1), build_data_to_signal(w2_2));
  b_sign2=horzcat(build_data_to_signal(b2_1), build_data_to_signal(b2_2));
  [decodedwt2,  BERoptwt2(i)]=spat_mod_wt(Nt,Nr,SNRdB,w_sign2,dist,repw);
 [wBeam2_1,wBeam2_2]=signals_wt(decodedwt2,w2_1,w2_2,neurons);
  [decodedbias2, BERoptb2(i)]=spat_mod_wt(Nt,Nr,SNRdB,b_sign2,dist,repb);
   [bBeam2_1,bBeam2_2]=beaming_bias(decodedbias2,b2_1,b2_2,neurons);
 w_sign3=horzcat(build_data_to_signal(w3_1), build_data_to_signal(w3_2));
  b_sign3=horzcat(build_data_to_signal(b3_1), build_data_to_signal(b3_2));
[decodedwt3, BERoptwt3(i)]=spat_mod_wt(Nt,Nr,SNRdB,w_sign3,dist,repw);
 [wBeam3_1,wBeam3_2]=signals_wt(decodedwt3,w3_1,w3_2,neurons);
  [decodedbias3, BERoptb3(i)]=spat_mod_wt(Nt,Nr,SNRdB,b_sign3,dist,repb);
   [bBeam3_1,bBeam3_2]=beaming_bias(decodedbias3,b3_1,b3_2,neurons);
w_sign4=horzcat(build_data_to_signal(w4_1), build_data_to_signal(w4_2));
  b_sign4=horzcat(build_data_to_signal(b4_1), build_data_to_signal(b4_2));
[decodedwt4, BERoptwt4(i)]=spat_mod_wt(Nt,Nr,SNRdB,w_sign4,dist,repw);
 [wBeam4_1,wBeam4_2]=signals_wt(decodedwt4,w4_1,w4_2,neurons);
  [decodedbias4, BERoptb4(i)]=spat_mod_wt(Nt,Nr,SNRdB,b_sign4,dist,repb);
   [bBeam4_1,bBeam4_2]=beaming_bias(decodedbias4,b4_1,b4_2,neurons);
   
 w1_1=wBeam1_1;
 w2_1=wBeam2_1;
 w3_1=wBeam3_1;
 w4_1=wBeam4_1;
 w1_2=wBeam1_2;
 w2_2=wBeam2_2;
 w3_2=wBeam3_2;
 w4_2=wBeam4_2;
 b1_1=bBeam1_1;
 b2_1=bBeam2_1;
 b3_1=bBeam3_1;
 b4_1=bBeam4_1;
 b1_2=bBeam1_2;
 b2_2=bBeam2_2;
 b3_2=bBeam3_2;
 b4_2=bBeam4_2;
tmp_net1 = net1.saveobj;
tmp_net1.Layers(2).Weights = w1_1;
tmp_net1.Layers(5).Weights = w1_2;
tmp_net1.Layers(2).Bias = b1_1;
tmp_net1.Layers(5).Bias = b1_2;
net1 = net1.loadobj(tmp_net1);
YPred1=classify(net1, XTrain{1}');
dev1(i) = sum(YPred1 == YTrain{1}')./numel(YTrain{1});
tmp_net2 = net2.saveobj;
tmp_net2.Layers(2).Weights = w2_1;
tmp_net2.Layers(5).Weights = w2_2;
tmp_net2.Layers(2).Bias = b2_1;
tmp_net2.Layers(5).Bias = b2_2;
net2 = net2.loadobj(tmp_net2);
YPred2=classify(net2, XTrain{2}');
dev2(i) = sum(YPred2 == YTrain{2}')./numel(YTrain{2});
tmp_net3 = net3.saveobj;
tmp_net3.Layers(2).Weights = w3_1;
tmp_net3.Layers(5).Weights = w3_2;
tmp_net3.Layers(2).Bias = b3_1;
tmp_net3.Layers(5).Bias = b3_2;
net3 = net3.loadobj(tmp_net3);
YPred3=classify(net3, XTrain{3}');
dev3(i) = sum(YPred3 == YTrain{3}')./numel(YTrain{3});
tmp_net4 = net4.saveobj;
tmp_net4.Layers(2).Weights = w4_1;
tmp_net4.Layers(5).Weights = w4_2;
tmp_net4.Layers(2).Bias = b4_1;
tmp_net4.Layers(5).Bias = b4_2;
net4 = net4.loadobj(tmp_net4);
YPred4=classify(net4, XTrain{4}');
dev4(i) = sum(YPred4 == YTrain{4}')./numel(YTrain{4});
end
plot(accg)
hold on
plot(dev1)
plot(dev2)
plot(dev3)
plot(dev4)
legend('global','dev1','dev2','dev3');
%     