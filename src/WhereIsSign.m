function LabelMap = WhereIsSign(I)
%% inputs
% limit1,limit2: rgb limit
% limit2,limit3: hsv limit
% limit3,limit4: feature limit
%% read image
% imshow(I);
limit1 = [63.5468577783733;3.94677523699969;0.151529676904801];
limit2 = [264.658286894250;125.305874812958;139.640640327722];
limit3 = [0.0188364547575912;0.403929405422917;0.249146451781176];
limit4 = [1.61358207601935;0.829788486471207;1.03803666151819];
limit5 = [13.2320053152898;19.1287759274938;5.67358199392345];
limit6 = [69.2879316279913;60.3597866511264;30.5454050324186];
limit7 = [931.200000000000;82.0172741438557;11.5763806948310;0.590216220208253;-4.70024810480174;931.200000000000;30.7979306325264;0.753054662379421;0.392515974131588;0.109473684210526];
limit8 = [1624360.40000000;2199.93965161501;1280.74322231386;1.28962409821671;112.654707964290;1624360.40000000;1639.71334843514;1.29765706634797;1.28043010752688;2.14556277056277];
%% rgb & hsv extraction
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);
im1 = r>=limit1(1)&r<=limit2(1)&g>=limit1(2)&g<=limit2(2)&b>=limit1(3)&b<=limit2(3);
% figure;
% imshow(im1);
I1 = rgb2hsv(I);
I1 = double(I1);
% imshow(I1);
h = I1(:,:,1);
s = I1(:,:,2);
v = I1(:,:,3);
im2 = s>=limit3(2)&s<=limit4(2)&v>=limit3(3)&v<=limit4(3);  % only use s as a criteria
I1 = rgb2lab(I);
I1 = double(I1);
% imshow(I1);
l = I1(:,:,1);
a = I1(:,:,2);
b = I1(:,:,3);
im2_2 = l>=limit5(1)&l<=limit6(1)&a>=limit5(2)&a<=limit6(2)&b>=limit5(3)&b<=limit6(3);  % only use s as a criteria
% figure;
% imshow(im2_2);
im3 = im1&im2&im2_2;
% figure;
% imshow(im3);
% morphological operation
se = strel('rectangle',[2,7]);
im4 = imopen(im3,se);
% im4 = medfilt2(im4,[4,4]);
% figure;
% imshow(im4);
%% BLOB Analysis
seg = bwlabel(im4,8);
% figure;
% imagesc(seg);
% feature extraction
features = blob(seg);
% imagesc(seg);
% feature = features(:,idx);
% blob selection
im5 = ones(size(I,1),size(I,2));
for i = [1,4,9,10]
    index = find(features(:,i)>=limit7(i)&features(:,i)<=limit8(i));
    temp = ismember(seg,index);
    im5 = im5&temp;
end
%     figure;
%     imshow(im5);
im6 = bwlabel(im5,8); % im6 contains all the signs
% % im6 = imdilate(im6,se);
% figure;
% imagesc(im6);
%% Create the blocked corner (e.g. by leaves)
sign_num = max(max(im6)); % the number of signs in the picture
LabelMap = zeros(size(I,1),size(I,2));
if sign_num ~=0
    for i = 1:sign_num
        sign = im6==i; % analyze each sign
        %imagesc(sign);
        %imshow(label_map);
        area = regionprops(sign,'Area');
        area = [area.Area];
%         if area<1100
%             Map = zeros(size(sign,1),size(sign,2)); % merge two images
%         else
            scale = 1;
            width = 10;
            label_map = box(sign,scale,width); % create the virtual box
            sign = fill_it(sign);
            Map = sign|label_map; % merge two images
%         end
        %imshow(Map);
        %loss = xor(im,label_map); % measure the function of the virtual box
        %imshow(loss);
        LabelMap = LabelMap+i*Map;
    end
else
    LabelMap = im5;
end
% se = strel('square',5);
% LabelMap = imopen(LabelMap,se);
end

%%
function features=blob(img)
%% input bwlabel image
% BLOB Analysis
% All the features should be scalar
feature = regionprops(img,'Area','MajoraxisLength','MinoraxisLength','Eccentricity','Orientation','FilledArea','EquivDiameter','Solidity','Extent');
if size([feature.Area])
    features(:,1) = [feature.Area];
    features(:,2) = [feature.MajorAxisLength];
    features(:,3) = [feature.MinorAxisLength];
    features(:,4) = [feature.Eccentricity];
    features(:,5) = [feature.Orientation];
    features(:,6) = [feature.FilledArea];
    features(:,7) = [feature.EquivDiameter];
    features(:,8) = [feature.Solidity];
    features(:,9) = [feature.Extent];
    bo = regionprops(img,'BoundingBox');
    b = [bo.BoundingBox];
    for i = 1:size(features,1)
        b = bo(i).BoundingBox;
        features(i,10) = b(4)/b(3);  % ratio: delta_y/delta_x
    end
    %%
    % add the result of feature selection
%     features = features(:,[1,4,8,9,10]);
else
    features = zeros(1,10);
end
end
%%
function label_map = box(img,scale,width)
% create a virtual box to fit the real edge of the sign
% because the sign is quadrilateral
% restore the segment to a quadrilateral area (virtual box)
% cov: points of the convex polygon that can contain the region
% bo: points of the rectangle that can contain the region
% label_map: the virtual box
[r, c] = size(img);
% imshow(img);
bo = regionprops(img,'BoundingBox');
area = regionprops(img,'Area');
area = [area.Area];
bo = [bo.BoundingBox];
%% find edge of the sign
% we don't use sobel here because sobel will also extract the edge of the
% characters in the sign
% use vertical and horizonal scan to extract the first and last not-zero points of
% each row or column
cox = [];
coy = [];
edge = zeros(r,c);
% vertical scan
for i = 1:r
    seg = img(i,:);
    idx = find(seg==1);
    if sum(seg)~=0
        cox = [cox;i];
        coy = [coy;idx(1)];
        cox = [cox;i];
        coy = [coy;idx(size(idx,2))];
    end
end
% horizonal scan
for i = 1:c
    seg = img(:,i);
    idx = find(seg==1);
    if sum(seg)~=0
        cox = [cox;idx(1)];
        coy = [coy;i];
        cox = [cox;idx(size(idx,2))];
        coy = [coy;i];
    end
end
cov = [cox,coy];% note that the plot frame is different from the imshow
for i = 1:size(cox,1)
    edge(cox(i),coy(i))=1;
end
% figure;
% imshow(edge);

y = [bo(1),bo(1),bo(1)+bo(3),bo(1)+bo(3)];
x = [bo(2),bo(2)+bo(4),bo(2),bo(2)+bo(4)];

% figure;
% plot(cov(:,1),cov(:,2),'.g');
% hold on;
% plot(x,y,'.r');

%% slope and distance
% slope is the slope of the line between the bo points and edge points
% distance is the distance between each bo points and each points
for i = 1:4
    delta_x = cov(:,1)-x(i);
    delta_y = cov(:,2)-y(i);
    dis(i,:) = sqrt(delta_y.^2+delta_x.^2);
    if find(delta_x==0)
        delta_x(find(delta_x==0)) = delta_x(find(delta_x==0))+0.01; % avoid infinite slope
    end
    slope(i,:) = abs(delta_y./delta_x);
end

%% define an index as a criteria
% the results are called target points, which are the candidates of the
% quadrilateral points
tar_p = zeros(4,2);
tar_idx = zeros(4,1);
dist = zeros(4,1);
for i = 1:4
    % vertical
    % normalization
    new_slope = norm(slope(i,:));
    new_dist = norm(dis(i,:));
    index = new_slope./new_dist;
    [~,idx1] = max(index);
    % horizonal
    % normalization
    new_slope = norm(slope(i,:));
    index = new_slope.*new_dist;
    [~,idx2] = min(index);
    index = [idx1,idx2];
    [~, idx] = min(new_dist(index));
    idx = index(idx);
    tar_p(i,1) = cov(idx,1);
    tar_p(i,2) = cov(idx,2);% store the candidates
    tar_idx(i) = idx;
    %     hold on;
    %     plot(cov(idx,1),cov(idx,2),'.b','Linewidth',15);
end

%% check if the points are "outliers"
% when a part of the corner of the sign is blocked by something, such as
% the leaves, the target points
% the trick is the distance between the target point and the corresponding bo points,
% if the distance is considered to be "different" from other points, it
% will be regard as the "outlier"
% there should be 5 cases, 0,1,2,3,4 point(s) wrong
% However, we must assume that there is at least two lines(three points)
% right, or we can't know the real size of the quadrilateral area
% In other words, we only consider the case that there is no more than 1
% wrong point
%% The selection is based on the standard deviation and mean
mx = mean(x);
my = mean(y);
for i = 1:4
    dist(i) = sqrt((tar_p(i,1)-mx)^2+(tar_p(i,2)-my)^2);
end
m = mean(dist);
st = std(dist);
if find(dist<=(m-st))% note that for the dist, the smaller the better
%     disp('wrong point! Fixing...');
    [~,idx] = min(dist); % point(idx) is the wrong point
    %% assume the area to be a parallelogram
    %     switch(idx)
    %         case 1
    %             delta_x = tar_p(4,1)-tar_p(2,1);
    %             delta_y = tar_p(4,2)-tar_p(2,2);
    %             tar_p(1,1) = tar_p(3,1)-delta_x;
    %             tar_p(1,2) = tar_p(3,2)-delta_y;
    %         case 2
    %             delta_x = tar_p(3,1)-tar_p(1,1);
    %             delta_y = tar_p(3,2)-tar_p(1,2);
    %             tar_p(2,1) = tar_p(4,1)-delta_x;
    %             tar_p(2,2) = tar_p(4,2)-delta_y;
    %         case 3
    %             delta_x = tar_p(4,1)-tar_p(2,1);
    %             delta_y = tar_p(4,2)-tar_p(2,2);
    %             tar_p(3,1) = tar_p(1,1)+delta_x;
    %             tar_p(3,2) = tar_p(1,2)+delta_y;
    %         case 4
    %             delta_x = tar_p(3,1)-tar_p(1,1);
    %             delta_y = tar_p(3,2)-tar_p(1,2);
    %             tar_p(4,1) = tar_p(2,1)+delta_x;
    %             tar_p(4,2) = tar_p(2,2)+delta_y;
    %     end
    
    %% polyfit
    switch(idx)
        case 1
            [tar_p(1,1),tar_p(1,2)] = fit_it(1,2,3,scale,width,bo,tar_p,cox,coy);
        case 2
            [tar_p(2,1),tar_p(2,2)] = fit_it(2,1,4,scale,width,bo,tar_p,cox,coy);
        case 3
            [tar_p(3,1),tar_p(3,2)] = fit_it(3,4,1,scale,width,bo,tar_p,cox,coy);
        case 4
            [tar_p(4,1),tar_p(4,2)] = fit_it(4,3,2,scale,width,bo,tar_p,cox,coy);
    end
%     hold on;
%     plot(tar_p(idx,1),tar_p(idx,2),'.k','Linewidth',15);
end
%% fill in the blanks
% h_map = zeros(r,c);
% v_map = zeros(r,c);
%% roipoly
p = tar_p;
% point 3 and 4 should swap
p(3,1) = tar_p(4,1);
p(3,2) = tar_p(4,2);
p(4,1) = tar_p(3,1);
p(4,2) = tar_p(3,2);
label_map = uint8(roipoly(img,p(:,2),p(:,1)));
%% line and point
% [x1,y1,k1,b1] = gen_line(tar_p(1,1),tar_p(1,2),tar_p(2,1),tar_p(2,2));%bottom->left
% [x2,y2,k2,b2] = gen_line(tar_p(2,1),tar_p(2,2),tar_p(4,1),tar_p(4,2));%right->bottom
% [x3,y3,k3,b3] = gen_line(tar_p(3,1),tar_p(3,2),tar_p(4,1),tar_p(4,2));%top->right
% [x4,y4,k4,b4] = gen_line(tar_p(1,1),tar_p(1,2),tar_p(3,1),tar_p(3,2));%left->top
% hold on;
% plot(x1,y1,'-k');
% hold on;
% plot(x2,y2,'-k');
% hold on;
% plot(x3,y3,'-k');
% hold on;
% plot(x4,y4,'-k');

% scan vertically
% the points must be between the left and right lines
% for i = 1:r
%     j1 = floor(k1*i+b1);
%     j2 = ceil(k3*i+b3);
%     if j1>0 & j2<c & j1<c & j2>0
%         v_map(i,j1:j2)=1;
%     end
% end
% scan horizontally
% the points must be between the top and bottom lines
% for j = 1:c
%     i1 = floor((j-b4)/k4);
%     i2 = ceil((j-b2)/k2);
%     if i1>0 & i2<r & i1<r & i2>0
%         h_map(i1:i2,j)=1;
%     end
% end
% figure;
% imshow(h_map);
% figure;
% imshow(v_map);
% label_map = uint8(h_map&v_map);
% figure;
% imagesc(label_map);
end


function norm_array = norm(array)
norm_array = (array-min(array))./(max(array)-min(array))+0.01;
end

% function [x,y,k,b] = gen_line(x1,y1,x2,y2)
% % generate a line linking two points (x1,y1) (x2,y2)
% k = (y2-y1)/(x2-x1);%slope
% b = y1-k*x1;%distance
% if abs(k)>1
%     % use y-axis as index
%     x = [];
%     y = y1:y2;
%     for i = y1:y2
%         x = [x,(i-b)/k];
%     end
% else
%     % use x-axis as index
%     y = [];
%     x = x1:x2;
%     for i = x1:x2
%         y = [y,k*i+b];
%     end
% end
% end

function p = rectangle(x1,x2,y1,y2,cox,coy)
% fit the points in a rectangle to a poly
% the line [(x1,y1);(x2,y2)] is the diagonal of the rectangle
x3 = min(x1,x2);
x4 = max(x1,x2);
y3 = min(y1,y2);
y4 = max(y1,y2);
s = size(cox,1);
co1 = [];
co2 = [];
for i = 1:s
    if cox(i)>=x3 & cox(i)<=x4 & coy(i)>=y3 & coy(i)<=y4
        co1 = [co1;cox(i)];
        co2 = [co2;coy(i)];
    end
end
p = polyfit(co1,co2,1);
t = [x3:x4];
y = polyval(p,t);
% figure;
% plot(t,y);
end
function [x,y] = fit_it(flag1,flag2,flag3,scale,width,bo,tar_p,cox,coy)
%% flag1-3 means the target point, the vertical and horizonal line
delta_x = tar_p(flag1,1)-tar_p(flag2,1);
delta_y = tar_p(flag1,2)-tar_p(flag3,2);
delta_x = delta_x/scale;
delta_y = delta_y/scale;
p1 = rectangle(tar_p(flag2,1),tar_p(flag2,1)+delta_x,tar_p(flag2,2)-width,tar_p(flag2,2)+width,cox,coy);
p2 = rectangle(tar_p(flag3,1)-width,tar_p(flag3,1)+width,tar_p(flag3,2),tar_p(flag3,2)+delta_y,cox,coy);
%             y1 = polyval(p1,[tar_p(flag2,1):tar_p(flag2,1)+delta_x]);
%             hold on;
%             plot([tar_p(flag2,1):tar_p(flag2,1)+delta_x],y1,'-b','linewidth',1);
%             y1 = polyval(p2,[tar_p(flag3,1)-width:tar_p(flag3,1)+width]);
%             hold on;
%             plot([tar_p(flag3,1)-3:tar_p(flag3,1)+3],y1,'-b','linewidth',1);
f1 = @(x) p1(1)*x+p1(2);
f2 = @(x) p2(1)*x+p2(2);
f = @(x) f2(x)-f1(x);
a = fsolve(f,[bo(1),bo(1)+bo(3)]);
x = round(a(1));
y = polyval(p1,tar_p(flag1,1));
end

%%
function loss = cal_loss(I,map,limit1,limit2)
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);
im1 = r>=limit1(1)&r<=limit2(1);
im2 = g>=limit1(2)&g<=limit2(2);
im3 = b>=limit1(3)&b<=limit2(3);
for i = 1:max(max(map))
    map(find(map==i))=map(find(map==i))/i;
end
loss(1) = sum(sum(abs(uint8(im1)-map))); % shrink range if positive, expand range if negative
loss(2) = sum(sum(abs(uint8(im2)-map)));
loss(3) = sum(sum(abs(uint8(im3)-map)));
I1 = rgb2hsv(I);
I1 = double(I1);
h = I1(:,:,1);
s = I1(:,:,2);
v = I1(:,:,3);
im4 = h>=limit1(4)&h<=limit2(4);
im5 = s>=limit1(5)&s<=limit2(5);
im6 = v>=limit1(6)&v<=limit2(6);
loss(4) = sum(sum(abs(double(im4)-double(map))));
loss(5) = sum(sum(abs(double(im5)-double(map))));
loss(6) = sum(sum(abs(double(im6)-double(map))));
end
%%
function [m,s,limit1,limit2] = color(img)
% extract the RGB color features of a picture
img = double(img);
img_r = img(:,:,1);
img_g = img(:,:,2);
img_b = img(:,:,3);
% [mr,str,lr,hr] = hist_ana(img_r(find(img_r~=0)));
% [mg,stg,lg,hg] = hist_ana(img_g(find(img_g~=0)));
% [mb,stb,lb,hb] = hist_ana(img_b(find(img_b~=0)));
[mr,str,lr,hr] = layer_ana(img_r(find(img_r~=0)));
[mg,stg,lg,hg] = layer_ana(img_g(find(img_g~=0)));
[mb,stb,lb,hb] = layer_ana(img_b(find(img_b~=0)));
limit1 = [lr,lg,lb];
limit2 = [hr,hg,hb];
m = [mr,mg,mb];
s = [str,stg,stb];
end

function [m, st, low_lim, high_lim] = hist_ana(layer)
%% color features of a layer
% m:mean st:standard deviation
% low_lim: lower limit high_lim: higher limit
%%
% preprocessing
% remove the white words in red sign
% there should be serval peaks in the hist, remove
% the last main peak(color white)
% find peak
h = imhist(uint8(layer));
% figure;
% plot(h);
k = 1; % filter scale
y = medfilt1(h,k); % midian filter
peak = [];
for i = 2:255
    if (y(i-1)<y(i) & y(i+1)<y(i))
        peak = [peak, i]; % find and save peak
    end
end
y1 = y(peak);
peak1 = [];
for i = 2:size(peak,2)-1
    if (y1(i-1)<y1(i) & y1(i+1)<y1(i))
        peak1 = [peak1, i]; % find and save main peak
    end
end
peak2 = peak(peak1); % peak2 is the main peak(peak of peak)
% remove white words
end_peak = peak2(size(peak2,2));
if end_peak == peak(size(peak,2))
    d = 256-end_peak; % assign the distance
    h(end_peak-d:256) = 0;
else
    [~, d] = min(h(end_peak:peak(size(peak,2))));% the local min between the last two peaks
    h(end_peak-d:256) = 0;
end
layer(find(layer>=end_peak&layer<=256))=0;
% m = mean(layer);
% st = std(layer);
% layer = layer(find(layer>=(m-3*st)&layer<=(m+3*st)));
m = mean(layer);
st = std(layer);
% figure;
% plot(h);
% extract the color features
% m = [0:255]*h/sum(h);
% st = 0;
% for i = 1:256
%     st = st+((i-m)^2)*h(i);
% end
% st = st/sum(h);
% st = sqrt(st);
low_lim = m-2*st;
high_lim = m+2*st;
end

function [m, st, low_lim, high_lim] = layer_ana(layer)
layer = layer(find(layer~=0));
m = mean(layer);
st = std(layer);
layer = layer(find(layer>=(m-3*st)&layer<=(m+3*st)));
% feature extraction
m = mean(layer);
st = std(layer);
low_lim = m-3*st;
high_lim = m+3*st;
end
%%
function img = fill_it(im)
se = strel('square',15);
im1 = imclose(im,se);
im2 = regionprops(im1,'FilledImage');
bo = regionprops(im1,'BoundingBox');
bo = bo.BoundingBox;
x = [bo(1),bo(1)+bo(3)];
y = [bo(2),bo(2)+bo(4)];
im3 = im2.FilledImage;
img = im;
img(y(1):(y(2)-1),x(1):(x(2)-1))=im3;
end
%%
function [m,s,limit1,limit2] = hsv(img)
% extract the hsv features of a picture
img_h = img(:,:,1);
img_s = img(:,:,2);
img_v = img(:,:,3);
% subplot(3,1,1)
% imshow(img_s);
% subplot(3,1,2)
% imshow(img_h);
% subplot(3,1,3)
% imshow(img_v);
[mh,sth,lh,hh] = hsv_ana(img_h(find(img_h~=0)));
[ms,sts,ls,hs] = hsv_ana(img_s(find(img_s~=0)));
[mv,stv,lv,hv] = hsv_ana(img_v(find(img_v~=0)));
limit1 = [lh,ls,lv];
limit2 = [hh,hs,hv];
m = [mh,ms,mv];
s = [sth,sts,stv];
end

function [m, st, low_lim, high_lim] = hsv_ana(layer)
%% hsv features of a layer
% m:mean st:standard deviation
% low_lim: lower limit high_lim: higher limit
%%
% preprocessing based on PauTa Criterion(3-Sigma Criterion)
% remove outliers
layer = layer(find(layer~=0));
m = mean(layer);
st = std(layer);
layer = layer(find(layer>=(m-3*st)&layer<=(m+3*st)));
% feature extraction
m = mean(layer);
st = std(layer);
low_lim = m-3*st;
high_lim = m+3*st;
end



%%
function [m,s,limit1,limit2] = lab(img)
% extract the hsv features of a picture
img_h = img(:,:,1);
img_s = img(:,:,2);
img_v = img(:,:,3);
% subplot(3,1,1)
% imshow(img_s);
% subplot(3,1,2)
% imshow(img_h);
% subplot(3,1,3)
% imshow(img_v);
[mh,sth,lh,hh] = hsv_ana(img_h(find(img_h~=0)));
[ms,sts,ls,hs] = hsv_ana(img_s(find(img_s~=0)));
[mv,stv,lv,hv] = hsv_ana(img_v(find(img_v~=0)));
limit1 = [lh,ls,lv];
limit2 = [hh,hs,hv];
m = [mh,ms,mv];
s = [sth,sts,stv];
end


%%
function [img,label] = read_img(i)
%% read image and its label from the folder DTUSignPhotos
if i<10
    idx = '0'+string(i);
else
    idx = string(i);
end
img = imread("DTUSignPhotos\DTUSigns0"+idx+".jpg");
label = dlmread("DTUSignPhotos\DTUSigns0"+idx+".txt");
end

%%
function indx = sel_feature
%% feature selection
% select 5 features from 10 predefined features
% idx is the index of selected features
features = [];
for i = 1:2:67
    [img,label] = read_img(i);
    %imshow(img);
    Map = CreateLabelMapFromAnnotations(img,label);
    feature = blob(Map);
    features = [features;feature];
end
for i = 1:size(features,2)
   st(i) = std(features(:,i));
end
[~,indx] = sort(st);
%idx = indx(1:4);
end
function C = multi(A,B)
for i = 1:3
    C(:,:,i) = double(A(:,:,i)).*double(B);
end
end