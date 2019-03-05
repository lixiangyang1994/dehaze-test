clear

clc

close all

kenlRatio = .01;
maxAtomsLight = 240;
w0 = 0.95;   %调节因子
% image_name =  'test images\21.bmp';
image_name =  'E:\论文\picture\frog\haze1.jpg';
img=imread(image_name);
subplot(221)
imshow(uint8(img));
title('雾化图像');

[h w c]=size(img);



min_I = zeros(h,w);

for y=1:h

    for x=1:w

        min_I(y,x) = min(img(y,x,:));

    end

end

% subplot(322)
% imshow(uint8(min_I));  
% title('Min(R,G,B)');

krnlsz = floor(max([3, w*kenlRatio, h*kenlRatio]))

dc2 = minfilt2(min_I, [krnlsz,krnlsz]);   %区域最小值
dc2(h,w)=min_I(h,w);
subplot(222)
imshow(uint8(dc2)); 
title('暗原色');

%求大气光值
% A = min([minAtomsLight, max(max(dc2))])
imsize = h * w ;
numpx = floor(imsize/1000);
Jdarkvec = reshape(min_I,imsize,1);
Imvec = reshape(double(img),imsize,3);
[Jdarkvec,indices] = sort(Jdarkvec);
indices = indices(imsize-numpx+1:end);
% atmSum = zeros(1,3);
% for ind = 1:numpx
%     atmSum = atmSum + Imvec(indices(ind),:);
% end
% A = atmSum(1,1) / numpx ;
A = min([maxAtomsLight, max(max(Imvec(indices,:)))])

r_ij = zeros(h,w);   %晕光估计算子
r_ij = min_I./dc2;
S_ij = 1./r_ij;

t = (255 - dc2)/A;  %透射率
% for i=1:h
% 
%     for j=1:w
%       if(r_ij(i,j)==1)
%          t(i,j) = 1 - w0*(1-S_ij).*min_I/A + S_ij.*dc2/A;
%       else          
%          t(i,j) = 1 - w0*dc2/A;
%       end
%     end
% 
% end
% subplot(324)
% imshow(t);
% title('粗略透射率');

% t_d=double(t)/255;
% 
% sum(sum(t_d))/(h*w)
% 


J = zeros(h,w,3);

img_d = double(img);

% J(:,:,1) = (img_d(:,:,1) - (1-t)*A)./t;
% 
% J(:,:,2) = (img_d(:,:,2) - (1-t)*A)./t;
% 
% J(:,:,3) = (img_d(:,:,3) - (1-t)*A)./t;

% subplot(325)
% imshow(uint8(J));
% title('粗略无雾图');
% figure,imshow(rgb2gray(uint8(abs(J-img_d)))), title('J-img_d');
% a = sum(sum(rgb2gray(uint8(abs(J-img_d))))) / (h*w)
% return;
%----------------------------------
r = krnlsz*4
eps = 10^-6;

% filtered = guidedfilter_color(double(img)/255, t_d, r, eps);
filtered = guidedfilter(double(rgb2gray(img))/255, t, r, eps);
t0 = 0.1;
t_d = max(filtered,t0);

subplot(223)
imshow(t_d,[]);
title('精细透射率');

J(:,:,1) = (img_d(:,:,1) - (1-t_d)*A)./t_d;

J(:,:,2) = (img_d(:,:,2) - (1-t_d)*A)./t_d;

J(:,:,3) = (img_d(:,:,3) - (1-t_d)*A)./t_d;
% 

img_d(1,3,1)
imwrite(uint8(J),'.\12.bmp');
figure,imshow(uint8(J)),title('去雾图像');

%----------------------------------
%imwrite(uint8(J), ['_', image_name])