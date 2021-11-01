clear all
clc
F=[];
for i =1:50
    str=['image/img',num2str(i),'.jpg'];
    I=imread(str);
    mysize=size(I);
    if mysize(end)==3
        I=rgb2gray(I);
    elseif mysize(end)==4
        I=I(:,:,1:3);
        I=rgb2gray(I);
    end
    
    F_tmp=double(I)/255;%
    F_tmp=F_tmp(:);
    F=[F F_tmp];
end

run ./gspbox/gsp_start
G=gsp_2dgrid(size(I,1));
A=full(G.W);
A=A+A';
A(A>0)=1;

mask=zeros(size(I,1),size(I,1));
mask(3:end-2,3:end-2)=1;
mask=mask(:);

save 2Dgrid A F mask

subplot(2,2,1);imagesc(reshape(F(:,6),[100 100]));axis equal
title('given images');
subplot(2,2,2);imagesc(reshape(F(:,20),[100 100]));axis equal
title('given images');
subplot(2,2,3);imagesc(reshape(F(:,30),[100 100]));axis equal
title('band-pass images');
subplot(2,2,4);imagesc(reshape(F(:,40),[100 100]));axis equal
title('low-pass images');