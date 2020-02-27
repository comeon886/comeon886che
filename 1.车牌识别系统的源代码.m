[filename,pathname] = uigetfile('*.jpg','��ȡͼƬ');
if isequal(filename,0)
    msgbox('û��ͼƬ')
else
    pathfile=fullfile(pathname,filename);
    I=imread(pathfile);
end
I1=rgb2gray(I);
figure(2),subplot(1,2,1),imshow(I1);title('�Ҷ�ͼ');
figure(2),subplot(1,2,2),imhist(I1);title('�Ҷ�ͼֱ��ͼ');
I2=edge(I1,'sobel',0.15,'both');
figure(3),imshow(I2);title('���ӱ�Ե���');
se=[1;1;1];
I3=imerode(I2,se);
figure(4),imshow(I3);title('��ʴ��ͼ��');
se=strel('rectangle',[25,25]);
I4=imclose(I3,se);
figure(5),imshow(I4);title('ƽ��ͼ�������');
I5=bwareaopen(I4,2000);
figure(6),imshow(I5);title('�Ӷ������Ƴ�С����');
[y,x,z]=size(I5);
myI=double(I5);
tic
 Blue_y=zeros(y,1);
 for i=1:y
    for j=1:x
             if(myI(i,j,1)==1) 
  
                Blue_y(i,1)= Blue_y(i,1)+1;%��ɫ���ص�ͳ�� 
            end  
     end       
 end
 [temp MaxY]=max(Blue_y);%Y����������ȷ��
 PY1=MaxY;
 while ((Blue_y(PY1,1)>=5)&&(PY1>1))
        PY1=PY1-1;
 end    
 PY2=MaxY;
 while ((Blue_y(PY2,1)>=5)&&(PY2<y))
        PY2=PY2+1;
 end
 IY=I(PY1:PY2,:,:);
 %%%%%% X���� %%%%%%%%%
 Blue_x=zeros(1,x);%��һ��ȷ��x����ĳ�������
 for j=1:x
     for i=PY1:PY2
            if(myI(i,j,1)==1)
                Blue_x(1,j)= Blue_x(1,j)+1;               
            end  
     end       
 end
  
 PX1=1;
 while ((Blue_x(1,PX1)<3)&&(PX1<x))
       PX1=PX1+1;
 end    
 PX2=x;
 while ((Blue_x(1,PX2)<3)&&(PX2>PX1))
        PX2=PX2-1;
 end
 PX1=PX1-1;%�Գ��������У��
 PX2=PX2+1;
  dw=I(PY1:PY2-8,PX1:PX2,:);
 G=toc; 
figure(7),subplot(1,2,1),imshow(IY),title('�з����������');
figure(7),subplot(1,2,2),imshow(dw),title('��λ���к�Ĳ�ɫ����ͼ��');
imwrite(dw,'dw.jpg');
a=imread('dw.jpg');
b=rgb2gray(a);%�����ǽ����ɫͼ��ת��Ϊ�Ҷ�ͼ�񣬼��ҶȻ�����
imwrite(b,'1.���ƻҶ�ͼ��.jpg');
figure(8);subplot(3,2,1),imshow(b),title('1.���ƻҶ�ͼ��')
g_max=double(max(max(b)));
g_min=double(min(min(b)));
T=round(g_max-(g_max-g_min)/3); % T Ϊ��ֵ������ֵ   ������ķ���ȡ��
[m,n]=size(b);
d=(double(b)>=T);  % d:��ֵͼ��
imwrite(d,'2.���ƶ�ֵͼ��.jpg');
figure(8);subplot(3,2,2),imshow(d),title('2.���ƶ�ֵͼ��')
figure(8),subplot(3,2,3),imshow(d),title('3.��ֵ�˲�ǰ')
% ��ֵ�˲�����
h=fspecial('average',3);
d=im2bw(round(filter2(h,d)));%filter2(B,X),BΪ�˲���.XΪҪ�˲�������,���ｫB����X��,һ��һ���ƶ�����ģ���˲�. 
imwrite(d,'4.��ֵ�˲���.jpg');
figure(8),subplot(3,2,4),imshow(d),title('4.��ֵ�˲���')
se=eye(2);%����m��n�ĵ�λ����
[m,n]=size(d);
if bwarea(d)/m/n>=0.365 %bwarea�Ǽ����ֵͼ���ж����������ĺ���
    d=imerode(d,se);%��ʴ
elseif bwarea(d)/m/n<=0.235
    d=imdilate(d,se);%����
end
imwrite(d,'5.���ͻ�ʴ�����.jpg');
figure(8),subplot(3,2,5),imshow(d),title('5.���ͻ�ʴ�����')
H=toc;
d=qiege(d);
figure,subplot(2,1,1),imshow(d),title(n)
[word1,d]=getword(d);
% �ָ���ڶ����ַ�
[word2,d]=getword(d);
% �ָ���������ַ�
[word3,d]=getword(d);
% �ָ�����ĸ��ַ�
[word4,d]=getword(d);
% �ָ��������ַ�
[word5,d]=getword(d);
% �ָ���������ַ�
[word6,d]=getword(d);
% �ָ�����߸��ַ�
[word7,d]=getword(d);
subplot(5,7,1),imshow(word1),title('1');
subplot(5,7,2),imshow(word2),title('2');
subplot(5,7,3),imshow(word3),title('3');
subplot(5,7,4),imshow(word4),title('4');
subplot(5,7,5),imshow(word5),title('5');
subplot(5,7,6),imshow(word6),title('6');
subplot(5,7,7),imshow(word7),title('7');
end

figure,subplot(2,1,1),imshow(d),title(n)
[word1,d]=getword(d);
% �ָ���ڶ����ַ�
[word2,d]=getword(d);
% �ָ���������ַ�
[word3,d]=getword(d);
% �ָ�����ĸ��ַ�
[word4,d]=getword(d);
% �ָ��������ַ�
[word5,d]=getword(d);
% �ָ���������ַ�
[word6,d]=getword(d);
% �ָ�����߸��ַ�
[word7,d]=getword(d);
subplot(5,7,1),imshow(word1),title('1');
subplot(5,7,2),imshow(word2),title('2');
subplot(5,7,3),imshow(word3),title('3');
subplot(5,7,4),imshow(word4),title('4');
subplot(5,7,5),imshow(word5),title('5');
subplot(5,7,6),imshow(word6),title('6');
subplot(5,7,7),imshow(word7),title('7');
end

function [word,result]=getword(d)

word=[];flag=0;y1=8;y2=0.5;

    while flag==0

        [m,n]=size(d);

        wide=0;

        while sum(d(:,wide+1))~=0 && wide<=n-2

            wide=wide+1;

        end

        temp=qiege(imcrop(d,[1 1 wide m]));%����ͼ���һ���и����� 

        [m1,n1]=size(temp);

        if wide<y1 && n1/m1>y2

            d(:,[1:wide])=0;

            if sum(sum(d))~=0

                d=qiege(d);  % �и����С��Χ

            else word=[];flag=1;

            end

        else

            word=qiege(imcrop(d,[1 1 wide m]));

            d(:,[1:wide])=0;

            if sum(sum(d))~=0;

                d=qiege(d);flag=1;

            else d=[];

            end

        end

    end

%end

          result=d;
end

function e=qiege(d)

[m,n]=size(d);

top=1;bottom=m;left=1;right=n;   % init

while sum(d(top,:))==0 && top<=m

    top=top+1;

end

while sum(d(bottom,:))==0 && bottom>=1

    bottom=bottom-1;

end

while sum(d(:,left))==0 && left<=n

    left=left+1;

end

while sum(d(:,right))==0 && right>=1

    right=right-1;
end
dd=right-left;

hh=bottom-top;

e=imcrop(d,[left top dd hh]);%�ú������ڷ���ͼ���һ���ü�����
[m,n]=size(word1);
% ��һ����СΪ 40*20
word1=imresize(word1,[40 20]);
word2=imresize(word2,[40 20]);
word3=imresize(word3,[40 20]);
word4=imresize(word4,[40 20]);
word5=imresize(word5,[40 20]);
word6=imresize(word6,[40 20]);
word7=imresize(word7,[40 20]);
subplot(5,7,15),imshow(word1),title('1');
subplot(5,7,16),imshow(word2),title('2');
subplot(5,7,17),imshow(word3),title('3')
subplot(5,7,18),imshow(word4),title('4');
subplot(5,7,19),imshow(word5),title('5');
subplot(5,7,20),imshow(word6),title('6');
subplot(5,7,21),imshow(word7),title('7');
imwrite(word1,'1.jpg');
imwrite(word2,'2.jpg');
imwrite(word3,'3.jpg');
imwrite(word4,'4.jpg');
imwrite(word5,'5.jpg');
imwrite(word6,'6.jpg');
imwrite(word7,'7.jpg');
liccode=char(['0':'9' 'A':'Z' '���򻦽��ɼ�³������³ԥ�������º���']); 
JG=zeros(40,20);%����һ��40*20��С�������
l=1;
L=toc;
for I=1:7 %IΪ��ʶ����ַ�λ
       ii=int2str(I);%��������ת��Ϊ�ַ�������
      t=imread([ii,'.jpg']);
       MB=imresize(t,[40 20],'nearest');%���Ŵ���
         if l==1         %���ƺŵ�һλΪ����ʶ��ʹ��37-53��������
             kmin=37;
             kmax=53;
         elseif l==2        %��Ʊ�ŵڶ�λΪ A~Z ��д��ĸʶ��ʹ��11-36��������
             kmin=11;
             kmax=36;
         else l>=3;              %����λ�Ժ�����ĸ������ʶ��ʹ��1-36��������
             kmin=1; 
            kmax=36;        
         end 
        for k2=kmin:kmax      
            fname=strcat('������\',liccode(k2),'.bmp');
             YB = imread(fname); %����������ͼ���ļ� 
             for  i=1:40
                 for j=1:20
  JG(i,j)=MB(i,j)-YB(i,j); % �����ǽ���ʶ��ͼ����ģ��ͼ������ͼ����õ�������ͼ
                 end
             end
            
            Dmax=0;
 
            for k1=1:40
 
                for l1=1:20
 
                    if  ( JG(k1,l1) > 0 || JG(k1,l1) <0 )
 
                        Dmax=Dmax+1;
 
                    end
 
                end
 
            end
 
            Error(k2)=Dmax;
 
        end
 
        Error1=Error(kmin:kmax);
 
        MinError=min(Error1);
 
        findc=find(Error1==MinError);
 
        Code(l*2-1)=liccode(findc(1)+kmin-1);
 
        Code(l*2)=' ';
 
        l=l+1;
 
end
t=toc;
figure(10),imshow(dw),title (['���ƺ�:', Code],'Color','k' ,'font','20');

