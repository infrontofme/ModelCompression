clc;
clear;
disp('start prepocessing data...')
% load data
block_0101_expand = load("block_0101_expand.txt",'r');
block_0101_expand_mask = load("block_0101_expand_mask.txt",'r');
block_0101_project = load("block_0101_project.txt",'r');
block_0101_project_mask = load("block_0101_project_mask.txt",'r');
for i=1:size(block_0101_expand, 2)
    weight01 = block_0101_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0101_expand_mask(:,i)==1); %(24,32)
end
block_0101_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0101_project, 2)
    weight01 = block_0101_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0101_project_mask(:,i)==1); %(24,16)
end
block_0101_project = weight02;
clear weight01;
clear weight02;

block_0201_expand = load("block_0201_expand.txt",'r');
block_0201_expand_mask = load("block_0201_expand_mask.txt",'r');
block_0201_project = load("block_0201_project.txt",'r');
block_0201_project_mask = load("block_0201_project_mask.txt",'r');
for i=1:size(block_0201_expand, 2)
    weight01 = block_0201_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0201_expand_mask(:,i)==1); %(24,32)
end
block_0201_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0201_project, 2)
    weight01 = block_0201_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0201_project_mask(:,i)==1); %(24,16)
end
block_0201_project = weight02;
clear weight01;
clear weight02;

block_0202_expand = load("block_0202_expand.txt",'r');
block_0202_expand_mask = load("block_0202_expand_mask.txt",'r');
block_0202_project = load("block_0202_project.txt",'r');
block_0202_project_mask = load("block_0202_project_mask.txt",'r');
for i=1:size(block_0202_expand, 2)
    weight01 = block_0202_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0202_expand_mask(:,i)==1); %(24,32)
end
block_0202_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0202_project, 2)
    weight01 = block_0202_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0202_project_mask(:,i)==1); %(24,16)
end
block_0202_project = weight02;
clear weight01;
clear weight02;

block_0301_expand = load("block_0301_expand.txt",'r');
block_0301_expand_mask = load("block_0301_expand_mask.txt",'r');
block_0301_project = load("block_0301_project.txt",'r');
block_0301_project_mask = load("block_0301_project_mask.txt",'r');
for i=1:size(block_0301_expand, 2)
    weight01 = block_0301_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0301_expand_mask(:,i)==1); %(24,32)
end
block_0301_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0301_project, 2)
    weight01 = block_0301_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0301_project_mask(:,i)==1); %(24,16)
end
block_0301_project = weight02;
clear weight01;
clear weight02;

block_0302_expand = load("block_0302_expand.txt",'r');
block_0302_expand_mask = load("block_0302_expand_mask.txt",'r');
block_0302_project = load("block_0302_project.txt",'r');
block_0302_project_mask = load("block_0302_project_mask.txt",'r');
for i=1:size(block_0302_expand, 2)
    weight01 = block_0302_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0302_expand_mask(:,i)==1); %(24,32)
end
block_0302_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0302_project, 2)
    weight01 = block_0302_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0302_project_mask(:,i)==1); %(24,16)
end
block_0302_project = weight02;
clear weight01;
clear weight02;

block_0303_expand = load("block_0303_expand.txt",'r');
block_0303_expand_mask = load("block_0303_expand_mask.txt",'r');
block_0303_project = load("block_0303_project.txt",'r');
block_0303_project_mask = load("block_0303_project_mask.txt",'r');
for i=1:size(block_0303_expand, 2)
    weight01 = block_0303_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0303_expand_mask(:,i)==1); %(24,32)
end
block_0303_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0303_project, 2)
    weight01 = block_0303_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0303_project_mask(:,i)==1); %(24,16)
end
block_0303_project = weight02;
clear weight01;
clear weight02;

block_0401_expand = load("block_0401_expand.txt",'r');
block_0401_expand_mask = load("block_0401_expand_mask.txt",'r');
block_0401_project = load("block_0401_project.txt",'r');
block_0401_project_mask = load("block_0401_project_mask.txt",'r');
for i=1:size(block_0401_expand, 2)
    weight01 = block_0401_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0401_expand_mask(:,i)==1); %(24,32)
end
block_0401_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0401_project, 2)
    weight01 = block_0401_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0401_project_mask(:,i)==1); %(24,16)
end
block_0401_project = weight02;
clear weight01;
clear weight02;

block_0402_expand = load("block_0402_expand.txt",'r');
block_0402_expand_mask = load("block_0402_expand_mask.txt",'r');
block_0402_project = load("block_0402_project.txt",'r');
block_0402_project_mask = load("block_0402_project_mask.txt",'r');
for i=1:size(block_0402_expand, 2)
    weight01 = block_0402_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0402_expand_mask(:,i)==1); %(24,32)
end
block_0402_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0402_project, 2)
    weight01 = block_0402_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0402_project_mask(:,i)==1); %(24,16)
end
block_0402_project = weight02;
clear weight01;
clear weight02;

block_0403_expand = load("block_0403_expand.txt",'r');
block_0403_expand_mask = load("block_0403_expand_mask.txt",'r');
block_0403_project = load("block_0403_project.txt",'r');
block_0403_project_mask = load("block_0403_project_mask.txt",'r');
for i=1:size(block_0403_expand, 2)
    weight01 = block_0403_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0403_expand_mask(:,i)==1); %(24,32)
end
block_0403_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0403_project, 2)
    weight01 = block_0403_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0403_project_mask(:,i)==1); %(24,16)
end
block_0403_project = weight02;
clear weight01;
clear weight02;

block_0404_expand = load("block_0404_expand.txt",'r');
block_0404_expand_mask = load("block_0404_expand_mask.txt",'r');
block_0404_project = load("block_0404_project.txt",'r');
block_0404_project_mask = load("block_0404_project_mask.txt",'r');
for i=1:size(block_0404_expand, 2)
    weight01 = block_0404_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0404_expand_mask(:,i)==1); %(24,32)
end
block_0404_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0404_project, 2)
    weight01 = block_0404_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0404_project_mask(:,i)==1); %(24,16)
end
block_0404_project = weight02;
clear weight01;
clear weight02;

block_0501_expand = load("block_0501_expand.txt",'r');
block_0501_expand_mask = load("block_0501_expand_mask.txt",'r');
block_0501_project = load("block_0501_project.txt",'r');
block_0501_project_mask = load("block_0501_project_mask.txt",'r');
for i=1:size(block_0501_expand, 2)
    weight01 = block_0501_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0501_expand_mask(:,i)==1); %(24,32)
end
block_0501_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0501_project, 2)
    weight01 = block_0501_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0501_project_mask(:,i)==1); %(24,16)
end
block_0501_project = weight02;
clear weight01;
clear weight02;

block_0502_expand = load("block_0502_expand.txt",'r');
block_0502_expand_mask = load("block_0502_expand_mask.txt",'r');
block_0502_project = load("block_0502_project.txt",'r');
block_0502_project_mask = load("block_0502_project_mask.txt",'r');
for i=1:size(block_0502_expand, 2)
    weight01 = block_0502_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0502_expand_mask(:,i)==1); %(24,32)
end
block_0502_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0502_project, 2)
    weight01 = block_0502_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0502_project_mask(:,i)==1); %(24,16)
end
block_0502_project = weight02;
clear weight01;
clear weight02;

block_0503_expand = load("block_0503_expand.txt",'r');
block_0503_expand_mask = load("block_0503_expand_mask.txt",'r');
block_0503_project = load("block_0503_project.txt",'r');
block_0503_project_mask = load("block_0503_project_mask.txt",'r');
for i=1:size(block_0503_expand, 2)
    weight01 = block_0503_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0503_expand_mask(:,i)==1); %(24,32)
end
block_0503_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0503_project, 2)
    weight01 = block_0503_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0503_project_mask(:,i)==1); %(24,16)
end
block_0503_project = weight02;
clear weight01;
clear weight02;

block_0601_expand = load("block_0601_expand.txt",'r');
block_0601_expand_mask = load("block_0601_expand_mask.txt",'r');
block_0601_project = load("block_0601_project.txt",'r');
block_0601_project_mask = load("block_0601_project_mask.txt",'r');
for i=1:size(block_0601_expand, 2)
    weight01 = block_0601_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0601_expand_mask(:,i)==1); %(24,32)
end
block_0601_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0601_project, 2)
    weight01 = block_0601_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0601_project_mask(:,i)==1); %(24,16)
end
block_0601_project = weight02;
clear weight01;
clear weight02;

block_0602_expand = load("block_0602_expand.txt",'r');
block_0602_expand_mask = load("block_0602_expand_mask.txt",'r');
block_0602_project = load("block_0602_project.txt",'r');
block_0602_project_mask = load("block_0602_project_mask.txt",'r');
for i=1:size(block_0602_expand, 2)
    weight01 = block_0602_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0602_expand_mask(:,i)==1); %(24,32)
end
block_0602_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0602_project, 2)
    weight01 = block_0602_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0602_project_mask(:,i)==1); %(24,16)
end
block_0602_project = weight02;
clear weight01;
clear weight02;

block_0603_expand = load("block_0603_expand.txt",'r');
block_0603_expand_mask = load("block_0603_expand_mask.txt",'r');
block_0603_project = load("block_0603_project.txt",'r');
block_0603_project_mask = load("block_0603_project_mask.txt",'r');
for i=1:size(block_0603_expand, 2)
    weight01 = block_0603_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0603_expand_mask(:,i)==1); %(24,32)
end
block_0603_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0603_project, 2)
    weight01 = block_0603_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0603_project_mask(:,i)==1); %(24,16)
end
block_0603_project = weight02;
clear weight01;
clear weight02;

block_0701_expand = load("block_0701_expand.txt",'r');
block_0701_expand_mask = load("block_0701_expand_mask.txt",'r');
block_0701_project = load("block_0701_project.txt",'r');
block_0701_project_mask = load("block_0701_project_mask.txt",'r');
for i=1:size(block_0701_expand, 2)
    weight01 = block_0701_expand(:,i); %(:,32)
    weight02(:,i) = weight01(block_0701_expand_mask(:,i)==1); %(24,32)
end
block_0701_expand = weight02;
clear weight01;
clear weight02;
for i=1:size(block_0701_project, 2)
    weight01 = block_0701_project(:,i); %(:,16)
    weight02(:,i) = weight01(block_0701_project_mask(:,i)==1); %(24,16)
end
block_0701_project = weight02;
clear weight01;
clear weight02;

%calculate l1-norm importantance by group
%block_0101
for i=1:4
    for j=1:4
        for k=1:size(block_0101_expand, 1)
            block_0101_expand_groupl1norm(k, j) = sum(abs(block_0101_expand(k, (size(block_0101_expand, 2)/4)*(j-1)+1:(size(block_0101_expand, 2)/4)*j)));
        end
    end
    block_0101_expand_groupStd(i) = std(block_0101_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0101_project, 1)
            block_0101_project_groupl1norm(k, j) = sum(abs(block_0101_project(k, (size(block_0101_project, 2)/4)*(j-1)+1:(size(block_0101_project, 2)/4)*j)));
        end
    end
    block_0101_project_groupStd(i) = std(block_0101_project_groupl1norm(:,i),0);
end
%block_0201
for i=1:4
    for j=1:4
        for k=1:size(block_0201_expand, 1)
            block_0201_expand_groupl1norm(k, j) = sum(abs(block_0201_expand(k, (size(block_0201_expand, 2)/4)*(j-1)+1:(size(block_0201_expand, 2)/4)*j)));
        end
    end
    block_0201_expand_groupStd(i) = std(block_0201_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0201_project, 1)
            block_0201_project_groupl1norm(k, j) = sum(abs(block_0201_project(k, (size(block_0201_project, 2)/4)*(j-1)+1:(size(block_0201_project, 2)/4)*j)));
        end
    end
    block_0201_project_groupStd(i) = std(block_0201_project_groupl1norm(:,i),0);
end
%block_0202
for i=1:4
    for j=1:4
        for k=1:size(block_0202_expand, 1)
            block_0202_expand_groupl1norm(k, j) = sum(abs(block_0202_expand(k, (size(block_0202_expand, 2)/4)*(j-1)+1:(size(block_0202_expand, 2)/4)*j)));
        end
    end
    block_0202_expand_groupStd(i) = std(block_0202_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0202_project, 1)
            block_0202_project_groupl1norm(k, j) = sum(abs(block_0202_project(k, (size(block_0202_project, 2)/4)*(j-1)+1:(size(block_0202_project, 2)/4)*j)));
        end
    end
    block_0202_project_groupStd(i) = std(block_0202_project_groupl1norm(:,i),0);
end
%block_0301
for i=1:4
    for j=1:4
        for k=1:size(block_0301_expand, 1)
            block_0301_expand_groupl1norm(k, j) = sum(abs(block_0301_expand(k, (size(block_0301_expand, 2)/4)*(j-1)+1:(size(block_0301_expand, 2)/4)*j)));
        end
    end
    block_0301_expand_groupStd(i) = std(block_0301_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0301_project, 1)
            block_0301_project_groupl1norm(k, j) = sum(abs(block_0301_project(k, (size(block_0301_project, 2)/4)*(j-1)+1:(size(block_0301_project, 2)/4)*j)));
        end
    end
    block_0301_project_groupStd(i) = std(block_0301_project_groupl1norm(:,i),0);
end
%block_0302
for i=1:4
    for j=1:4
        for k=1:size(block_0302_expand, 1)
            block_0302_expand_groupl1norm(k, j) = sum(abs(block_0302_expand(k, (size(block_0302_expand, 2)/4)*(j-1)+1:(size(block_0302_expand, 2)/4)*j)));
        end
    end
    block_0302_expand_groupStd(i) = std(block_0302_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0302_project, 1)
            block_0302_project_groupl1norm(k, j) = sum(abs(block_0302_project(k, (size(block_0302_project, 2)/4)*(j-1)+1:(size(block_0302_project, 2)/4)*j)));
        end
    end
    block_0302_project_groupStd(i) = std(block_0302_project_groupl1norm(:,i),0);
end
%block_0303
for i=1:4
    for j=1:4
        for k=1:size(block_0303_expand, 1)
            block_0303_expand_groupl1norm(k, j) = sum(abs(block_0303_expand(k, (size(block_0303_expand, 2)/4)*(j-1)+1:(size(block_0303_expand, 2)/4)*j)));
        end
    end
    block_0303_expand_groupStd(i) = std(block_0303_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0303_project, 1)
            block_0303_project_groupl1norm(k, j) = sum(abs(block_0303_project(k, (size(block_0303_project, 2)/4)*(j-1)+1:(size(block_0303_project, 2)/4)*j)));
        end
    end
    block_0303_project_groupStd(i) = std(block_0303_project_groupl1norm(:,i),0);
end
%block_0401
for i=1:4
    for j=1:4
        for k=1:size(block_0401_expand, 1)
            block_0401_expand_groupl1norm(k, j) = sum(abs(block_0401_expand(k, (size(block_0401_expand, 2)/4)*(j-1)+1:(size(block_0401_expand, 2)/4)*j)));
        end
    end
    block_0401_expand_groupStd(i) = std(block_0401_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0401_project, 1)
            block_0401_project_groupl1norm(k, j) = sum(abs(block_0401_project(k, (size(block_0401_project, 2)/4)*(j-1)+1:(size(block_0401_project, 2)/4)*j)));
        end
    end
    block_0401_project_groupStd(i) = std(block_0401_project_groupl1norm(:,i),0);
end
%block_0402
for i=1:4
    for j=1:4
        for k=1:size(block_0402_expand, 1)
            block_0402_expand_groupl1norm(k, j) = sum(abs(block_0402_expand(k, (size(block_0402_expand, 2)/4)*(j-1)+1:(size(block_0402_expand, 2)/4)*j)));
        end
    end
    block_0402_expand_groupStd(i) = std(block_0402_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0402_project, 1)
            block_0402_project_groupl1norm(k, j) = sum(abs(block_0402_project(k, (size(block_0402_project, 2)/4)*(j-1)+1:(size(block_0402_project, 2)/4)*j)));
        end
    end
    block_0402_project_groupStd(i) = std(block_0402_project_groupl1norm(:,i),0);
end
%block_0403
for i=1:4
    for j=1:4
        for k=1:size(block_0403_expand, 1)
            block_0403_expand_groupl1norm(k, j) = sum(abs(block_0403_expand(k, (size(block_0403_expand, 2)/4)*(j-1)+1:(size(block_0403_expand, 2)/4)*j)));
        end
    end
    block_0403_expand_groupStd(i) = std(block_0403_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0403_project, 1)
            block_0403_project_groupl1norm(k, j) = sum(abs(block_0403_project(k, (size(block_0403_project, 2)/4)*(j-1)+1:(size(block_0403_project, 2)/4)*j)));
        end
    end
    block_0403_project_groupStd(i) = std(block_0403_project_groupl1norm(:,i),0);
end
%block_0404
for i=1:4
    for j=1:4
        for k=1:size(block_0404_expand, 1)
            block_0404_expand_groupl1norm(k, j) = sum(abs(block_0404_expand(k, (size(block_0404_expand, 2)/4)*(j-1)+1:(size(block_0404_expand, 2)/4)*j)));
        end
    end
    block_0404_expand_groupStd(i) = std(block_0404_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0404_project, 1)
            block_0404_project_groupl1norm(k, j) = sum(abs(block_0404_project(k, (size(block_0404_project, 2)/4)*(j-1)+1:(size(block_0404_project, 2)/4)*j)));
        end
    end
    block_0404_project_groupStd(i) = std(block_0404_project_groupl1norm(:,i),0);
end
%block_0501
for i=1:4
    for j=1:4
        for k=1:size(block_0501_expand, 1)
            block_0501_expand_groupl1norm(k, j) = sum(abs(block_0501_expand(k, (size(block_0501_expand, 2)/4)*(j-1)+1:(size(block_0501_expand, 2)/4)*j)));
        end
    end
    block_0501_expand_groupStd(i) = std(block_0501_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0501_project, 1)
            block_0501_project_groupl1norm(k, j) = sum(abs(block_0501_project(k, (size(block_0501_project, 2)/4)*(j-1)+1:(size(block_0501_project, 2)/4)*j)));
        end
    end
    block_0501_project_groupStd(i) = std(block_0501_project_groupl1norm(:,i),0);
end
%block_0502
for i=1:4
    for j=1:4
        for k=1:size(block_0502_expand, 1)
            block_0502_expand_groupl1norm(k, j) = sum(abs(block_0502_expand(k, (size(block_0502_expand, 2)/4)*(j-1)+1:(size(block_0502_expand, 2)/4)*j)));
        end
    end
    block_0502_expand_groupStd(i) = std(block_0502_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0502_project, 1)
            block_0502_project_groupl1norm(k, j) = sum(abs(block_0502_project(k, (size(block_0502_project, 2)/4)*(j-1)+1:(size(block_0502_project, 2)/4)*j)));
        end
    end
    block_0502_project_groupStd(i) = std(block_0502_project_groupl1norm(:,i),0);
end
%block_0503
for i=1:4
    for j=1:4
        for k=1:size(block_0503_expand, 1)
            block_0503_expand_groupl1norm(k, j) = sum(abs(block_0503_expand(k, (size(block_0503_expand, 2)/4)*(j-1)+1:(size(block_0503_expand, 2)/4)*j)));
        end
    end
    block_0503_expand_groupStd(i) = std(block_0503_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0503_project, 1)
            block_0503_project_groupl1norm(k, j) = sum(abs(block_0503_project(k, (size(block_0503_project, 2)/4)*(j-1)+1:(size(block_0503_project, 2)/4)*j)));
        end
    end
    block_0503_project_groupStd(i) = std(block_0503_project_groupl1norm(:,i),0);
end
%block_0601
for i=1:4
    for j=1:4
        for k=1:size(block_0601_expand, 1)
            block_0601_expand_groupl1norm(k, j) = sum(abs(block_0601_expand(k, (size(block_0601_expand, 2)/4)*(j-1)+1:(size(block_0601_expand, 2)/4)*j)));
        end
    end
    block_0601_expand_groupStd(i) = std(block_0601_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0601_project, 1)
            block_0601_project_groupl1norm(k, j) = sum(abs(block_0601_project(k, (size(block_0601_project, 2)/4)*(j-1)+1:(size(block_0601_project, 2)/4)*j)));
        end
    end
    block_0601_project_groupStd(i) = std(block_0601_project_groupl1norm(:,i),0);
end
%block_0602
for i=1:4
    for j=1:4
        for k=1:size(block_0602_expand, 1)
            block_0602_expand_groupl1norm(k, j) = sum(abs(block_0602_expand(k, (size(block_0602_expand, 2)/4)*(j-1)+1:(size(block_0602_expand, 2)/4)*j)));
        end
    end
    block_0602_expand_groupStd(i) = std(block_0602_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0602_project, 1)
            block_0602_project_groupl1norm(k, j) = sum(abs(block_0602_project(k, (size(block_0602_project, 2)/4)*(j-1)+1:(size(block_0602_project, 2)/4)*j)));
        end
    end
    block_0602_project_groupStd(i) = std(block_0602_project_groupl1norm(:,i),0);
end
%block_0603
for i=1:4
    for j=1:4
        for k=1:size(block_0603_expand, 1)
            block_0603_expand_groupl1norm(k, j) = sum(abs(block_0603_expand(k, (size(block_0603_expand, 2)/4)*(j-1)+1:(size(block_0603_expand, 2)/4)*j)));
        end
    end
    block_0603_expand_groupStd(i) = std(block_0603_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0603_project, 1)
            block_0603_project_groupl1norm(k, j) = sum(abs(block_0603_project(k, (size(block_0603_project, 2)/4)*(j-1)+1:(size(block_0603_project, 2)/4)*j)));
        end
    end
    block_0603_project_groupStd(i) = std(block_0603_project_groupl1norm(:,i),0);
end
%block_0701
for i=1:4
    for j=1:4
        for k=1:size(block_0701_expand, 1)
            block_0701_expand_groupl1norm(k, j) = sum(abs(block_0701_expand(k, (size(block_0701_expand, 2)/4)*(j-1)+1:(size(block_0701_expand, 2)/4)*j)));
        end
    end
    block_0701_expand_groupStd(i) = std(block_0701_expand_groupl1norm(:,i),0);
end
for i=1:4
    for j=1:4
        for k=1:size(block_0701_project, 1)
            block_0701_project_groupl1norm(k, j) = sum(abs(block_0701_project(k, (size(block_0701_project, 2)/4)*(j-1)+1:(size(block_0701_project, 2)/4)*j)));
        end
    end
    block_0701_project_groupStd(i) = std(block_0701_project_groupl1norm(:,i),0);
end
disp('prepocessing finished...') 