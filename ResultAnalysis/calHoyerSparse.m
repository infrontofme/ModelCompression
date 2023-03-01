%calculate Hoyer Sparsity of all input feature on eachgroup
%input features importance sparsity on eachgroup
%{
(sqrt(size(A)) - norm(A,1)/norm(A,2)) / (sqrt(size(A)) - 1)

0 - least sparse
1 - most sparse
%}
disp('Calculate Hoyer Sparsity for Expansion and Projection layer...')
HoyerSparseExpandOnGroup = zeros(17,4);
%block_0101
for i=1:4
    HoyerSparseExpandOnGroup(1,i) = (sqrt(size(block_0101_expand_groupl1norm, 1)) - norm(block_0101_expand_groupl1norm(:,i), 1) / max(norm(block_0101_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0101_expand_groupl1norm, 1)) - 1);
end
%block_0201
for i=1:4
    HoyerSparseExpandOnGroup(2,i) = (sqrt(size(block_0201_expand_groupl1norm, 1)) - norm(block_0201_expand_groupl1norm(:,i), 1) / max(norm(block_0201_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0201_expand_groupl1norm, 1)) - 1);
end
%block_0202
for i=1:4
    HoyerSparseExpandOnGroup(3,i) = (sqrt(size(block_0202_expand_groupl1norm, 1)) - norm(block_0202_expand_groupl1norm(:,i), 1) / max(norm(block_0202_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0202_expand_groupl1norm, 1)) - 1);
end
%block_0301
for i=1:4
    HoyerSparseExpandOnGroup(4,i) = (sqrt(size(block_0301_expand_groupl1norm, 1)) - norm(block_0301_expand_groupl1norm(:,i), 1) / max(norm(block_0301_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0301_expand_groupl1norm, 1)) - 1);
end
%block_0302
for i=1:4
    HoyerSparseExpandOnGroup(5,i) = (sqrt(size(block_0302_expand_groupl1norm, 1)) - norm(block_0302_expand_groupl1norm(:,i), 1) / max(norm(block_0302_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0302_expand_groupl1norm, 1)) - 1);
end
%block_0303
for i=1:4
    HoyerSparseExpandOnGroup(6,i) = (sqrt(size(block_0303_expand_groupl1norm, 1)) - norm(block_0303_expand_groupl1norm(:,i), 1) / max(norm(block_0303_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0303_expand_groupl1norm, 1)) - 1);
end
%block_0401
for i=1:4
    HoyerSparseExpandOnGroup(7,i) = (sqrt(size(block_0401_expand_groupl1norm, 1)) - norm(block_0401_expand_groupl1norm(:,i), 1) / max(norm(block_0401_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0401_expand_groupl1norm, 1)) - 1);
end
%block_0402
for i=1:4
    HoyerSparseExpandOnGroup(8,i) = (sqrt(size(block_0402_expand_groupl1norm, 1)) - norm(block_0402_expand_groupl1norm(:,i), 1) / max(norm(block_0402_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0402_expand_groupl1norm, 1)) - 1);
end
%block_0403
for i=1:4
    HoyerSparseExpandOnGroup(9,i) = (sqrt(size(block_0403_expand_groupl1norm, 1)) - norm(block_0403_expand_groupl1norm(:,i), 1) / max(norm(block_0403_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0403_expand_groupl1norm, 1)) - 1);
end
%block_0404
for i=1:4
    HoyerSparseExpandOnGroup(10,i) = (sqrt(size(block_0404_expand_groupl1norm, 1)) - norm(block_0404_expand_groupl1norm(:,i), 1) / max(norm(block_0404_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0404_expand_groupl1norm, 1)) - 1);
end
%block_0501
for i=1:4
    HoyerSparseExpandOnGroup(11,i) = (sqrt(size(block_0501_expand_groupl1norm, 1)) - norm(block_0501_expand_groupl1norm(:,i), 1) / max(norm(block_0501_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0501_expand_groupl1norm, 1)) - 1);
end
%block_0502
for i=1:4
    HoyerSparseExpandOnGroup(12,i) = (sqrt(size(block_0502_expand_groupl1norm, 1)) - norm(block_0502_expand_groupl1norm(:,i), 1) / max(norm(block_0502_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0502_expand_groupl1norm, 1)) - 1);
end
%block_0503
for i=1:4
    HoyerSparseExpandOnGroup(13,i) = (sqrt(size(block_0503_expand_groupl1norm, 1)) - norm(block_0503_expand_groupl1norm(:,i), 1) / max(norm(block_0503_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0503_expand_groupl1norm, 1)) - 1);
end
%block_0601
for i=1:4
    HoyerSparseExpandOnGroup(14,i) = (sqrt(size(block_0601_expand_groupl1norm, 1)) - norm(block_0601_expand_groupl1norm(:,i), 1) / max(norm(block_0601_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0601_expand_groupl1norm, 1)) - 1);
end
%block_0602
for i=1:4
    HoyerSparseExpandOnGroup(15,i) = (sqrt(size(block_0602_expand_groupl1norm, 1)) - norm(block_0602_expand_groupl1norm(:,i), 1) / max(norm(block_0602_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0602_expand_groupl1norm, 1)) - 1);
end
%block_0603
for i=1:4
    HoyerSparseExpandOnGroup(16,i) = (sqrt(size(block_0603_expand_groupl1norm, 1)) - norm(block_0603_expand_groupl1norm(:,i), 1) / max(norm(block_0603_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0603_expand_groupl1norm, 1)) - 1);
end
%block_0701
for i=1:4
    HoyerSparseExpandOnGroup(17,i) = (sqrt(size(block_0701_expand_groupl1norm, 1)) - norm(block_0701_expand_groupl1norm(:,i), 1) / max(norm(block_0701_expand_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0701_expand_groupl1norm, 1)) - 1);
end


%calculate Hoyer Sparsity of all input feature on eachgroup
%input features importance sparsity on eachgroup
%{
(sqrt(size(A)) - norm(A,1)/norm(A,2)) / (sqrt(size(A)) - 1)

0 - least sparse
1 - most sparse
%}
HoyerSparseProjectOnGroup = zeros(17,4);
%block_0101
for i=1:4
    HoyerSparseProjectOnGroup(1,i) = (sqrt(size(block_0101_project_groupl1norm, 1)) - norm(block_0101_project_groupl1norm(:,i), 1) / max(norm(block_0101_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0101_project_groupl1norm, 1)) - 1);
end
%block_0201
for i=1:4
    HoyerSparseProjectOnGroup(2,i) = (sqrt(size(block_0201_project_groupl1norm, 1)) - norm(block_0201_project_groupl1norm(:,i), 1) / max(norm(block_0201_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0201_project_groupl1norm, 1)) - 1);
end
%block_0202
for i=1:4
    HoyerSparseProjectOnGroup(3,i) = (sqrt(size(block_0202_project_groupl1norm, 1)) - norm(block_0202_project_groupl1norm(:,i), 1) / max(norm(block_0202_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0202_project_groupl1norm, 1)) - 1);
end
%block_0301
for i=1:4
    HoyerSparseProjectOnGroup(4,i) = (sqrt(size(block_0301_project_groupl1norm, 1)) - norm(block_0301_project_groupl1norm(:,i), 1) / max(norm(block_0301_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0301_project_groupl1norm, 1)) - 1);
end
%block_0302
for i=1:4
    HoyerSparseProjectOnGroup(5,i) = (sqrt(size(block_0302_project_groupl1norm, 1)) - norm(block_0302_project_groupl1norm(:,i), 1) / max(norm(block_0302_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0302_project_groupl1norm, 1)) - 1);
end
%block_0303
for i=1:4
    HoyerSparseProjectOnGroup(6,i) = (sqrt(size(block_0303_project_groupl1norm, 1)) - norm(block_0303_project_groupl1norm(:,i), 1) / max(norm(block_0303_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0303_project_groupl1norm, 1)) - 1);
end
%block_0401
for i=1:4
    HoyerSparseProjectOnGroup(7,i) = (sqrt(size(block_0401_project_groupl1norm, 1)) - norm(block_0401_project_groupl1norm(:,i), 1) / max(norm(block_0401_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0401_project_groupl1norm, 1)) - 1);
end
%block_0402
for i=1:4
    HoyerSparseProjectOnGroup(8,i) = (sqrt(size(block_0402_project_groupl1norm, 1)) - norm(block_0402_project_groupl1norm(:,i), 1) / max(norm(block_0402_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0402_project_groupl1norm, 1)) - 1);
end
%block_0403
for i=1:4
    HoyerSparseProjectOnGroup(9,i) = (sqrt(size(block_0403_project_groupl1norm, 1)) - norm(block_0403_project_groupl1norm(:,i), 1) / max(norm(block_0403_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0403_project_groupl1norm, 1)) - 1);
end
%block_0404
for i=1:4
    HoyerSparseProjectOnGroup(10,i) = (sqrt(size(block_0404_project_groupl1norm, 1)) - norm(block_0404_project_groupl1norm(:,i), 1) / max(norm(block_0404_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0404_project_groupl1norm, 1)) - 1);
end
%block_0501
for i=1:4
    HoyerSparseProjectOnGroup(11,i) = (sqrt(size(block_0501_project_groupl1norm, 1)) - norm(block_0501_project_groupl1norm(:,i), 1) / max(norm(block_0501_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0501_project_groupl1norm, 1)) - 1);
end
%block_0502
for i=1:4
    HoyerSparseProjectOnGroup(12,i) = (sqrt(size(block_0502_project_groupl1norm, 1)) - norm(block_0502_project_groupl1norm(:,i), 1) / max(norm(block_0502_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0502_project_groupl1norm, 1)) - 1);
end
%block_0503
for i=1:4
    HoyerSparseProjectOnGroup(13,i) = (sqrt(size(block_0503_project_groupl1norm, 1)) - norm(block_0503_project_groupl1norm(:,i), 1) / max(norm(block_0503_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0503_project_groupl1norm, 1)) - 1);
end
%block_0601
for i=1:4
    HoyerSparseProjectOnGroup(14,i) = (sqrt(size(block_0601_project_groupl1norm, 1)) - norm(block_0601_project_groupl1norm(:,i), 1) / max(norm(block_0601_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0601_project_groupl1norm, 1)) - 1);
end
%block_0602
for i=1:4
    HoyerSparseProjectOnGroup(15,i) = (sqrt(size(block_0602_project_groupl1norm, 1)) - norm(block_0602_project_groupl1norm(:,i), 1) / max(norm(block_0602_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0602_project_groupl1norm, 1)) - 1);
end
%block_0603
for i=1:4
    HoyerSparseProjectOnGroup(16,i) = (sqrt(size(block_0603_project_groupl1norm, 1)) - norm(block_0603_project_groupl1norm(:,i), 1) / max(norm(block_0603_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0603_project_groupl1norm, 1)) - 1);
end
%block_0701
for i=1:4
    HoyerSparseProjectOnGroup(17,i) = (sqrt(size(block_0701_project_groupl1norm, 1)) - norm(block_0701_project_groupl1norm(:,i), 1) / max(norm(block_0701_project_groupl1norm(:,i), 2), 1e-10)) / ...,
        (sqrt(size(block_0701_project_groupl1norm, 1)) - 1);
end
disp('Calculation finished...');
figure(1);
HSparseExpand = heatmap(HoyerSparseExpandOnGroup);
figure(2);
HSparseProject = heatmap(HoyerSparseProjectOnGroup);