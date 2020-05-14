function LabelMap = CreateLabelMapFromAnnotations(I, LM)

Isize = size(I);
nLM = size(LM,1);
nSquares = ceil(nLM / 4); % the number of signs

LabelMap = zeros(Isize(1),Isize(2),'uint8');
for S = 1:nSquares
    ids = (S-1) * 4 + 1:(S-1) * 4 + 4;
    cc = LM(ids,1);
    rc = LM(ids,2);
    BW = roipoly(I, cc, rc);
    LabelMap = LabelMap + uint8(BW) * S;
end

