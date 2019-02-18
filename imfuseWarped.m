function [C,RC] = imfuseWarped(A,RA,B,RB)
% This function fuses two images (A,B) that have been warped with Matlab's
% "imwarp" function.  RA and RB are the corresponding spatial referencing
% objects returned by "imwarp".  This code is very similar to, and is
% adapted from, Matlab's "imfuse". Unlike "imfuse", this function treats
% pixels that have zero values as "unknown".  So, at pixels where both
% values are known, the output is the average of the two.  But at pixels
% where only one is known, the output uses that value.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First calculate output referencing object. World limits are minimum
% bounding box that contains world limits of both images. Resolution is the
% minimum resolution in each dimension. We don't want to down sample either
% image.
outputWorldLimitsX = [min(RA.XWorldLimits(1),RB.XWorldLimits(1)),max(RA.XWorldLimits(2),RB.XWorldLimits(2))];
outputWorldLimitsY = [min(RA.YWorldLimits(1),RB.YWorldLimits(1)),max(RA.YWorldLimits(2),RB.YWorldLimits(2))];
goalResolutionX = min(RA.PixelExtentInWorldX,RB.PixelExtentInWorldX);
goalResolutionY = min(RA.PixelExtentInWorldY,RB.PixelExtentInWorldY);
widthOutputRaster = ceil(diff(outputWorldLimitsX) / goalResolutionX);
heightOutputRaster = ceil(diff(outputWorldLimitsY) / goalResolutionY);
RC = imref2d([heightOutputRaster, widthOutputRaster]);
RC.XWorldLimits = outputWorldLimitsX;
RC.YWorldLimits = outputWorldLimitsY;
fillVal = 0;
A_padded = images.spatialref.internal.resampleImageToNewSpatialRef(A,RA,RC,'bilinear',fillVal);
B_padded = images.spatialref.internal.resampleImageToNewSpatialRef(B,RB,RC,'bilinear',fillVal);
% Create masks: assume that valid pixels are those that are not zero.
A_mask = (sum(A_padded,3) > 0);
B_mask = (sum(B_padded,3) > 0);
% Erode the masks a little, to avoid erroneous border values.
A_mask = imerode(A_mask, strel('disk',1));
B_mask = imerode(B_mask, strel('disk',1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute regions of overlap.
onlyA = A_mask & ~B_mask;
onlyB = ~A_mask & B_mask;
bothAandB = A_mask & B_mask;
% Weight each image equally.
weight1 = 0.5;
weight2 = 0.5;
% Allocate result image.
C = zeros([size(A_padded,1) size(A_padded,2) size(A_padded,3)], class(A_padded));
% For each color band, compute blended output band.
for i = 1:size(A_padded,3)
a = A_padded(:,:,i);
b = B_padded(:,:,i);
r = C(:,:,i);
r(onlyA) = a(onlyA);
r(onlyB) = b(onlyB);
r(bothAandB) = uint8( weight1 .* single(a(bothAandB)) + weight2 .* single(b(bothAandB)));
C(:,:,i) = r;
end
end


