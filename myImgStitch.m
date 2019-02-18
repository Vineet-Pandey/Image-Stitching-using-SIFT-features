% Author: Vineet Pandey
% CWID: 10826588
% Date: 3rd November, 2018

clc
clear all
close all

%% Image Pre-processing
% imageFiles = {'Painting/mural01.jpg','Painting/mural02.jpg','Painting/mural03.jpg','Painting/mural04.jpg','Painting/mural05.jpg','Painting/mural06.jpg','Painting/mural07.jpg','Painting/mural08.jpg','Painting/mural09.jpg','Painting/mural10.jpg','Painting/mural11.jpg','Painting/mural12.jpg'};
imageFiles = {'myImages/1.jpeg','myImages/2.jpeg','myImages/3.jpeg','myImages/4.jpeg','myImages/5.jpeg','myImages/6.jpeg'};
% imageFiles = {'Images/1.jpeg','Images/2.jpeg','Images/3.jpeg','Images/4.jpeg','Images/5.jpeg','Images/6.jpeg'};
width_lim = 400; % Maximum assumed width of the image
imgPrev = imread(imageFiles{1});
imgWidth = size( imgPrev,2);
% Resizing incase the Image exceeds the assumed size
if imgWidth > width_lim
imgPrev = imresize(imgPrev, width_lim/imgWidth);
end
figure(1), imshow(imgPrev,[]);
%% Converting the first image to orthophoto
% Taking the four co-ordinates from the user as an 4X2 array and creating a
% rectangle. The first I/P should be from the top-left corner
for i=1:4
    usrCoord(i,:) = ginput(1);
    x = usrCoord(i,1);
    y = usrCoord(i,2);
    rectangle('Position', [x-2, y-2, 4,4], 'FaceColor', 'r');
    if i>1
        line([x,usrCoord(i-1,1)], [y,usrCoord(i-1,2)], 'Color', 'r');
    end
    if i==4
        line([x,usrCoord(1,1)], [y,usrCoord(1,2)], 'Color', 'r');
    end
end
% Image properties
imgCentroid = mean(usrCoord);
imgWidth = mean([abs(usrCoord(1,1)-usrCoord(2,1)), abs(usrCoord(3,1)-usrCoord(4,1))]);
imgHeight = mean([abs(usrCoord(1,2)-usrCoord(4,2)), abs(usrCoord(2,2)-usrCoord(3,2))]);

% Coordinates for the orthophoto
orthoCoord = [imgCentroid(1)-imgWidth/2, imgCentroid(2)-imgHeight/2;
              imgCentroid(1)+imgWidth/2, imgCentroid(2)-imgHeight/2;
              imgCentroid(1)+imgWidth/2, imgCentroid(2)+imgHeight/2;
              imgCentroid(1)-imgWidth/2, imgCentroid(2)+imgHeight/2;];
          
% Getting the transformation from the fixed and moving points using
% fitgeotrans and applying the transformation to the first image        
orthoTrans = fitgeotrans(usrCoord, orthoCoord, 'projective');
[imgPrev, ~] = imwarp(imgPrev,orthoTrans);
figure(4), imshow(imgPrev, []);
%pause;

%% Working with the remaining Images
% Creating a reference image (from the image I1)
H_prev_ref = eye(3,3);
I_ref = imgPrev; 
R = imref2d(size(I_ref));

for j=2:length(imageFiles)
    %% Preprocessing the remaining images
    
    I_nxt = imread(imageFiles{j});
    I_nxt_width = size(I_nxt,2);
    if I_nxt_width > width_lim
        I_nxt = imresize(I_nxt, width_lim/I_nxt_width);
    end
    
    % Converting the images to grey scale if they are RGB
    if (size(imgPrev,3)>1)
        I_prev = rgb2gray(imgPrev);
    else
        I_prev = imgPrev;
    end
    if (size(I_nxt,3)>1)
        Inext = rgb2gray(I_nxt);
    else
        Inext = I_nxt;
    end
    
    %% Extracting SIFT features
    peak_threshold = 5;
    edge_threshold = 15;
    % Running the vlfeat toolbox
    run('F:\Grad\Fall 2018\CV\Homeworks\HW5\vlfeat-0.9.21\toolbox\vl_setup')
    I_prev = single(I_prev);
    figure(1), imshow(I_prev,[]);
    
    % detecting SIFT features for the previous image
    [features_prev,descriptors_prev] = vl_sift(I_prev,'PeakThresh', peak_threshold,'edgethresh', edge_threshold );
    
    % Over laying the SIFT feature location on the image
    features_prev_loc = vl_plotframe(features_prev) ;
    set(features_prev_loc,'color','y','linewidth',1.0) ;
    Inext = single(Inext);
    figure(2), imshow(Inext,[]);
    
    % detecting SIFT features for the next image
    [features_next,descriptors_next] = vl_sift(Inext,'PeakThresh', peak_threshold,'edgethresh', edge_threshold );
    % Overlaying the SIFT features location on the image
    features_prev_loc = vl_plotframe(features_next) ;
    set(features_prev_loc,'color','y','linewidth',1.0) ;
    
    %% Matching the Features
    % Index of initial match and the closest descriptor is stored in
    % 'match', while the distance between the pairs is stored in
    % 'euclid_dist'
    thresh=1.5;
    [match, euclid_dist] = vl_ubcmatch(descriptors_prev, descriptors_next, thresh);
    indices_prev = match(1,:);
    features_prev_match = features_prev(:,indices_prev);
    descriptors_prev_match = descriptors_prev(:,indices_prev);
    indices_next = match(2,:);
    features_next_match = features_next(:,indices_next);
    descriptors_next_match = descriptors_next(:,indices_next);
    Corres_points = size(features_prev_match,2);
    sigma = 2;
    confidence = 0.99;
    % fitting a homography from the points in I1, to the points in
    % I_nxt using 'fitHomographyRansac' function provided
    [tform12, indices, rmsErr] = fitHomographyRansac(features_prev_match,features_next_match,size(Inext,1),size(Inext,2),sigma,500,confidence );
    
    %% Overlaying the final matches over the image
    features_prev_Inlier = features_prev_match(:,indices);
    features_next_Inlier = features_next_match(:,indices);
    figure(3), subplot(1,2,1), imshow(I_prev,[]);
    features_prev_loc = vl_plotframe(features_prev_Inlier) ;
    set(features_prev_loc,'color','y','linewidth',1.5) ;
    subplot(1,2,2), imshow(Inext,[]);
    features_prev_loc = vl_plotframe(features_next_Inlier) ;
    set(features_prev_loc,'color','y','linewidth',1.5) ;
    
    % Homography from imgPrev to I_nxt
    H_prev_nxt = tform12.T';
    H_nxt_prev = inv(H_prev_nxt);
    H_nxt_ref = H_prev_ref * H_nxt_prev;
    tform_next_ref = projective2d(H_nxt_ref');
    
    % warping next image to align with the reference image
    [Inext_ref, Rnext] = imwarp(I_nxt,tform_next_ref);
    
    %% image fusion using imfuseWarped function provided
    [I_ref, R] = imfuseWarped(I_ref, R, Inext_ref, Rnext);
    figure(4), imshow(I_ref, []);
    imgPrev = I_nxt;
    H_prev_ref = H_nxt_ref;
end