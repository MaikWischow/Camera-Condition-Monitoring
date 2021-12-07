clear;
close all;
addpath(genpath('.\Graph_Based_BID_p1.1'));

%% Configuration parameters
k_estimate_size=25; % set approximate kernel size
show_intermediate=false; % show intermediate output or not
border=20;% cut boundaries of the input blurry image (default border length: 20) 
imgPatchSize = 192

basePath = '../../data/udacity';
corruptionDir = 'GT'
maxNumFiles = 1000;
dirIn = fullfile(basePath, 'img', corruptionDir);
dirOut = fullfile(basePath, 'labels_blur_gbb_patchwise', corruptionDir);
whiteListPath = fullfile(basePath, 'blurEstimationWhitelist.txt');
objectDetectionsPath = fullfile(basePath, 'label');
classWhiteList = {'car', 'pedestrian'};
imgFileEnding = '.jpg'

%% Parse list of image names to be analysed
whiteList = fileread(whiteListPath);
whiteList = strsplit(whiteList);

%% Iterate sub folders
subFolders = dir(dirIn);
subFolders = subFolders([subFolders(:).isdir]);
subFolders = subFolders(~ismember({subFolders(:).name},{'..'}))
for subFolderIdx = 1:length(subFolders)
    subFolderName = subFolders(subFolderIdx).name;
    fileNames = dir(fullfile(dirIn, subFolderName, strcat('*', imgFileEnding)));
    
    subDirOut = fullfile(dirOut, subFolderName);
    if ~exist(subDirOut, 'dir')
       mkdir(subDirOut);
    end
    
    %% Iterate image files
    for fileNameIdx = 1:length(fileNames)
        filefolder = fileNames(fileNameIdx).folder;
        filename = fileNames(fileNameIdx).name;
        dotLocation = find(filename == '.');
        rawfilename = filename(1:dotLocation(1)-1);
        
        %% Check if filename is not a blur kernel and if in whitelist
        if contains(rawfilename, 'kernel') || ~any(strcmp(whiteList, rawfilename))
            continue
        end
        
        %% Get object classes of image
        labelsFile = dir(fullfile(objectDetectionsPath, strcat('*', rawfilename, '*')));
        objList = fileread(fullfile(labelsFile.folder, labelsFile.name));
        objList = splitlines(objList);
        
        %% Filter object classes
        objList = objList(contains(objList, classWhiteList, 'IgnoreCase', true) &  ...
                                ~contains(objList, 'DontCare', 'IgnoreCase', true));
        
        try
            %% Read image and convert to grayscale
            y = imread(fullfile(filefolder, filename));
            y = y(border+1:end-border,border+1:end-border, :);
            imgSize = size(y);
            if size(y,3)==3
                y = im2double(rgb2gray(y));
            else
                y = im2double(y);
            end
            
            %% Init cache
            objDetRoICacheCoord = cell(0);
            objDetRoICacheResult = cell(0);
            for objDetIdx = 1:length(objList)
                %% Parse object detection
                objDet = objList(objDetIdx);
                objDetSplit = strsplit(objDet{1});
                [objClass, objx1, objy1, objx2, objy2] = deal(objDetSplit{:});
                [startX, startY, endX, endY] = getObjDetRoI(imgSize, imgPatchSize, objx1, objy1, objx2, objy2);
                
                %% Iterate corresponding image patches of detection
                numPatchesX = ceil((endX - startX) / imgPatchSize);
                numPatchesY = ceil((endY - startY) / imgPatchSize);
                for idxX = 0 : numPatchesX - 1
                    for idxY = 0 : numPatchesY - 1
                        
                        %% Check pre-defined file limit
                        if length(dir(fullfile(subDirOut, strcat("*", imgFileEnding)))) >= maxNumFiles
                            break
                        end 
                        
                        %% Get image patch
                        [startX_, startY_, endX_, endY_] = getPatchRoI(imgSize, imgPatchSize, startX, startY, idxX, idxY);
                        patchIdx = idxX * numPatchesY + idxY;
                        resultFileName = fullfile(subDirOut, strcat(rawfilename, '_', objClass, '_', num2str(objDetIdx), '_', num2str(patchIdx), imgFileEnding));
                        
                        %% Search cache for fitting entry
                        cacheHit = false;
                        for cacheIdx = 1:length(objDetRoICacheCoord)
                            objDetCacheCoord = objDetRoICacheCoord{cacheIdx};
                            [x1, y1, x2, y2] = objDetCacheCoord{:};
                            iou = calculateIoU(startX_, startY_, endX_, endY_, x1, y1, x2, y2);
                            if iou >= 0.7
                                objDetResult = objDetRoICacheResult{cacheIdx};
                                imwrite(im2uint8(objDetResult), resultFileName);
                                cacheHit = true;
                            end
                        end
                        
                        if cacheHit
                            continue
                        end
                        
                        %% Check if kernel was already processed
                        if ~isfile(resultFileName)
                            fprintf('Computing file: %s \n', resultFileName);
                            
                            %% Get patch
                            patch = y(startY_:endY_, startX_:endX_);

                            %% Calculate Kernel
                            tic;
                            [ kernel,Y_intermediate ] = bid_rgtv_c2f_cg(patch, k_estimate_size, show_intermediate );
                            t=toc;
                            fprintf('Grpah based Blind Deblurring Running Time:%f s\n',t);
                            
                            %% Save kernel
                            kernel=k_rescale(kernel);
                            imwrite(im2uint8(kernel), resultFileName);
                            objDetRoICacheCoord{end+1} = {startX_, startY_, endX_, endY_};
                            objDetRoICacheResult{end+1} = kernel;

                        end
                    end
                end
            end
        catch
        end
    end
end

function [startX, startY, endX, endY] = getObjDetRoI(imgSize, imgPatchSize, objx1, objy1, objx2, objy2)
    startX = str2double(objx1);
    startY = str2double(objy1);
    endX = str2double(objx2);
    endY = str2double(objy2);
    
    xRange = endX - startX;
    yRange = endY - startY;
    addX = (imgPatchSize - mod(xRange, imgPatchSize));
    addY = (imgPatchSize - mod(yRange, imgPatchSize));
    endX = endX + addX;
    endY = endY + addY;
    
    if startX == 0
        startX = 1;
    end
    if startY == 0
        startY = 1;
    end
    if endX > imgSize(2)
        endX = imgSize(2);
    end
    if endY > imgSize(1)
        endY = imgSize(1);
    end
end

function [startX_, startY_, endX_, endY_] = getPatchRoI(imgSize, imgPatchSize, startX, startY, idxX, idxY)
    startX_ = startX + idxX * imgPatchSize;
    startY_ = startY + idxY * imgPatchSize;
    endX_ = startX_ + imgPatchSize - 1;
    endY_ = startY_ + imgPatchSize - 1;
    if endX_ > imgSize(2)
        offset = endX_ - imgSize(2);
        endX_ = endX_ - offset; 
        startX_ = startX_ - offset;
    end
    if endY_ > imgSize(1)
        offset = endY_ - imgSize(1);
        endY_ = endY_ - offset; 
        startY_ = startY_ - offset;
    end
end

function iou = calculateIoU(startX_, startY_, endX_, endY_, x1, y1, x2, y2)
    x_left = max(startX_, x1);
    y_top = max(startY_, y1);
    x_right = min(endX_, x2);
    y_bottom = min(endY_, y2);

    if x_right < x_left || y_bottom < y_top
        iou = 0.0;
        return
    end

    intersection_area = (x_right - x_left) * (y_bottom - y_top);
    
    bb1_area = (endX_ - startX_) * (endY_ - startY_);
    bb2_area = (x2 - x1) * (y2 - y1);

    iou = intersection_area / (bb1_area + bb2_area - intersection_area);
end

function patches = img2patches(img, imgPatchSize)

   imSz = size(img);
   xIdxs = [1:imgPatchSize:imSz(2) imSz(2)+1];
   yIdxs = [1:imgPatchSize:imSz(1) imSz(1)+1];
   len = (length(yIdxs) - 1) * (length(xIdxs) - 1)
   patches = cell(len, 1);
    
  for i = 1:length(yIdxs)-1
      Isub = img(yIdxs(i):yIdxs(i+1)-1,:,:);
      for j = 1:length(xIdxs)-1
          patches{((i-1) * length(yIdxs)) + j,1} = Isub(:, xIdxs(j):xIdxs(j+1) - 1, :);
      end
  end
end