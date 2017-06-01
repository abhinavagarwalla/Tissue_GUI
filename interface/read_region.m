function res = read_region(image_path, level, region)
    res = imread(image_path, 'ReductionLevel', level, 'PixelRegion', {[region(1),region(2)],[region(3),region(4)]});
end