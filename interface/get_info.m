function [levels, width, height] = get_info(image_path)
    a = imfinfo(image_path, 'JP2');
    levels = a.WaveletDecompositionLevels;
    width = a.Width;
    height = a.Height;
end