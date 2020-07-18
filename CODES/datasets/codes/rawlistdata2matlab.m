function [] = rawlistdata2matlab(root)
dir_list = dir(root);
for i=3:length(dir_list)
    subdir_list = dir(fullfile(root,dir_list(i).name));
    for j=3:length(subdir_list)
        filelist = dir(fullfile(root, dir_list(i).name, subdir_list(j).name, '*.aedat'));
        for k=1:length(filelist)
            filelist(k).name
            rawdata2matlab(fullfile(root, dir_list(i).name, subdir_list(j).name, filelist(k).name),['../data/' filelist(k).name(1:end-6) '/']);
        end
    end
end