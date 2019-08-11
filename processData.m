

org_root = 'shape_net_core_uniform_samples_2048_split/train';

folder_list = dir(org_root);

del_ratio=[20, 30, 40, 50, 70];

%for d=1:length(del_ratio)
d=5;
    new_root = strcat(org_root,'_', num2str(del_ratio(d)));
    mkdir(new_root);

    for f=1:length(folder_list)
       if(folder_list(f).name(1) == '.')
           continue;
       end
       category_folder= folder_list(f).name;
       mkdir(strcat(new_root, '/', category_folder));

       ply_list = dir(strcat(org_root, '/', category_folder));
       for p=1:length(ply_list)
           if(ply_list(p).isdir == 1)
               continue;
           end
           pc_org = pcread(strcat(org_root, '/', category_folder, '/', ply_list(p).name));

           %% remove a portion
           pc_removed=removePts(pc_org, del_ratio(d));
           pcwrite(pc_removed, strcat(new_root, '/', category_folder, '/', ply_list(p).name));
       end
    end
%end


function pc_removed =  removePts(pc_org, del_ratio)

pc = pc_org.Location;
seed_idx = round(rand() * pc_org.Count);
if seed_idx <1
    seed_idx=1;
end
if seed_idx > pc_org.Count
    pc_org.Count
end
seed = pc(seed_idx,:);

diff = pc - repmat(seed, pc_org.Count, 1);
dist_sq = sum(diff.*diff, 2);
[~, sort_idx] = sort(dist_sq);

pc(sort_idx(1:round(pc_org.Count*del_ratio/100)),:)=[];

pc_removed = pointCloud(pc);

end



