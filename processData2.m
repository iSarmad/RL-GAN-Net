

org_root = 'shape_net_core_uniform_samples_2048';

folder_list = dir(org_root);

split_ratio=85


new_root = strcat(org_root,'_', 'split');
train_root = strcat(new_root,'/','train');
test_root = strcat(new_root,'/','test');
mkdir(train_root);
mkdir(test_root);

    for f=1:length(folder_list)
       if(folder_list(f).name(1) == '.'||isdir(strcat(org_root, '/',folder_list(f).name)) ==0)
           continue;
       end
       category_folder= folder_list(f).name;
       mkdir(strcat(test_root, '/', category_folder));
       mkdir(strcat(train_root, '/', category_folder));

       ply_list = dir(strcat(org_root, '/', category_folder));
       nrow = length(ply_list);
       ntrain = floor(nrow * split_ratio/100);
       train_ind = randperm(nrow, ntrain);
       
       for p=1:length(ply_list)
           if(ply_list(p).isdir == 1)
               continue;
           end
        file_tocopy =strcat(org_root, '/', category_folder, '/', ply_list(p).name);
        if(ismember(p,train_ind))
            copyfile(file_tocopy,strcat(train_root, '/', category_folder, '/', ply_list(p).name))
        else
            copyfile(file_tocopy,strcat(test_root, '/', category_folder, '/', ply_list(p).name))
        end
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



