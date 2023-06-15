#include "fiery_process.h"


// 更新instance的id
// correct
torch::Tensor update_instance_ids(const torch::Tensor& instance_segmentation, const torch::Tensor& old_ids, const torch::Tensor& new_ids)
{
    torch::Tensor instance_seg = instance_segmentation;
    // 更新instance的id
    auto max_id = (at::max(old_ids)).item().toInt() + 1;
    torch::Tensor indices = at::arange(max_id);
    auto temp_min = std::min(old_ids.size(0), new_ids.size(0));
    for(auto id=0;id<temp_min;++id)
    {
        indices.index_put_({old_ids[id].item().toInt()},new_ids[id].item().toInt());
    }
    // 在instance_seg上更新id
    for(int height_=0;height_<160;++height_)
    {
        for(int width_=0;width_<64;++width_)
        {
            if(instance_seg[0][height_][width_].item().toInt() != 0)
            {
                instance_seg.index_put_({0,height_,width_},indices[instance_seg[0][height_][width_].item().toInt()].item().toInt());
            }
        }
    }
    return instance_seg;
}


// 将实例分割的id更新为连续的id
// correct
torch::Tensor make_instance_seg_consecutive(torch::Tensor instance_seg)
{
    torch::Tensor temp = instance_seg.reshape({160*64});
    temp = get<0>(at::_unique(temp));
    torch::Tensor new_ids = torch::arange(temp.size(0));
    instance_seg = update_instance_ids(instance_seg, temp, new_ids);
    return instance_seg;
}


// Correct
torch::Tensor find_instance_centers(const torch::Tensor& instance_center)
{
    torch::Tensor center_prediction = at::threshold(instance_center, 0.1, -1);
    // 对center_prediction做一个max_pooling
    torch::Tensor maxpooled_center_prediction = at::max_pool2d(center_prediction, 3, 1, 1);
    // 然后将center_prediction与max_pooling的结果对比，值不同的位置均赋值-1
    center_prediction = at::masked_fill(center_prediction,center_prediction != maxpooled_center_prediction,-1);
    // 取出center_prediction中所有值>0的位置的索引，即车辆中心点的坐标
    return at::nonzero(center_prediction > 0).index({"...", Slice({1, None})});
}

// correct
torch::Tensor group_pixels(const torch::Tensor& ins_centers, const torch::Tensor& offset_predictions)
{
    torch::Tensor centers = ins_centers.view({ins_centers.size(0), 1, 2});
    // 每个点的原始位置加上offset得到真实位置
    torch::Tensor x_grid = torch::arange(160).reshape({1,160,1}).repeat({1,1,64});
    torch::Tensor y_grid = torch::arange(64).reshape({1,1,64}).repeat({1,160,1});
    torch::Tensor pixel_grid = torch::cat({x_grid,y_grid},0).reshape({2,160,64});
    // 调整一下shape方便后续做norm运算
    torch::Tensor center_locations = (offset_predictions+pixel_grid).view({2, 160*64, 1}).permute({2, 1, 0});
    // 计算各个中心点到BEV上其他所有位置的距离
    torch::Tensor temp = centers - center_locations;// temp的维度是[centers的数量，center_locations的数量，2]
    // 对temp最后一维进行norm
    temp = temp.reshape({centers.size(0)*center_locations.size(1),2});
    torch::Tensor distances = at::float_power(at::float_power(temp.index({"...",0}),2) + at::float_power(temp.index({"...",1}),2),0.5).reshape({centers.size(0),center_locations.size(1)});
    // distances的维度是[centers的数量，center_locations的数量]
    // 0为背景，每个instance_id需要+1
    torch::Tensor instance_id = at::argmin(distances, 0).reshape({1,160,64}) + 1;
    return instance_id;
}

// 计算实例分割和中心点
// correct
torch::Tensor get_instance_segmentation_and_centers(const torch::Tensor& center, const torch::Tensor& offset, const torch::Tensor& seg)
{
    torch::Tensor instance_center = center.view({1, 160, 64});
    torch::Tensor instance_offset = offset.view({2, 160, 64});
    torch::Tensor segmentation = seg.view({1, 160, 64});
    // 计算中心点
    torch::Tensor centers = find_instance_centers(instance_center);
    // 车辆数量太多的话，只保留前100个
    if(centers.size(0) > 100)
    {
        centers = centers.index({Slice({0, 100})});
    }
    // above correct.将160*64的图上的各个像素分类到各个车，用segmentation做一次mask
    torch::Tensor instance_ids = group_pixels(centers, instance_offset);
    // segmentation的值只有0和1两种可能，为0说明是背景
    instance_ids = at::masked_fill(instance_ids,segmentation==0,0);
    torch::Tensor instance_seg = make_instance_seg_consecutive(instance_ids);
    return instance_seg;
}

// 匹配跟踪
torch::Tensor make_instance_id_temporally_consistent(const torch::Tensor& pred_inst_result, const torch::Tensor& instance_flow)
{
    // pred_inst_result：1, 5, 1, 160, 64;instance_flow：1, 5, 2, 160, 64
    // 取出第0个时刻的instance_segmentation结果
    // outer_consistent_instance_seg用来做stack
    vector<torch::Tensor> outer_consistent_instance_seg;
    outer_consistent_instance_seg.push_back(pred_inst_result[0][0]);

    // 计算当前实例分割的最大的instance id
    torch::Tensor temp = outer_consistent_instance_seg[0].reshape({160*64});
    int largest_instance_id = at::max(temp).item().toInt();
    // correct
    at::TensorOptions opts=at::TensorOptions().dtype(at::kInt);;
    for(int seq=0;seq<4;++seq)
    {
        // 计算当前的全部instance id（即consistent_instance_seg上的不重复元素）
        temp = outer_consistent_instance_seg[outer_consistent_instance_seg.size()-1].reshape({160*64});
        temp = get<0>(at::_unique(temp)); // temp存的是当前实例分割的全部instance id，包括0
        // 没有instance，无需更新
        if(temp.size(0)==1)
        {
            outer_consistent_instance_seg.push_back(pred_inst_result[0][seq+1]);
            continue;
        }

        // 给每个网格坐标添加future flow
        torch::Tensor x_grid = torch::arange(160).reshape({1,160,1}).repeat({1,1,64});
        torch::Tensor y_grid = torch::arange(64).reshape({1,1,64}).repeat({1,160,1});
        torch::Tensor grid = torch::cat({x_grid,y_grid},0).reshape({2,160,64});
        torch::Tensor future_flow = instance_flow[0][seq] + grid; // future_flow维度为[2,160,64]

        vector<Point> warped_centers;
        // temp存的是当前的全部instance id（即consistent_instance_seg上的不重复元素），有序
        // 从第一个开始取,0表示背景
        torch::Tensor temp_consistent_seg = outer_consistent_instance_seg[outer_consistent_instance_seg.size()-1];
        for(int cut_tensor=1;cut_tensor<temp.size(0);++cut_tensor)
        {
            torch::Tensor mask_temp = temp_consistent_seg[0] == temp[cut_tensor];
            torch::Tensor centers_x = future_flow[0].masked_select(mask_temp);
            torch::Tensor centers_y = future_flow[1].masked_select(mask_temp);
            float a = at::sum(centers_x).item().toFloat() / centers_x.size(0);
            float b = at::sum(centers_y).item().toFloat() / centers_x.size(0);
            Point pa(a,b);
            warped_centers.push_back(pa);
        }
        // 计算实际的future instance的均值
        vector<Point> centers;
        torch::Tensor temp_instance = pred_inst_result[0][seq+1].reshape({160*64});
        // 最大的instance id
        int n_instances = at::max(temp_instance).item().toInt();

        // 如果没有instance就不需要更新
        if(n_instances==0)
        {
            outer_consistent_instance_seg.push_back(pred_inst_result[0][seq+1]);
            continue;
        }

        // 全部instance id
        // tuple<torch::Tensor, torch::Tensor> unique_id = at::_unique(temp_instance);
        // torch::Tensor t_instance_ids = get<0>(unique_id);
        torch::Tensor t_instance_ids = temp;
        for(int ins=1;ins<n_instances+1;++ins)
        {
            torch::Tensor mask_temp = pred_inst_result[0][seq+1][0] == ins ;
            torch::Tensor centers_x = grid[0].masked_select(mask_temp);
            torch::Tensor centers_y = grid[1].masked_select(mask_temp);
            float a = at::sum(centers_x).item().toFloat() / centers_x.size(0);
            float b = at::sum(centers_y).item().toFloat() / centers_x.size(0);
            Point pa(a,b);
            centers.push_back(pa);
        }
        // 将centers和warped_centers做一个类型转换,方便后续计算
        int center_len = warped_centers.size();
        torch::Tensor t_warped_centers = torch::ones({center_len,2});
        for(int center_seq=0;center_seq<center_len;++center_seq)
        {
            t_warped_centers.index_put_({center_seq,0},warped_centers[center_seq].first);
            t_warped_centers.index_put_({center_seq,1},warped_centers[center_seq].second);
        }
        center_len = centers.size();
        torch::Tensor t_centers = torch::ones({center_len,2});
        for(int center_seq=0;center_seq<center_len;++center_seq)
        {
            t_centers.index_put_({center_seq,0},centers[center_seq].first);
            t_centers.index_put_({center_seq,1},centers[center_seq].second);
        }

        // Compute distance matrix between warped centers and actual centers
        torch::Tensor temp1 = t_centers.unsqueeze(0) - t_warped_centers.unsqueeze(1);
        // temp1 = temp1.view({t_centers.size(0)*t_warped_centers.size(0),2});
        // torch::Tensor distances = at::float_power(at::float_power(temp1.index({"...",0}),2) + at::float_power(temp1.index({"...",1}),2),0.5).view({t_centers.size(0),t_warped_centers.size(0)});
        torch::Tensor distances=torch::zeros({temp1.size(0),temp1.size(1)});;
        for(int d_i=0;d_i<temp1.size(0);++d_i)
        {
            for(int d_j=0;d_j<temp1.size(1);++d_j)
            {
                float dis = pow(pow(temp1[d_i][d_j][0].item().toFloat(),2.0)+pow(temp1[d_i][d_j][1].item().toFloat(),2.0),0.5);
                distances.index_put_({d_i,d_j},dis);
            }
        }
        
        vector<int> ids_t, ids_t_one; // ids_t为frame t的index，ids_t_one为t+1的index

        // 类型转换
        vector<vector<double>> DistMatrix(distances.size(0),vector<double>(distances.size(1)));
        for(int d_i=0;d_i<distances.size(0);++d_i)
        {
            vector<float> c(distances[d_i].data_ptr<float>(), distances[d_i].data_ptr<float>() + distances[d_i].numel());
            vector<double> temp_c(c.begin(),c.end());
            DistMatrix[d_i] = temp_c;
        }

        HungarianAlgorithm HungAlgo;
        vector<int> assignment;
        HungAlgo.Solve(DistMatrix,assignment);
        vector<vector<int>> temp_matrix(DistMatrix.size(),vector<int>(DistMatrix.size()));
        for(int x=0;x<DistMatrix.size();++x)
        {
            if(assignment[x]==-1)
            {
                continue;
            }else
            {
                if(DistMatrix[x][assignment[x]] > 1e16)
                {
                    continue;
                }
                ids_t.push_back(x);
                ids_t_one.push_back(assignment[x]);
            }
        }
        // cout<<seq<<" 317:"<<ids_t<<endl;
        // cout<<seq<<" 318:"<<ids_t_one<<endl;
        int temp_len = ids_t.size();
        torch::Tensor matching_distances = torch::ones({temp_len});
        for(int temp_seq=0;temp_seq<temp_len;++temp_seq)
        {
            matching_distances.index_put_({temp_seq}, distances[ids_t[temp_seq]][ids_t_one[temp_seq]].item().toFloat());
        }
        // Offset by 1, 0是background
        for(int & id : ids_t)
        {
            id += 1;
        }
        for(int & id : ids_t_one)
        {
            id += 1;
        }
        // swap ids_t with real ids. as those ids correspond to the position in the distance matrix
        map<int,int> id_mapping;
        for(int map_seq=1;map_seq<t_instance_ids.size(0);++map_seq)
        {
            id_mapping[map_seq] = t_instance_ids[map_seq].item().toInt();
        }
        // swap ids_t with real ids
        for(int id_seq=0;id_seq<temp_len;++id_seq)
        {
            ids_t[id_seq] = id_mapping[ids_t[id_seq]];
        }
        // Filter low quality match
        float matching_threshold = 3.0;
        vector<int> new_ids_t;
        vector<int> new_ids_t_one;
        for(int er_seq=0;er_seq<ids_t.size();++er_seq)
        {
            if(matching_distances[er_seq].item().toFloat()<matching_threshold)
            {
                new_ids_t.push_back(ids_t[er_seq]);
                new_ids_t_one.push_back(ids_t_one[er_seq]);
            }
        }
        // Elements that are in t+1, but weren't matched
        torch::Tensor next_temp_instance = pred_inst_result[0][seq+1].reshape({160*64});
        // 全部instance id
        // tuple<torch::Tensor, torch::Tensor> next_unique_id = at::_unique(next_temp_instance);
        torch::Tensor next_t_instance_ids = get<0>(at::_unique(next_temp_instance));
        set<int> last_id(new_ids_t_one.begin(),new_ids_t_one.end());
        set<int> remaining_ids;
        for(int i=0;i<next_t_instance_ids.size(0);++i)
        {
            if(last_id.find(next_t_instance_ids[i].item().toInt()) == last_id.end())
            {
                remaining_ids.insert(next_t_instance_ids[i].item().toInt());
            }
        }
        // remove background
        remaining_ids.erase(0);
        // Set remaining_ids to a new unique id
        for(int remaining_id : remaining_ids)
        {
            largest_instance_id += 1;
            new_ids_t.push_back(largest_instance_id);
            new_ids_t_one.push_back(remaining_id);
        }
        // 类型转换
        torch::Tensor old_id=at::from_blob(new_ids_t_one.data(),new_ids_t_one.size(),opts).clone();
        torch::Tensor new_id=at::from_blob(new_ids_t.data(),new_ids_t_one.size(),opts).clone();
        outer_consistent_instance_seg.push_back(update_instance_ids(pred_inst_result[0][seq+1], old_id, new_id));
    }
    torch::Tensor consistent_instance_seg = torch::stack({outer_consistent_instance_seg[0],outer_consistent_instance_seg[1],outer_consistent_instance_seg[2],outer_consistent_instance_seg[3],outer_consistent_instance_seg[4]},0).unsqueeze(0);
    return consistent_instance_seg;
}


vector<pair<int, vector<Point>>> predict_instance_segmentation_and_trajectories(const torch::Tensor& segmentation, const torch::Tensor& instance_center, const torch::Tensor& instance_offset, const torch::Tensor& instance_flow)
{
    // 1 instance_segmentation
    vector<torch::Tensor> pred_inst;
    for(int seq=0;seq<5;++seq)
    {
        // 得到实例分割和中心点
        torch::Tensor pred_inst_t = get_instance_segmentation_and_centers(instance_center[0][seq], instance_offset[0][seq], segmentation[0][seq]);
        pred_inst.emplace_back(pred_inst_t); // pred_inst_t：[1,160,64]
    }

    // 这里可以得到5个时刻的instance_segmentation结果，但是同一辆车在不同时刻的id不统一，需要进行匹配
    torch::Tensor pred_inst_result = torch::stack({pred_inst[0],pred_inst[1], pred_inst[2], pred_inst[3], pred_inst[4]},0).unsqueeze(0);// pred_inst_result:[1, 5, 160, 64]

    // 2 匹配
    torch::Tensor consistent_instance_seg;
    bool make_consistent = 1;
    if(make_consistent)
    {
        consistent_instance_seg = make_instance_id_temporally_consistent(pred_inst_result, instance_flow);
    }

    // 3
    vector<pair<int, vector<Point>>> matched_centers;
    // 填充一个grid
    torch::Tensor x_grid = torch::arange(160).reshape({1,160,1}).repeat({1,1,64});
    torch::Tensor y_grid = torch::arange(64).reshape({1,1,64}).repeat({1,160,1});
    torch::Tensor grid = torch::cat({x_grid,y_grid},0).reshape({2,160,64});

    torch::Tensor temp = consistent_instance_seg[0][0].reshape({160*64});
    // 取出consistent_instance_seg上的不重复元素
    temp = get<0>(at::_unique(temp));
    torch::Tensor instance_mask;
    torch::Tensor mask_temp;
    torch::Tensor centers_x;
    torch::Tensor centers_y;
    float a = 0;
    float b=0;
    for(int id = 1 ; id < temp.size(0); id++)
    {
        vector<Point> Centers;
        for(int t = 0 ; t < 5 ; t++)
        {
            // make a mask of this instance
            instance_mask = torch::zeros({160,64});
            instance_mask = at::masked_fill(instance_mask,consistent_instance_seg[0][t][0] == temp[id],1);
            if(at::sum(instance_mask).item().toFloat() > 0)
            {
                mask_temp = instance_mask !=0 ;
                centers_x = grid[0].masked_select(mask_temp);
                centers_y = grid[1].masked_select(mask_temp);
                a = at::sum(centers_x).item().toFloat() / centers_x.size(0);
                b = at::sum(centers_y).item().toFloat() / centers_x.size(0);
                Point pa(a,b);
                Centers.push_back(pa);
            }
        }
        pair<int, vector<Point>> po(temp[id].item().toInt(),Centers);
        matched_centers.push_back(po);
    }
    return matched_centers;
}