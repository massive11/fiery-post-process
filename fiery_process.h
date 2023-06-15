#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "Hungarian.h"
#include <cmath>
using namespace std;
using namespace torch::indexing;
typedef pair<float, float> Point;


vector<pair<int, vector<Point>>> predict_instance_segmentation_and_trajectories(const torch::Tensor& segmentation, const torch::Tensor& instance_center, const torch::Tensor& instance_offset, const torch::Tensor& instance_flow);
torch::Tensor get_instance_segmentation_and_centers(const torch::Tensor& center, const torch::Tensor& offset, const torch::Tensor& seg);
torch::Tensor find_instance_centers(const torch::Tensor& instance_center);
torch::Tensor group_pixels(const torch::Tensor& ins_centers, const torch::Tensor& offset_predictions);
torch::Tensor make_instance_seg_consecutive(torch::Tensor instance_seg);
torch::Tensor update_instance_ids(const torch::Tensor& instance_segmentation, const torch::Tensor& old_ids, const torch::Tensor& new_ids);
torch::Tensor make_instance_id_temporally_consistent(const torch::Tensor& pred_inst, const torch::Tensor& future_flow);
