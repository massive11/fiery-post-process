#include "fiery_process.h"
using namespace std;

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}


int main()
{
    std::vector<char> f4 = get_the_bytes("/media/data/Git_repos/fiery_post_process/data/segmentation.pt");
    torch::IValue x = torch::pickle_load(f4);
    torch::Tensor my_tensor = x.toTensor();
    auto segmentation_prediction = my_tensor.squeeze(2);

    std::vector<char> f1 = get_the_bytes("/media/data/Git_repos/fiery_post_process/data/instance_center.pt");
    x = torch::pickle_load(f1);
    my_tensor = x.toTensor();
    auto center_prediction = my_tensor.squeeze(2);

    std::vector<char> f2 = get_the_bytes("/media/data/Git_repos/fiery_post_process/data/instance_offset.pt");
    x = torch::pickle_load(f2);
    auto offset_prediction = x.toTensor();

    std::vector<char> f3 = get_the_bytes("/media/data/Git_repos/fiery_post_process/data/instance_flow.pt");
    x = torch::pickle_load(f3);
    auto flow_prediction = x.toTensor();

    // 后处理，计算各辆车的中心点
    std::clock_t start_time;
    std::clock_t end_time;
    start_time = clock();
    vector<pair<int, vector<Point>>> matched_centers = predict_instance_segmentation_and_trajectories(segmentation_prediction, center_prediction, offset_prediction, flow_prediction);
    end_time = clock();
    cout<<"The run time is :"<<(double)(end_time-start_time) / CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"The matched_centers is :"<<matched_centers<<endl;
    return 0;
}