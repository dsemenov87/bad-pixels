#include <map> 
#include <math.h>
#include <opencv4/opencv2/core/matx.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include "opencv4/opencv2/highgui.hpp"
#include "iostream"
#include "numeric"

using namespace std;

const int MAX_DISTANCE = 4;
const int DISTANCE_DIFFERENCE_THRESHOLD = 2;

typedef pair<int, int> point_t ;

static point_t zero = make_pair(0, 0);

static float distance(point_t a, point_t b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

struct WeightPoint {
    point_t coords;
    float weight;
};

struct RecoveryParams
{
    point_t self;
    vector<WeightPoint> neighboors;
};

struct RecoveryContext
{
    const cv::Mat &img;
    vector<RecoveryParams> &recover_params;
};

static point_t get_closest(const RecoveryContext &ctx,
                                int deep,
                                point_t coords,
                                point_t step(point_t),
                                bool is_healthy(uchar)) {
    if (deep > MAX_DISTANCE)
        return zero;

    auto next = step(coords);
    
    // вышел за границу
    if (next.first < 0 || next.first >= ctx.img.rows || next.second < 0 || next.second >= ctx.img.cols)
        return zero;

    auto value = ctx.img.at<uchar>(next.first, next.second);   
    if (is_healthy(value))
        return next;
    
    return get_closest(ctx, deep + 1, next, step, is_healthy);
}

static point_t x1_step(point_t coords) { return { coords.first - 1, coords.second }; }
static point_t x2_step(point_t coords) { return { coords.first + 1, coords.second }; }
static point_t y1_step(point_t coords) { return { coords.first, coords.second + 1 }; }
static point_t y2_step(point_t coords) { return { coords.first, coords.second - 1 }; }

static point_t d1_step(point_t coords) { return { coords.first - 1, coords.second + 1 }; }
static point_t d2_step(point_t coords) { return { coords.first + 1, coords.second + 1 }; }
static point_t d3_step(point_t coords) { return { coords.first - 1, coords.second + 1 }; }
static point_t d4_step(point_t coords) { return { coords.first + 1, coords.second - 1 }; }

static vector<WeightPoint> make_interpol_line(const RecoveryContext& ctx,
                            bool is_healthy(uchar),
                            point_t self,
                            point_t dir1_step(point_t),
                            point_t dir2_step(point_t)) {
    auto p1 = get_closest(ctx, 0, self, dir1_step, is_healthy); 
    auto p2 = get_closest(ctx, 0, self, dir2_step, is_healthy); 
        
    vector<WeightPoint> result;
    
    auto dp1 = p1 == zero ? 0.0f : distance(p1, self);
    auto dp2 = p2 == zero ? 0.0f : distance(p2, self);
    
    if (abs(dp1 - dp2) > DISTANCE_DIFFERENCE_THRESHOLD) {
        if (dp1 > dp2) {
            p1 = zero;
        }
        else
            p2 = zero; 
    } 
    
    auto w1 = p1 == zero ? 0.0f : p2 == zero ? 0.5f : 0.5f * dp2 / (dp1 + dp2);
    auto w2 = p2 == zero ? 0.0f : p1 == zero ? 0.5f : 0.5f * dp1 / (dp1 + dp2);
 
    return vector<WeightPoint> { {p1, w1 }, {p2, w2 } };
}   

static bool is_line_complete(const vector<WeightPoint> &line) {
    for (const WeightPoint &point : line)
       if (point.coords == zero) return false; 
    return true;
}

/* --------------------------------
 * |d1            |y1           d2|
 * |              |               |
 * |              |               |
 * |              |               |
 * |--------------|---------------|
 * |x1            |             x2|
 * |              |               |
 * |              |               |
 * |d3            |y2           d4|
 * |--------------|---------------|
 */

static void make_recovery_params(const RecoveryContext& ctx, bool is_healthy(uchar), point_t self) {
    // horizontal - vertical neighboors
    auto x_line = make_interpol_line(ctx, is_healthy, self, x1_step, x2_step);
    auto y_line = make_interpol_line(ctx, is_healthy, self, y1_step, y2_step);
    
    RecoveryParams result;
    result.self = self;
    if (is_line_complete(x_line) && is_line_complete(y_line))
    {
        result.neighboors.push_back(x_line.at(0));
        result.neighboors.push_back(x_line.at(1));
        result.neighboors.push_back(y_line.at(0));
        result.neighboors.push_back(y_line.at(1));
        ctx.recover_params.push_back(result); 
        return;
    }
    
    // diagonal neighboors
    auto d12_line = make_interpol_line(ctx, is_healthy, self, d1_step, d2_step);
    auto d34_line = make_interpol_line(ctx, is_healthy, self, d3_step, d4_step);
    
    if (is_line_complete(d12_line) && is_line_complete(d34_line))
    {
        result.neighboors.push_back(d12_line.at(0));
        result.neighboors.push_back(d12_line.at(1));
        result.neighboors.push_back(d34_line.at(0));
        result.neighboors.push_back(d34_line.at(1));
        ctx.recover_params.push_back(result); 
        return;
    }
    
    result.neighboors.push_back({x_line.at(0).coords, 1});
    result.neighboors.push_back({x_line.at(1).coords, 1});
    result.neighboors.push_back({y_line.at(0).coords, 1});
    result.neighboors.push_back({y_line.at(1).coords, 1});
    ctx.recover_params.push_back(result); 
} 

static bool is_healthy(uchar pixel) { return (int)pixel < 155; } 

static void prepare_recovery(RecoveryContext &ctx) {
    for (int i = 0; i < ctx.img.rows; ++i) {
        for (int j = 0; j < ctx.img.cols; ++j) {
            auto intensity = ctx.img.at<uchar>(i, j);
            if (!is_healthy(intensity)) {
                make_recovery_params(ctx, is_healthy, make_pair(i, j)); 
            } 
        }
    }    
}

static void make_recovery(const vector<RecoveryParams> &recover_params, cv::Mat &img) {
    for (RecoveryParams rp : recover_params) {
        float approx_val = 0;
        for (WeightPoint wp: rp.neighboors) {
            approx_val += wp.weight * img.at<uchar>(wp.coords.first, wp.coords.second);
        }
        img.at<uchar>(rp.self.first, rp.self.second) = approx_val;
    }   
}

static pair<float, float> convert_to_opengl_display(point_t point, int x_size, int y_size) {
    auto x = (2.0f * (0.5f + point.second) / x_size) - 1.0f;
    auto y = 1.0f - (2.0f * (0.5f + point.first) / y_size);
    return make_pair(x ,y);
}

static pair<float, float> convert_to_opengl_texture(point_t point, int x_size, int y_size) {
    auto x = (point.second + 0.5f) / x_size;
    auto y = 1.0f - (point.first + 0.5f) / y_size;
    cout << x << ", " << y << endl;
    return make_pair(x ,y);
}

void get_pixel_recovery_params(const cv::Mat &image, vector<float*> &result) {
    auto *recovered_params = new vector<RecoveryParams>;
    RecoveryContext recovery_ctx = {image, *recovered_params};

    prepare_recovery(recovery_ctx); 
    
    for (RecoveryParams rp : recovery_ctx.recover_params) {
        float item[14];
        auto p = convert_to_opengl_display(rp.self, image.cols, image.rows);
        item[0] = p.first;
        item[1] = p.second;
        for (int i = 0; i < 12; i+=3) {
            auto n = rp.neighboors.at(i / 3); 
            p = convert_to_opengl_texture(n.coords, image.cols, image.rows);
            item[i + 2] = p.first;
            item[i + 3] = p.second;
            item[i + 4] = n.weight;
        }
        result.push_back(item);
    }   
    
    delete(recovered_params);
}

int main(int argc, char **argv) {
    auto image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    
    if(!image.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    auto *recovered_params = new vector<RecoveryParams>;
    RecoveryContext recovery_ctx = {image, *recovered_params};

    prepare_recovery(recovery_ctx); 
    
    make_recovery(*recovered_params, image);
    cv::namedWindow("Display window 1");
    cv::imshow("Display window 1", image);
  
    cv::waitKey(0); 
    delete(recovered_params);
    
    /*auto res = new vector<float*>;
    get_pixel_recovery_params(image, *res);
    for (auto item : *res) {
        for (int i = 0; i < 4; i++)
            cout << item[i] << ",";
        cout << endl;      
    }*/
    return 0;
}
