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

struct RecoveryParams
{
    point_t self;
    point_t neighboors[4];    
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

static void make_interpol_line(const RecoveryContext& ctx,
                            point_t p[2],
                            bool is_healthy(uchar),
                            point_t self,
                            point_t dir1_step(point_t),
                            point_t dir2_step(point_t)) {
    p[0] = get_closest(ctx, 0, self, dir1_step, is_healthy); 
    p[1] = get_closest(ctx, 0, self, dir2_step, is_healthy); 
    
    auto dp1 = distance(p[0], self);
    auto dp2 = distance(p[1], self);
    
    if (abs(dp1 - dp2) > DISTANCE_DIFFERENCE_THRESHOLD) {
        if (dp1 > dp2) p[0] = zero; else p[1] = zero; 
    } 
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
    // x - y neighboors
    auto x_line = new point_t[2];
    make_interpol_line(ctx, x_line, is_healthy, self, x1_step, x2_step);

    auto y_line = new point_t[2];
    make_interpol_line(ctx, y_line, is_healthy, self, y1_step, y2_step);
    
    if ((x_line[0] != zero && x_line[1] != zero) // найдена горизонтальная линия интерполяции
     || (y_line[0] != zero && y_line[1] != zero))// найдена вертикальная линия интерполяции 
    {
        ctx.recover_params.push_back(RecoveryParams {self, } 
        return;
    }
} 

bool is_healthy(uchar pixel) { return (int)pixel < 155; } 

void first_iteration(RecoveryContext ctx) {
    for (int i = 0; i < ctx.img.rows; ++i) {
        for (int j = 0; j < ctx.img.cols; ++j) {
            auto pix = ctx.img.at<uchar>(i, j);
            if (!is_healthy(pix)) {
                auto coords = make_pair(i, j);
                auto recover_value = recover_pixel(ctx, is_healthy, coords); 
                ctx.recovered->insert(make_pair(coords, recover_value));
            } 
        }
    }    
    
    for (pair<point_t, uchar> pixel: *ctx.recovered) {
        auto coords = pixel.first;
        auto recover_value = pixel.second; 
        ctx.img.at<uchar>(coords.first, coords.second) = recover_value;
    }   
}

void second_iteration(RecoveryContext ctx) {
    for (pair<point_t, uchar> pixel: *ctx.recovered) {
        auto coords = pixel.first;
        auto intensity = ctx.img.at<uchar>(coords.first, coords.second);
        if (!is_healthy(intensity)) {
            auto recovered = recover_pixel(ctx, is_healthy, coords); 
            ctx.img.at<uchar>(coords.first, coords.second) = recovered;
        } 
    }
}

int main(int argc, char **argv) {
    auto image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    
    if(!image.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    auto *recovered_pixels = new map<point_t, uchar>;
    RecoveryContext recovery_ctx = {image, recovered_pixels};

    // восстанавливаем битые пиксели, используя соседние НЕ БИТЫЕ пиксели.
    first_iteration(recovery_ctx); 
    
    /* проделываем ту же процедуру осреднения, что и в первый раз,
     * но используем пиксели, которые были в прошлый раз битые, а теперь - нет.
     */
    second_iteration(recovery_ctx);
    
    cv::namedWindow("Display window 1");
    cv::imshow("Display window 1", image);
  
    cv::waitKey(0); 
    
    delete(recovered_pixels);
    return 0;
}

