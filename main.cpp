#include <map> 
#include <math.h>

using namespace std;

const int MAX_DISTANCE = 4;
const int DISTANCE_DIFFERENCE_THRESHOLD = 4;

struct Pixel { pair<int, int> coords; float value; };

static float dist(pair<int, int> x, pair<int, int> y) {
    return sqrt(pow(x.first - y.first, 2) + pow(x.second - y.second, 2));
}

struct RecoveryContext {
    float **image;     
    int size_x;
    int size_y;
    map<pair<int, int>, float> recover_pixels;
};

static pair<int, int> left_step(pair<int, int> coords) { return pair<int, int>{ coords.first - 1, coords.second }; }
static pair<int, int> right_step(pair<int, int> coords) { return pair<int, int>{ coords.first + 1, coords.second }; }
static pair<int, int> up_step(pair<int, int> coords) { return pair<int, int>{ coords.first, coords.second + 1 }; }
static pair<int, int> down_step(pair<int, int> coords) { return pair<int, int>{ coords.first, coords.second - 1 }; }

static Pixel get_closest(const RecoveryContext& ctx,
                                pair<int, int> coords,
                                pair<int, int> step(pair<int, int>),
                                bool is_bad_pixel(float)) {
    pair<int, int> next = step(coords);
    if (next.first < 0 || next.first >= ctx.size_x)
        return Pixel {pair<int, int>{0, 0}, 0};

    if (next.second < 0 || next.second >= ctx.size_y)
        return Pixel {pair<int, int>{0, 0}, 0};

    auto value = ctx.image[next.first][next.second];   
    if (!is_bad_pixel(value))
        return Pixel {next, value};
    
    return get_closest(ctx, next, step, is_bad_pixel);
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

static float approximate_pixel(const RecoveryContext& ctx, bool is_bad_pixel(float), pair<int, int> coords) {
    auto x1 = get_closest(ctx, coords, left_step, is_bad_pixel); 
    auto x2 = get_closest(ctx, coords, right_step, is_bad_pixel); 
    auto y1 = get_closest(ctx, coords, up_step, is_bad_pixel); 
    auto y2 = get_closest(ctx, coords, down_step, is_bad_pixel); 
    float dx1 = dist(x1.coords, coords);
    float dx2 = dist(x2.coords, coords);
    float dy1 = dist(y1.coords, coords);
    float dy2 = dist(y2.coords, coords);
    float vx = (dx1 * x1.value + dx2 * x2.value) / (dx1 + dx2);
    float vy = (dy1 * y1.value + dy2 * y2.value) / (dy1 + dy2);
    return 0.5 * (vx + vy);
} 

RecoveryContext& make_recovery(RecoveryContext& ctx, const Pixel& pix, bool is_bad_pixel(float)) {
    if (!is_bad_pixel(pix.value)) {
        return ctx;
    }
    else {
        auto recover_value = approximate_pixel(ctx, is_bad_pixel, pix.coords); 

        ctx.recover_pixels.insert({pix.coords, recover_value});
    } 
    
    return ctx;    
}



