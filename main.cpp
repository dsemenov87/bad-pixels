#include <map> 
#include <math.h>

using namespace std;

const int MAX_DISTANCE = 4;
const int DISTANCE_DIFFERENCE_THRESHOLD = 2;

struct Pixel { pair<int, int> coords; float value; };

static float distance(pair<int, int> x, pair<int, int> y) {
    return sqrt(pow(x.first - y.first, 2) + pow(x.second - y.second, 2));
}

struct RecoveryContext {
    float **image;     
    int size_x;
    int size_y;
    map<pair<int, int>, float> recover_pixels;
};

static Pixel get_closest(const RecoveryContext& ctx,
                                int deep,
                                pair<int, int> coords,
                                pair<int, int> step(pair<int, int>),
                                bool is_healthy(float)) {
    Pixel zero = {pair<int, int>{0, 0}, 0};

    if (deep > MAX_DISTANCE)
        return zero;

    auto next = step(coords);
    if (next.first < 0 || next.first >= ctx.size_x)
        return zero;

    if (next.second < 0 || next.second >= ctx.size_y)
        return zero;

    auto value = ctx.image[next.first][next.second];   
    if (is_healthy(value))
        return Pixel {next, value};
    
    return get_closest(ctx, deep + 1, next, step, is_healthy);
}

static pair<int, int> x1_step(pair<int, int> coords) { return pair<int, int>{ coords.first - 1, coords.second }; }
static pair<int, int> x2_step(pair<int, int> coords) { return pair<int, int>{ coords.first + 1, coords.second }; }
static pair<int, int> y1_step(pair<int, int> coords) { return pair<int, int>{ coords.first, coords.second + 1 }; }
static pair<int, int> y2_step(pair<int, int> coords) { return pair<int, int>{ coords.first, coords.second - 1 }; }

static pair<int, int> d1_step(pair<int, int> coords) { return pair<int, int>{ coords.first - 1, coords.second + 1 }; }
static pair<int, int> d2_step(pair<int, int> coords) { return pair<int, int>{ coords.first + 1, coords.second + 1 }; }
static pair<int, int> d3_step(pair<int, int> coords) { return pair<int, int>{ coords.first - 1, coords.second + 1 }; }
static pair<int, int> d4_step(pair<int, int> coords) { return pair<int, int>{ coords.first + 1, coords.second - 1 }; }

static float calc_average_value(pair<int, int> coords, const Pixel& x1, const Pixel& x2,
                                    const Pixel& y1, const Pixel& y2) {
    auto dx1 = distance(x1.coords, coords);
    auto dx2 = distance(x2.coords, coords);
    auto dy1 = distance(y1.coords, coords);
    auto dy2 = distance(y2.coords, coords);
    
    auto vx =
         (dx1 - dx2) <= DISTANCE_DIFFERENCE_THRESHOLD ? (dx2 - dx1) <= DISTANCE_DIFFERENCE_THRESHOLD 
       ? (dx1 * x1.value + dx2 * x2.value) / (dx1 + dx2) : x2.value
       : x1.value;
    
    auto vy =
         (dy1 - dy2) <= DISTANCE_DIFFERENCE_THRESHOLD ? (dy2 - dy1) <= DISTANCE_DIFFERENCE_THRESHOLD 
       ? (dy1 * y1.value + dy2 * y2.value) / (dy1 + dy2) : y2.value 
       : y1.value;

    return 0.5 * (vx + vy);
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

static float recover_pixel(const RecoveryContext& ctx, bool is_healthy(float), pair<int, int> coords) {
    auto x1 = get_closest(ctx, 0, coords, x1_step, is_healthy); 
    auto x2 = get_closest(ctx, 0, coords, x2_step, is_healthy); 
    auto y1 = get_closest(ctx, 0, coords, y1_step, is_healthy); 
    auto y2 = get_closest(ctx, 0, coords, y2_step, is_healthy); 
    
    if (!x1.value && !x2.value && !y1.value && !y2.value) {
        auto d1 = get_closest(ctx, 0, coords, d1_step, is_healthy); 
        auto d2 = get_closest(ctx, 0, coords, d2_step, is_healthy); 
        auto d3 = get_closest(ctx, 0, coords, d3_step, is_healthy); 
        auto d4 = get_closest(ctx, 0, coords, d4_step, is_healthy); 
        
        return calc_average_value(coords, d1, d2, d3, d4);
    }
    else
        return calc_average_value(coords, x1, x2, y1, y2);
} 

RecoveryContext& check_pixel(RecoveryContext& ctx, const Pixel& pix, bool is_healthy_pixel(float)) {
    if (is_healthy_pixel(pix.value)) {
        return ctx;
    }
    else {
        auto recover_value = recover_pixel(ctx, is_healthy_pixel, pix.coords); 
        ctx.recover_pixels.insert({pix.coords, recover_value});
    } 
    return ctx;    
}



