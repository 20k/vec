#ifndef VEC_HPP_INCLUDED
#define VEC_HPP_INCLUDED

#include <math.h>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <float.h>
#include <random>

#define M_PI		3.14159265358979323846
#define M_PIf ((float)M_PI)

///bad, only for temporary debugging
#define EXPAND_3(vec) vec.v[0], vec.v[1], vec.v[2]
#define EXPAND_2(vec) vec.v[0], vec.v[1]

template<int N, typename T>
struct vec
{
    T v[N];

    vec(std::initializer_list<T> init)
    {
        if(init.size() == 1)
        {
            for(int i=0; i<N; i++)
            {
                v[i] = *(init.begin());
            }
        }
        else
        {
            int i;

            for(i=0; i<init.size(); i++)
            {
                v[i] = *(init.begin() + i);
            }

            for(; i<N; i++)
            {
                v[i] = 0.f;
            }
        }
    }

    ///lets modernise the code a little
    vec()
    {
        for(int i=0; i<N; i++)
        {
            v[i] = T();
        }
    }

    vec(T val)
    {
        for(int i=0; i<N; i++)
        {
            v[i] = val;
        }
    }

    vec<N, T>& operator=(T val)
    {
        for(int i=0; i<N; i++)
        {
            v[i] = val;
        }

        return *this;
    }

    vec<N, T> operator+(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] + other.v[i];
        }

        return r;
    }

    vec<N, T> operator+(T other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] + other;
        }

        return r;
    }

    vec<N, T>& operator+=(T other)
    {
        *this = *this + other;

        return *this;
    }

    vec<N, T>& operator+=(const vec<N, T>& other)
    {
        *this = *this + other;

        return *this;
    }

    vec<N, T> operator-(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] - other.v[i];
        }

        return r;
    }


    vec<N, T> operator-(T other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] - other;
        }

        return r;
    }


    vec<N, T> operator*(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] * other.v[i];
        }

        return r;
    }

    ///beginnings of making this actually work properly
    template<typename U>
    vec<N, T> operator*(U other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] * other;
        }

        return r;
    }

    vec<N, T> operator/(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] / other.v[i];
        }

        return r;
    }

    vec<N, T> operator/(T other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] / other;
        }

        return r;
    }

    inline
    T squared_length() const
    {
        T sqsum = 0;

        for(int i=0; i<N; i++)
        {
            sqsum += v[i]*v[i];
        }

        return sqsum;
    }

    inline
    T length() const
    {
        T l = squared_length();

        T val = sqrt(l);

        return val;
    }

    inline
    T lengthf() const
    {
        T l = squared_length();

        T val = sqrtf(l);

        return val;
    }

    double length_d() const
    {
        double l = 0;

        for(int i=0; i<N; i++)
        {
            l += v[i]*v[i];
        }

        return sqrt(l);
    }

    float sum() const
    {
        float accum = 0;

        for(int i=0; i<N; i++)
        {
            accum += v[i];
        }

        return accum;
    }

    float sum_absolute() const
    {
        float accum = 0;

        for(int i=0; i<N; i++)
        {
            accum += fabs(v[i]);
        }

        return accum;
    }

    float max_elem() const
    {
        float val = -FLT_MAX;

        for(const auto& s : v)
        {
            if(s > val)
                val = s;
        }

        return val;
    }

    float min_elem() const
    {
        float val = FLT_MAX;

        for(const auto& s : v)
        {
            if(s < val)
                val = s;
        }

        return val;
    }

    int which_element_minimum() const
    {
        float val = FLT_MAX;
        int num = -1;

        for(int i=0; i<N; i++)
        {
            if(v[i] < val)
            {
                val = v[i];
                num = i;
            }
        }

        return num;
    }

    float largest_elem() const
    {
        float val = -1;

        for(const auto& s : v)
        {
            float r = fabs(s);

            if(r > val)
                val = r;
        }

        return val;
    }


    vec<N, T> norm() const
    {
        T len = length();

        if(len < 0.00001f)
        {
            vec<N, T> ret;

            for(int i=0; i<N; i++)
                ret.v[i] = 0.f;

            return ret;
        }

        return (*this) / len;
    }

    ///only makes sense for a vec3f
    ///swap to matrices once I have any idea what is life
    ///glm or myself (;_;)
    vec<N, T> rot(const vec<3, float>& pos, const vec<3, float>& rotation) const
    {
        vec<3, float> c;
        vec<3, float> s;

        for(int i=0; i<3; i++)
        {
            c.v[i] = cos(rotation.v[i]);
            s.v[i] = sin(rotation.v[i]);
        }

        vec<3, float> rel = *this - pos;

        vec<3, float> ret;

        ret.v[0] = c.v[1] * (s.v[2] * rel.v[1] + c.v[2]*rel.v[0]) - s.v[1]*rel.v[2];
        ret.v[1] = s.v[0] * (c.v[1] * rel.v[2] + s.v[1]*(s.v[2]*rel.v[1] + c.v[2]*rel.v[0])) + c.v[0]*(c.v[2]*rel.v[1] - s.v[2]*rel.v[0]);
        ret.v[2] = c.v[0] * (c.v[1] * rel.v[2] + s.v[1]*(s.v[2]*rel.v[1] + c.v[2]*rel.v[0])) - s.v[0]*(c.v[2]*rel.v[1] - s.v[2]*rel.v[0]);

        return ret;
    }

    vec<N, T> back_rot(const vec<3, float>& position, const vec<3, float>& rotation) const
    {
        vec<3, float> new_pos = this->rot(position, (vec<3, float>){-rotation.v[0], 0, 0});
        new_pos = new_pos.rot(position, (vec<3, float>){0, -rotation.v[1], 0});
        new_pos = new_pos.rot(position, (vec<3, float>){0, 0, -rotation.v[2]});

        return new_pos;
    }

    ///only valid for a 2-vec
    ///need to rejiggle the templates to work this out
    vec<2, T> rot(T rot_angle)
    {
        T len = length();

        if(len < 0.00001f)
            return *this;

        T cur_angle = angle();

        T new_angle = cur_angle + rot_angle;

        T nx = len * cos(new_angle);
        T ny = len * sin(new_angle);

        return {nx, ny};
    }

    T angle()
    {
        return atan2(v[1], v[0]);
    }

    ///from top
    vec<3, T> get_euler() const
    {
        static_assert(N == 3, "Can only convert 3 element vectors into euler angles");

        vec<3, T> dir = *this;

        float cangle = dot((vec<3, T>){0, 1, 0}, dir.norm());

        float angle2 = acos(cangle);

        float y = atan2(dir.v[2], dir.v[0]);

        ///z y x then?
        vec<3, T> rot = {0, y, angle2};

        return rot;
    }

    ///so y is along rectangle axis
    ///remember, z is not along the rectangle axis, its perpendicular to x
    ///extrinsic x, y, z
    ///intrinsic z, y, x

    /*vec<3, T> get_euler_alt() const
    {
        vec<3, T> dir = *this;

        vec<3, T> pole = {0, 1, 0};

        float angle_to_pole = -atan2(dir.v[2], dir.v[1]);//acos(dot(pole, dir.norm()));

        float zc = angle_to_pole;//M_PI/2.f - angle_to_pole;

        float xc = atan2(dir.v[0], dir.v[1]) + M_PI;

        //static float test = 0.f;
        //test += 0.001f;

        vec<3, T> rot = {zc, 0.f, -xc};

        return rot;
    }*/

    operator float() const
    {
        static_assert(N == 1, "Implicit float can conversion only be used on vec<1,T> types");

        return v[0];
    }

    friend std::ostream& operator<<(std::ostream& os, const vec<N, T>& v1)
    {
        for(int i=0; i<N-1; i++)
        {
            os << std::to_string(v1.v[i]) << " ";
        }

        os << std::to_string(v1.v[N-1]);

        return os;
    }
};

template<int N, typename T>
inline
vec<2, T> s_xz(const vec<N, T>& v)
{
    static_assert(N >= 3, "Not enough elements for xz swizzle");

    return {v.v[0], v.v[2]};
}

template<int N, typename T>
inline
vec<2, T> s_xy(const vec<N, T>& v)
{
    static_assert(N >= 2, "Not enough elements for xy swizzle");

    return {v.v[0], v.v[1]};
}

template<int N, typename T>
inline
vec<2, T> s_yz(const vec<N, T>& v)
{
    static_assert(N >= 3, "Not enough elements for xy swizzle");

    return {v.v[1], v.v[2]};
}

template<int N, typename T>
inline
vec<2, T> s_zy(const vec<N, T>& v)
{
    static_assert(N >= 3, "Not enough elements for xy swizzle");

    return {v.v[2], v.v[1]};
}

template<int N, typename T>
inline
vec<3, T> s_xz_to_xyz(const vec<2, T>& v)
{
    return {v.v[0], 0.f, v.v[1]};
}

template<int N, typename T>
inline
vec<N, T> sqrtf(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = sqrtf(v.v[i]);
    }

    return ret;
}

///r theta, phi
template<typename T>
inline
vec<3, T> cartesian_to_polar(const vec<3, T>& cartesian)
{
    float r = cartesian.length();
    float theta = acos(cartesian.v[2] / r);
    float phi = atan2(cartesian.v[1], cartesian.v[0]);

    return {r, theta, phi};
}

template<typename T>
inline
vec<3, T> polar_to_cartesian(const vec<3, T>& polar)
{
    float r = polar.v[0];
    float theta = polar.v[1];
    float phi = polar.v[2];

    float x = r * sin(theta) * cos(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(theta);

    return {x, y, z};
}

template<typename T>
inline
vec<2, T> radius_angle_to_vec(T radius, T angle)
{
    vec<2, T> ret;

    ret.v[0] = cos(angle) * radius;
    ret.v[1] = sin(angle) * radius;

    return ret;
}

template<int N, typename T>
inline
vec<N, T> round(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = roundf(v.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> round_to_multiple(const vec<N, T>& v, int multiple)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v.v[i] / multiple;
        ret.v[i] = round(ret.v[i]);
        ret.v[i] *= multiple;
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> vcos(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = cosf(v.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> vsin(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = sinf(v.v[i]);
    }

    return ret;
}

/*template<int N, typename T>
bool operator<(const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
    {
        if(v1.v[i] < v2.v[i])
            return true;
        if(v1.v[i] > v2.v[i])
            return false;
    }

    return false;
}*/

template<int N, typename T>
bool operator<(const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
        if(v1.v[i] >= v2.v[i])
            return false;

    return true;
}

template<int N, typename T>
bool operator>(const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
        if(v1.v[i] <= v2.v[i])
            return false;

    return true;
}

template<int N, typename T>
bool operator== (const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
        if(v1.v[i] != v2.v[i])
            return false;

    return true;
}

template<int N, typename T>
bool operator>= (const vec<N, T>& v1, const vec<N, T>& v2)
{
    return v1 > v2 || v1 == v2;
}

#define V3to4(x) {x.v[0], x.v[1], x.v[2], x.v[3]}

typedef vec<4, float> vec4f;
typedef vec<3, float> vec3f;
typedef vec<2, float> vec2f;

typedef vec<3, int> vec3i;
typedef vec<2, int> vec2i;

template<int N, typename T>
inline
vec<N, T> val_to_vec(float val)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = val;
    }

    return ret;
}
inline float randf_s()
{
    return (float)rand() / (RAND_MAX + 1.f);
}

///both of these functions are stolen shamelessly off stackoverflow
///https://stackoverflow.com/questions/686353/c-random-float-number-generation
inline float randf_s(float M, float N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;
}

template<int N, typename T>
bool any_nan(const vec<N, T>& v)
{
    for(int i=0; i<N; i++)
    {
        if(std::isnan(v.v[i]))
            return true;
    }

    return false;
}

template<int N, typename T>
inline
vec<N, T> randf(float M, float MN)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s(M, MN);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> randv(const vec<N, T>& vm, const vec<N, T>& vmn)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s(vm.v[i], vmn.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> randf()
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s();
    }

    return ret;
}

inline
float rand_det_s(std::minstd_rand& rnd, float M, float MN)
{
    float scaled = (rnd() - rnd.min()) / (float)(rnd.max() - rnd.min());

    return scaled * (MN - M) + M;
}

template<int N, typename T>
inline
vec<N, T> rand_det(std::minstd_rand& rnd, const vec<N, T>& M, const vec<N, T>& MN)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = rand_det_s(rnd, M.v[i], MN.v[i]);
    }

    return ret;
}

template<int N, typename T, typename U, typename V>
inline
vec<N, T> clamp(vec<N, T> v1, U p1, V p2)
{
    for(int i=0; i<N; i++)
    {
        v1.v[i] = v1.v[i] < p1 ? p1 : v1.v[i];
        v1.v[i] = v1.v[i] > p2 ? p2 : v1.v[i];
    }

    return v1;
}

template<typename T, typename U, typename V>
inline
T clamp(T v1, U p1, V p2)
{
    v1 = v1 < p1 ? p1 : v1;
    v1 = v1 > p2 ? p2 : v1;

    return v1;
}

template<int N, typename T, typename U, typename V>
inline
vec<N, T> clamp(const vec<N, T>& v1, const vec<N, U>& p1, const vec<N, V>& p2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v1.v[i] < p1.v[i] ? p1.v[i] : v1.v[i];
        ret.v[i] = ret.v[i] > p2.v[i] ? p2.v[i] : ret.v[i];
    }

    return ret;
}


///0 -> 1, returns packed RGBA uint
inline
uint32_t rgba_to_uint(const vec<4, float>& rgba)
{
    vec<4, float> val = clamp(rgba, 0.f, 1.f);

    uint8_t r = val.v[0] * 255;
    uint8_t g = val.v[1] * 255;
    uint8_t b = val.v[2] * 255;
    uint8_t a = val.v[3] * 255;

    uint32_t ret = (r << 24) | (g << 16) | (b << 8) | a;

    return ret;
}

inline
uint32_t rgba_to_uint(const vec<3, float>& rgb)
{
    return rgba_to_uint((vec4f){rgb.v[0], rgb.v[1], rgb.v[2], 1.f});
}

inline vec3f rot(const vec3f& p1, const vec3f& pos, const vec3f& rot)
{
    return p1.rot(pos, rot);
}

inline vec3f back_rot(const vec3f& p1, const vec3f& pos, const vec3f& rot)
{
    return p1.back_rot(pos, rot);
}

inline vec3f cross(const vec3f& v1, const vec3f& v2)
{
    vec3f ret;

    ret.v[0] = v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1];
    ret.v[1] = v1.v[2] * v2.v[0] - v1.v[0] * v2.v[2];
    ret.v[2] = v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0];

    return ret;
}


///counterclockwise
template<int N, typename T>
inline vec<N, T> perpendicular(const vec<N, T>& v1)
{
    return {-v1.v[1], v1.v[0]};
}

template<int N, typename T>
inline T dot(const vec<N, T>& v1, const vec<N, T>& v2)
{
    T ret = 0;

    for(int i=0; i<N; i++)
    {
        ret += v1.v[i] * v2.v[i];
    }

    return ret;
}

template<int N, typename T>
inline
T angle_between_vectors(const vec<N, T>& v1, const vec<N, T>& v2)
{
    return acos(dot(v1.norm(), v2.norm()));
}

template<int N, typename T>
inline vec<N, T> operator-(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = -v1.v[i];
    }

    return ret;
}

template<int N, typename T, typename U>
inline vec<N, U> operator*(T v, const vec<N, U>& v1)
{
    return v1 * v;
}

template<int N, typename T>
inline vec<N, T> operator+(T v, const vec<N, T>& v1)
{
    return v1 + v;
}

/*inline vec3f operator-(float v, const vec3f& v1)
{
    return v1 - v;
}*/

///should convert these functions to be N/T

template<int N, typename T>
inline vec<N, T> operator/(T v, const vec<N, T>& v1)
{
    vec<N, T> top;

    for(int i=0; i<N; i++)
        top.v[i] = v;

    return top / v1;
}

inline float r2d(float v)
{
    return (v / (M_PI*2.f)) * 360.f;
}

template<int N, typename T>
inline vec<N, T> fabs(const vec<N, T>& v)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = fabs(v.v[i]);
    }

    return v1;
}

template<int N, typename T, typename U>
inline vec<N, T> fmod(const vec<N, T>& v, U n)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = fmod(v.v[i], n);
    }

    return v1;
}

template<int N, typename T>
inline vec<N, T> frac(const vec<N, T>& v)
{
    vec<N, T> v1 = fmod(v, (T)1.);

    return v1;
}

template<int N, typename T>
inline vec<N, T> floor(const vec<N, T>& v)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = floorf(v.v[i]);
    }

    return v1;
}

template<int N, typename T>
inline vec<N, T> ceil(const vec<N, T>& v)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = ceilf(v.v[i]);
    }

    return v1;
}

template<int N, typename T>
inline vec<N, T> ceil_away_from_zero(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        if(v1.v[i] >= 0)
            ret.v[i] = ceil(v1.v[i]);
        if(v1.v[i] < 0)
            ret.v[i] = floor(v1.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> min(const vec<N, T>& v1, const vec<N, T>& v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::min(v1.v[i], v2.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> max(const vec<N, T>& v1, const vec<N, T>& v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::max(v1.v[i], v2.v[i]);
    }

    return ret;
}

template<int N, typename T, typename U>
inline vec<N, T> min(const vec<N, T>& v1, U v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::min(v1.v[i], (T)v2);
    }

    return ret;
}

template<int N, typename T, typename U>
inline vec<N, T> max(const vec<N, T>& v1, U v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::max(v1.v[i], (T)v2);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> axis_angle(const vec<N, T>& v, const vec<N, T>& axis, float angle)
{
    return cos(angle) * v + sin(angle) * cross(axis, v) + (1.f - cos(angle)) * dot(axis, v) * axis;
}


inline
vec3f aa_to_euler(const vec3f& axis, float angle)
{
    vec3f naxis = axis.norm();

    float s = sin(angle);
	float c = cos(angle);
	float t = 1-c;

	float x = naxis.v[0];
	float y = naxis.v[0];
	float z = naxis.v[0];

	if ((x*y*t + z*s) > 0.998f)
    { // north pole singularity detected
		float heading = 2*atan2(x*sin(angle/2), cos(angle/2));
		float attitude = M_PI/2;
		float bank = 0;

		return {attitude, heading, bank};
	}

	if ((x*y*t + z*s) < -0.998)
    { // south pole singularity detected
		float heading = -2*atan2(x*sin(angle/2), cos(angle/2));
		float attitude = -M_PI/2;
		float bank = 0;

        return {attitude, heading, bank};
	}

	float heading = atan2(y * s- x * z * t , 1 - (y*y+ z*z ) * t);
	float attitude = asin(x * y * t + z * s) ;
	float bank = atan2(x * s - y * z * t , 1 - (x*x + z*z) * t);

	return {attitude, heading, bank};
}

template<typename U>
inline vec<4, float> rgba_to_vec(const U& rgba)
{
    vec<4, float> ret;

    ret.v[0] = rgba.r;
    ret.v[1] = rgba.g;
    ret.v[2] = rgba.b;
    ret.v[3] = rgba.a;

    return ret;
}

template<typename U>
inline vec<3, float> xyz_to_vec(const U& xyz)
{
    vec<3, float> ret;

    ret.v[0] = xyz.x;
    ret.v[1] = xyz.y;
    ret.v[2] = xyz.z;

    return ret;
}

template<typename U>
inline vec<2, float> xy_to_vec(const U& xyz)
{
    vec<2, float> ret;

    ret.v[0] = xyz.x;
    ret.v[1] = xyz.y;

    return ret;
}

template<typename U, typename T, int N>
inline vec<N, T> conv(const vec<N, U>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v1.v[i];
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> d2r(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = (v1.v[i] / 360.f) * M_PI * 2;
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> mix(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v1.v[i] * (1.f - a) + v2.v[i] * a;//v1.v[i] + (v2.v[i] - v1.v[i]) * a;
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> mix3(const vec<N, T>& v1, const vec<N, T>& mid, const vec<N, T>& v2, float a)
{
    if(a <= 0.5f)
    {
        return mix(v1, mid, a * 2.f);
    }
    else
    {
        return mix(mid, v2, (a - 0.5f) * 2.f);
    }
}

/*template<int N, typename T>
inline vec<N, T> slerp(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    vec<N, T> ret;

    ///im sure you can convert the cos of a number to the sign, rather than doing this
    float angle = acos(dot(v1.norm(), v2.norm()));

    if(angle < 0.0001f && angle >= -0.0001f)
        return mix(v1, v2, a);

    float a1 = sin((1 - a) * angle) / sin(angle);
    float a2 = sin(a * angle) / sin(angle);

    ret = a1 * v1 + a2 * v2;

    return ret;
}*/

template<int N, typename T>
inline vec<N, T> slerp(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    vec<N, T> ret;

    ///im sure you can convert the cos of a number to the sign, rather than doing this
    float angle = acos(dot(v1, v2) / (v1.length() * v2.length()));

    if(angle < 0.00001f && angle >= -0.00001f)
        return mix(v1, v2, a);

    float a1 = sin((1 - a) * angle) / sin(angle);
    float a2 = sin(a * angle) / sin(angle);

    ret = a1 * v1 + a2 * v2;

    return ret;
}

template<int N, typename T>
inline vec<N, T> slerp3(const vec<N, T>& v1, const vec<N, T>& mid, const vec<N, T>& v2, float a)
{
    if(a <= 0.5f)
    {
        return slerp(v1, mid, a * 2.f);
    }
    else
    {
        return slerp(mid, v2, (a - 0.5f) * 2.f);
    }
}

template<int N, typename T>
inline
vec<N, T> cosint(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    float mu2 = (1.f-cosf(a*M_PI))/2.f;

    return v1*(1-mu2) + v2*mu2;
}

template<int N, typename T>
inline
vec<N, T> cosint3(const vec<N, T>& v1, const vec<N, T>& mid, const vec<N, T>& v2, float a)
{
    if(a <= 0.5f)
    {
        return cosint(v1, mid, a * 2.f);
    }
    else
    {
        return cosint(mid, v2, (a - 0.5f) * 2.f);
    }
}

/*template<int N, typename T>
inline
vec<N, T> rejection(const vec<N, T>& v1, const vec<N, T>& v2)
{
    /*vec<N, T> me_to_them = v2 - v1;

    me_to_them = me_to_them.norm();

    float scalar_proj = dot(move_dir, me_to_them);

    vec<N, T> to_them_relative = scalar_proj * me_to_them;

    vec<N, T> perp = move_dir - to_them_relative;

    return perp;
}*/

/*float cosif3(float y1, float y2, float y3, float frac)
{
    float fsin = sin(frac * M_PI);

    float val;

    if(frac < 0.5f)
    {
        val = (1.f - fsin) * y1 + fsin * y2;
    }
    else
    {
        val = (fsin) * y2 + (1.f - fsin) * y3;
    }

    return val;
}*/

inline vec3f generate_flat_normal(const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
    return cross(p2 - p1, p3 - p1).norm();
}

///t should be some container of vec3f
///sorted via 0 -> about vector, plane perpendicular to that
///I should probably make the second version an overload/specialisation, rather than a runtime check
///performance isnt 100% important though currently
template<typename T>
inline std::vector<vec3f> sort_anticlockwise(const T& in, const vec3f& about, std::vector<std::pair<float, int>>* pass_out = nullptr)
{
    int num = in.size();

    std::vector<vec3f> out;
    std::vector<std::pair<float, int>> intermediate;

    out.reserve(num);
    intermediate.reserve(num);

    vec3f euler = about.get_euler();

    vec3f centre_point = about.back_rot(0.f, euler);

    vec2f centre_2d = (vec2f){centre_point.v[0], centre_point.v[2]};

    for(int i=0; i<num; i++)
    {
        vec3f vec_pos = in[i];

        vec3f rotated = vec_pos.back_rot(0.f, euler);

        vec2f rot_2d = (vec2f){rotated.v[0], rotated.v[2]};

        vec2f rel = rot_2d - centre_2d;

        float angle = rel.angle();

        intermediate.push_back({angle, i});
    }

    std::sort(intermediate.begin(), intermediate.end(),
              [](auto i1, auto i2)
              {
                  return i1.first < i2.first;
              }
              );

    for(auto& i : intermediate)
    {
        out.push_back(in[i.second]);
    }

    if(pass_out != nullptr)
    {
        *pass_out = intermediate;
    }

    return out;
}

template<int N, typename T, typename U>
inline
void line_draw_helper(const vec<N, T>& start, const vec<N, T>& finish, vec<N, T>& out_dir, U& num)
{
    vec<N, T> dir = (finish - start);
    T dist = dir.largest_elem();

    dir = dir / dist;

    out_dir = dir;
    num = dist;
}

///there is almost certainly a better way to do this
///this function doesn't work properly, don't use it
inline
float circle_minimum_distance(float v1, float v2)
{
    v1 = fmodf(v1, M_PI * 2.f);
    v2 = fmodf(v2, M_PI * 2.f);

    float d1 = fabs(v2 - v1);
    float d2 = fabs(v2 - v1 - M_PI*2.f);
    float d3 = fabs(v2 - v1 + M_PI*2.f);

    vec3f v = (vec3f){d1, d2, d3};

    if(v.which_element_minimum() == 0)
    {
        return v2 - v1;
    }

    if(v.which_element_minimum() == 1)
    {
        return v2 - v1 - M_PI*2.f;
    }

    if(v.which_element_minimum() == 2)
    {
        return v2 - v1 + M_PI*2.f;
    }

    //float result = std::min(d1, std::min(d2, d3));
}

///rename this function
template<int N, typename T>
inline
vec<N, T> point2line_shortest(const vec<N, T>& lp, const vec<N, T>& ldir, const vec<N, T>& p)
{
    vec<N, T> ret;

    auto n = ldir.norm();

    ret = (lp - p) - dot(lp - p, n) * n;

    return ret;
}

///https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
template<typename T>
inline
bool is_left_side(const vec<2, T>& l1, const vec<2, float>& l2, const vec<2, float>& lp)
{
    return ((l2.v[0] - l1.v[0]) * (lp.v[1] - l1.v[1]) - (l2.v[1] - l1.v[1]) * (lp.v[0] - l1.v[0])) > 0;
}


template<int N, typename T>
struct mat
{
    T v[N][N];

    mat()
    {
        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
            {
                v[j][i] = 0;
            }
        }
    }

    mat<N, T> from_vec(vec3f v1, vec3f v2, vec3f v3) const
    {
        mat<N, T> m;

        for(int i=0; i<3; i++)
            m.v[0][i] = v1.v[i];

        for(int i=0; i<3; i++)
            m.v[1][i] = v2.v[i];

        for(int i=0; i<3; i++)
            m.v[2][i] = v3.v[i];

        return m;
    }

    void load(vec3f v1, vec3f v2, vec3f v3)
    {
        for(int i=0; i<3; i++)
            v[0][i] = v1.v[i];

        for(int i=0; i<3; i++)
            v[1][i] = v2.v[i];

        for(int i=0; i<3; i++)
            v[2][i] = v3.v[i];
    }

    float det() const
    {
        float a11, a12, a13, a21, a22, a23, a31, a32, a33;

        a11 = v[0][0];
        a12 = v[0][1];
        a13 = v[0][2];

        a21 = v[1][0];
        a22 = v[1][1];
        a23 = v[1][2];

        a31 = v[2][0];
        a32 = v[2][1];
        a33 = v[2][2];

        ///get determinant
        float d = a11*a22*a33 + a21*a32*a13 + a31*a12*a23 - a11*a32*a23 - a31*a22*a13 - a21*a12*a33;

        return d;
    }

    mat<N, T> invert() const
    {
        float d = det();

        float a11, a12, a13, a21, a22, a23, a31, a32, a33;

        a11 = v[0][0];
        a12 = v[0][1];
        a13 = v[0][2];

        a21 = v[1][0];
        a22 = v[1][1];
        a23 = v[1][2];

        a31 = v[2][0];
        a32 = v[2][1];
        a33 = v[2][2];


        vec3f ir1, ir2, ir3;

        ir1.v[0] = a22 * a33 - a23 * a32;
        ir1.v[1] = a13 * a32 - a12 * a33;
        ir1.v[2] = a12 * a23 - a13 * a22;

        ir2.v[0] = a23 * a31 - a21 * a33;
        ir2.v[1] = a11 * a33 - a13 * a31;
        ir2.v[2] = a13 * a21 - a11 * a23;

        ir3.v[0] = a21 * a32 - a22 * a31;
        ir3.v[1] = a12 * a31 - a11 * a32;
        ir3.v[2] = a11 * a22 - a12 * a21;

        ir1 = ir1 * d;
        ir2 = ir2 * d;
        ir3 = ir3 * d;

        return from_vec(ir1, ir2, ir3);
    }

    vec<3, T> get_v1() const
    {
        return {v[0][0], v[0][1], v[0][2]};
    }
    vec<3, T> get_v2() const
    {
        return {v[1][0], v[1][1], v[1][2]};
    }
    vec<3, T> get_v3() const
    {
        return {v[2][0], v[2][1], v[2][2]};
    }

    mat<N, T> identity()
    {
        mat<N, T> ret;

        for(int i=0; i<N; i++)
        {
            ret.v[i][i] = 1;
        }

        return ret;
    }

    mat<3, float> skew_symmetric_cross_product(vec3f cr)
    {
        mat<3, float> ret;

        ret.load({0, -cr.v[2], cr.v[1]},
                 {cr.v[3], 0, -cr.v[0]},
                 {-cr.v[1], cr.v[0], 0});

        return ret;
    }

    #if 0
    void from_dir(vec3f dir)
    {
        /*vec3f up = {0, 1, 0};

        vec3f xaxis = cross(up, dir).norm();
        vec3f yaxis = cross(dir, xaxis).norm();

        v[0][0] = xaxis.v[0];
        v[0][1] = yaxis.v[0];
        v[0][2] = dir.v[0];

        v[1][0] = xaxis.v[1];
        v[1][1] = yaxis.v[1];
        v[1][2] = dir.v[1];

        v[2][0] = xaxis.v[2];
        v[2][1] = yaxis.v[2];
        v[2][2] = dir.v[2];*/


    }
    #endif

    void load_rotation_matrix(vec3f _rotation)
    {
        /*vec3f c;
        vec3f s;

        for(int i=0; i<3; i++)
        {
            c.v[i] = cos(rotation.v[i]);
            s.v[i] = sin(rotation.v[i]);
        }

        ///rotation matrix
        //vec3f r1 = {c.v[1]*c.v[2], -c.v[1]*s.v[2], s.v[1]};
        //vec3f r2 = {c.v[0]*s.v[2] + c.v[2]*s.v[0]*s.v[1], c.v[0]*c.v[2] - s.v[0]*s.v[1]*s.v[2], -c.v[1]*s.v[0]};
        //vec3f r3 = {s.v[0]*s.v[2] - c.v[0]*c.v[2]*s.v[1], c.v[2]*s.v[0] + c.v[0]*s.v[1]*s.v[2], c.v[1]*c.v[0]};

        vec3f r1 = {c.v[1] * c.v[2], c.v[0] * s.v[2] + s.v[0] * s.v[1] * c.v[2], s.v[0] * s.v[2] - c.v[0] * s.v[1] * c.v[2]};
        vec3f r2 = {-c.v[1] * s.v[2], c.v[0] * c.v[2] - s.v[0] * s.v[1] * s.v[2], s.v[0] * c.v[2] + c.v[0] * s.v[1] * s.v[2]};
        vec3f r3 = {s.v[1], -s.v[0] * c.v[1], c.v[0] * c.v[1]};

        load(r1, r2, r3);*/

        vec3f rotation = {_rotation.v[0], _rotation.v[1], _rotation.v[2]};

        //rotation = rotation + M_PI/2.f;

        vec3f c = vcos(rotation);
        vec3f s = vsin(rotation);

        mat<3, float> v1;

        v1.load({1, 0, 0}, {0, c.v[0], s.v[0]}, {0, -s.v[0], c.v[0]});

        mat<3, float> v2;

        v2.load({c.v[1], 0, -s.v[1]}, {0, 1, 0}, {s.v[1], 0, c.v[1]});

        mat<3, float> v3;

        v3.load({c.v[2], s.v[2], 0}, {-s.v[2], c.v[2], 0}, {0, 0, 1});


        *this = v1 * v2 * v3;
    }

    vec3f get_rotation()
    {
        vec3f rotation;

        //rotation.v[0] = atan2(v[2][0], v[2][1]);
        //rotation.v[1] = acos(v[2][2]);
        //rotation.v[2] = -atan2(v[0][2], v[1][2]);

        //rotation.v[2] = -atan2(v[2][1], v[2][2]);
        //rotation.v[1] = asin(v[2][0]);
        //rotation.v[0] = -atan2(v[1][0], v[0][0]);

        rotation.v[0] = atan2(v[1][2], v[2][2]);
        rotation.v[1] = -asin(v[0][2]);
        rotation.v[2] = atan2(v[0][1], v[0][0]);

        //printf("[2][2] %f\n", v[2][2]);

        return rotation;
    }

    vec<3, T> operator*(const vec<3, T>& other) const
    {
        vec<3, T> val;

        val.v[0] = v[0][0] * other.v[0] + v[0][1] * other.v[1] + v[0][2] * other.v[2];
        val.v[1] = v[1][0] * other.v[0] + v[1][1] * other.v[1] + v[1][2] * other.v[2];
        val.v[2] = v[2][0] * other.v[0] + v[2][1] * other.v[1] + v[2][2] * other.v[2];

        return val;
    }

    mat<3, T> operator*(const mat<3, T>& other) const
    {
        mat<3, T> ret;

        for(int j=0; j<3; j++)
        {
            for(int i=0; i<3; i++)
            {
                //float val = v[j][0] * other.v[0][i] + v[j][1] * other.v[1][i] + v[j][2] * other.v[2][i];

                float accum = 0;

                for(int k=0; k<3; k++)
                {
                    accum += v[j][k] * other.v[k][i];
                }

                ret.v[j][i] = accum;
            }
        }

        return ret;
    }

    mat<3, T> operator*(T other) const
    {
        mat<3, T> ret;

        for(int j=0; j<3; j++)
        {
            for(int i=0; i<3; i++)
            {
                ret.v[j][i] = v[j][i] * other;
            }
        }

        return ret;
    }

    mat<3, T> operator+(const mat<3, T>& other) const
    {
        mat<3, T> ret;

        for(int j=0; j<3; j++)
        {
            for(int i=0; i<3; i++)
            {
                ret.v[j][i] = v[j][i] + other.v[j][i];
            }
        }

        return ret;
    }

    mat<3, T> transp()
    {
        mat<3, T> ret;

        for(int j=0; j<3; j++)
        {
            for(int i=0; i<3; i++)
            {
                ret.v[j][i] = v[i][j];
            }
        }

        return ret;
    }

    friend std::ostream& operator<<(std::ostream& os, const mat<N, T>& v1)
    {
        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
                os << std::to_string(v1.v[j][i]) << " ";

            os << std::endl;
        }

        return os;
    }
};

typedef mat<3, float> mat3f;

inline
mat3f tensor_product(vec3f v1, vec3f v2)
{
    mat3f ret;

    for(int j=0; j<3; j++)
    {
        for(int i=0; i<3; i++)
        {
            ret.v[j][i] = v1.v[j] * v2.v[i];
        }
    }

    return ret;
}

///need to fix for a == b = I, and a == -b = -I
///although the latter does not seem valid
inline
vec3f mat_from_dir(vec3f d1, vec3f d2)
{
    d1 = d1.norm();
    d2 = d2.norm();

    vec3f d = cross(d1, d2).norm();
    float c = dot(d1, d2);

    float s = sin(acos(c));

    ///so, d, acos(c) would be axis angle

    mat<3, float> skew_symmetric;
    skew_symmetric = skew_symmetric.skew_symmetric_cross_product(d);

    mat<3, float> unit;

    unit = unit.identity();

    mat<3, float> ret = unit * c + skew_symmetric * s + tensor_product(d, d) * (1.f - c);

    //std::cout << d << std::endl;

    return ret.get_rotation();
}


/*template<typename T>
vec<3, T> operator*(const mat<3, T> m, const vec<3, T>& other)
{
    vec<3, T> val;

    val.v[0] = m.v[0][0] * other.v[0] + m.v[0][1] * other.v[1] + m.v[0][2] * other.v[2];
    val.v[1] = m.v[1][0] * other.v[0] + m.v[1][1] * other.v[1] + m.v[1][2] * other.v[2];
    val.v[2] = m.v[2][0] * other.v[0] + m.v[2][1] * other.v[1] + m.v[2][2] * other.v[2];

    return val;
}*/


#endif // VEC_HPP_INCLUDED