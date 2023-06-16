#ifndef TENSOR_HPP_INCLUDED
#define TENSOR_HPP_INCLUDED

#include "vec.hpp"
#include <cmath>

template<int... N>
constexpr int N_to_size()
{
    return (N * ...);
}

template<int N, int... M>
constexpr int get_first_of()
{
    return N;
}

template<int N, int... M>
inline
constexpr int get_second_of()
{
    return get_first_of<M...>();
}

template<int N, int... M>
inline
constexpr int get_third_of()
{
    return get_second_of<M...>();
}

template<typename T, size_t size, size_t... sizes>
struct md_array_impl
{
    using type = std::array<typename md_array_impl<T, sizes...>::type, size>;
};

template<typename T, size_t size>
struct md_array_impl<T, size>
{
    using type = std::array<T, size>;
};

template<typename T, size_t... sizes>
using md_array = typename md_array_impl<T, sizes...>::type;

template<typename T, typename Next>
constexpr
auto& index_md_array(T& arr, Next v)
{
    return arr[v];
}

template<typename T, typename Next, typename... Rest>
constexpr
auto& index_md_array(T& arr, Next v, Rest... r)
{
    return index_md_array(arr[v], r...);
}

/*
void metric_inverse(const float m[16], float invOut[16])
{
    float inv[16], det;
    int i;

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[11] -
             m[6] * m[6]  * m[15] +
             m[6] * m[7]  * m[11] +
             m[7] * m[6]  * m[11] -
             m[7] * m[7]  * m[10];

    inv[1] = -m[1] * m[10] * m[15] +
              m[1] * m[11] * m[11] +
              m[6] * m[2] * m[15] -
              m[6] * m[3] * m[11] -
              m[7] * m[2] * m[11] +
              m[7] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[11] -
             m[2] * m[2] * m[15] +
             m[2] * m[3] * m[11] +
             m[3] * m[2] * m[11] -
             m[3] * m[3] * m[10];


    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[11] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[11] +
             m[7] * m[2] * m[7] -
             m[7] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
              m[0] * m[7] * m[11] +
              m[1] * m[2] * m[15] -
              m[1] * m[3] * m[11] -
              m[3] * m[2] * m[7] +
              m[3] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[7] -
              m[1] * m[1] * m[15] +
              m[1] * m[3] * m[7] +
              m[3] * m[1] * m[7] -
              m[3] * m[3] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[6] * m[2] * m[7] +
              m[6] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[1] * m[2] * m[11] +
             m[1] * m[3] * m[10] +
             m[2] * m[2] * m[7] -
             m[2] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[6] +
               m[1] * m[1] * m[11] -
               m[1] * m[3] * m[6] -
               m[2] * m[1] * m[7] +
               m[2] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[6] -
              m[1] * m[1] * m[10] +
              m[1] * m[2] * m[6] +
              m[2] * m[1] * m[6] -
              m[2] * m[2] * m[5];

    inv[4] = inv[1];
    inv[8] = inv[2];
    inv[12] = inv[3];
    inv[9] = inv[6];
    inv[13] = inv[7];
    inv[14] = inv[11];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;
}*/

template<template<typename T, int... N> typename Concrete, typename T, int... N>
struct tensor_base;

template<typename T, int... N>
struct tensor;

template<template<typename T, int... N> typename Concrete, typename U, typename T, int... N>
inline
Concrete<T, N...> tensor_for_each_binary(const Concrete<T, N...>& v1, const Concrete<T, N...>& v2, U&& u);

template<template<typename T, int... N> typename Concrete, typename U, typename T, int... N>
inline
auto tensor_for_each_unary(const Concrete<T, N...>& v, U&& u);

///https://pharr.org/matt/blog/2019/11/03/difference-of-floats
///a * b - c * d
template<typename T>
inline
T difference_of_products(const T& a, const T& b, const T& c, const T& d)
{
    auto cd = c * d;
    auto err = fma(-c, d, cd);
    auto dop = fma(a, b, -cd);

    return dop + err;
}

template<template<typename T, int... N> typename Concrete, typename T, int... N>
struct tensor_base
{
    template<typename U, int... M>
    using concrete_t = Concrete<U, M...>;

    template<typename U>
    using value_t = T;

    md_array<T, N...> data{};

    Concrete<T, N...> to_concrete() const
    {
        return *this;
    }

    tensor<T, N...> to_tensor() const
    {
        tensor<T, N...> ret;

        ret.data = data;

        return ret;
    }

    auto begin()
    {
        static_assert(sizeof...(N) == 1);

        return data.begin();
    }

    auto end()
    {
        static_assert(sizeof...(N) == 1);

        return data.end();
    }

    auto begin() const
    {
        static_assert(sizeof...(N) == 1);

        return data.begin();
    }

    auto end() const
    {
        static_assert(sizeof...(N) == 1);

        return data.end();
    }

    template<typename... V>
    T& idx(V... vals)
    {
        static_assert(sizeof...(V) == sizeof...(N));

        return index_md_array(data, vals...);
    }

    template<typename... V>
    const T& idx(V... vals) const
    {
        static_assert(sizeof...(V) == sizeof...(N));

        return index_md_array(data, vals...);
    }

    template<typename... V>
    T& operator[](V... vals)
    {
        return idx(vals...);
    }

    template<typename... V>
    const T& operator[](V... vals) const
    {
        return idx(vals...);
    }

    T symmetric_det3() const
    {
        assert(sizeof...(N) == 2);
        assert(((N == 3) && ...));

        T a11 = idx(0, 0);
        T a12 = idx(0, 1);
        T a13 = idx(0, 2);

        T a21 = idx(1, 0);
        T a22 = idx(1, 1);
        T a23 = idx(1, 2);

        T a31 = idx(2, 0);
        T a32 = idx(2, 1);
        T a33 = idx(2, 2);

        a21 = a12;
        a31 = a13;
        a32 = a23;

        return a11 * difference_of_products(a22, a33, a32, a23) + a12 * difference_of_products(a23, a31, a21, a33) + a13 * difference_of_products(a21, a32, a22, a31);
    }

    Concrete<T, N...> symmetric_unit_invert() const
    {
        assert(sizeof...(N) == 2);
        assert(((N == 3) && ...));

        Concrete<T, N...> ret;

        ret[0, 0] = difference_of_products(idx(1, 1), idx(2, 2), idx(2, 1), idx(1, 2));
        ret[0, 1] = difference_of_products(idx(0, 2), idx(2, 1), idx(0, 1), idx(2, 2));
        ret[0, 2] = difference_of_products(idx(0, 1), idx(1, 2), idx(0, 2), idx(1, 1));
        ret[1, 0] = difference_of_products(idx(1, 2), idx(2, 0), idx(1, 0), idx(2, 2));
        ret[1, 1] = difference_of_products(idx(0, 0), idx(2, 2), idx(0, 2), idx(2, 0));
        ret[1, 2] = difference_of_products(idx(1, 0), idx(0, 2), idx(0, 0), idx(1, 2));
        ret[2, 0] = difference_of_products(idx(1, 0), idx(2, 1), idx(2, 0), idx(1, 1));
        ret[2, 1] = difference_of_products(idx(2, 0), idx(0, 1), idx(0, 0), idx(2, 1));
        ret[2, 2] = difference_of_products(idx(0, 0), idx(1, 1), idx(1, 0), idx(0, 1));

        ret.idx(1, 0) = ret.idx(0, 1);
        ret.idx(2, 0) = ret.idx(0, 2);
        ret.idx(2, 1) = ret.idx(1, 2);

        return ret;
    }

    Concrete<T, N...> symmetric_invert() const
    {
        assert(sizeof...(N) == 2);
        assert((((N == 3) && ...)) || ((N == 4) && ...));

        if constexpr(((N == 3) && ...))
        {
            T d = 1/symmetric_det3();

            Concrete<T, N...> ret = symmetric_unit_invert();

            ret.idx(0, 0) = ret.idx(0, 0) * d;
            ret.idx(0, 1) = ret.idx(0, 1) * d;
            ret.idx(0, 2) = ret.idx(0, 2) * d;
            ret.idx(1, 0) = ret.idx(1, 0) * d;
            ret.idx(1, 1) = ret.idx(1, 1) * d;
            ret.idx(1, 2) = ret.idx(1, 2) * d;
            ret.idx(2, 0) = ret.idx(2, 0) * d;
            ret.idx(2, 1) = ret.idx(2, 1) * d;
            ret.idx(2, 2) = ret.idx(2, 2) * d;

            return ret;
        }

        if constexpr(((N == 4) && ...))
        {
            ///[0, 1, 2, 3]
            ///[4, 5, 6, 7]
            ///[8, 9, 10,11]
            ///[12,13,14,15]

            std::array<T, 16> m;
            m[0] = idx(0, 0);
            m[1] = idx(0, 1);
            m[2] = idx(0, 2);
            m[3] = idx(0, 3);
            m[4] = idx(1, 0);
            m[5] = idx(1, 1);
            m[6] = idx(1, 2);
            m[7] = idx(1, 3);
            m[8] = idx(2, 0);
            m[9] = idx(2, 1);
            m[10] = idx(2, 2);
            m[11] = idx(2, 3);
            m[12] = idx(3, 0);
            m[13] = idx(3, 1);
            m[14] = idx(3, 2);
            m[15] = idx(3, 3);

            std::array<T, 16> inv;

            T det = T();
            Concrete<T, N...> ret;

            inv[0] = m[5] * m[10] * m[15] -
                     m[5] * m[11] * m[11] -
                     m[6] * m[6]  * m[15] +
                     m[6] * m[7]  * m[11] +
                     m[7] * m[6]  * m[11] -
                     m[7] * m[7]  * m[10];

            inv[1] = -m[1] * m[10] * m[15] +
                      m[1] * m[11] * m[11] +
                      m[6] * m[2] * m[15] -
                      m[6] * m[3] * m[11] -
                      m[7] * m[2] * m[11] +
                      m[7] * m[3] * m[10];

            inv[5] = m[0] * m[10] * m[15] -
                     m[0] * m[11] * m[11] -
                     m[2] * m[2] * m[15] +
                     m[2] * m[3] * m[11] +
                     m[3] * m[2] * m[11] -
                     m[3] * m[3] * m[10];


            inv[2] = m[1] * m[6] * m[15] -
                     m[1] * m[7] * m[11] -
                     m[5] * m[2] * m[15] +
                     m[5] * m[3] * m[11] +
                     m[7] * m[2] * m[7] -
                     m[7] * m[3] * m[6];

            inv[6] = -m[0] * m[6] * m[15] +
                      m[0] * m[7] * m[11] +
                      m[1] * m[2] * m[15] -
                      m[1] * m[3] * m[11] -
                      m[3] * m[2] * m[7] +
                      m[3] * m[3] * m[6];

            inv[10] = m[0] * m[5] * m[15] -
                      m[0] * m[7] * m[7] -
                      m[1] * m[1] * m[15] +
                      m[1] * m[3] * m[7] +
                      m[3] * m[1] * m[7] -
                      m[3] * m[3] * m[5];

            inv[3] = -m[1] * m[6] * m[11] +
                      m[1] * m[7] * m[10] +
                      m[5] * m[2] * m[11] -
                      m[5] * m[3] * m[10] -
                      m[6] * m[2] * m[7] +
                      m[6] * m[3] * m[6];

            inv[7] = m[0] * m[6] * m[11] -
                     m[0] * m[7] * m[10] -
                     m[1] * m[2] * m[11] +
                     m[1] * m[3] * m[10] +
                     m[2] * m[2] * m[7] -
                     m[2] * m[3] * m[6];

            inv[11] = -m[0] * m[5] * m[11] +
                       m[0] * m[7] * m[6] +
                       m[1] * m[1] * m[11] -
                       m[1] * m[3] * m[6] -
                       m[2] * m[1] * m[7] +
                       m[2] * m[3] * m[5];

            inv[15] = m[0] * m[5] * m[10] -
                      m[0] * m[6] * m[6] -
                      m[1] * m[1] * m[10] +
                      m[1] * m[2] * m[6] +
                      m[2] * m[1] * m[6] -
                      m[2] * m[2] * m[5];

            inv[4] = inv[1];
            inv[8] = inv[2];
            inv[12] = inv[3];
            inv[9] = inv[6];
            inv[13] = inv[7];
            inv[14] = inv[11];

            det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

            det = 1.0f / det;

            for(int x=0; x < 4; x++)
            {
                for(int y=0; y < 4; y++)
                {
                    ret.idx(y, x) = inv[y * 4 + x] * det;
                }
            }

            return ret;
        }

        assert(false);
    }

    T x() const
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 3) && ...));

        return idx(0);
    }

    T y() const
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 3) && ...));

        return idx(1);
    }

    T z() const
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 3) && ...));

        return idx(2);
    }

    T w() const
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 4) && ...));

        return idx(3);
    }

    T& x()
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 3) && ...));

        return idx(0);
    }

    T& y()
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 3) && ...));

        return idx(1);
    }

    T& z()
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 3) && ...));

        return idx(2);
    }

    T& w()
    {
        static_assert(sizeof...(N) == 1);
        static_assert(((N >= 4) && ...));

        return idx(3);
    }

    T squared_length() const
    {
        static_assert(sizeof...(N) == 1);

        T ret = 0;

        for(size_t i = 0; i < get_first_of<N...>(); i++)
        {
            ret += idx(i) * idx(i);
        }

        return ret;
    }

    T length() const
    {
        return sqrt(squared_length());
    }

    friend Concrete<T, N...> operator+(const Concrete<T, N...>& t1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_binary(t1, t2, [](const T& v1, const T& v2){return v1 + v2;});
    }

    friend Concrete<T, N...>& operator+=(Concrete<T, N...>& t1, const Concrete<T, N...>& t2)
    {
        t1 = t1 + t2;

        return t1;
    }

    friend Concrete<T, N...> operator-(const Concrete<T, N...>& t1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_binary(t1, t2, [](const T& v1, const T& v2){return v1 - v2;});
    }

    friend Concrete<T, N...> operator*(const Concrete<T, N...>& t1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_binary(t1, t2, [](const T& v1, const T& v2){return v1 * v2;});
    }

    friend Concrete<T, N...> operator/(const Concrete<T, N...>& t1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_binary(t1, t2, [](const T& v1, const T& v2){return v1 / v2;});
    }

    friend Concrete<T, N...> operator+(const Concrete<T, N...>& t1, const T& v2)
    {
        return tensor_for_each_unary(t1, [&](const T& v1){return v1 + v2;});
    }

    friend Concrete<T, N...> operator-(const Concrete<T, N...>& t1, const T& v2)
    {
        return tensor_for_each_unary(t1, [&](const T& v1){return v1 - v2;});
    }

    friend Concrete<T, N...> operator*(const Concrete<T, N...>& t1, const T& v2)
    {
        return tensor_for_each_unary(t1, [&](const T& v1){return v1 * v2;});
    }

    friend Concrete<T, N...> operator/(const Concrete<T, N...>& t1, const T& v2)
    {
        return tensor_for_each_unary(t1, [&](const T& v1){return v1 / v2;});
    }

    friend Concrete<T, N...> operator+(const T& v1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_unary(t2, [&](const T& v2){return v1 + v2;});
    }

    friend Concrete<T, N...> operator-(const T& v1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_unary(t2, [&](const T& v2){return v1 - v2;});
    }

    friend Concrete<T, N...> operator*(const T& v1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_unary(t2, [&](const T& v2){return v1 * v2;});
    }

    friend Concrete<T, N...> operator/(const T& v1, const Concrete<T, N...>& t2)
    {
        return tensor_for_each_unary(t2, [&](const T& v2){return v1 / v2;});
    }

    friend Concrete<T, N...> operator-(const Concrete<T, N...>& t1)
    {
        return tensor_for_each_unary(t1, [](const T& v1){return -v1;});
    }
};

template<typename T, int... N>
struct tensor : tensor_base<tensor, T, N...>
{
    template<typename U>
    tensor<U, N...> as() const
    {
        return tensor_for_each_unary(*this, [](const T& v1){return U{v1};});
    }
};

template<typename TestTensor, typename T, int... N>
concept SizedTensor = std::is_base_of_v<tensor_base<TestTensor::template concrete_t, T, N...>, TestTensor>;

template<typename T, int N>
inline
T sum_multiply(const tensor<T, N>& t1, const tensor<T, N>& t2)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        ret += t1.idx(i) * t2.idx(i);
    }

    return ret;
}

template<typename T, int N>
inline
T sum_multiply(const tensor<T, N, N>& t1, const tensor<T, N, N>& t2)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            ret += t1.idx(i, j) * t2.idx(i, j);
        }
    }

    return ret;
}

template<typename T, int N>
inline
T sum(const tensor<T, N>& t1)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        ret = ret + t1.idx(i);
    }

    return ret;
}

template<template<typename T, int... N> typename Concrete, typename U, typename T, int... N>
inline
auto tensor_for_each_unary(const Concrete<T, N...>& v, U&& u)
{
    using result_t = decltype(u(std::declval<T>()));

    Concrete<result_t, N...> ret;

    if constexpr(sizeof...(N) == 1)
    {
        int len = get_first_of<N...>();

        for(int i=0; i < len; i++)
        {
            ret.idx(i) = u(v.idx(i));
        }
    }
    else if constexpr(sizeof...(N) == 2)
    {
        int l1 = get_first_of<N...>();
        int l2 = get_second_of<N...>();

        for(int i=0; i < l1; i++)
        {
            for(int j=0; j < l2; j++)
            {
                ret.idx(i, j) = u(v.idx(i, j));
            }
        }
    }
    else if constexpr(sizeof...(N) == 3)
    {
        int l1 = get_first_of<N...>();
        int l2 = get_second_of<N...>();
        int l3 = get_third_of<N...>();

        for(int i=0; i < l1; i++)
        {
            for(int j=0; j < l2; j++)
            {
                for(int k=0; k < l3; k++)
                {
                    ret.idx(i, j, k) = u(v.idx(i, j, k));
                }
            }
        }
    }
    else
    {
        assert(false);
    }

    return ret;
}

template<template<typename T, int... N> typename Concrete, typename U, typename T, int... N>
inline
Concrete<T, N...> tensor_for_each_binary(const Concrete<T, N...>& v1, const Concrete<T, N...>& v2, U&& u)
{
    Concrete<T, N...> ret;

    if constexpr(sizeof...(N) == 1)
    {
        int len = get_first_of<N...>();

        for(int i=0; i < len; i++)
        {
            ret.idx(i) = u(v1.idx(i), v2.idx(i));
        }
    }
    else if constexpr(sizeof...(N) == 2)
    {
        int l1 = get_first_of<N...>();
        int l2 = get_second_of<N...>();

        for(int i=0; i < l1; i++)
        {
            for(int j=0; j < l2; j++)
            {
                ret.idx(i, j) = u(v1.idx(i, j), v2.idx(i, j));
            }
        }
    }
    else if constexpr(sizeof...(N) == 3)
    {
        int l1 = get_first_of<N...>();
        int l2 = get_second_of<N...>();
        int l3 = get_third_of<N...>();

        for(int i=0; i < l1; i++)
        {
            for(int j=0; j < l2; j++)
            {
                for(int k=0; k < l3; k++)
                {
                    ret.idx(i, j, k) = u(v1.idx(i, j, k), v2.idx(i, j, k));
                }
            }
        }
    }
    else
    {
        assert(false);
    }

    return ret;
}

template<int... Indices>
struct tensor_indices
{
    std::array<int, sizeof...(Indices)> indices = {Indices...};
};

#if 0
template<typename T, int... N, int... M, int... N1, int... M1>
inline
auto sum_multiply_fat(const tensor<T, N...>& t1, const tensor<T, M...>& t2, const tensor_indices<N1...>& b1, const tensor_indices<M1...>& b2)
{
    constexpr int total_dimensionality = sizeof...(N) + sizeof...(M);
    constexpr int argument_summation = sizeof...(N1) + sizeof...(M1);

    constexpr int return_dimensions = total_dimensionality - argument_summation;

    ///need to turn return_dimensions into a parameter pack of return_dimensions long, where each element has a value of the components of N...

    //tensor<T, return_dimension>
}
#endif // 0

template<typename T, int... N>
struct inverse_metric : tensor_base<inverse_metric, T, N...>
{

};

template<template<typename T, int... N> typename Concrete, typename T, int... N>
struct metric_base : tensor_base<Concrete, T, N...>
{
    T det() const
    {
        return tensor_base<Concrete, T, N...>::symmetric_det3();
    }

    virtual inverse_metric<T, N...> invert() const
    {
        inverse_metric<T, N...> r;
        r.data = tensor_base<Concrete, T, N...>::symmetric_invert().data;

        return r;
    }

    virtual ~metric_base(){}
};

template<typename T, int... N>
struct metric : metric_base<metric, T, N...>
{

};

template<typename T, int... N>
struct unit_metric : metric<T, N...>
{
    virtual inverse_metric<T, N...> invert() const override
    {
        inverse_metric<T, N...> r;
        r.data = tensor_base<metric, T, N...>::symmetric_unit_invert().data;

        return r;
    }
};

template<typename TestTensor, typename T, int... N>
concept MetricTensor = std::is_base_of_v<metric_base<TestTensor::template concrete_t, T, N...>, TestTensor>;


template<typename T, int... N>
inline
tensor<T, N...> round(const tensor<T, N...>& v)
{
    return tensor_for_each_unary(v, [](const T& in)
    {
        using namespace std;

        return round(in);
    });
}

template<typename T, int N, typename U>
inline
tensor<T, N> sum_symmetric(const tensor<T, N>& mT, const U& met, int index = 0)
{
    assert(index == 0);

    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        T sum = 0;

        for(int s=0; s < N; s++)
        {
            sum = sum + met.idx(i, s) * mT.idx(s);
        }

        ret.idx(i) = sum;
    }

    return ret;
}

template<typename T, int N, typename U>
inline
tensor<T, N, N> sum_symmetric(const tensor<T, N, N>& mT, const U& met, int index)
{
    tensor<T, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int s=0; s < N; s++)
            {
                if(index == 0)
                {
                    sum = sum + met.idx(i, s) * mT.idx(s, j);
                }

                if(index == 1)
                {
                    sum = sum + met.idx(j, s) * mT.idx(i, s);
                }
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}

template<typename T, int N, typename U>
inline
tensor<T, N, N, N> sum_symmetric(const tensor<T, N, N, N>& mT, const U& met, int index)
{
    tensor<T, N, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            for(int k=0; k < N; k++)
            {
                T sum = 0;

                for(int s=0; s < N; s++)
                {
                    if(index == 0)
                    {
                        sum = sum + met.idx(i, s) * mT.idx(s, j, k);
                    }

                    if(index == 1)
                    {
                        sum = sum + met.idx(j, s) * mT.idx(i, s, k);
                    }

                    if(index == 2)
                    {
                        sum = sum + met.idx(k, s) * mT.idx(i, j, s);
                    }
                }

                ret.idx(i, j, k) = sum;
            }
        }
    }

    return ret;
}

template<typename T, int U, int... N>
inline
tensor<T, N...> raise_index(const tensor<T, N...>& mT, const inverse_metric<T, U, U>& met, int index)
{
    return sum_symmetric(mT, met, index);
}

template<typename T, int U, int... N>
inline
tensor<T, N...> lower_index(const tensor<T, N...>& mT, const metric<T, U, U>& met, int index)
{
    return sum_symmetric(mT, met, index);
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

#endif
