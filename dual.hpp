#ifndef DUAL_HPP_INCLUDED
#define DUAL_HPP_INCLUDED

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <type_traits>
#include <array>
#include <complex>
#include "vec.hpp"

namespace dual_types
{
    template<typename T>
    using complex_v = complex_type::complex<T>;

    template<typename Test, template<typename...> class Ref>
    struct is_specialization : std::false_type {};

    template<template<typename...> class Ref, typename... Args>
    struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

    template<typename T>
    constexpr
    bool is_complex()
    {
        return is_specialization<T, complex_v>{};
    }

    template<typename T>
    struct dual_v
    {
        T real = T();
        T dual = T();

        dual_v(){}

        dual_v(const T& _real, const T& _dual)
        {
            real = _real;
            dual = _dual;
        }

        template<typename U>
        requires std::is_constructible_v<T, U>
        dual_v(const U& _real) : real(_real)
        {
            dual = T();
        }

        template<typename U>
        requires std::is_constructible_v<T, U>
        dual_v(const dual_v<U>& other) : real(other.real), dual(other.dual)
        {

        }

        void make_constant(T val)
        {
            real = val;
            dual.set_dual_constant();
        }

        void make_variable(T val)
        {
            real = val;
            dual.set_dual_variable();
        }
    };

    template<typename T, typename U>
    concept DualValue = std::is_constructible_v<dual_v<T>, U>;

    template<typename T>
    inline
    complex_v<T> makefinite(const complex_v<T>& c1)
    {
        return complex_v<T>(makefinite(c1.real), makefinite(c1.imaginary));
    }

    template<typename T>
    inline
    dual_v<T> operator+(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real + d2.real, d1.dual + d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator+(const dual_v<T>& d1, const U& v)
    {
        return dual_v<T>(d1.real + T(v), d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator+(const U& v, const dual_v<T>& d1)
    {
        return dual_v<T>(T(v) + d1.real, d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator+(const dual_v<T>& d1, const dual_v<U>& d2)
    {
        return d1 + dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator+(const dual_v<U>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real, d1.dual) + d2;
    }

    template<typename T>
    inline
    void operator+=(dual_v<T>& d1, const dual_v<T>& d2)
    {
        d1 = d1 + d2;
    }

    template<typename T>
    inline
    dual_v<T> operator-(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real - d2.real, d1.dual - d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator-(const dual_v<T>& d1, const U& v)
    {
        return dual_v<T>(d1.real - T(v), d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator-(const U& v, const dual_v<T>& d1)
    {
        return dual_v<T>(T(v) - d1.real, -d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator-(const dual_v<T>& d1, const dual_v<U>& d2)
    {
        return d1 - dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator-(const dual_v<U>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real, d1.dual) - d2;
    }

    template<typename T>
    inline
    dual_v<T> operator-(const dual_v<T>& d1)
    {
        return dual_v<T>(-d1.real, -d1.dual);
    }

    template<typename T>
    inline
    dual_v<T> operator*(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real * d2.real, d1.real * d2.dual + d2.real * d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator*(const dual_v<T>& d1, const U& v)
    {
        return d1 * dual_v<T>(T(v), T());
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator*(const U& v, const dual_v<T>& d1)
    {
        return dual_v<T>(T(v), T()) * d1;
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator*(const dual_v<T>& d1, const dual_v<U>& d2)
    {
        return d1 * dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator*(const dual_v<U>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real, d1.dual) * d2;
    }

    template<typename T>
    inline
    dual_v<T> operator/(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real / d2.real, ((d1.dual * d2.real - d1.real * d2.dual) / (d2.real * d2.real)));
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator/(const dual_v<T>& d1, const U& v)
    {
        return d1 / dual_v<T>(T(v), T());
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator/(const U& v, const dual_v<T>& d1)
    {
        return dual_v<T>(T(v), T()) / d1;
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator/(const dual_v<T>& d1, const dual_v<U>& d2)
    {
        return d1 / dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> operator/(const dual_v<U>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(d1.real, d1.dual) / d2;
    }

    template<typename T>
    inline
    dual_v<T> sqrt(const dual_v<T>& d1)
    {
        return dual_v<T>(sqrt(d1.real), T(0.5f) * d1.dual / sqrt(d1.real));
    }

    ///if this has no imaginary components, its guaranteed to be >= 0
    ///if it has imaginary components, all bets are off
    template<typename T>
    inline
    dual_v<T> psqrt(const dual_v<T>& d1)
    {
        return dual_v<T>(psqrt(d1.real), T(0.5f) * d1.dual / psqrt(d1.real));
    }

    template<typename T>
    inline
    dual_v<complex_v<T>> csqrt(const dual_v<T>& d1)
    {
        return dual_v<complex_v<T>>(::csqrt(d1.real), complex_v<T>(0.5f * d1.dual, 0) / ::csqrt(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> pow(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(pow(d1.real, d2.real), pow(d1.real, d2.real) * (d1.dual * (d2.real / d1.real) + d2.dual * log(d1.real)));
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_v<T> pow(const dual_v<T>& d1, const U& d2)
    {
        if constexpr(is_complex<T>())
        {
            static_assert(std::is_same_v<U, int>);

            return dual_v<T>(pow(d1.real, d2), pow(d1.real, d2 - 1) * T(d2) * d1.dual);
        }
        else
        {
            return dual_v<T>(pow(d1.real, T(d2)), pow(d1.real, T(d2 - 1)) * T(d2) * d1.dual);
        }
    }

    template<typename T>
    inline
    dual_v<T> log(const dual_v<T>& d1)
    {
        return dual_v<T>(log(d1.real), d1.dual / d1.real);
    }

    template<typename T>
    inline
    dual_v<T> fabs(const dual_v<T>& d1)
    {
        return dual_v<T>(fabs(d1.real), d1.real * d1.dual / fabs(d1.real));
    }

    ///https://math.stackexchange.com/questions/2352341/the-derivative-of-absolute-value-of-complex-function-fx-z-where-x-in-math
    template<typename T>
    inline
    dual_v<T> fabs(const dual_v<complex_v<T>>& d1)
    {
        return dual_v<T>(fabs(d1.real), Real(d1.real * conjugate(d1.dual)) / fabs(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> fma(const dual_v<T>& d1, const dual_v<T>& d2, const dual_v<T>& d3)
    {
        return d1 * d2 + d3;
    }

    template<typename T>
    inline
    dual_v<T> exp(const dual_v<T>& d1)
    {
        return dual_v<T>(exp(d1.real), d1.dual * exp(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> sin(const dual_v<T>& d1)
    {
        return dual_v<T>(sin(d1.real), d1.dual * cos(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> cos(const dual_v<T>& d1)
    {
        return dual_v<T>(cos(d1.real), -d1.dual * sin(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> sec(const dual_v<T>& d1)
    {
        return 1/cos(d1);
    }

    template<typename T>
    inline
    dual_v<T> tan(const dual_v<T>& d1)
    {
        return dual_v<T>(tan(d1.real), d1.dual / (cos(d1.real) * cos(d1.real)));
    }

    template<typename T>
    inline
    dual_v<T> sinh(const dual_v<T>& d1)
    {
        return dual_v<T>(sinh(d1.real), d1.dual * cosh(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> cosh(const dual_v<T>& d1)
    {
        return dual_v<T>(cosh(d1.real), d1.dual * sinh(d1.real));
    }

    template<typename T>
    inline
    dual_v<T> tanh(const dual_v<T>& d1)
    {
        return dual_v<T>(tanh(d1.real), d1.dual * (1 - tanh(d1.real) * tanh(d1.real)));
    }

    template<typename T>
    inline
    dual_v<T> asin(const dual_v<T>& d1)
    {
        return dual_v<T>(asin(d1.real), d1.dual / sqrt(1 - d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_v<T> acos(const dual_v<T>& d1)
    {
        return dual_v<T>(acos(d1.real), -d1.dual / sqrt(1 - d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_v<T> atan(const dual_v<T>& d1)
    {
        return dual_v<T>(atan(d1.real), d1.dual / (1 + d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_v<T> atan2(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_v<T>(atan2(d1.real, d2.real), (-d1.real * d2.dual / (d2.real * d2.real + d1.real * d1.real)) + d1.dual * d2.real / (d2.real * d2.real + d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_v<T> lambert_w0(const dual_v<T>& d1)
    {
        return dual_v<T>(lambert_w0(d1.real), d1.dual * lambert_w0(d1.real) / (d1.real * lambert_w0(d1.real) + d1.real));
    }

    ///https://math.stackexchange.com/questions/1052500/what-is-the-general-definition-of-the-conjugate-of-a-multiple-component-number
    template<typename T>
    inline
    dual_v<T> conjugate(const dual_v<T>& d1)
    {
        return dual_v<T>(conjugate(d1.real), conjugate(d1.dual));
    }

    template<typename T>
    inline
    dual_v<T> length(const dual_v<T>& d1, const dual_v<T>& d2, const dual_v<T>& d3)
    {
        T bottom = 2 * length(d1.real, d2.real, d3.real);

        return dual_v<T>(length(d1.real, d2.real, d3.real), (2 * d1.real * d1.dual + 2 * d2.real * d2.dual + 2 * d3.real * d3.dual) / bottom);
    }

    template<typename T>
    inline
    dual_v<T> smooth_fmod(const dual_v<T>& d1, const T& d2)
    {
        return {smooth_fmod(d1.real, d2), d1.dual};
    }

    /*template<typename T>
    inline
    dual_v<T> fast_length(const dual_v<T>& d1, const dual_v<T>& d2, const dual_v<T>& d3)
    {
        T bottom = 2 * fast_length(d1.real, d2.real, d3.real);

        return dual_v<T>(fast_length(d1.real, d2.real, d3.real), (2 * d1.real * d1.dual + 2 * d2.real * d2.dual + 2 * d3.real * d3.dual) / bottom);
    }*/

    template<typename T>
    inline
    dual_v<T> fast_length(const dual_v<T>& d1, const dual_v<T>& d2, const dual_v<T>& d3)
    {
        return sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    }

    template<typename T>
    inline
    dual_v<T> Real(const dual_v<complex_v<T>>& c1)
    {
        return dual_v<T>(Real(c1.real), Real(c1.dual));
    }

    template<typename T>
    inline
    dual_v<T> Imaginary(const dual_v<complex_v<T>>& c1)
    {
        return dual_v<T>(Imaginary(c1.real), Imaginary(c1.dual));
    }

    ///(a + bi) (a - bi) = a^2 + b^2
    template<typename T>
    inline
    dual_v<T> self_conjugate_multiply(const dual_v<complex_v<T>>& c1)
    {
        return Real(c1 * conjugate(c1));
    }

    template<typename T>
    inline
    dual_v<T> self_conjugate_multiply(const dual_v<T>& c1)
    {
        return c1 * c1;
    }

    template<typename T>
    inline
    complex_v<T> unit_i()
    {
        return complex_v<T>(0, 1);
    }

    template<typename T>
    inline
    dual_v<T> select(const dual_v<T>& d1, const dual_v<T>& d2, const T& d3)
    {
        return dual_v<T>(select(d1.real, d2.real, d3), select(d1.dual, d2.dual, d3));
    }

    template<typename T, typename U, typename V>
    inline
    auto dual_if(const T& condition, U&& if_true, V&& if_false)
    {
        return select(if_false(), if_true(), condition);
    }

    template<typename T>
    inline
    T operator<(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return d1.real < d2.real;
    }

    template<typename T>
    inline
    T operator<=(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return d1.real <= d2.real;
    }

    template<typename T>
    inline
    T operator>(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return d1.real > d2.real;
    }

    template<typename T>
    inline
    T operator>=(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return d1.real >= d2.real;
    }

    template<typename T>
    inline
    T operator==(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return d1.real == d2.real;
    }

    template<typename T>
    inline
    auto max(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_if(d1 < d2, [&](){return d2;}, [&](){return d1;});
    }

    template<typename T>
    inline
    auto min(const dual_v<T>& d1, const dual_v<T>& d2)
    {
        return dual_if(d1 < d2, [&](){return d1;}, [&](){return d2;});
    }
};

//using dual = dual_v<symbol>;
//using dual_complex = dual_v<complex<symbol>>;

#endif // DUAL_HPP_INCLUDED
