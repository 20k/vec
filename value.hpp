#ifndef DUAL2_HPP_INCLUDED
#define DUAL2_HPP_INCLUDED

#include "dual.hpp"
#include <set>
#include <optional>
#include <cmath>
#include <map>
#include <assert.h>
#include <variant>
#include <vec/tensor.hpp>
#include <CL/cl.h>
#include <concepts>
#include <source_location>
#include <memory>

#ifndef __clang__
#include <stdfloat>
#endif

#ifndef __clang__
using float16 = std::float16_t;
#else
using float16 = _Float16;
#endif

std::string get_stacktrace();

namespace dual_types
{
    template<typename T>
    struct value;

    template<typename T>
    struct value_mut;

    template<typename T>
    inline
    std::string name_type(T tag)
    {
        #define CMAP(x, y) if constexpr(std::is_same_v<T, x>) {return #y;}

        CMAP(float, float)
        //CMAP(cl_float, float);

        else CMAP(double, double)
        //CMAP(cl_double, double);

        else CMAP(float16, half)
        //CMAP(cl_half, half);

        //CMAP(cl_int, int);
        else CMAP(int, int)

        //CMAP(cl_short, short);
        else CMAP(short, short)

        //CMAP(cl_uint, unsigned int);
        else CMAP(unsigned int, unsigned int)

        //CMAP(cl_ushort, unsigned short);
        else CMAP(unsigned short, unsigned short)

        else CMAP(char, char)
        else CMAP(unsigned char, unsigned char)

        else CMAP(cl_float4, float4)
        else CMAP(cl_float3, float3)
        else CMAP(cl_float2, float2)

        else CMAP(cl_int4, int4)
        else CMAP(cl_int3, int3)
        else CMAP(cl_int2, int2)

        else CMAP(cl_uint4, uint4)
        else CMAP(cl_uint3, uint3)
        else CMAP(cl_uint2, uint2)

        else CMAP(cl_short4, short4)
        else CMAP(cl_short3, short3)
        else CMAP(cl_short2, short2)

        else CMAP(cl_ushort4, ushort4)
        else CMAP(cl_ushort3, ushort3)
        else CMAP(cl_ushort2, ushort2)

        else if constexpr(std::is_same_v<T, std::monostate>)
            return "monostate##neverused";

        else if constexpr(std::is_same_v<T, tensor<value<float>, 4>>)
            return "float4";

        else if constexpr(std::is_same_v<T, tensor<value<float>, 3>>)
            return "float3";

        else if constexpr(std::is_same_v<T, tensor<value<float>, 2>>)
            return "float2";

        else if constexpr(std::is_same_v<T, tensor<value<int>, 4>>)
            return "int4";

        else if constexpr(std::is_same_v<T, tensor<value<int>, 3>>)
            return "int3";

        else if constexpr(std::is_same_v<T, tensor<value<int>, 2>>)
            return "int2";

        else if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 4>>)
            return "ushort4";

        else if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 3>>)
            return "ushort3";

        else if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 2>>)
            return "ushort2";

        else if constexpr(std::is_same_v<T, tensor<value<float16>, 4>>)
            return "half4";

        else if constexpr(std::is_same_v<T, tensor<value<float16>, 3>>)
            return "half3";

        else if constexpr(std::is_same_v<T, tensor<value<float16>, 2>>)
            return "half2";

        else
        {
            #ifndef __clang__
            static_assert(false);
            #endif
        }

        #undef CMAP
    }

    template<typename T>
    inline
    std::string name_type(value<T> tag)
    {
        return name_type(typename value<T>::value_type());
    }

    template<typename T>
    inline
    std::string name_type(value_mut<T> tag)
    {
        return name_type(typename value_mut<T>::value_type());
    }

    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    inline
    std::string to_string_s(float v)
    {
        std::ostringstream oss;
        oss << std::setprecision(32) << std::fixed << std::showpoint << v;
        std::string str = oss.str();

        while(str.size() > 0 && str.back() == '0')
            str.pop_back();

        if(str.size() > 0 && str.back() == '.')
            str += "0";

        //str += "f";

        return str;
    }

    inline
    std::string to_string_s(std::integral auto v)
    {
        return std::to_string(v);
    }

    inline
    std::string to_string_s(std::monostate m)
    {
        throw std::runtime_error("Bad to_string_s std::monostate");
        return "";
    }

    template<typename T>
    inline
    std::optional<T> get_value(std::string_view in)
    {
        if(in.size() == 0)
            throw std::runtime_error("Bad in size, 0");

        while(in.size() > 2 && in.front() == '(' && in.back() == ')')
        {
            in.remove_prefix(1);
            in.remove_suffix(1);
        }

        if(in == "nan")
            throw std::runtime_error("Nan in get_value");

        std::string cstr(in);

        char* ptr = nullptr;
        double val = std::strtod(cstr.c_str(), &ptr);

        if(ptr == cstr.c_str() + cstr.size())
            return val;

        return std::nullopt;
    }

    template<typename T>
    inline
    std::variant<std::string, T> get_value_or_string(std::string_view in)
    {
        auto result_opt = get_value<T>(in);

        if(result_opt.has_value())
        {
            return result_opt.value();
        }

        return std::string(in);
    }

    template<typename T>
    inline
    std::string to_string_either(const std::variant<std::string, T>& in)
    {
        if(in.index() == 0)
            return std::get<0>(in);

        if constexpr(std::is_same_v<T, std::monostate>)
        {
            assert(false);
        }
        else
        {
            const T& val = std::get<1>(in);

            std::string result = std::visit([](const auto& in)
            {
                return to_string_s(in);
            }, val.storage);

            return result;

            //return to_string_s(std::get<1>(in));
        }
    }

    struct operation_desc
    {
        bool is_infix = false;
        std::string_view sym;
        bool is_semicolon_terminated = true;
        bool introduces_block = false;
        bool reordering_hazard = false;
    };

    namespace ops
    {
        enum type_t
        {
            PLUS,
            COMBO_PLUS,
            UMINUS,
            MINUS,
            MULTIPLY,
            COMBO_MULTIPLY,
            DIVIDE,
            MODULUS, ///c style %
            AND,
            ASSIGN,
            RETURN,
            BREAK,
            IF_S,
            IF_START,
            FOR_START,
            BLOCK_START,
            BLOCK_END,

            LESS,
            LESS_EQUAL,

            GREATER,
            GREATER_EQUAL,

            EQUAL,
            NOT_EQUAL,

            LAND,
            LOR,
            LNOT,

            COMMA,

            IDOT,
            CONVERT,

            SIN,
            COS,
            TAN,

            ASIN,
            ACOS,
            ATAN,
            ATAN2,

            EXP,

            SQRT,

            SINH,
            COSH,
            TANH,

            ASINH,
            ACOSH,
            ATANH,

            LOG,

            ISFINITE,
            SIGNBIT,
            SIGN,
            FABS,
            ABS,

            SELECT,
            POW,
            MAX,
            MIN,
            CLAMP,

            SMOOTH_FMOD, ///assumes that the derivative is smooth, and that we don't need to touch it

            LAMBERT_W0,

            FMA,
            MAD,

            FLOOR,
            CEIL,
            ROUND,

            BRACKET,
            BRACKET2,
            BRACKET_LINEAR,
            DECLARE,
            DECLARE_ARRAY,

            UNKNOWN_FUNCTION,
            SIDE_EFFECT,

            VALUE,
            NONE,
        };
    }

    inline
    operation_desc get_description(ops::type_t type)
    {
        using namespace ops;

        operation_desc ret;

        if(type == PLUS || type == MINUS || type == MULTIPLY || type == MODULUS || type == AND || type == ASSIGN ||
           type == LESS || type == LESS_EQUAL || type == GREATER || type == GREATER_EQUAL ||
           type == EQUAL || type == NOT_EQUAL || type == LAND || type == LOR || type == LNOT || type == COMMA || type == IDOT)
        {
            ret.is_infix = true;
        }

        #define NATIVE_DIVIDE
        #ifndef NATIVE_DIVIDE
        if(type == DIVIDE)
        {
            ret.is_infix = true;
        }
        #endif // NATIVE_DIVIDE

        std::array syms = {
            "+",
            "+",
            "-",
            "-",
            "*",
            "*",
            #ifdef NATIVE_DIVIDE
            "native_divide",
            #else
            "/",
            #endif // NATIVE_DIVIDE
            "%",
            "&",
            "=",
            "return",
            "break",
            "if#err",
            "if#err",
            "for#err",
            "block_start#err",
            "block_end#err",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "&&",
            "||",
            "!",
            ",",
            ".",
            "(#err)",
            "native_sin",
            "native_cos",
            "native_tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "native_exp",
            "native_sqrt",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "native_log",
            "isfinite",
            "signbit",
            "sign",
            "fabs",
            "abs",
            "select",
            "pow",
            "max",
            "min",
            "clamp",
            "smooth_fmod",
            "lambert_w0",
            "fma",
            "mad",
            "floor",
            "ceil",
            "round",
            "bad#bracket",
            "bad#bracket2",
            "bad#bracketlinear",
            "bad#declare",
            "bad#declarearray",
            "generated#function#failure",
            "side#effect",
            "ERROR#"
        };

        if(type == ops::FOR_START || type == ops::BLOCK_START || type == ops::BLOCK_END || type == ops::IF_START)
            ret.is_semicolon_terminated = false;

        if(type == ops::FOR_START || type == ops::IF_START)
            ret.introduces_block = true;

        if(type == ops::RETURN || type == ops::BREAK || type == ops::SIDE_EFFECT || type == ops::BLOCK_START || type == ops::BLOCK_END)
            ret.reordering_hazard = true;

        static_assert(syms.size() == ops::type_t::NONE);

        ret.sym = syms[(int)type];

        return ret;
    }

    template<typename T>
    inline
    T signbit(T in)
    {
        return in < 0;
    }

    template<typename T>
    inline
    T sign(T in)
    {
        if(in == T(-0.0))
            return T(-0.0);

        if(in == T(0.0))
            return T(0.0);

        if(in > 0)
            return 1;

        if(in < 0)
            return -1;

        if(std::isnan(in))
            return 0;

        throw std::runtime_error("Bad sign function");
    }

    template<typename T>
    requires std::is_arithmetic_v<T>
    inline
    T select(T v1, T v2, T v3)
    {
        if(v3 > 0)
            return v2;

        return v1;
    }

    template<typename T>
    inline
    T mad(T a, T b, T c)

    {
        return a * b + c;
    }

    struct codegen
    {
        inline virtual std::optional<std::string> bracket2(const value<std::monostate>& op) const;
        inline virtual std::optional<std::string> bracket_linear(const value<std::monostate>& op) const;
    };

    template<typename T>
    std::string type_to_string(const value<T>& op, const codegen& cg = codegen());

    template<typename U, typename... T>
    value<U> make_op(ops::type_t type, T&&... args);

    template<typename T>
    inline
    value<T> make_op(ops::type_t type, const std::vector<value<T>>& args);

    template<auto N>
    struct is_value_equal
    {
        template<typename T>
        bool operator()(const T& in)
        {
            if constexpr(std::is_same_v<T, std::monostate>)
                return false;
            else
                return in == N;
        }
    };

    template<typename T>
    bool equivalent(const value<T>& d1, const value<T>& d2);

    template<typename T>
    value<T> fma(const value<T>&, const value<T>&, const value<T>&);
    template<typename T>
    value<T> mad(const value<T>&, const value<T>&, const value<T>&);

    static inline auto length_sorter = [](const auto& v1, const auto& v2)
    {
        return type_to_string(v1) < type_to_string(v2);
    };

    template<typename T, typename Reduce>
    inline
    auto pairwise_reduce(std::vector<T> pending, Reduce&& reduce)
    {
        assert(pending.size() != 0);

        if(pending.size() == 1)
            return pending.at(0);

        ///take the first two elements a + b, bracket them as (a + b) as a singular element, then push them to the back of the queue
        ///eg a + b + c + d + e + f
        ///-> c + d + e + f + (a + b)
        ///-> e + f + (a + b) + (c + d)
        ///-> (a + b) + (c + d) + (e + f)
        ///-> (e + f) + ((a + b) + (c + d))
        ///-> ((e + f) + ((a + b) + (c + d)))
        while(pending.size() >= 2)
        {
            #ifdef ORDER_PRESERVING
            for(int i=0; i < pending.size(); i++)
            {
                if(i == (int)pending.size() - 1)
                    break;

                ///a b c d e f
                T v1 = pending[i];
                T v2 = pending[i + 1];

                ///groups ab
                T reduced = reduce(v1, v2);

                ///abcdef -> cdef
                ///-> (ab)abcdef
                pending.insert(pending.begin() + i, reduced);

                ///-> (ab)bcdef
                pending.erase(pending.begin() + i+1);
                ///-> (ab)cdef
                pending.erase(pending.begin() + i+1);
                //i--;
            }
            #else
            T v1 = pending.at(0);
            T v2 = pending.at(1);

            pending.erase(pending.begin());
            pending.erase(pending.begin());

            pending.push_back(reduce(v1, v2));
            #endif
        }

        assert(pending.size() == 1);

        return pending.at(0);
    }

    template<typename T, typename U>
    struct mutable_value
    {
        const T& v;
        U& ctx;
        std::source_location loc;

        mutable_value(const T& _v, U& _ctx, const std::source_location _loc = std::source_location::current()) : v(_v), ctx(_ctx), loc(_loc){}

        template<typename V>
        void operator=(const V& to_set)
        {
            if(!v.is_mutable)
            {
                std::cout << "file: "
                << loc.file_name() << '('
                << loc.line() << ':'
                << loc.column() << ") `"
                << loc.function_name() << "\n";
            }

            assert(v.is_mutable);

            ctx.exec(assign(v, to_set).as_generic());
        }
    };

    struct value_generic
    {
        ops::type_t type = ops::NONE;

        std::variant<std::string, std::monostate,
                     float16, float, double,
                     int64_t, uint64_t,
                     int, unsigned int,
                     uint16_t, int16_t,
                     uint8_t, int8_t>  value_payload;


        std::string original_type;
        bool is_mutable = false;
        using is_complex = std::false_type;
    };

    template<typename T>
    struct value : value_generic
    {
        using value_type = T;

        std::vector<value<T>> args;

        value()
        {
            value_payload = T{}; type = ops::VALUE;
            original_type = name_type(T());
        }

        template<typename U>
        requires std::is_arithmetic_v<U>
        value(U u)
        {
            value_payload = static_cast<T>(u); type = ops::VALUE;
            original_type = name_type(T());
        }

        value(const std::string& str){value_payload = str; type = ops::VALUE;original_type = name_type(T());}
        value(const char* str){assert(str); value_payload = std::string(str); type = ops::VALUE;original_type = name_type(T());}

        void set_from(const std::variant<std::string, T>& val)
        {
            std::visit([&](auto& in)
            {
                value_payload = in;
            }, val);
        }

        bool is_memory_access()
        {
            return type == ops::BRACKET || type == ops::BRACKET2 || type == ops::BRACKET_LINEAR;
        }

        template<typename U>
        U reinterpret_as() const
        {
            U result;
            result.type = type;
            result.original_type = original_type;
            result.is_mutable = is_mutable;
            result.value_payload = value_payload;

            for(const value& v : args)
            {
                result.args.push_back(v.template reinterpret_as<U>());
            }

            return result;
        }

        template<typename U>
        value<U> assert_as() const
        {
            assert(original_type == name_type(U()));

            return reinterpret_as<value<U>>();
        }

        template<typename U>
        value<U> convert() const
        {
            value<U> op = make_op<U>(ops::CONVERT, reinterpret_as<value<U>>(), name_type(U()));
            op.original_type = name_type(U());
            return op;
        }

        value<std::monostate> as_generic() const
        {
            if constexpr(std::is_same_v<T, std::monostate>)
                return *this;
            else
            {
                value<std::monostate> result;
                result.type = type;
                result.original_type = original_type;
                result.is_mutable = is_mutable;
                result.value_payload = value_payload;

                for(const value<T>& v : args)
                {
                    result.args.push_back(v.as_generic());
                }

                return result;
            }
        }

        template<typename U>
        explicit operator value<U>() const
        {
            return convert<U>();
        }

        bool is_value() const
        {
            return type == ops::VALUE;
        }

        bool is_constant() const
        {
            return is_value() && value_payload.index() != 0;
        }

        template<typename U>
        bool is_constant_constraint(U&& func) const
        {
            if(!is_constant())
                return false;

            auto wrapper_func = [&]<typename V>(const V& in)
            {
                if constexpr(std::is_same_v<V, std::monostate> || std::is_same_v<V, std::string>)
                {
                    assert(false);
                    return false;
                }
                else
                    return func(in);
            };

            return constant_callback(wrapper_func);
        }

        T get_constant() const
        {
            assert(is_constant());

            return std::get<T>(value_payload);
        }

        T get(int idx) const
        {
            return std::get<T>(args[idx].value_payload);
        }

        template<typename Func>
        auto constant_callback(Func&& f) const
        {
            assert(is_constant());

            return std::visit(std::forward<Func>(f), value_payload);
        }

        void make_value(const std::string& str)
        {
            set_from(get_value_or_string<T>(str));

            type = ops::VALUE;
        }

        void set_dual_constant()
        {
            value_payload = T(0.);
            type = ops::VALUE;
        }

        void set_dual_variable()
        {
            value_payload = T(1.);
            type = ops::VALUE;
        }

        #define PROPAGATE1(x, y) if(type == ops::x){return y(get(0));}
        #define PROPAGATE2(x, y) if(type == ops::x){return y(get(0), get(1));}
        #define PROPAGATE3(x, y) if(type == ops::x){return y(get(0), get(1), get(2));}

        value flatten(bool recurse = false) const
        {
            #ifdef __clang__
            if constexpr(std::is_same_v<T, float16>)
                return *this;
            else
            {
            #endif

            if(type == ops::VALUE)
                return *this;

            bool all_constant = true;

            for(auto& i : args)
            {
                if(!i.is_constant())
                {
                    all_constant = false;
                    break;
                }
            }

            if(all_constant)
            {
                if(type == ops::PLUS)
                    return get(0) + get(1);

                if(type == ops::UMINUS)
                    return -get(0);

                if(type == ops::MINUS)
                    return get(0) - get(1);

                if(type == ops::MULTIPLY)
                    return get(0) * get(1);

                if(type == ops::DIVIDE)
                    return get(0) / get(1);

                //if(type == ops::MODULUS)
                //    return make_op_value(get(0) % get(1));

                ///can't propagate because we only do doubles oops
                //if(type == ops::AND)
                //    return make_op_value(get(0) & get(1));

                if(type == ops::LESS)
                    return get(0) < get(1);

                if(type == ops::LESS_EQUAL)
                    return get(0) <= get(1);

                if(type == ops::GREATER)
                    return get(0) > get(1);

                if(type == ops::GREATER_EQUAL)
                    return get(0) >= get(1);

                if(type == ops::EQUAL)
                    return get(0) == get(1);

                if(type == ops::NOT_EQUAL)
                    return get(0) != get(1);

                PROPAGATE1(SIN, std::sin);
                PROPAGATE1(COS, std::cos);
                PROPAGATE1(TAN, std::tan);
                PROPAGATE1(ASIN, std::asin);
                PROPAGATE1(ACOS, std::acos);
                PROPAGATE1(ATAN, std::atan);
                PROPAGATE2(ATAN2, std::atan2);
                PROPAGATE1(EXP, std::exp);
                PROPAGATE1(SQRT, std::sqrt);
                PROPAGATE1(SINH, std::sinh);
                PROPAGATE1(COSH, std::cosh);
                PROPAGATE1(TANH, std::tanh);
                PROPAGATE1(ASINH, std::asinh);
                PROPAGATE1(ACOSH, std::acosh);
                PROPAGATE1(ATANH, std::atanh);
                PROPAGATE1(LOG, std::log);
                PROPAGATE1(ISFINITE, std::isfinite);
                PROPAGATE1(SIGNBIT, signbit);
                PROPAGATE1(SIGN, sign);
                PROPAGATE1(FABS, std::fabs);
                PROPAGATE1(ABS, std::abs);
                PROPAGATE3(SELECT, select);
                PROPAGATE2(POW, std::pow);
                PROPAGATE2(MAX, std::max);
                PROPAGATE2(MIN, std::min);
                PROPAGATE3(CLAMP, std::clamp);
                //PROPAGATE2(LAMBERT_W0, lambert_w0);

                ///FMA is not propagated as we can't actually simulate it? Does that matter?
                PROPAGATE3(MAD, mad);
            }

            if(type == ops::SELECT)
            {
                if(args[2].is_constant())
                {
                    bool gt_zero = args[2].is_constant_constraint([](const auto& in){return in > 0;});

                    return gt_zero ? args[1] : args[0];
                }
            }

            auto is_zero = []<typename U>(const U& f)
            {
                return f == 0;
            };

            if(type == ops::MULTIPLY)
            {
                if(args[0].is_constant_constraint(is_zero) || args[1].is_constant_constraint(is_zero))
                    return value(0);

                //std::cout << "hello " << type_to_string(args[0]) << " with " << type_to_string(args[1]) << std::endl;

                if(args[0].is_constant_constraint(is_value_equal<1>{}))
                    return args[1].flatten();

                if(args[1].is_constant_constraint(is_value_equal<1>{}))
                    return args[0].flatten();
            }

            if(type == ops::PLUS)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return args[1].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return 2 * args[0].flatten();
            }

            if(type == ops::COMBO_PLUS)
            {
                int const_num = 0;

                for(const auto& i : args)
                {
                    if(i.is_constant())
                        const_num++;
                }

                if(const_num > 1)
                {
                    value cst = 0;

                    value copied = *this;

                    for(int i=0; i < (int)copied.args.size(); i++)
                    {
                        if(copied.args[i].is_constant())
                        {
                            cst += copied.args[i];

                            copied.args.erase(copied.args.begin() + i);
                            i--;
                            continue;
                        }
                    }

                    copied.args.push_back(cst);

                    return copied.flatten();
                }

                ///still need to implement equivalence checking to do n * thing
            }

            if(type == ops::COMBO_MULTIPLY)
            {
                int const_num = 0;

                for(const auto& i : args)
                {
                    if(i.is_constant())
                        const_num++;
                }

                if(const_num > 1)
                {
                    value cst = 1;

                    value copied = *this;

                    for(int i=0; i < (int)copied.args.size(); i++)
                    {
                        if(copied.args[i].is_constant())
                        {
                            cst = cst * copied.args[i];

                            copied.args.erase(copied.args.begin() + i);
                            i--;
                            continue;
                        }
                    }

                    copied.args.push_back(cst);

                    return copied.flatten();
                }

                ///still need to implement equivalence checking to do n * thing
            }

            if(type == ops::MINUS)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return -args[1].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return 0;
            }

            if(type == ops::UMINUS)
            {
                if(args[0].type == ops::MULTIPLY)
                {
                    if(args[0].args[0].is_constant())
                    {
                        return (-args[0].args[0]) * args[0].args[1];
                    }

                    if(args[0].args[1].is_constant())
                    {
                        return args[0].args[0] * (-args[0].args[1]);
                    }
                }

                if(args[0].type == ops::DIVIDE)
                {
                    if(args[0].args[0].is_constant())
                    {
                        return (-args[0].args[0]) / args[0].args[1];
                    }

                    if(args[0].args[1].is_constant())
                    {
                        return args[0].args[0] / (-args[0].args[1]);
                    }
                }
            }

            if(type == ops::DIVIDE)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return 0;

                if(args[1].is_constant_constraint(is_value_equal<1>{}))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return 1;
            }

            ///ops::MODULUS

            if(type == ops::POW)
            {
                if(args[1].is_constant_constraint(is_value_equal<1>{}))
                    return args[0].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return 1;

                if constexpr(std::is_arithmetic_v<T>)
                {
                    ///according to amd, this is an unconditional win
                    if(args[1].is_constant())
                    {
                        T cst = args[1].get_constant();

                        if(cst == floor(cst) && abs(cst) < 32)
                        {
                            value ret = args[0];

                            for(int i=0; i < abs(cst) - 1; i++)
                            {
                                ret = ret * args[0];
                            }

                            if(cst > 0)
                                return ret;
                            else
                                return 1/ret;
                        }
                    }
                }
            }

            if(type == ops::FMA || type == ops::MAD)
            {
                ///a * 0 + c or 0 * b + c
                if(args[0].is_constant_constraint(is_zero) || args[1].is_constant_constraint(is_zero))
                    return args[2].flatten();

                ///1 * b + c
                if(args[0].is_constant_constraint(is_value_equal<1>{}))
                    return (args[1] + args[2]).flatten();

                ///a * 1 + c
                if(args[1].is_constant_constraint(is_value_equal<1>{}))
                    return (args[0] + args[2]).flatten();

                ///a * b + 0
                if(args[2].is_constant_constraint(is_zero))
                {
                    return (args[0] * args[1]).flatten();
                }
            }


            //#define FMA_REPLACE
            ///much worse than letting the compiler do it, even with mad
            ///better with new backend (!)
            #ifdef FMA_REPLACE
            if(type == ops::PLUS && args[0].original_type == "float")
            {
                if(args[0].type == ops::MULTIPLY)
                {
                    value c = args[1];
                    value a = args[0].args[0];
                    value b = args[0].args[1];

                    return fma(a, b, c);
                }
                else if(args[1].type == ops::MULTIPLY)
                {
                    value c = args[0];
                    value a = args[1].args[0];
                    value b = args[1].args[1];

                    return fma(a, b, c);
                }
            }
            #endif

            if(recurse)
            {
                value ret = *this;

                for(auto& i : ret.args)
                    i = i.flatten(true);

                return ret;
            }

            return *this;

            #ifdef __clang__
            }
            #endif
        }

        value group_associative_operators() const
        {
            #define NO_REGROUP_ASSOCIATIVE
            #ifdef NO_REGROUP_ASSOCIATIVE
            return *this;
            #endif

            if(type == ops::PLUS)
            {
                ///note to self, don't do this recursively
                if(type == ops::COMBO_PLUS)
                    return *this;

                std::vector<value<T>> final_args;

                for(int i=0; i < (int)args.size(); i++)
                {
                    if(args[i].type == ops::PLUS || args[i].type == ops::COMBO_PLUS)
                        final_args.insert(final_args.end(), args[i].args.begin(), args[i].args.end());
                    else
                        final_args.push_back(args[i]);
                }

                //std::sort(final_args.begin(), final_args.end(), length_sorter);

                ///could disable combo plus promotion, keeps flattening etc but not that helpful
                //if(final_args.size() == 2)
                //    return *this;

                return make_op_vec<T>(ops::COMBO_PLUS, final_args);
            }

            #ifdef COMBO_MULT
            if(type == ops::MULTIPLY)
            {
                ///note to self, don't do this recursively
                if(type == ops::COMBO_MULTIPLY)
                    return *this;

                std::vector<value<T>> final_args;

                for(int i=0; i < (int)args.size(); i++)
                {
                    if(args[i].type == ops::MULTIPLY || args[i].type == ops::COMBO_MULTIPLY)
                        final_args.insert(final_args.end(), args[i].args.begin(), args[i].args.end());
                    else
                        final_args.push_back(args[i]);
                }

                //std::sort(final_args.begin(), final_args.end(), length_sorter);

                ///could disable combo plus promotion, keeps flattening etc but not that helpful
                //if(final_args.size() == 2)
                //    return *this;

                return make_op<T>(ops::COMBO_MULTIPLY, final_args);
            }
            #endif

            return *this;
        }

        #define DUAL_CHECK1(x, y) if(type == x) { return y(args[0].dual(sym)); }
        #define DUAL_CHECK2(x, y) if(type == x) { return y(args[0].dual(sym), args[1].dual(sym)); }
        #define DUAL_CHECK3(x, y) if(type == x) { return y(args[0].dual(sym), args[1].dual(sym), args[2].dual(sym)); }

        #define DUAL_CHECK21(x, y) if(type == x) { return y(forw(args[0])); }
        #define DUAL_CHECK22(x, y) if(type == x) { return y(forw(args[0]), forw(args[1])); }
        #define DUAL_CHECK23(x, y) if(type == x) { return y(forw(args[0]), forw(args[1]), forw(args[2])); }

        dual_types::dual_v<value<T>> dual2(const std::vector<std::pair<value<T>, value<T>>>& substitutions) const
        {
            using namespace ops;

            auto forw = [&](const value<T>& in)
            {
                return in.dual2(substitutions);
            };

            for(const auto& [root, derivative] : substitutions)
            {
                if(equivalent(*this, root))
                {
                    dual_types::dual_v<value<T>> ret;

                    ret.real = root;
                    ret.dual = derivative;

                    return ret;
                }
            }

            if(type == VALUE)
            {
                dual_types::dual_v<value<T>> ret;

                ret.real = *this;
                ret.dual = 0;

                return ret;
            }
            if(type == PLUS)
            {
                return forw(args[0]) + forw(args[1]);
            }
            if(type == COMBO_PLUS)
            {
                std::vector<dual_types::dual_v<value<T>>> vals;

                for(const auto& i : args)
                {
                    vals.push_back(forw(i));
                }

                return pairwise_reduce(vals, [](const auto& v1, const auto& v2){return v1 + v2;});
            }
            if(type == COMBO_MULTIPLY)
            {
                std::vector<dual_types::dual_v<value<T>>> vals;

                for(const auto& i : args)
                {
                    vals.push_back(forw(i));
                }

                return pairwise_reduce(vals, [](const auto& v1, const auto& v2){return v1 * v2;});
            }
            if(type == UMINUS)
            {
                return -forw(args[0]);
            }
            if(type == MINUS)
            {
                return forw(args[0]) - forw(args[1]);
            }
            if(type == MULTIPLY)
            {
                return forw(args[0]) * forw(args[1]);
            }
            if(type == DIVIDE)
            {
                return forw(args[0]) / forw(args[1]);
            }
            /*if(type == MODULUS)
            {

            }*/
            /*if(type == AND)
            {

            }*/
            if(type == LESS)
            {
                return forw(args[0]) < forw(args[1]);
            }
            if(type == LESS_EQUAL)
            {
                return forw(args[0]) <= forw(args[1]);
            }

            if(type == FMA || type == MAD)
            {
                return forw(args[0]) * forw(args[1]) + forw(args[2]);
            }

            /*if(type == EQUAL)
            {
                return args[0].dual(sym) == args[1].dual(sym);
            }*/

            /*if(type == GREATER)
            {
                return args[0].dual(sym) > args[1].dual(sym);
            }
            if(type == GREATER_EQUAL)
            {
                return args[0].dual(sym) >= args[1].dual(sym);
            }*/

            DUAL_CHECK21(SIN, sin);
            DUAL_CHECK21(COS, cos);
            DUAL_CHECK21(TAN, tan);
            DUAL_CHECK21(ASIN, asin);
            DUAL_CHECK21(ACOS, acos);
            DUAL_CHECK21(ATAN, atan);
            DUAL_CHECK22(ATAN2, atan2);
            DUAL_CHECK21(EXP, exp);
            DUAL_CHECK21(SQRT, sqrt);
            DUAL_CHECK21(SINH, sinh);
            DUAL_CHECK21(COSH, cosh);
            DUAL_CHECK21(TANH, tanh);
            DUAL_CHECK21(LOG, log);
            //DUAL_CHECK1(ISFINITE, isfinite);
            //DUAL_CHECK1(SIGNBIT, signbit);
            //DUAL_CHECK1(SIGN, sign);
            DUAL_CHECK21(FABS, fabs);
            //DUAL_CHECK1(ABS, abs);

            if(type == SELECT)
            {
                return dual_types::select(forw(args[0]), forw(args[1]), args[2]);
            }

            DUAL_CHECK22(POW, pow);
            DUAL_CHECK22(MAX, max);
            DUAL_CHECK22(MIN, min);
            DUAL_CHECK21(LAMBERT_W0, lambert_w0);

            assert(false);
        }

        dual_types::dual_v<value<T>> dual(const std::string& sym) const
        {
            using namespace ops;

            if(type == VALUE)
            {
                dual_types::dual_v<value<T>> ret;

                if(value_payload.index() == 0 && std::get<0>(value_payload) == sym)
                {
                    ret = dual_types::dual_v<value<T>>(*this, 1.f);
                }
                else
                {
                    ret = dual_types::dual_v<value<T>>(*this, 0.f);
                }

                return ret;
            }
            if(type == PLUS)
            {
                return args[0].dual(sym) + args[1].dual(sym);
            }
            if(type == COMBO_PLUS)
            {
                std::vector<dual_types::dual_v<value<T>>> vals;

                for(const auto& i : args)
                {
                    vals.push_back(i.dual(sym));
                }

                return pairwise_reduce(vals, [](const auto& v1, const auto& v2){return v1 + v2;});
            }
            if(type == COMBO_MULTIPLY)
            {
                std::vector<dual_types::dual_v<value<T>>> vals;

                for(const auto& i : args)
                {
                    vals.push_back(i.dual(sym));
                }

                return pairwise_reduce(vals, [](const auto& v1, const auto& v2){return v1 * v2;});
            }
            if(type == UMINUS)
            {
                return -args[0].dual(sym);
            }
            if(type == MINUS)
            {
                return args[0].dual(sym) - args[1].dual(sym);
            }
            if(type == MULTIPLY)
            {
                return args[0].dual(sym) * args[1].dual(sym);
            }
            if(type == DIVIDE)
            {
                return args[0].dual(sym) / args[1].dual(sym);
            }
            /*if(type == MODULUS)
            {

            }*/
            /*if(type == AND)
            {

            }*/
            if(type == LESS)
            {
                return args[0].dual(sym) < args[1].dual(sym);
            }
            if(type == LESS_EQUAL)
            {
                return args[0].dual(sym) <= args[1].dual(sym);
            }

            if(type == FMA || type == MAD)
            {
                return args[0].dual(sym) * args[1].dual(sym) + args[2].dual(sym);
            }

            /*if(type == EQUAL)
            {
                return args[0].dual(sym) == args[1].dual(sym);
            }*/

            /*if(type == GREATER)
            {
                return args[0].dual(sym) > args[1].dual(sym);
            }
            if(type == GREATER_EQUAL)
            {
                return args[0].dual(sym) >= args[1].dual(sym);
            }*/

            DUAL_CHECK1(SIN, sin);
            DUAL_CHECK1(COS, cos);
            DUAL_CHECK1(TAN, tan);
            DUAL_CHECK1(ASIN, asin);
            DUAL_CHECK1(ACOS, acos);
            DUAL_CHECK1(ATAN, atan);
            DUAL_CHECK2(ATAN2, atan2);
            DUAL_CHECK1(EXP, exp);
            DUAL_CHECK1(SQRT, sqrt);
            DUAL_CHECK1(SINH, sinh);
            DUAL_CHECK1(COSH, cosh);
            DUAL_CHECK1(TANH, tanh);
            DUAL_CHECK1(LOG, log);
            //DUAL_CHECK1(ISFINITE, isfinite);
            //DUAL_CHECK1(SIGNBIT, signbit);
            //DUAL_CHECK1(SIGN, sign);
            DUAL_CHECK1(FABS, fabs);
            //DUAL_CHECK1(ABS, abs);

            if(type == SELECT)
            {
                return dual_types::select(args[0].dual(sym), args[1].dual(sym), args[2]);
            }

            DUAL_CHECK2(POW, pow);
            DUAL_CHECK2(MAX, max);
            DUAL_CHECK2(MIN, min);
            DUAL_CHECK1(LAMBERT_W0, lambert_w0);

            assert(false);
        }

        value<T> differentiate(const std::string& sym) const
        {
            return dual(sym).dual;
        }

        void substitute_impl(const std::string& sym, T value)
        {
            if(type == ops::VALUE && value_payload.index() == 0 && std::get<0>(value_payload) == sym)
            {
                value_payload = double{value};
                return;
            }

            int start = 0;

            if(type == ops::UNKNOWN_FUNCTION)
                start = 1;

            for(int kk=start; kk < (int)args.size(); kk++)
            {
                args[kk].substitute_impl(sym, value);
                args[kk] = args[kk].flatten();
            }

            *this = flatten();
        }

        value substitute(const std::string& sym, T value) const
        {
            auto cp = *this;

            cp.substitute_impl(sym, value);

            return cp;
        }

        void get_all_variables_impl(std::set<std::string>& v) const
        {
            if(type == ops::VALUE)
            {
                if(is_constant())
                    return;

                v.insert(std::get<0>(value_payload));
                return;
            }

            for_each_real_arg([&](const value& me)
            {
                me.get_all_variables_impl(v);
            });
        }

        std::vector<std::string> get_all_variables() const
        {
            std::set<std::string> v;
            get_all_variables_impl(v);

            std::vector<std::string> ret(v.begin(), v.end());

            return ret;
        }

        template<typename U>
        void for_each_real_arg(U&& in) const
        {
            if(type == ops::IDOT)
                return;

            if(type == ops::SIDE_EFFECT)
                return;

            for(int i=0; i < (int)args.size(); i++)
            {
                if(type == ops::DECLARE && i < 2)
                    continue;

                if(type == ops::DECLARE_ARRAY)
                    continue;

                if(type == ops::UNKNOWN_FUNCTION && i == 0)
                    continue;

                if(type == ops::CONVERT && i == 1)
                    continue;

                if(type == ops::ASSIGN && i == 0)
                    continue;

                if((type == ops::BRACKET || type == ops::BRACKET2 || type == ops::BRACKET_LINEAR) && i == 0)
                    continue;

                in(args[i]);
            }
        }

        template<typename U>
        void for_each_real_arg(U&& in)
        {
            if(type == ops::IDOT)
                return;

            if(type == ops::SIDE_EFFECT)
                return;

            for(int i=0; i < (int)args.size(); i++)
            {
                if(type == ops::DECLARE && i < 2)
                    continue;

                if(type == ops::DECLARE_ARRAY)
                    continue;

                if(type == ops::UNKNOWN_FUNCTION && i == 0)
                    continue;

                if(type == ops::CONVERT && i == 1)
                    continue;

                if(type == ops::ASSIGN && i == 0)
                    continue;

                if((type == ops::BRACKET || type == ops::BRACKET2 || type == ops::BRACKET_LINEAR) && i == 0)
                    continue;

                in(args[i]);
            }
        }

        template<typename U>
        void recurse_arguments(U&& in) const
        {
            in(*this);

            for_each_real_arg([&](const value& me)
            {
                me.recurse_arguments(std::forward<U>(in));
            });
        }

        template<typename U>
        void recurse_arguments(U&& in)
        {
            in(*this);

            for_each_real_arg([&](value& me)
            {
                me.recurse_arguments(std::forward<U>(in));
            });
        }

        void substitute(const std::map<std::string, std::string>& variables)
        {
            if(type == ops::VALUE)
            {
                if(is_constant())
                    return;

                auto it = variables.find(std::get<0>(value_payload));

                if(it == variables.end())
                    return;

                set_from(get_value_or_string<T>(it->second));
                return;
            }

            for(auto& val : args)
            {
                val.substitute(variables);
            }
        }

        void substitute(const std::vector<std::pair<value<T>, value<T>>>& variables)
        {
            for(auto& [conc, rep] : variables)
            {
                if(equivalent(*this, conc))
                {
                    *this = rep;
                    break;
                }
            }

            for(auto& val : args)
            {
                val.substitute(variables);
            }
        }

        template<typename U>
        void substitute_generic(U&& func)
        {
            auto sub_opt = func(*this);

            if(sub_opt.has_value())
            {
                *this = *sub_opt;
                return;
            }

            for(auto& val : args)
            {
                val.substitute_generic(func);
            }

            *this = flatten();
        }

        value x()
        {
            return make_op<T>(ops::IDOT, *this, "x");
        }

        value y()
        {
            return make_op<T>(ops::IDOT, *this, "y");
        }

        value z()
        {
            return make_op<T>(ops::IDOT, *this, "z");
        }

        value w()
        {
            return make_op<T>(ops::IDOT, *this, "w");
        }

        value property(const std::string& name)
        {
            return make_op<T>(ops::IDOT, *this, name);
        }

        value<T> index(int idx)
        {
            if(idx == 0)
                return x();
            if(idx == 1)
                return y();
            if(idx == 2)
                return z();
            if(idx == 3)
                return w();

            assert(false);
        }

        value<T>& operator+=(const value<T>& d1)
        {
            *this = *this + d1;

            return *this;
        }

        value<T>& operator-=(const value<T>& d1)
        {
            *this = *this - d1;

            return *this;
        }

        friend
        value<T> operator<(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LESS, d1, d2);
        }

        friend
        value<T> operator<=(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LESS_EQUAL, d1, d2);
        }

        friend
        value<T> operator>(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::GREATER, d1, d2);
        }

        friend
        value<T> operator>=(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::GREATER_EQUAL, d1, d2);
        }

        friend
        value<T> operator==(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::EQUAL, d1, d2);
        }

        friend
        value<T> operator!=(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::NOT_EQUAL, d1, d2);
        }

        friend
        value<T> operator+(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::PLUS, d1, d2);
        }

        friend
        value<T> operator-(const value<T>& d1, const value<T>& d2)
        {
            #ifdef NO_REGROUP_ASSOCIATIVE
            return make_op<T>(ops::MINUS, d1, d2);
            #else
            return make_op<T>(ops::PLUS, d1, make_op<T>(ops::UMINUS, d2));
            #endif
        }

        friend
        value<T> operator-(const value& d1)
        {
            return make_op<T>(ops::UMINUS, d1);
        }

        friend
        value<T> operator*(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MULTIPLY, d1, d2);
        }

        friend
        value<T> operator*(const T& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MULTIPLY, value<T>(d1), d2);
        }

        friend
        value<T> operator*(const value<T>& d1, const T& d2)
        {
            return make_op<T>(ops::MULTIPLY, d1, value<T>(d2));
        }

        friend
        value<T> operator/(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::DIVIDE, d1, d2);
        }

        friend
        value<T> operator%(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MODULUS, d1, d2);
        }

        friend
        value<T> operator&(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::AND, d1, d2);
        }

        friend
        value<T> operator||(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LOR, d1, d2);
        }

        friend
        value<T> operator&&(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LAND, d1, d2);
        }

        friend
        value<T> operator!(const value<T>& d1)
        {
            return make_op<T>(ops::LNOT, d1);
        }

        /*template<typename U>
        friend
        value<std::monostate> operator,(const value<T>& d1, const value<U>& d2)
        {
            return make_op<std::monostate>(ops::COMMA, d1.as_generic(), d2.as_generic());
        }*/
    };

    std::optional<std::string> codegen::bracket2(const value<std::monostate>& op) const
    {
        ///the way this works then is that we pass in dim information, eg
        ///"name", px, py, pz, dx, dy, dz

        int argc = op.args.size();

        assert(argc == 2 || argc == 3 || argc == 5 || argc == 7 || argc == 9);

        std::string name = type_to_string(op.args[0], *this);

        ///"name", px, dx
        if(argc == 2 || argc == 3)
        {
            return "(" + name + "[" + type_to_string(op.args[1], *this) + "])";;
        }

        ///2d indexing
        ///"name", px, py, dx, dy
        if(argc == 5)
        {
            auto x = op.args[1];
            auto y = op.args[2];
            auto width = op.args[3];

            return "(" + name + "[" + type_to_string(y * width + x, *this) + "])";
        }

        ///3d indexing
        ///"name", px, py, pz, dx, dy, dz
        if(argc == 7)
        {
            auto x = op.args[1];
            auto y = op.args[2];
            auto z = op.args[3];
            auto width = op.args[4];
            auto height = op.args[5];

            return "(" + name + "[" + type_to_string(z * width * height + y * width + x) + "])";
        }

        ///4d indexing
        ///"name", px, py, pz, pw, dx, dy, dz, dw
        if(argc == 9)
        {
            auto x = op.args[1];
            auto y = op.args[2];
            auto z = op.args[3];
            auto w = op.args[4];
            auto width = op.args[5];
            auto height = op.args[6];
            auto depth = op.args[7];

            return "(" + name + "[" + type_to_string(w * width * height * depth + z * width * height + y * width + x) + "])";
        }

        return std::nullopt;
    }

    std::optional<std::string> codegen::bracket_linear(const value<std::monostate>& op) const
    {
        ///same as bracket2, except that px etc can be fractional, and we do linear interpolation

        return std::nullopt;
    }

    template<typename T>
    struct value_mut : value<T>
    {
        value_mut() : value<T>()
        {
            value<T>::is_mutable = true;
        }

        const value<T>& as_constant() const
        {
            return *this;
        }

        void set_from_constant(const value<T>& in)
        {
            static_cast<value<T>&>(*this) = in;
            value<T>::is_mutable = true;
        }

        operator value<T>() const
        {
            return as_constant();
        }

        template<typename U>
        auto as_mutable(U& executor, const std::source_location loc = std::source_location::current()) const
        {
            return mutable_value(*this, executor, loc);
        }

        template<typename U>
        auto mut(U& executor, const std::source_location loc = std::source_location::current()) const
        {
            return mutable_value(*this, executor, loc);
        }
    };

    template<typename T>
    inline
    const T& as_constant(const T& in)
    {
        return in;
    }

    template<typename T>
    inline
    value<T> as_constant(const value_mut<T>& in)
    {
        return in.as_constant();
    }

    template<typename T>
    inline
    value_mut<T> to_mutable(const value<T>& in)
    {
        value_mut<T> v;
        v.set_from_constant(in);
        return v;
    }

    template<typename T>
    inline
    const T& to_mutable(const T& in)
    {
        return in;
    }

    template<typename T>
    inline
    std::string get_function_name(const value<T>& v)
    {
        assert(!v.is_value());

        if(v.type == ops::UNKNOWN_FUNCTION)
            return type_to_string(v.args.at(0));
        else
            return std::string(get_description(v.type).sym);
    }

    template<typename T>
    inline
    std::vector<value<T>> get_function_args(const value<T>& v)
    {
        assert(!v.is_value());

        if(v.type == ops::UNKNOWN_FUNCTION)
            return std::vector<value<T>>(v.args.begin() + 1, v.args.end());
        else
            return v.args;

    }
    template<typename T>
    inline
    bool equivalent(const value<T>& d1, const value<T>& d2)
    {
        if(d1.type != d2.type)
            return false;

        if(d1.args.size() != d2.args.size())
            return false;

        if(d1.is_mutable != d2.is_mutable)
            return false;

        if(d1.original_type != d2.original_type)
            return false;

        if(d1.type == ops::VALUE)
        {
            if(d1.is_constant() != d2.is_constant())
                return false;

            if(d1.is_constant())
            {
                auto comparator = []<typename T1, typename T2>(const T1& v1, const T2& v2)
                {
                    if constexpr(!std::is_same_v<T1, T2>)
                        return false;
                    else
                        return v1 == v2;
                };

                return std::visit(comparator, d1.value_payload, d2.value_payload);
            }

            if(d1.original_type != d2.original_type)
                return false;

            return d1.value_payload == d2.value_payload;
        }

        if(d1.args.size() == 2 && (d1.type == ops::MULTIPLY || d1.type == ops::PLUS))
        {
            return (equivalent(d1.args[0], d2.args[0]) && equivalent(d1.args[1], d2.args[1])) ||
                   (equivalent(d1.args[1], d2.args[0]) && equivalent(d1.args[0], d2.args[1]));
        }
        for(int i=0; i < (int)d1.args.size(); i++)
        {
            if(!equivalent(d1.args.at(i), d2.args.at(i)))
                return false;
        }

        return true;
    }

    /*template<typename T>
    inline
    bool negates(const value<T>& v1, const value<T>& v2)
    {
        if(v1.is_value() && v2.type == dual_type::ops::UMINUS)
        {
            if(equivalent(v1, v2.args.at(0)))
                return true;
        }

        if(v2.is_value() && v1.type == dual_type::ops::UMINUS)
        {
            if(equivalent(v2, v1.args.at(0)))
                return true;
        }

        return false;
    }

    template<typename T>
    inline
    bool evaluates_to_zero(const value<T>& v1)
    {
        if(v1.type == dual_type::ops::PLUS)
            return negates(v1.args.at(0), v1.args.at(1));

        return false;
    }*/

    inline
    std::string type_to_string(const std::string& in)
    {
        return in;
    }

    template<typename Unused>
    inline
    std::string type_to_string(const value<Unused>& op, const codegen& cg)
    {
        //make it difficult to call type_to_string without the code generator
        auto type_to_string = [&](const value<Unused>& op)
        {
            return ::dual_types::type_to_string(op, cg);
        };

        //static_assert(std::is_same_v<T, std::float16_t> || std::is_same_v<T, double> || std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, short> || std::is_same_v<T, unsigned short> || std::is_same_v<T, std::monostate>);

        ///the type system is becoming really not ideal, if we promote a half to a float, we have no way of detecting that. Relying on stringyness
        if(op.type == ops::VALUE)
        {
            if(op.value_payload.index() == 0)
                return std::get<0>(op.value_payload);

            return std::visit([]<typename T>(const T& in)
            {
                if constexpr(std::is_same_v<T, std::string>)
                {
                    assert(false);
                    return std::string();
                }

                std::string suffix;

                if constexpr(std::is_same_v<T, float16>)
                    suffix = "h";
                if constexpr(std::is_same_v<T, float>)
                    suffix = "f";

                if constexpr(!std::is_same_v<T, std::monostate> && !std::is_same_v<T, std::string>)
                {
                    if(in < 0)
                        return "(" + to_string_s(in) + suffix + ")";
                    else
                        return to_string_s(in) + suffix;
                }

                assert(false);
                return std::string();
            }, op.value_payload);
        }

        if(op.type == ops::BRACKET)
        {
            return "(" + type_to_string(op.args[0]) + "[" + type_to_string(op.args[1]) + "])";
        }

        if(op.type == ops::BRACKET2)
            return cg.bracket2(op.as_generic()).value();

        if(op.type == ops::BRACKET_LINEAR)
            return cg.bracket_linear(op.as_generic()).value();

        if(op.type == ops::RETURN)
        {
            if(op.args.size() == 0)
                return "return";
            else
                return "return " + type_to_string(op.args.at(0));
        }

        if(op.type == ops::BREAK)
            return "break";

        if(op.type == ops::FOR_START)
        {
            return "for(" + type_to_string(op.args.at(0)) + " " + type_to_string(op.args.at(1)) + "=" + type_to_string(op.args.at(2)) + ";" +
                            type_to_string(op.args.at(3)) + ";" + type_to_string(op.args.at(4)) + ")";
        }

        if(op.type == ops::BLOCK_START)
        {
            return "{";
        }

        if(op.type == ops::BLOCK_END)
        {
            return "}";
        }

        if(op.type == ops::IF_S)
        {
            return "if(" + type_to_string(op.args[0]) + "){" + type_to_string(op.args[1]) + ";}";
        }

        if(op.type == ops::IF_START)
        {
            return "if(" + type_to_string(op.args[0]) + ")";
        }

        if(op.type == ops::COMMA)
        {
            return type_to_string(op.args[0]) + ";" + type_to_string(op.args[1]);
        }

        if(op.type == ops::CONVERT)
        {
            return "((" + type_to_string(op.args[1]) + ")" + type_to_string(op.args[0]) + ")";
        }

        if(op.type == ops::LNOT)
        {
            return "(!(" + type_to_string(op.args[0]) + "))";
        }

        if(op.type == ops::DECLARE)
        {
            std::string prefix = "";

            if(!op.is_mutable)
                prefix = "const ";

            return prefix + type_to_string(op.args[0]) + " " + type_to_string(op.args[1]) + "=" + type_to_string(op.args[2]) + ";";
        }

        if(op.type == ops::DECLARE_ARRAY)
        {
            return type_to_string(op.args[0]) + " " + type_to_string(op.args[1]) + "[" + type_to_string(op.args[2]) + "];";
        }

        if(op.type == ops::COMBO_PLUS)
        {
            auto bracket_pair = [](const std::string& v1, const std::string& v2)
            {
                return "(" + v1 + "+" + v2 + ")";
            };

            std::vector<std::string> pending;

            for(const auto& i : op.args)
            {
                pending.push_back(type_to_string(i));
            }

            std::sort(pending.begin(), pending.end());

            assert(pending.size() >= 2);

            return pairwise_reduce(pending, bracket_pair);
        }

        if(op.type == ops::COMBO_MULTIPLY)
        {
            auto bracket_pair = [](const std::string& v1, const std::string& v2)
            {
                return "(" + v1 + "*" + v2 + ")";
            };

            std::vector<std::string> pending;

            for(const auto& i : op.args)
            {
                pending.push_back(type_to_string(i));
            }

            std::sort(pending.begin(), pending.end());

            assert(pending.size() >= 2);

            return pairwise_reduce(pending, bracket_pair);
        }

        if(op.type == ops::SIDE_EFFECT)
        {
            return type_to_string(op.args[0]) + ";";
        }

        const operation_desc desc = get_description(op.type);

        if(desc.is_infix)
        {
            assert(op.args.size() == 2);

            std::vector<std::string> expanded;
            expanded.reserve(2);

            for(const auto& i : op.args)
            {
                expanded.push_back(type_to_string(i));
            }

            #ifdef SORT_ASSOCIATIVE
            if(op.type == ops::PLUS || op.type == ops::MULTIPLY)
                std::sort(expanded.begin(), expanded.end());
            #endif

            return "(" + expanded[0] + std::string(desc.sym) + expanded[1] + ")";
        }
        else
        {
           if(op.type == ops::UMINUS)
                return "(-(" + type_to_string(op.args[0]) + "))";

            std::string build;

            if(op.type == ops::UNKNOWN_FUNCTION)
            {
                assert(op.args[0].is_value());

                std::string name = type_to_string(op.args[0]);

                build = name + "(";

                ///starts at one, ignoring the first argument which is the function name
                for(int i=1; i < (int)op.args.size() - 1; i++)
                {
                    build += type_to_string(op.args[i]) + ",";
                }

                if(op.args.size() > 1)
                    build += type_to_string(op.args.back());

                build += ")";
            }
            else
            {
                build = std::string(desc.sym) + "(";

                for(int i=0; i < (int)op.args.size() - 1; i++)
                {
                    build += type_to_string(op.args[i]) + ",";
                }

                if(op.args.size() > 0)
                    build += type_to_string(op.args.back());

                build += ")";
            }

            return build;
        }
    }

    template<typename T, typename U>
    inline
    value<T> create_as(value<U>&& in)
    {
        return in.template reinterpret_as<value<T>>();
    }

    template<typename T, typename U>
    inline
    value<T> create_as(const value<U>& in)
    {
        return in.template reinterpret_as<value<T>>();
    }

    template<typename T, typename U>
    requires (std::is_arithmetic_v<U> || std::same_as<U, std::string> || std::same_as<std::decay_t<U>, const char*>)
    inline
    value<T> create_as(U&& in)
    {
        return value<T>(in);
    }

    template<typename T, typename U>
    requires (std::is_arithmetic_v<U> || std::same_as<U, std::string> || std::same_as<std::decay_t<U>, const char*>)
    inline
    value<T> create_as(const U& in)
    {
        return value<T>(in);
    }

    template<typename U, typename... T>
    inline
    value<U> make_op(ops::type_t type, T&&... args)
    {
        value<U> val;
        val.type = type;
        val.args = {create_as<U>(std::forward<T>(args))...};

        if constexpr(std::is_same_v<U, std::monostate>)
            return val;
        else
        {
            return val.flatten().group_associative_operators();
        }
    }

    template<typename T>
    inline
    value<T> make_op_vec(ops::type_t type, const std::vector<value<T>>& args)
    {
        value<T> val;
        val.type = type;
        val.args = args;

        if constexpr(std::is_same_v<T, std::monostate>)
            return val;
        else
        {
            return val.flatten().group_associative_operators();
        }
    }

    inline
    value<std::monostate> block_start()
    {
        return make_op<std::monostate>(ops::BLOCK_START);
    }

    inline
    value<std::monostate> block_end()
    {
        return make_op<std::monostate>(ops::BLOCK_END);
    }

    #define UNARY(x, y) template<typename T> inline value<T> x(const value<T>& d1){return make_op<T>(ops::y, d1);}
    #define BINARY(x, y) template<typename T, typename U> inline value<T> x(const value<T>& d1, const U& d2){return make_op<T>(ops::y, d1, d2);}
    #define TRINARY(x, y) template<typename T, typename U, typename V> inline value<T> x(const value<T>& d1, const U& d2, const V& d3){return make_op<T>(ops::y, d1, d2, d3);}

    UNARY(floor, FLOOR);
    UNARY(ceil, CEIL);
    UNARY(round, ROUND);
    UNARY(sqrt, SQRT);
    UNARY(psqrt, SQRT);
    UNARY(log, LOG);
    UNARY(fabs, FABS);
    UNARY(abs, ABS);
    UNARY(exp, EXP);
    UNARY(sin, SIN);
    UNARY(cos, COS);
    UNARY(tan, TAN);
    UNARY(sinh, SINH);
    UNARY(cosh, COSH);
    UNARY(tanh, TANH);
    UNARY(asinh, ASINH);
    UNARY(acosh, ACOSH);
    UNARY(atanh, ATANH);
    UNARY(asin, ASIN);
    UNARY(acos, ACOS);
    UNARY(atan, ATAN);
    BINARY(atan2, ATAN2);
    UNARY(isfinite, ISFINITE);
    UNARY(signbit, SIGNBIT);
    UNARY(sign, SIGN);
    //TRINARY(select, SELECT);
    BINARY(pow, POW);
    BINARY(max, MAX);
    BINARY(min, MIN);
    UNARY(lambert_w0, LAMBERT_W0);
    BINARY(smooth_fmod, SMOOTH_FMOD);

    template<typename T>
    inline
    value<T> mad(const value<T>& v1, const value<T>& v2, const value<T>& v3)
    {
        return make_op<T>(ops::MAD, v1, v2, v3);
    }

    template<typename T>
    inline
    value<T> fma(const value<T>& v1, const value<T>& v2, const value<T>& v3)
    {
        return make_op<T>(ops::FMA, v1, v2, v3);
    }

    template<typename T, typename V>
    inline
    value<T> select(const value<T>& v1, const value<T>& v2, const value<V>& v3)
    {
        if(equivalent(v1, v2))
            return v1;

        return make_op<T>(ops::SELECT, v1, v2, v3.template reinterpret_as<value<T>>());
    }

    template<typename T>
    inline
    value<T> clamp(const value<T>& val, const value<T>& lower, const value<T>& upper)
    {
        return make_op<T>(ops::CLAMP, val, lower, upper);
    }

    ///https://man.opencl.org/mix.html, use the exact spec
    template<typename T>
    inline
    value<T> mix(const value<T>& v1, const value<T>& v2, const value<T>& a)
    {
        return v1 + (v2 - v1) * a;
    }

    ///select
    template<typename T, typename U>
    inline
    value<T> if_v(const value<U>& condition, const value<T>& if_true, const value<T>& if_false)
    {
        if(condition.is_constant())
        {
            U val = condition.get_constant();

            if(val == 0)
                return if_false;
            else
                return if_true;
        }

        return select<T, U>(if_false, if_true, condition);
    }

    namespace cf
    {
        [[nodiscard]]
        inline
        value<std::monostate> break_s()
        {
            return make_op<std::monostate>(ops::BREAK);
        }

        template<typename T>
        [[nodiscard]]
        inline
        value<std::monostate> return_v(const value<T>& in)
        {
            return make_op<std::monostate>(ops::RETURN, in);
        }

        [[nodiscard]]
        inline
        value<std::monostate> return_v()
        {
            return make_op<std::monostate>(ops::RETURN);
        }
    }

    ///true branch
    ///if with to-execute on true. This should be removed
    /*template<typename T, typename U>
    inline
    value<std::monostate> if_s(const value<T>& condition, const value<U>& to_execute)
    {
        return make_op<std::monostate>(ops::IF_S, condition.as_generic(), to_execute.as_generic());
    }*/

    ///if block start
    template<typename T>
    inline
    value<std::monostate> if_b(const value<T>& condition)
    {
        return make_op<std::monostate>(ops::IF_START, condition.as_generic());
    }

    ///if + execute. This is more the direction I would like to go
    ///Long term: Do I want a local executor stack?
    ///I've never once needed them to do anything fancy at all
    template<typename Ctx, typename T, typename Func>
    inline
    void if_e(const value<T>& condition, Ctx& ctx, Func&& func)
    {
        ctx.exec(if_b(condition));

        ctx.exec(block_start());

        func();

        ctx.exec(block_end());
    }

    template<typename T>
    inline
    value<std::monostate> for_b(const std::string& loop_variable_name, const value<T>& init, const value<T>& condition, const value<T>& post)
    {
        return make_op<std::monostate>(ops::FOR_START, name_type(T()), loop_variable_name, init.as_generic(), condition.as_generic(), post.as_generic());
    }

    template<typename Ctx, typename T, typename U, typename Func>
    inline
    void for_e(Ctx& ctx, const U& loop_variable_name, const value<T>& init, const value<T>& condition, const value<T>& post, Func&& func)
    {
        //int id = ctx.get_id();

        ctx.exec(for_b(type_to_string(loop_variable_name), init, condition, post));

        value<T> val = loop_variable_name;

        ctx.exec(block_start());

        func(val);

        ctx.exec(block_end());
    }

    namespace implicit
    {
        struct context_base
        {
            virtual void exec(const value<std::monostate>& st)
            {
                assert(false);
            }

            virtual int get_id()
            {
                assert(false);
            }
        };

        namespace detail
        {
            #define MANAGE_CONTEXT(ctx) dual_types::implicit::detail::context_manager hello(&ctx)

            inline
            std::vector<std::variant<std::shared_ptr<context_base>, context_base*>>& get_contexts()
            {
                thread_local std::vector<std::variant<std::shared_ptr<context_base>, context_base*>> contexts;

                return contexts;
            }

            inline
            void push_context(context_base* base)
            {
                get_contexts().push_back(base);
            }

            inline
            void push_context(std::shared_ptr<context_base> base)
            {
                get_contexts().push_back(base);
            }

            inline
            void pop_context()
            {
                assert(get_contexts().size() > 0);
                get_contexts().pop_back();
            }

            template<typename T>
            struct context_manager
            {
                context_manager(const context_manager&) = delete;
                context_manager& operator=(const context_manager&) = delete;
                context_manager(context_manager&&) = delete;
                context_manager& operator=(context_manager&&) = delete;

                context_manager(T base)
                {
                    push_context(base);
                }

                ~context_manager()
                {
                    pop_context();
                }
            };

            ///pushes it to the global stack
            template<typename T>
            inline
            context_manager<std::shared_ptr<T>> make_context()
            {
                std::shared_ptr<T> ctx = std::make_shared<T>();

                return context_manager<std::shared_ptr<T>>(ctx);
            }

            template<typename T>
            inline
            T* get_context()
            {
                T* out = nullptr;

                std::visit([&](auto&& arg)
                {
                    using U = std::decay_t<decltype(arg)>;

                    if constexpr(std::is_same_v<U, std::shared_ptr<context_base>>)
                    {
                        out = dynamic_cast<T*>(arg.get());
                    }

                    else if constexpr(std::is_same_v<U, context_base*>)
                    {
                        out = dynamic_cast<T*>(arg);
                    }
                    else
                    {
                        #ifndef __clang__
                        static_assert(false);
                        #endif
                    }
                }, get_contexts().back());

                return out;
            }

            inline
            context_base* get_context_base()
            {
                return get_context<context_base>();
            }
        }

        template<typename T, typename U, typename Func>
        inline
        void for_e(const U& loop_variable_name, const value<T>& init, const value<T>& condition, const value<T>& post, Func&& func)
        {
            auto ctx_ptr = detail::get_context_base();

            for_e(*ctx_ptr, loop_variable_name, init, condition, post, std::forward<Func>(func));
        }

        template<typename T, typename Func>
        inline
        void if_e(const value<T>& condition, Func&& func)
        {
            auto ctx_ptr = detail::get_context_base();

            if_e(condition, *ctx_ptr, std::forward<Func>(func));
        }

        inline
        void break_e()
        {
            auto ctx_ptr = detail::get_context_base();

            ctx_ptr->exec(dual_types::cf::break_s());
        }

        template<typename T>
        inline
        void return_e(const value<T>& in)
        {
            auto ctx_ptr = detail::get_context_base();

            ctx_ptr->exec(dual_types::cf::return_v(in));
        }

        inline
        void return_e()
        {
            auto ctx_ptr = detail::get_context_base();

            ctx_ptr->exec(dual_types::cf::return_v());
        }

        template<typename T, typename U>
        inline
        void assign_e(T&& t, U&& u)
        {
            auto ctx_ptr = detail::get_context_base();

            ctx_ptr->exec(assign(std::forward<T>(t), std::forward<U>(u)).template reinterpret_as<value<std::monostate>>());
        }

        template<typename T>
        inline
        auto mut(T&& in)
        {
            return in.as_mutable(*detail::get_context_base());
        }

        template<typename T>
        inline
        auto as_mutable(T&& in)
        {
            return in.as_mutable(*detail::get_context_base());
        }

        template<typename T>
        inline
        value<T> declare(const value<T>& v1, const std::string& name = "")
        {
            return declare(*detail::get_context_base(), v1, name);
        }

        template<typename T, int N>
        inline
        tensor<value<T>, N> declare(const tensor<value<T>, N>& v1)
        {
            return declare(*detail::get_context_base(), v1);
        }

        template<typename T>
        inline
        value_mut<T> declare_mut(const value<T>& v1, const std::string& name = "")
        {
            return declare_mut(*detail::get_context_base(), v1, name);
        }

        template<typename T, int N>
        inline
        tensor<value_mut<T>, N> declare_mut(const tensor<value<T>, N>& v1)
        {
            return declare_mut(*detail::get_context_base(), v1);
        }
    }

    ///select
    template<typename T>
    requires std::is_arithmetic_v<T>
    auto if_v(bool condition, const T& if_true, const T& if_false)
    {
        return condition ? if_true : if_false;
    }

    template<typename T, typename U>
    inline
    T divide_with_limit(const T& top, const T& bottom, const U& limit, float tol = 0.001f)
    {
        if constexpr(std::is_arithmetic_v<T>)
            return dual_types::if_v(std::fabs(bottom) >= tol, top / bottom, limit);
        else
            return dual_types::if_v(fabs(bottom) >= tol, top / bottom, T{limit});
    }

    template<typename U, typename... T>
    inline
    value<U> apply(const value<U>& name, T&&... args)
    {
        return make_op<U>(ops::UNKNOWN_FUNCTION, name, std::forward<T>(args)...);
    }

    template<typename... T>
    inline
    value<std::monostate> print(const std::string& fmt, T&&... args)
    {
        std::vector<std::string> sargs;

        (sargs.push_back(type_to_string(args)), ...);

        std::string root = "printf(\"" + fmt + "\"";

        for(int i=0; i < (int)sargs.size(); i++)
        {
            root += "," + sargs[i];
        }

        return root + ")";
    }

    template<typename T>
    inline
    std::pair<value<std::monostate>, value<T>> declare_raw(const value<T>& v1, const std::string& name, bool is_mutable)
    {
        value declare_op = make_op<T>(ops::DECLARE, v1.original_type, name, v1);

        declare_op.is_mutable = is_mutable;

        value<T> result = name;
        result.is_mutable = declare_op.is_mutable;

        return {declare_op.as_generic(), result};
    }

    template<typename T, int... N>
    struct stack_array
    {
        T prototype;

        int expanded_size = (N * ...);
        std::string name;

        value_mut<typename T::value_type> operator[](const value<int>& in)
        {
            T v = prototype.bracket(in);
            v.is_mutable = true;
            return to_mutable(v);
        }

        value_mut<typename T::value_type> operator[](const value<int>& v1, const value<int>& v2)
        {
            tensor<int, N...> dims = {N...};

            T v = prototype.bracket(v1 + v2 * dims[0]);
            v.is_mutable = true;
            return to_mutable(v);
        }

        value_mut<typename T::value_type> operator[](const value<int>& v1, const value<int>& v2, const value<int>& v3)
        {
            tensor<int, N...> dims = {N...};

            T v = prototype.bracket(v1 + v2 * dims[0] + v3 * dims[0] * dims[1]);
            v.is_mutable = true;
            return to_mutable(v);
        }
    };

    template<typename T, int... N>
    inline
    std::pair<value<std::monostate>, stack_array<T, N...>> declare_array_raw(const std::string& name)
    {
        int length = (N * ...);

        value declare_op = make_op<typename T::value_type>(ops::DECLARE_ARRAY, name_type(typename T::value_type()), name, value<int>{length}.reinterpret_as<value<typename T::value_type>>());
        declare_op.is_mutable = true;

        stack_array<T, N...> result;
        result.prototype = name;
        result.prototype.is_mutable = declare_op.is_mutable;

        return {declare_op.as_generic(), result};
    }

    template<typename T, int... N>
    inline
    stack_array<T, N...> declare_array(auto& executor, const std::string& name = "")
    {
        if(name == "")
        {
            int id = executor.get_id();

            std::string fname = "declared" + std::to_string(id);
            return declare_array<T, N...>(executor, fname);
        }

        auto [declare_op, result] = declare_array_raw<T, N...>(name);

        executor.exec(declare_op);

        return result;
    }

    template<typename U, typename T>
    inline
    value<T> declare_impl(U& executor, const value<T>& v1, const std::string& name, bool is_mutable)
    {
        if(name == "")
        {
            int id = executor.get_id();

            std::string fname = "declared" + std::to_string(id);
            return declare_impl(executor, v1, fname, is_mutable);
        }

        value declare_op = make_op<T>(ops::DECLARE, v1.original_type, name, v1);

        declare_op.is_mutable = is_mutable;

        executor.exec(declare_op.as_generic());

        value<T> result = name;
        result.is_mutable = declare_op.is_mutable;

        return result;
    }

    template<typename U, typename T, int N>
    inline
    tensor<value<T>, N> declare_impl(U& executor, const tensor<value<T>, N>& v1, bool is_mutable)
    {
        tensor<value<T>, N> ret;

        for(int i=0; i < N; i++)
        {
            ret[i] = declare_impl(executor, v1[i], "", is_mutable);
        }

        return ret;
    }

    template<typename T, typename U>
    inline
    value<T> declare(U& executor, const value<T>& v1, const std::string& name = "")
    {
        return declare_impl(executor, v1, name, false);
    }

    template<typename T, typename U, int N>
    inline
    tensor<value<T>, N> declare(U& executor, const tensor<value<T>, N>& v1)
    {
        return declare_impl(executor, v1, false);
    }

    template<typename T, typename U>
    inline
    value_mut<T> declare_mut(U& executor, const value<T>& v1, const std::string& name = "")
    {
        return to_mutable(declare_impl(executor, v1, name, true));
    }

    template<typename T, typename U, int N>
    inline
    tensor<value_mut<T>, N> declare_mut(U& executor, const tensor<value<T>, N>& v1)
    {
        return to_mutable(declare_impl(executor, v1, true));
    }

    template<typename T>
    void side_effect(T& executor, const std::string& effect)
    {
        executor.exec(make_op<std::monostate>(ops::SIDE_EFFECT, effect));
    }

    /*template<typename T>
    auto assert_s(const value<T>& is_true)
    {
        value<std::monostate> print = "printf(\"Failed: %s\",\"" + type_to_string(is_true) + "\")";

        return if_s(!is_true, print);
    }*/

    template<typename T>
    inline
    value<T> conjugate(const value<T>& d1)
    {
        return d1;
    }

    template<typename T>
    inline
    dual_types::complex_v<value<T>> psqrt(const dual_types::complex_v<value<T>>& d1)
    {
        if(Imaginary(d1).is_constant() && Imaginary(d1).get_constant() == 0)
        {
            return sqrt(Real(d1));
        }

        return sqrt(d1);
    }

    template<typename T>
    inline
    value_mut<T> assign(const value_mut<T>& location, const value<T>& what)
    {
        value_mut<T> ret;

        ret.set_from_constant(make_op<T>(ops::ASSIGN, location.as_constant(), what));

        return ret;
    }

    template<typename T, typename U>
    requires std::is_arithmetic_v<U>
    inline
    value_mut<T> assign(const value_mut<T>& location, const U& what)
    {
        value_mut<T> ret;

        ret.set_from_constant(make_op<T>(ops::ASSIGN, location.as_constant(), what));

        return ret;
    }

    template<typename T, typename U, int N>
    inline
    tensor<value_mut<T>, N> assign(const tensor<value_mut<T>, N>& location, const tensor<U, N>& what)
    {
        tensor<value_mut<T>, N> ret;

        for(int i=0; i < N; i++)
        {
            ret[i] = assign(location[i], what[i]);
        }

        return ret;
    }

    template<typename T>
    struct block
    {
        T& context;

        block(T& in) : context(in)
        {
            in.exec(block_start());
        }

        ~block()
        {
            context.exec(block_end());
        }
    };

    inline
    void test_operation()
    {
        value<float> v1 = 1;
        value<float> v2 = 2;

        value<float> sum = v1 + v2;

        value<float> root_3 = sqrt(sum);

        dual_v<value<float>> test_dual = 1234;

        dual_v<value<float>> test_operator = test_dual * 1111;

        value<float> v = "v";

        value<float> test_op = 2 * (v/2);

        assert(type_to_string(test_op) == "v");

        value<float> test_op2 = (v * 2)/2;

        assert(type_to_string(test_op2) == "v");

        value<float> test_op3 = (2 * ((2 * v/2) / 2) * 2) / 2;

        assert(type_to_string(test_op3) == "v");

        value<float> test_op4 = (2 * v) / v;

        assert(type_to_string(test_op4) == "2.0");

        value<float> test_op5 = (2 * sin(v)) / sin(v);

        assert(type_to_string(test_op5) == "2.0");

        //std::cout << type_to_string(root_3) << std::endl;
    }
}

namespace tensor_impl
{
    template<typename T, int... N>
    inline
    auto as_constant(const tensor<T, N...>& in)
    {
        return tensor_for_each_unary(in, [](const T& v)
        {
            return as_constant(v);
        });
    }

    template<typename T, int... N>
    inline
    auto to_mutable(const tensor<T, N...>& in)
    {
        return tensor_for_each_unary(in, [](const T& v)
        {
            return to_mutable(v);
        });
    }
}

template<typename T, typename U>
inline
T divide_with_callback(const T& top, const T& bottom, U&& if_nonfinite)
{
    using namespace std;

    T result = top / bottom;

    auto is_finite = isfinite(result);

    if constexpr(std::is_same_v<T, float>)
    {
        static_assert(std::is_same_v<decltype(is_finite), bool>);
    }

    return dual_types::if_v(is_finite, result, if_nonfinite(top, bottom));
}

using dual = dual_types::dual_v<dual_types::value<float>>;
using dual_complex = dual_types::dual_v<dual_types::complex_v<dual_types::value<float>>>;

using namespace dual_types::cf;

template<typename T>
using value_base = dual_types::value<T>;
using value = dual_types::value<float>;
using value_i = dual_types::value<int>;
using value_s = dual_types::value<short>;
using value_us = dual_types::value<unsigned short>;
using value_v = dual_types::value<std::monostate>;
using value_h = dual_types::value<float16>;

template<typename T>
using value_base_mut = dual_types::value_mut<T>;
using value_mut = dual_types::value_mut<float>;
using value_i_mut = dual_types::value_mut<int>;
using value_s_mut = dual_types::value_mut<short>;
using value_us_mut = dual_types::value_mut<unsigned short>;
using value_v_mut = dual_types::value_mut<std::monostate>;
using value_h_mut = dual_types::value_mut<float16>;

using v4f = tensor<value, 4>;
using v4i = tensor<value_i, 4>;
using v3f = tensor<value, 3>;
using v3i = tensor<value_i, 3>;
using v2f = tensor<value, 2>;
using v2i = tensor<value_i, 2>;
using v1f = tensor<value, 1>;
using v1i = tensor<value_i, 1>;

using v4f_mut = tensor<value_mut, 4>;
using v4i_mut = tensor<value_i_mut, 4>;
using v3f_mut = tensor<value_mut, 3>;
using v3i_mut = tensor<value_i_mut, 3>;
using v2f_mut = tensor<value_mut, 2>;
using v2i_mut = tensor<value_i_mut, 2>;
using v1f_mut = tensor<value_mut, 1>;
using v1i_mut = tensor<value_i_mut, 1>;

template<typename T, int... N>
using stack_array = dual_types::stack_array<T, N...>;

#endif // DUAL2_HPP_INCLUDED
