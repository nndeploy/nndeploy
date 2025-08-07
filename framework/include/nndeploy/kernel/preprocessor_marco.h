/*
ref： oneflow
*/
#ifndef _NNDEPLOY_KERNEL_PREPROCESSOR_MARCO_H_
#define _NNDEPLOY_KERNEL_PREPROCESSOR_MARCO_H_


#define NNDEPLOY_PP_TUPLE2VARADIC(t) NNDEPLOY_PP_TUPLE2VARADIC_I(t)

#define NNDEPLOY_PP_TUPLE2VARADIC_I(t) NNDEPLOY_PP_TUPLE2VARADIC_II t

#define NNDEPLOY_PP_TUPLE2VARADIC_II(...) __VA_ARGS__

#define NNDEPLOY_PP_INTERNAL_STRINGIZE(...) \
  NNDEPLOY_PP_INTERNAL_STRINGIZE_I(__VA_ARGS__)
#define NNDEPLOY_PP_INTERNAL_STRINGIZE_I(...) #__VA_ARGS__

#define NNDEPLOY_PP_INTERNAL_CAT(a, b) NNDEPLOY_PP_INTERNAL_CAT_I(a, b)
#define NNDEPLOY_PP_INTERNAL_CAT_I(a, b) a##b

#define NNDEPLOY_PP_INTERNAL_JOIN(glue, ...)                       \
  NNDEPLOY_PP_INTERNAL_CAT(                                        \
      NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_JOIN_,         \
                               NNDEPLOY_PP_INTERNAL_VARIADIC_SIZE( \
                                   __VA_ARGS__))(glue, __VA_ARGS__), )

#define NNDEPLOY_PP_INTERNAL_JOIN_0(glue)
#define NNDEPLOY_PP_INTERNAL_JOIN_1(glue, x) x
#define NNDEPLOY_PP_INTERNAL_JOIN_2(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_1(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_3(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_2(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_4(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_3(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_5(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_4(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_6(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_5(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_7(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_6(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_8(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_7(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_9(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                       \
      NNDEPLOY_PP_INTERNAL_CAT(                   \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),      \
          NNDEPLOY_PP_INTERNAL_JOIN_8(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_10(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                        \
      NNDEPLOY_PP_INTERNAL_CAT(                    \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),       \
          NNDEPLOY_PP_INTERNAL_JOIN_9(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_11(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                        \
      NNDEPLOY_PP_INTERNAL_CAT(                    \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),       \
          NNDEPLOY_PP_INTERNAL_JOIN_10(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_12(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                        \
      NNDEPLOY_PP_INTERNAL_CAT(                    \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),       \
          NNDEPLOY_PP_INTERNAL_JOIN_11(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_13(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                        \
      NNDEPLOY_PP_INTERNAL_CAT(                    \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),       \
          NNDEPLOY_PP_INTERNAL_JOIN_12(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_14(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                        \
      NNDEPLOY_PP_INTERNAL_CAT(                    \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),       \
          NNDEPLOY_PP_INTERNAL_JOIN_13(glue, __VA_ARGS__)), )
#define NNDEPLOY_PP_INTERNAL_JOIN_15(glue, x, ...) \
  NNDEPLOY_PP_INTERNAL_CAT(                        \
      NNDEPLOY_PP_INTERNAL_CAT(                    \
          NNDEPLOY_PP_INTERNAL_CAT(x, glue),       \
          NNDEPLOY_PP_INTERNAL_JOIN_14(glue, __VA_ARGS__)), )

#define NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq) \
  NNDEPLOY_PP_INTERNAL_PAIR_FIRST(NNDEPLOY_PP_INTERNAL_SEQ_TO_PAIR(seq))
#define NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq) \
  NNDEPLOY_PP_INTERNAL_PAIR_SECOND(NNDEPLOY_PP_INTERNAL_SEQ_TO_PAIR(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_TO_PAIR(seq) \
  (NNDEPLOY_PP_INTERNAL_SEQ_TO_PAIR_ seq)
#define NNDEPLOY_PP_INTERNAL_SEQ_TO_PAIR_(x) x, NNDEPLOY_PP_INTERNAL_NIL
#define NNDEPLOY_PP_INTERNAL_NIL

#define NNDEPLOY_PP_INTERNAL_PAIR_FIRST(t) NNDEPLOY_PP_INTERNAL_PAIR_FIRST_I(t)
#define NNDEPLOY_PP_INTERNAL_PAIR_FIRST_I(t) NNDEPLOY_PP_INTERNAL_FIRST_ARG t
#define NNDEPLOY_PP_INTERNAL_PAIR_SECOND(t) \
  NNDEPLOY_PP_INTERNAL_PAIR_SECOND_I(t)
#define NNDEPLOY_PP_INTERNAL_PAIR_SECOND_I(t) NNDEPLOY_PP_INTERNAL_SECOND_ARG t
#define NNDEPLOY_PP_INTERNAL_PAIR_THIRD(t) NNDEPLOY_PP_INTERNAL_PAIR_THIRD_I(t)
#define NNDEPLOY_PP_INTERNAL_PAIR_THIRD_I(t) NNDEPLOY_PP_INTERNAL_THIRD_ARG t

#define NNDEPLOY_PP_INTERNAL_FIRST_ARG(x, ...) x
#define NNDEPLOY_PP_INTERNAL_SECOND_ARG(x, y, ...) y
#define NNDEPLOY_PP_INTERNAL_THIRD_ARG(x, y, z, ...) z

#define NNDEPLOY_PP_INTERNAL_MAKE_TUPLE(...) (__VA_ARGS__)
#define NNDEPLOY_PP_INTERNAL_MAKE_TUPLE_SEQ(...) \
  (NNDEPLOY_PP_INTERNAL_MAKE_TUPLE(__VA_ARGS__))

// Tuple

#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT(tuple, x)            \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_, \
                           NNDEPLOY_PP_INTERNAL_TUPLE_SIZE(tuple)) \
  (tuple, x)

#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_0(tuple, x) (x)
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_1(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_2(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_3(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(2, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_4(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(2, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(3, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_5(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(2, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(3, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(4, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_6(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(2, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(3, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(4, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(5, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_7(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(2, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(3, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(4, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(5, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(6, tuple))
#define NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT_8(tuple, x) \
  (x, NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(0, tuple),          \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(1, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(2, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(3, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(4, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(5, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(6, tuple),             \
   NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(7, tuple))

#define NNDEPLOY_PP_INTERNAL_TUPLE_ELEM(n, t) \
  NNDEPLOY_PP_INTERNAL_TUPLE_ELEM_I(n, t)
#define NNDEPLOY_PP_INTERNAL_TUPLE_ELEM_I(n, t) \
  NNDEPLOY_PP_INTERNAL_CAT(                     \
      NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_ARG_, n) t, )

#define NNDEPLOY_PP_INTERNAL_ARG_0(a0, ...) a0
#define NNDEPLOY_PP_INTERNAL_ARG_1(a0, a1, ...) a1
#define NNDEPLOY_PP_INTERNAL_ARG_2(a0, a1, a2, ...) a2
#define NNDEPLOY_PP_INTERNAL_ARG_3(a0, a1, a2, a3, ...) a3
#define NNDEPLOY_PP_INTERNAL_ARG_4(a0, a1, a2, a3, a4, ...) a4
#define NNDEPLOY_PP_INTERNAL_ARG_5(a0, a1, a2, a3, a4, a5, ...) a5
#define NNDEPLOY_PP_INTERNAL_ARG_6(a0, a1, a2, a3, a4, a5, a6, ...) a6
#define NNDEPLOY_PP_INTERNAL_ARG_7(a0, a1, a2, a3, a4, a5, a6, a7, ...) a7

#define NNDEPLOY_PP_INTERNAL_TUPLE_SIZE(tuple)                         \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_TUPLE_SIZE_,           \
                           NNDEPLOY_PP_INTERNAL_IS_TUPLE_EMPTY(tuple)) \
  (tuple)

#define NNDEPLOY_PP_INTERNAL_TUPLE_SIZE_1(t) 0
#define NNDEPLOY_PP_INTERNAL_TUPLE_SIZE_0(t) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SIZE_0_I(t)
#define NNDEPLOY_PP_INTERNAL_TUPLE_SIZE_0_I(t) \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_VARIADIC_SIZE t, )

#define NNDEPLOY_PP_INTERNAL_VARIADIC_SIZE(...)                                \
  NNDEPLOY_PP_INTERNAL_CAT(                                                    \
      NNDEPLOY_PP_INTERNAL_VARIADIC_SIZE_I(                                    \
          __VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, \
          50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34,  \
          33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  \
          16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, ), )
#define NNDEPLOY_PP_INTERNAL_VARIADIC_SIZE_I(                                  \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, size, ...)                                                       \
  size

#define NNDEPLOY_PP_INTERNAL_IS_TUPLE_EMPTY(t) \
  NNDEPLOY_PP_INTERNAL_IS_TUPLE_EMPTY_I(t)
#define NNDEPLOY_PP_INTERNAL_IS_TUPLE_EMPTY_I(t) \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_IS_VARIADIC_EMPTY t, )

#define NNDEPLOY_PP_INTERNAL_IS_VARIADIC_EMPTY(...)                                                          \
  NNDEPLOY_PP_INTERNAL_IS_VARIADIC_EMPTY_(/* test if there is just one                                       \
                              argument, eventually an empty one */                                           \
                                          NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                              __VA_ARGS__), /* test if                                       \
                                                               _NNDEPLOY_PP_INTERNAL_TRIGGER_PARENTHESIS_    \
                                                               together with                                 \
                                                               the argument                                  \
                                                               adds a comma                                  \
                                                             */                                              \
                                          NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                              _NNDEPLOY_PP_INTERNAL_TRIGGER_PARENTHESIS_                     \
                                                  __VA_ARGS__), /* test if the                               \
                                                                   argument                                  \
                                                                   together                                  \
                                                                   with a                                    \
                                                                   parenthesis                               \
                                                                   adds a                                    \
                                                                   comma                                     \
                                                                 */                                          \
                                          NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                              __VA_ARGS__(                                                   \
                                                  /*empty*/)), /* test if                                    \
                                                                  placing it                                 \
                                                                  between                                    \
                                                                  _NNDEPLOY_PP_INTERNAL_TRIGGER_PARENTHESIS_ \
                                                                  and the                                    \
                                                                  parenthesis                                \
                                                                  adds a comma                               \
                                                                */                                           \
                                          NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                              _NNDEPLOY_PP_INTERNAL_TRIGGER_PARENTHESIS_                     \
                                                  __VA_ARGS__(/*empty*/)))

#define NNDEPLOY_PP_INTERNAL_IS_VARIADIC_EMPTY_(e0, e1, e2, e3)      \
  NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA(NNDEPLOY_PP_INTERNAL_CAT5( \
      NNDEPLOY_PP_INTERNAL_IS_EMPTY_CASE_, e0, e1, e2, e3))

#define NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA(...)                           \
  NNDEPLOY_PP_INTERNAL_CAT(                                                    \
      NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA_I(                               \
          __VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   \
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    \
          0), )
#define NNDEPLOY_PP_INTERNAL_VARIADIC_HAS_COMMA_I(                             \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, has_comma, ...)                                                  \
  has_comma

#define _NNDEPLOY_PP_INTERNAL_TRIGGER_PARENTHESIS_(...) ,

#define NNDEPLOY_PP_INTERNAL_CAT5(e0, e1, e2, e3, e4) e0##e1##e2##e3##e4
#define NNDEPLOY_PP_INTERNAL_IS_EMPTY_CASE_0001 ,

// Seq Product

#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_FOR_EACH_TUPLE(macro, seq0, ...) \
  NNDEPLOY_PP_INTERNAL_SEQ_FOR_EACH_TUPLE(                                \
      macro, _, NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT(seq0, __VA_ARGS__))

#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT(seq0, ...)                \
  NNDEPLOY_PP_INTERNAL_CAT(                                        \
      NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_,  \
                               NNDEPLOY_PP_INTERNAL_VARIADIC_SIZE( \
                                   seq0, __VA_ARGS__))(seq0, __VA_ARGS__), )

#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_0()
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_1(seq0) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ((()), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_2(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_1(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_3(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_2(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_4(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_3(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_5(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_4(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_6(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_5(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_7(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_6(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_8(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_7(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_9(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(        \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_8(__VA_ARGS__), seq0)
#define NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_10(seq0, ...) \
  NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(         \
      NNDEPLOY_PP_INTERNAL_SEQ_PRODUCT_9(__VA_ARGS__), seq0)

// Seq ForEach

#define NNDEPLOY_PP_INTERNAL_OUTTER_FOR_EACH_TUPLE(macro, seq) \
  NNDEPLOY_PP_INTERNAL_OUTTER_SEQ_FOR_EACH_TUPLE(macro, _, seq)
#define NNDEPLOY_PP_INTERNAL_FOR_EACH_TUPLE(macro, seq) \
  NNDEPLOY_PP_INTERNAL_SEQ_FOR_EACH_TUPLE(macro, _, seq)
#define NNDEPLOY_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(tuple_seq, atomic_seq) \
  NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH(                                    \
      NNDEPLOY_PP_INTERNAL_D1_APPLY_ATOMIC_WITH_DATA,                      \
      NNDEPLOY_PP_INTERNAL_TUPLE_X_ATOMIC_SEQ, atomic_seq, tuple_seq)

#define NNDEPLOY_PP_INTERNAL_TUPLE_X_ATOMIC_SEQ(atomic_seq, tuple) \
  NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH(                            \
      NNDEPLOY_PP_INTERNAL_D2_APPLY_ATOMIC_WITH_DATA,              \
      NNDEPLOY_PP_INTERNAL_MAKE_SEQ_TUPLE_PUSH_FRONT, tuple, atomic_seq)

#define NNDEPLOY_PP_INTERNAL_D1_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)
#define NNDEPLOY_PP_INTERNAL_D2_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define NNDEPLOY_PP_INTERNAL_MAKE_SEQ_TUPLE_PUSH_FRONT(tuple, x) \
  (NNDEPLOY_PP_INTERNAL_TUPLE_PUSH_FRONT(tuple, x))

// Seq Size

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE(seq) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_I(seq)
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_I(seq)               \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_SEQ_SIZE_, \
                           NNDEPLOY_PP_INTERNAL_SEQ_SIZE_0 seq)

#define NNDEPLOY_PP_INTERNAL_OUTTER_SEQ_FOR_EACH_TUPLE \
  NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_TUPLE
#define NNDEPLOY_PP_INTERNAL_SEQ_FOR_EACH_TUPLE \
  NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_TUPLE

#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_TUPLE(m, d, seq)                  \
  NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH(NNDEPLOY_PP_INTERNAL_D0_APPLY_TUPLE, m, \
                                       d, seq)
#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_TUPLE(m, d, seq)                  \
  NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH(NNDEPLOY_PP_INTERNAL_APPLY_TUPLE, m, d, \
                                       seq)
#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_TUPLE(m, d, seq)                  \
  NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH(NNDEPLOY_PP_INTERNAL_APPLY_TUPLE, m, d, \
                                       seq)

#define NNDEPLOY_PP_INTERNAL_SEQ_FOR_EACH_ATOMIC \
  NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_ATOMIC

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_ATOMIC(m, d, seq)               \
  NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH(NNDEPLOY_PP_INTERNAL_APPLY_ATOMIC, m, \
                                       d, seq)
#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_ATOMIC(m, d, seq)               \
  NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH(NNDEPLOY_PP_INTERNAL_APPLY_ATOMIC, m, \
                                       d, seq)

#define NNDEPLOY_PP_INTERNAL_D0_APPLY_TUPLE(m, d, t) \
  NNDEPLOY_PP_INTERNAL_D0_APPLY_TUPLE_I(m, d, t)
#define NNDEPLOY_PP_INTERNAL_D0_APPLY_TUPLE_I(m, d, t) m t

#define NNDEPLOY_PP_INTERNAL_APPLY_TUPLE(m, d, t) \
  NNDEPLOY_PP_INTERNAL_APPLY_TUPLE_I(m, d, t)
#define NNDEPLOY_PP_INTERNAL_APPLY_TUPLE_I(m, d, t) m t
#define NNDEPLOY_PP_INTERNAL_APPLY_ATOMIC(m, d, x) m(x)
#define NNDEPLOY_PP_INTERNAL_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH(apply, m, d, seq)    \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_, \
                           NNDEPLOY_PP_INTERNAL_SEQ_SIZE(seq))    \
  (apply, m, d, seq)

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH(apply, m, d, seq)    \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_, \
                           NNDEPLOY_PP_INTERNAL_SEQ_SIZE(seq))    \
  (apply, m, d, seq)

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH(apply, m, d, seq)    \
  NNDEPLOY_PP_INTERNAL_CAT(NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_, \
                           NNDEPLOY_PP_INTERNAL_SEQ_SIZE(seq))    \
  (apply, m, d, seq)

#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_0(apply, m, d, seq)
#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_0(apply, m, d, seq)
#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_0(apply, m, d, seq)

// php code to generate iterator macro
// clang-format off
/*
<?php $limit = 512; for ($i = 0; $i < $limit; ++$i) {?> 
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_<?= $i?>(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_<?= $i + 1?> 

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_<?= $i?> <?= $i?>

<?php $dim = 2; for ($d = 0; $d <= $dim; ++$d) {?> 
#define NNDEPLOY_PP_INTERNAL_D<?= $d?>_SEQ_FOR_EACH_<?= $i + 1?>(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D<?= $d?>_SEQ_FOR_EACH_<?= $i?>(apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))
<?php }?> 

<?php }?> 
*/
// clang-format on

//  do not edit iterator macro directly, it's generated by the above php code.
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_0(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_1
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_0 0
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_0(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_0(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_0(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_1(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_2
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_1 1
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_1(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_1(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_1(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_2(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_3
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_2 2
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_2(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_2(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_2(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_3(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_4
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_3 3
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_3(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_3(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_3(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_4(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_5
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_4 4
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_4(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_4(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_4(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_5(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_6
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_5 5
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_5(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_5(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_5(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_6(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_7
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_6 6
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_6(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_6(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_6(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_7(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_8
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_7 7
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_7(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_7(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_7(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_8(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_9
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_8 8
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_8(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_8(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_8(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_9(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_10
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_9 9
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_9(                     \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_9(                     \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_9(                     \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_10(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_11
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_10 10
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_10(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_10(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_10(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_11(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_12
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_11 11
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_11(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_11(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_11(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_12(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_13
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_12 12
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_12(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_12(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_12(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_13(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_14
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_13 13
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_13(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_13(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_13(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_14(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_15
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_14 14
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_14(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_14(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_14(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_15(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_16
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_15 15
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_15(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_15(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_15(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_16(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_17
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_16 16
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_16(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_16(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_16(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_17(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_18
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_17 17
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_17(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_17(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_17(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_18(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_19
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_18 18
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_18(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_18(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_18(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_19(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_20
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_19 19
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_19(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_19(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_19(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_20(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_21
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_20 20
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_20(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_20(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_20(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_21(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_22
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_21 21
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_21(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_21(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_21(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_22(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_23
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_22 22
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_22(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_22(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_22(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_23(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_24
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_23 23
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_23(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_23(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_23(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_24(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_25
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_24 24
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_24(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_24(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_24(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_25(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_26
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_25 25
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_25(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_25(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_25(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_26(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_27
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_26 26
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_26(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_26(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_26(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_27(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_28
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_27 27
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_27(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_27(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_27(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_28(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_29
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_28 28
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_28(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_28(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_28(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_29(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_30
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_29 29
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_29(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_29(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_29(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_30(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_31
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_30 30
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_30(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_30(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_30(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_31(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_32
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_31 31
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_31(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_31(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_31(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_32(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_33
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_32 32
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_32(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_32(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_32(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_33(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_34
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_33 33
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_33(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_33(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_33(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_34(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_35
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_34 34
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_34(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_34(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_34(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_35(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_36
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_35 35
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_35(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_35(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_35(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_36(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_37
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_36 36
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_36(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_36(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_36(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_37(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_38
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_37 37
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_37(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_37(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_37(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_38(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_39
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_38 38
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_38(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_38(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_38(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_39(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_40
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_39 39
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_39(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_39(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_39(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_40(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_41
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_40 40
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_40(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_40(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_40(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_41(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_42
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_41 41
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_41(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_41(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_41(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_42(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_43
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_42 42
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_42(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_42(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_42(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_43(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_44
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_43 43
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_43(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_43(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_43(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_44(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_45
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_44 44
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_44(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_44(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_44(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_45(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_46
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_45 45
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_45(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_45(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_45(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_46(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_47
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_46 46
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_46(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_46(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_46(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_47(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_48
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_47 47
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_47(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_47(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_47(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_48(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_49
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_48 48
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_48(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_48(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_48(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_49(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_50
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_49 49
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_49(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_49(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_49(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_50(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_51
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_50 50
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_50(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_50(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_50(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_51(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_52
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_51 51
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_51(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_51(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_51(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_52(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_53
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_52 52
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_52(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_52(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_52(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_53(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_54
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_53 53
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_53(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_53(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_53(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_54(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_55
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_54 54
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_54(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_54(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_54(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_55(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_56
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_55 55
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_55(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_55(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_55(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_56(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_57
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_56 56
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_56(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_56(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_56(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_57(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_58
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_57 57
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_57(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_57(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_57(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_58(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_59
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_58 58
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_58(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_58(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_58(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_59(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_60
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_59 59
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_59(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_59(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_59(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_60(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_61
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_60 60
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_60(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_60(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_60(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_61(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_62
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_61 61
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_61(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_61(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_61(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_62(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_63
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_62 62
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_62(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_62(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_62(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_63(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_64
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_63 63
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_63(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_63(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_63(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_64(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_65
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_64 64
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_64(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_64(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_64(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_65(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_66
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_65 65
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_65(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_65(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_65(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_66(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_67
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_66 66
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_66(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_66(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_66(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_67(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_68
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_67 67
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_67(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_67(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_67(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_68(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_69
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_68 68
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_68(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_68(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_68(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_69(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_70
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_69 69
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_69(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_69(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_69(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_70(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_71
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_70 70
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_70(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_70(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_70(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_71(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_72
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_71 71
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_71(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_71(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_71(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_72(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_73
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_72 72
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_72(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_72(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_72(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_73(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_74
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_73 73
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_73(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_73(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_73(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_74(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_75
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_74 74
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_74(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_74(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_74(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_75(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_76
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_75 75
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_75(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_75(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_75(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_76(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_77
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_76 76
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_76(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_76(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_76(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_77(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_78
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_77 77
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_77(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_77(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_77(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_78(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_79
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_78 78
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_78(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_78(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_78(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_79(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_80
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_79 79
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_79(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_79(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_79(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_80(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_81
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_80 80
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_80(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_80(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_80(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_81(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_82
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_81 81
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_81(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_81(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_81(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_82(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_83
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_82 82
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_82(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_82(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_82(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_83(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_84
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_83 83
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_83(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_83(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_83(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_84(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_85
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_84 84
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_84(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_84(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_84(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_85(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_86
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_85 85
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_85(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_85(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_85(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_86(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_87
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_86 86
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_86(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_86(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_86(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_87(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_88
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_87 87
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_87(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_87(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_87(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_88(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_89
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_88 88
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_88(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_88(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_88(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_89(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_90
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_89 89
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_89(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_89(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_89(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_90(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_91
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_90 90
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_90(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_90(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_90(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_91(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_92
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_91 91
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_91(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_91(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_91(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_92(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_93
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_92 92
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_92(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_92(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_92(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_93(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_94
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_93 93
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_93(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_93(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_93(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_94(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_95
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_94 94
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_94(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_94(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_94(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_95(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_96
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_95 95
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_95(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_95(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_95(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_96(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_97
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_96 96
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_96(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_96(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_96(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_97(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_98
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_97 97
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_97(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_97(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_97(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_98(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_99
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_98 98
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_98(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_98(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                 \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_98(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_99(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_100
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_99 99
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_99(                     \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_99(                     \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_99(                     \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_100(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_101
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_100 100
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_100(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_100(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_100(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_101(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_102
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_101 101
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_101(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_101(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_101(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_102(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_103
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_102 102
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_102(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_102(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_102(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_103(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_104
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_103 103
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_103(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_103(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_103(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_104(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_105
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_104 104
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_104(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_104(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_104(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_105(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_106
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_105 105
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_105(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_105(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_105(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_106(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_107
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_106 106
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_106(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_106(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_106(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_107(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_108
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_107 107
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_107(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_107(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_107(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_108(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_109
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_108 108
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_108(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_108(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_108(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_109(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_110
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_109 109
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_109(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_109(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_109(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_110(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_111
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_110 110
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_110(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_110(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_110(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_111(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_112
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_111 111
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_111(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_111(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_111(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_112(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_113
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_112 112
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_112(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_112(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_112(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_113(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_114
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_113 113
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_113(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_113(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_113(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_114(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_115
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_114 114
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_114(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_114(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_114(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_115(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_116
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_115 115
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_115(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_115(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_115(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_116(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_117
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_116 116
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_116(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_116(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_116(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_117(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_118
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_117 117
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_117(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_117(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_117(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_118(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_119
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_118 118
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_118(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_118(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_118(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_119(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_120
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_119 119
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_119(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_119(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_119(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_120(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_121
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_120 120
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_120(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_120(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_120(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_121(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_122
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_121 121
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_121(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_121(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_121(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_122(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_123
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_122 122
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_122(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_122(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_122(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_123(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_124
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_123 123
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_123(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_123(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_123(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_124(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_125
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_124 124
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_124(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_124(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_124(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_125(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_126
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_125 125
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_125(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_125(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_125(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_126(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_127
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_126 126
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_126(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_126(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_126(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_127(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_128
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_127 127
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_127(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_127(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_127(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_128(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_129
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_128 128
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_128(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_128(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_128(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_129(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_130
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_129 129
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_129(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_129(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_129(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_130(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_131
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_130 130
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_130(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_130(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_130(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_131(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_132
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_131 131
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_131(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_131(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_131(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_132(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_133
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_132 132
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_132(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_132(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_132(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_133(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_134
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_133 133
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_133(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_133(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_133(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_134(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_135
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_134 134
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_134(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_134(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_134(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_135(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_136
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_135 135
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_135(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_135(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_135(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_136(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_137
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_136 136
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_136(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_136(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_136(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_137(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_138
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_137 137
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_137(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_137(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_137(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_138(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_139
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_138 138
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_138(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_138(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_138(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_139(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_140
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_139 139
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_139(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_139(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_139(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_140(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_141
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_140 140
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_140(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_140(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_140(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_141(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_142
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_141 141
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_141(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_141(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_141(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_142(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_143
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_142 142
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_142(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_142(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_142(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_143(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_144
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_143 143
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_143(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_143(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_143(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_144(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_145
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_144 144
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_144(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_144(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_144(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_145(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_146
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_145 145
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_145(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_145(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_145(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_146(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_147
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_146 146
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_146(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_146(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_146(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_147(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_148
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_147 147
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_147(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_147(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_147(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_148(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_149
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_148 148
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_148(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_148(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_148(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_149(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_150
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_149 149
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_149(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_149(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_149(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_150(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_151
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_150 150
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_150(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_150(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_150(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_151(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_152
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_151 151
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_151(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_151(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_151(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_152(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_153
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_152 152
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_152(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_152(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_152(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_153(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_154
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_153 153
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_153(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_153(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_153(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_154(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_155
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_154 154
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_154(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_154(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_154(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_155(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_156
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_155 155
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_155(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_155(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_155(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_156(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_157
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_156 156
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_156(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_156(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_156(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_157(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_158
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_157 157
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_157(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_157(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_157(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_158(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_159
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_158 158
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_158(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_158(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_158(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_159(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_160
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_159 159
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_159(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_159(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_159(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_160(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_161
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_160 160
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_160(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_160(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_160(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_161(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_162
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_161 161
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_161(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_161(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_161(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_162(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_163
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_162 162
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_162(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_162(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_162(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_163(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_164
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_163 163
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_163(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_163(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_163(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_164(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_165
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_164 164
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_164(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_164(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_164(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_165(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_166
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_165 165
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_165(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_165(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_165(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_166(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_167
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_166 166
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_166(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_166(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_166(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_167(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_168
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_167 167
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_167(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_167(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_167(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_168(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_169
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_168 168
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_168(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_168(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_168(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_169(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_170
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_169 169
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_169(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_169(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_169(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_170(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_171
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_170 170
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_170(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_170(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_170(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_171(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_172
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_171 171
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_171(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_171(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_171(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_172(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_173
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_172 172
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_172(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_172(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_172(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_173(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_174
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_173 173
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_173(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_173(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_173(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_174(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_175
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_174 174
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_174(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_174(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_174(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_175(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_176
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_175 175
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_175(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_175(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_175(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_176(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_177
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_176 176
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_176(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_176(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_176(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_177(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_178
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_177 177
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_177(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_177(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_177(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_178(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_179
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_178 178
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_178(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_178(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_178(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_179(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_180
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_179 179
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_179(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_179(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_179(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_180(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_181
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_180 180
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_180(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_180(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_180(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_181(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_182
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_181 181
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_181(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_181(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_181(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_182(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_183
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_182 182
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_182(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_182(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_182(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_183(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_184
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_183 183
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_183(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_183(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_183(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_184(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_185
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_184 184
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_184(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_184(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_184(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_185(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_186
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_185 185
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_185(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_185(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_185(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_186(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_187
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_186 186
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_186(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_186(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_186(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_187(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_188
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_187 187
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_187(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_187(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_187(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_188(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_189
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_188 188
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_188(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_188(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_188(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_189(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_190
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_189 189
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_189(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_189(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_189(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_190(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_191
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_190 190
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_190(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_190(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_190(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_191(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_192
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_191 191
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_191(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_191(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_191(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_192(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_193
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_192 192
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_192(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_192(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_192(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_193(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_194
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_193 193
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_193(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_193(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_193(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_194(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_195
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_194 194
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_194(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_194(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_194(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_195(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_196
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_195 195
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_195(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_195(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_195(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_196(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_197
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_196 196
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_196(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_196(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_196(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_197(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_198
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_197 197
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_197(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_197(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_197(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_198(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_199
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_198 198
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_198(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_198(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_198(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_199(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_200
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_199 199
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_199(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_199(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_199(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_200(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_201
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_200 200
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_200(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_200(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_200(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_201(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_202
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_201 201
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_201(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_201(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_201(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_202(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_203
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_202 202
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_202(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_202(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_202(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_203(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_204
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_203 203
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_203(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_203(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_203(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_204(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_205
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_204 204
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_204(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_204(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_204(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_205(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_206
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_205 205
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_205(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_205(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_205(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_206(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_207
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_206 206
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_206(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_206(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_206(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_207(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_208
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_207 207
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_207(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_207(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_207(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_208(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_209
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_208 208
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_208(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_208(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_208(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_209(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_210
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_209 209
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_209(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_209(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_209(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_210(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_211
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_210 210
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_210(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_210(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_210(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_211(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_212
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_211 211
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_211(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_211(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_211(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_212(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_213
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_212 212
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_212(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_212(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_212(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_213(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_214
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_213 213
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_213(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_213(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_213(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_214(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_215
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_214 214
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_214(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_214(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_214(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_215(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_216
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_215 215
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_215(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_215(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_215(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_216(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_217
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_216 216
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_216(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_216(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_216(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_217(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_218
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_217 217
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_217(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_217(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_217(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_218(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_219
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_218 218
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_218(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_218(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_218(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_219(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_220
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_219 219
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_219(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_219(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_219(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_220(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_221
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_220 220
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_220(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_220(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_220(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_221(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_222
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_221 221
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_221(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_221(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_221(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_222(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_223
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_222 222
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_222(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_222(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_222(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_223(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_224
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_223 223
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_223(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_223(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_223(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_224(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_225
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_224 224
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_224(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_224(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_224(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_225(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_226
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_225 225
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_225(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_225(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_225(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_226(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_227
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_226 226
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_226(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_226(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_226(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_227(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_228
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_227 227
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_227(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_227(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_227(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_228(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_229
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_228 228
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_228(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_228(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_228(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_229(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_230
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_229 229
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_229(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_229(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_229(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_230(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_231
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_230 230
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_230(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_230(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_230(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_231(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_232
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_231 231
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_231(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_231(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_231(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_232(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_233
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_232 232
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_232(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_232(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_232(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_233(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_234
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_233 233
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_233(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_233(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_233(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_234(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_235
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_234 234
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_234(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_234(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_234(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_235(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_236
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_235 235
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_235(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_235(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_235(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_236(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_237
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_236 236
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_236(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_236(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_236(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_237(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_238
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_237 237
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_237(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_237(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_237(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_238(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_239
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_238 238
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_238(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_238(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_238(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_239(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_240
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_239 239
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_239(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_239(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_239(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_240(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_241
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_240 240
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_240(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_240(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_240(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_241(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_242
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_241 241
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_241(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_241(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_241(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_242(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_243
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_242 242
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_242(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_242(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_242(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_243(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_244
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_243 243
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_243(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_243(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_243(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_244(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_245
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_244 244
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_244(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_244(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_244(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_245(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_246
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_245 245
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_245(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_245(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_245(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_246(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_247
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_246 246
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_246(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_246(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_246(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_247(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_248
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_247 247
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_247(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_247(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_247(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_248(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_249
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_248 248
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_248(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_248(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_248(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_249(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_250
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_249 249
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_249(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_249(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_249(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_250(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_251
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_250 250
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_250(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_250(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_250(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_251(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_252
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_251 251
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_251(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_251(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_251(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_252(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_253
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_252 252
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_252(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_252(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_252(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_253(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_254
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_253 253
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_253(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_253(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_253(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_254(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_255
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_254 254
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_254(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_254(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_254(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_255(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_256
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_255 255
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_255(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_255(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_255(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_256(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_257
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_256 256
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_257(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_256(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_257(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_256(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_257(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_256(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_257(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_258
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_257 257
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_258(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_257(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_258(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_257(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_258(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_257(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_258(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_259
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_258 258
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_259(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_258(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_259(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_258(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_259(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_258(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_259(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_260
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_259 259
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_260(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_259(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_260(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_259(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_260(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_259(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_260(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_261
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_260 260
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_261(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_260(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_261(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_260(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_261(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_260(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_261(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_262
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_261 261
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_262(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_261(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_262(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_261(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_262(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_261(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_262(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_263
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_262 262
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_263(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_262(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_263(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_262(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_263(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_262(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_263(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_264
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_263 263
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_264(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_263(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_264(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_263(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_264(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_263(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_264(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_265
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_264 264
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_265(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_264(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_265(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_264(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_265(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_264(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_265(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_266
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_265 265
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_266(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_265(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_266(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_265(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_266(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_265(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_266(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_267
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_266 266
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_267(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_266(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_267(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_266(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_267(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_266(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_267(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_268
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_267 267
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_268(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_267(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_268(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_267(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_268(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_267(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_268(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_269
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_268 268
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_269(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_268(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_269(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_268(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_269(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_268(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_269(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_270
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_269 269
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_270(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_269(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_270(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_269(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_270(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_269(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_270(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_271
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_270 270
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_271(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_270(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_271(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_270(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_271(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_270(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_271(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_272
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_271 271
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_272(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_271(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_272(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_271(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_272(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_271(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_272(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_273
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_272 272
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_273(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_272(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_273(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_272(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_273(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_272(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_273(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_274
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_273 273
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_274(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_273(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_274(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_273(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_274(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_273(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_274(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_275
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_274 274
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_275(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_274(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_275(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_274(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_275(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_274(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_275(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_276
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_275 275
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_276(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_275(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_276(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_275(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_276(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_275(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_276(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_277
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_276 276
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_277(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_276(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_277(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_276(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_277(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_276(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_277(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_278
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_277 277
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_278(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_277(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_278(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_277(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_278(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_277(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_278(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_279
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_278 278
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_279(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_278(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_279(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_278(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_279(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_278(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_279(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_280
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_279 279
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_280(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_279(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_280(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_279(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_280(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_279(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_280(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_281
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_280 280
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_281(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_280(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_281(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_280(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_281(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_280(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_281(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_282
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_281 281
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_282(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_281(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_282(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_281(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_282(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_281(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_282(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_283
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_282 282
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_283(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_282(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_283(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_282(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_283(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_282(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_283(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_284
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_283 283
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_284(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_283(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_284(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_283(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_284(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_283(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_284(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_285
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_284 284
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_285(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_284(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_285(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_284(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_285(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_284(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_285(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_286
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_285 285
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_286(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_285(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_286(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_285(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_286(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_285(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_286(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_287
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_286 286
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_287(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_286(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_287(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_286(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_287(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_286(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_287(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_288
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_287 287
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_288(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_287(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_288(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_287(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_288(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_287(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_288(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_289
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_288 288
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_289(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_288(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_289(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_288(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_289(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_288(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_289(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_290
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_289 289
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_290(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_289(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_290(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_289(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_290(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_289(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_290(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_291
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_290 290
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_291(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_290(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_291(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_290(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_291(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_290(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_291(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_292
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_291 291
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_292(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_291(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_292(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_291(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_292(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_291(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_292(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_293
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_292 292
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_293(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_292(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_293(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_292(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_293(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_292(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_293(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_294
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_293 293
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_294(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_293(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_294(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_293(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_294(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_293(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_294(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_295
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_294 294
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_295(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_294(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_295(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_294(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_295(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_294(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_295(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_296
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_295 295
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_296(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_295(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_296(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_295(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_296(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_295(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_296(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_297
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_296 296
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_297(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_296(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_297(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_296(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_297(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_296(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_297(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_298
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_297 297
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_298(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_297(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_298(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_297(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_298(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_297(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_298(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_299
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_298 298
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_299(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_298(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_299(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_298(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_299(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_298(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_299(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_300
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_299 299
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_300(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_299(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_300(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_299(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_300(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_299(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_300(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_301
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_300 300
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_301(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_300(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_301(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_300(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_301(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_300(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_301(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_302
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_301 301
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_302(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_301(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_302(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_301(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_302(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_301(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_302(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_303
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_302 302
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_303(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_302(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_303(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_302(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_303(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_302(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_303(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_304
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_303 303
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_304(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_303(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_304(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_303(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_304(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_303(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_304(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_305
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_304 304
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_305(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_304(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_305(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_304(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_305(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_304(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_305(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_306
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_305 305
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_306(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_305(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_306(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_305(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_306(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_305(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_306(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_307
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_306 306
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_307(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_306(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_307(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_306(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_307(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_306(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_307(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_308
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_307 307
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_308(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_307(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_308(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_307(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_308(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_307(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_308(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_309
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_308 308
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_309(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_308(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_309(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_308(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_309(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_308(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_309(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_310
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_309 309
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_310(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_309(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_310(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_309(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_310(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_309(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_310(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_311
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_310 310
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_311(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_310(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_311(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_310(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_311(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_310(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_311(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_312
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_311 311
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_312(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_311(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_312(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_311(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_312(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_311(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_312(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_313
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_312 312
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_313(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_312(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_313(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_312(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_313(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_312(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_313(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_314
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_313 313
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_314(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_313(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_314(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_313(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_314(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_313(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_314(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_315
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_314 314
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_315(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_314(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_315(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_314(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_315(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_314(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_315(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_316
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_315 315
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_316(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_315(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_316(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_315(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_316(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_315(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_316(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_317
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_316 316
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_317(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_316(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_317(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_316(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_317(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_316(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_317(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_318
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_317 317
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_318(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_317(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_318(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_317(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_318(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_317(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_318(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_319
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_318 318
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_319(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_318(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_319(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_318(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_319(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_318(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_319(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_320
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_319 319
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_320(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_319(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_320(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_319(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_320(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_319(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_320(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_321
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_320 320
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_321(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_320(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_321(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_320(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_321(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_320(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_321(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_322
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_321 321
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_322(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_321(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_322(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_321(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_322(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_321(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_322(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_323
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_322 322
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_323(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_322(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_323(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_322(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_323(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_322(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_323(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_324
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_323 323
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_324(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_323(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_324(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_323(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_324(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_323(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_324(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_325
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_324 324
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_325(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_324(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_325(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_324(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_325(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_324(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_325(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_326
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_325 325
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_326(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_325(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_326(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_325(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_326(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_325(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_326(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_327
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_326 326
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_327(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_326(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_327(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_326(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_327(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_326(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_327(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_328
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_327 327
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_328(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_327(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_328(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_327(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_328(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_327(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_328(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_329
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_328 328
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_329(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_328(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_329(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_328(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_329(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_328(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_329(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_330
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_329 329
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_330(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_329(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_330(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_329(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_330(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_329(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_330(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_331
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_330 330
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_331(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_330(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_331(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_330(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_331(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_330(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_331(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_332
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_331 331
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_332(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_331(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_332(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_331(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_332(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_331(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_332(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_333
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_332 332
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_333(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_332(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_333(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_332(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_333(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_332(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_333(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_334
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_333 333
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_334(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_333(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_334(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_333(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_334(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_333(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_334(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_335
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_334 334
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_335(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_334(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_335(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_334(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_335(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_334(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_335(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_336
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_335 335
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_336(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_335(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_336(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_335(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_336(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_335(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_336(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_337
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_336 336
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_337(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_336(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_337(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_336(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_337(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_336(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_337(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_338
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_337 337
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_338(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_337(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_338(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_337(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_338(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_337(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_338(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_339
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_338 338
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_339(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_338(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_339(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_338(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_339(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_338(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_339(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_340
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_339 339
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_340(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_339(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_340(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_339(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_340(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_339(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_340(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_341
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_340 340
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_341(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_340(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_341(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_340(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_341(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_340(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_341(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_342
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_341 341
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_342(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_341(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_342(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_341(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_342(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_341(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_342(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_343
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_342 342
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_343(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_342(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_343(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_342(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_343(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_342(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_343(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_344
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_343 343
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_344(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_343(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_344(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_343(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_344(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_343(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_344(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_345
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_344 344
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_345(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_344(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_345(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_344(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_345(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_344(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_345(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_346
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_345 345
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_346(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_345(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_346(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_345(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_346(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_345(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_346(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_347
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_346 346
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_347(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_346(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_347(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_346(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_347(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_346(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_347(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_348
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_347 347
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_348(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_347(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_348(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_347(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_348(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_347(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_348(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_349
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_348 348
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_349(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_348(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_349(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_348(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_349(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_348(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_349(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_350
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_349 349
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_350(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_349(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_350(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_349(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_350(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_349(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_350(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_351
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_350 350
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_351(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_350(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_351(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_350(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_351(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_350(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_351(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_352
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_351 351
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_352(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_351(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_352(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_351(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_352(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_351(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_352(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_353
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_352 352
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_353(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_352(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_353(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_352(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_353(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_352(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_353(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_354
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_353 353
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_354(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_353(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_354(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_353(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_354(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_353(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_354(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_355
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_354 354
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_355(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_354(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_355(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_354(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_355(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_354(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_355(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_356
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_355 355
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_356(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_355(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_356(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_355(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_356(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_355(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_356(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_357
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_356 356
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_357(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_356(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_357(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_356(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_357(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_356(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_357(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_358
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_357 357
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_358(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_357(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_358(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_357(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_358(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_357(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_358(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_359
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_358 358
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_359(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_358(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_359(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_358(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_359(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_358(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_359(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_360
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_359 359
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_360(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_359(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_360(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_359(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_360(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_359(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_360(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_361
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_360 360
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_361(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_360(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_361(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_360(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_361(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_360(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_361(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_362
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_361 361
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_362(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_361(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_362(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_361(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_362(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_361(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_362(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_363
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_362 362
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_363(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_362(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_363(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_362(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_363(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_362(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_363(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_364
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_363 363
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_364(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_363(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_364(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_363(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_364(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_363(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_364(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_365
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_364 364
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_365(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_364(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_365(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_364(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_365(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_364(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_365(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_366
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_365 365
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_366(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_365(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_366(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_365(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_366(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_365(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_366(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_367
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_366 366
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_367(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_366(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_367(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_366(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_367(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_366(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_367(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_368
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_367 367
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_368(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_367(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_368(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_367(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_368(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_367(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_368(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_369
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_368 368
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_369(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_368(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_369(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_368(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_369(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_368(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_369(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_370
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_369 369
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_370(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_369(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_370(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_369(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_370(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_369(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_370(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_371
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_370 370
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_371(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_370(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_371(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_370(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_371(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_370(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_371(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_372
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_371 371
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_372(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_371(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_372(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_371(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_372(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_371(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_372(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_373
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_372 372
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_373(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_372(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_373(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_372(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_373(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_372(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_373(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_374
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_373 373
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_374(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_373(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_374(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_373(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_374(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_373(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_374(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_375
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_374 374
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_375(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_374(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_375(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_374(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_375(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_374(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_375(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_376
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_375 375
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_376(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_375(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_376(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_375(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_376(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_375(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_376(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_377
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_376 376
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_377(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_376(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_377(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_376(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_377(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_376(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_377(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_378
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_377 377
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_378(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_377(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_378(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_377(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_378(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_377(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_378(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_379
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_378 378
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_379(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_378(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_379(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_378(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_379(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_378(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_379(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_380
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_379 379
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_380(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_379(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_380(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_379(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_380(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_379(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_380(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_381
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_380 380
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_381(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_380(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_381(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_380(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_381(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_380(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_381(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_382
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_381 381
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_382(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_381(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_382(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_381(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_382(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_381(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_382(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_383
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_382 382
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_383(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_382(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_383(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_382(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_383(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_382(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_383(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_384
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_383 383
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_384(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_383(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_384(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_383(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_384(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_383(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_384(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_385
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_384 384
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_385(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_384(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_385(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_384(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_385(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_384(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_385(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_386
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_385 385
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_386(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_385(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_386(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_385(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_386(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_385(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_386(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_387
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_386 386
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_387(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_386(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_387(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_386(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_387(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_386(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_387(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_388
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_387 387
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_388(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_387(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_388(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_387(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_388(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_387(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_388(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_389
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_388 388
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_389(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_388(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_389(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_388(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_389(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_388(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_389(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_390
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_389 389
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_390(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_389(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_390(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_389(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_390(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_389(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_390(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_391
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_390 390
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_391(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_390(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_391(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_390(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_391(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_390(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_391(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_392
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_391 391
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_392(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_391(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_392(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_391(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_392(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_391(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_392(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_393
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_392 392
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_393(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_392(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_393(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_392(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_393(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_392(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_393(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_394
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_393 393
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_394(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_393(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_394(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_393(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_394(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_393(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_394(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_395
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_394 394
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_395(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_394(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_395(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_394(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_395(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_394(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_395(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_396
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_395 395
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_396(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_395(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_396(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_395(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_396(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_395(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_396(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_397
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_396 396
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_397(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_396(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_397(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_396(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_397(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_396(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_397(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_398
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_397 397
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_398(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_397(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_398(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_397(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_398(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_397(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_398(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_399
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_398 398
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_399(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_398(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_399(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_398(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_399(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_398(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_399(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_400
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_399 399
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_400(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_399(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_400(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_399(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_400(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_399(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_400(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_401
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_400 400
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_401(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_400(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_401(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_400(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_401(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_400(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_401(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_402
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_401 401
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_402(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_401(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_402(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_401(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_402(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_401(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_402(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_403
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_402 402
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_403(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_402(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_403(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_402(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_403(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_402(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_403(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_404
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_403 403
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_404(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_403(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_404(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_403(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_404(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_403(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_404(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_405
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_404 404
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_405(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_404(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_405(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_404(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_405(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_404(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_405(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_406
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_405 405
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_406(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_405(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_406(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_405(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_406(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_405(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_406(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_407
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_406 406
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_407(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_406(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_407(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_406(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_407(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_406(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_407(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_408
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_407 407
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_408(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_407(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_408(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_407(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_408(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_407(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_408(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_409
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_408 408
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_409(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_408(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_409(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_408(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_409(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_408(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_409(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_410
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_409 409
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_410(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_409(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_410(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_409(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_410(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_409(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_410(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_411
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_410 410
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_411(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_410(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_411(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_410(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_411(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_410(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_411(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_412
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_411 411
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_412(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_411(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_412(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_411(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_412(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_411(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_412(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_413
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_412 412
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_413(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_412(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_413(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_412(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_413(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_412(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_413(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_414
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_413 413
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_414(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_413(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_414(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_413(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_414(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_413(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_414(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_415
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_414 414
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_415(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_414(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_415(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_414(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_415(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_414(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_415(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_416
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_415 415
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_416(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_415(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_416(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_415(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_416(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_415(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_416(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_417
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_416 416
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_417(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_416(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_417(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_416(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_417(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_416(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_417(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_418
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_417 417
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_418(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_417(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_418(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_417(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_418(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_417(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_418(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_419
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_418 418
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_419(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_418(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_419(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_418(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_419(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_418(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_419(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_420
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_419 419
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_420(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_419(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_420(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_419(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_420(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_419(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_420(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_421
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_420 420
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_421(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_420(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_421(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_420(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_421(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_420(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_421(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_422
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_421 421
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_422(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_421(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_422(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_421(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_422(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_421(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_422(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_423
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_422 422
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_423(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_422(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_423(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_422(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_423(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_422(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_423(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_424
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_423 423
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_424(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_423(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_424(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_423(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_424(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_423(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_424(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_425
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_424 424
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_425(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_424(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_425(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_424(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_425(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_424(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_425(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_426
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_425 425
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_426(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_425(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_426(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_425(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_426(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_425(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_426(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_427
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_426 426
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_427(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_426(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_427(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_426(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_427(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_426(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_427(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_428
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_427 427
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_428(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_427(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_428(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_427(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_428(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_427(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_428(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_429
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_428 428
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_429(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_428(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_429(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_428(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_429(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_428(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_429(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_430
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_429 429
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_430(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_429(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_430(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_429(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_430(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_429(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_430(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_431
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_430 430
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_431(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_430(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_431(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_430(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_431(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_430(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_431(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_432
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_431 431
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_432(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_431(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_432(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_431(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_432(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_431(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_432(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_433
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_432 432
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_433(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_432(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_433(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_432(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_433(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_432(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_433(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_434
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_433 433
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_434(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_433(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_434(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_433(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_434(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_433(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_434(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_435
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_434 434
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_435(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_434(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_435(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_434(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_435(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_434(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_435(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_436
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_435 435
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_436(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_435(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_436(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_435(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_436(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_435(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_436(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_437
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_436 436
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_437(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_436(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_437(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_436(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_437(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_436(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_437(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_438
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_437 437
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_438(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_437(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_438(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_437(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_438(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_437(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_438(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_439
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_438 438
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_439(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_438(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_439(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_438(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_439(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_438(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_439(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_440
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_439 439
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_440(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_439(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_440(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_439(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_440(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_439(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_440(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_441
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_440 440
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_441(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_440(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_441(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_440(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_441(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_440(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_441(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_442
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_441 441
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_442(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_441(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_442(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_441(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_442(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_441(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_442(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_443
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_442 442
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_443(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_442(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_443(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_442(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_443(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_442(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_443(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_444
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_443 443
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_444(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_443(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_444(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_443(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_444(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_443(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_444(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_445
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_444 444
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_445(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_444(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_445(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_444(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_445(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_444(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_445(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_446
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_445 445
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_446(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_445(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_446(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_445(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_446(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_445(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_446(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_447
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_446 446
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_447(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_446(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_447(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_446(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_447(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_446(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_447(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_448
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_447 447
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_448(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_447(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_448(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_447(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_448(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_447(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_448(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_449
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_448 448
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_449(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_448(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_449(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_448(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_449(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_448(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_449(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_450
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_449 449
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_450(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_449(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_450(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_449(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_450(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_449(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_450(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_451
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_450 450
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_451(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_450(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_451(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_450(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_451(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_450(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_451(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_452
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_451 451
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_452(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_451(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_452(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_451(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_452(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_451(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_452(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_453
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_452 452
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_453(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_452(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_453(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_452(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_453(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_452(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_453(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_454
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_453 453
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_454(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_453(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_454(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_453(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_454(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_453(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_454(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_455
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_454 454
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_455(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_454(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_455(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_454(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_455(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_454(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_455(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_456
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_455 455
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_456(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_455(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_456(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_455(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_456(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_455(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_456(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_457
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_456 456
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_457(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_456(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_457(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_456(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_457(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_456(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_457(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_458
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_457 457
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_458(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_457(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_458(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_457(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_458(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_457(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_458(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_459
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_458 458
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_459(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_458(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_459(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_458(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_459(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_458(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_459(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_460
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_459 459
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_460(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_459(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_460(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_459(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_460(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_459(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_460(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_461
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_460 460
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_461(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_460(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_461(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_460(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_461(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_460(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_461(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_462
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_461 461
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_462(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_461(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_462(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_461(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_462(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_461(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_462(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_463
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_462 462
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_463(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_462(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_463(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_462(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_463(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_462(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_463(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_464
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_463 463
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_464(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_463(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_464(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_463(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_464(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_463(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_464(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_465
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_464 464
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_465(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_464(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_465(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_464(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_465(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_464(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_465(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_466
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_465 465
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_466(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_465(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_466(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_465(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_466(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_465(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_466(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_467
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_466 466
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_467(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_466(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_467(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_466(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_467(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_466(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_467(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_468
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_467 467
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_468(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_467(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_468(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_467(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_468(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_467(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_468(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_469
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_468 468
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_469(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_468(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_469(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_468(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_469(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_468(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_469(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_470
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_469 469
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_470(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_469(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_470(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_469(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_470(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_469(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_470(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_471
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_470 470
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_471(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_470(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_471(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_470(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_471(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_470(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_471(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_472
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_471 471
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_472(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_471(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_472(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_471(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_472(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_471(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_472(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_473
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_472 472
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_473(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_472(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_473(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_472(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_473(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_472(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_473(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_474
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_473 473
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_474(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_473(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_474(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_473(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_474(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_473(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_474(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_475
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_474 474
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_475(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_474(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_475(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_474(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_475(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_474(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_475(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_476
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_475 475
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_476(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_475(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_476(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_475(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_476(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_475(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_476(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_477
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_476 476
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_477(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_476(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_477(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_476(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_477(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_476(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_477(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_478
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_477 477
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_478(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_477(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_478(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_477(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_478(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_477(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_478(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_479
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_478 478
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_479(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_478(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_479(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_478(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_479(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_478(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_479(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_480
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_479 479
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_480(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_479(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_480(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_479(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_480(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_479(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_480(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_481
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_480 480
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_481(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_480(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_481(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_480(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_481(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_480(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_481(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_482
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_481 481
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_482(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_481(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_482(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_481(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_482(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_481(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_482(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_483
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_482 482
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_483(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_482(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_483(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_482(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_483(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_482(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_483(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_484
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_483 483
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_484(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_483(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_484(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_483(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_484(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_483(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_484(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_485
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_484 484
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_485(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_484(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_485(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_484(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_485(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_484(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_485(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_486
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_485 485
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_486(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_485(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_486(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_485(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_486(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_485(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_486(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_487
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_486 486
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_487(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_486(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_487(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_486(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_487(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_486(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_487(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_488
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_487 487
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_488(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_487(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_488(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_487(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_488(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_487(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_488(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_489
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_488 488
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_489(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_488(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_489(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_488(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_489(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_488(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_489(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_490
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_489 489
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_490(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_489(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_490(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_489(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_490(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_489(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_490(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_491
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_490 490
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_491(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_490(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_491(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_490(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_491(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_490(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_491(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_492
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_491 491
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_492(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_491(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_492(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_491(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_492(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_491(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_492(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_493
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_492 492
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_493(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_492(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_493(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_492(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_493(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_492(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_493(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_494
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_493 493
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_494(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_493(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_494(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_493(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_494(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_493(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_494(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_495
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_494 494
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_495(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_494(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_495(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_494(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_495(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_494(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_495(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_496
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_495 495
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_496(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_495(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_496(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_495(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_496(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_495(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_496(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_497
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_496 496
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_497(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_496(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_497(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_496(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_497(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_496(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_497(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_498
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_497 497
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_498(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_497(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_498(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_497(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_498(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_497(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_498(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_499
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_498 498
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_499(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_498(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_499(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_498(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_499(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_498(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_499(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_500
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_499 499
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_500(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_499(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_500(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_499(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_500(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_499(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_500(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_501
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_500 500
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_501(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_500(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_501(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_500(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_501(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_500(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_501(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_502
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_501 501
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_502(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_501(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_502(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_501(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_502(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_501(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_502(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_503
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_502 502
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_503(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_502(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_503(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_502(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_503(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_502(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_503(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_504
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_503 503
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_504(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_503(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_504(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_503(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_504(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_503(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_504(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_505
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_504 504
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_505(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_504(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_505(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_504(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_505(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_504(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_505(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_506
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_505 505
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_506(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_505(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_506(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_505(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_506(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_505(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_506(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_507
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_506 506
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_507(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_506(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_507(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_506(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_507(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_506(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_507(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_508
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_507 507
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_508(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_507(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_508(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_507(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_508(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_507(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_508(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_509
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_508 508
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_509(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_508(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_509(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_508(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_509(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_508(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_509(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_510
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_509 509
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_510(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_509(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_510(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_509(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_510(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_509(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_510(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_511
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_510 510
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_511(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_510(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_511(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_510(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_511(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_510(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_511(_) NNDEPLOY_PP_INTERNAL_SEQ_SIZE_512
#define NNDEPLOY_PP_INTERNAL_SEQ_SIZE_NNDEPLOY_PP_INTERNAL_SEQ_SIZE_511 511
#define NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_512(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D0_SEQ_FOR_EACH_511(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_512(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D1_SEQ_FOR_EACH_511(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#define NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_512(apply, m, d, seq) \
  apply(m, d, NNDEPLOY_PP_INTERNAL_SEQ_HEAD(seq))                  \
      NNDEPLOY_PP_INTERNAL_D2_SEQ_FOR_EACH_511(                    \
          apply, m, d, NNDEPLOY_PP_INTERNAL_SEQ_TAIL(seq))

#endif
