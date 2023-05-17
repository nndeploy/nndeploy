
#ifndef _NNDEPLOY_SOURCE_BASE_TYPE_H_
#define _NNDEPLOY_SOURCE_BASE_TYPE_H_

#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/saturate_cast.h"

namespace nndeploy {
namespace base {

enum PixelType : int32_t {
  kPixelTypeGRAY = 0x0000,
  kPixelTypeRGB,
  kPixelTypeBGR,
  kPixelTypeRGBA,
  kPixelTypeBGRA,

  // not sopport
  kPixelTypeNotSupport,
};

enum CvtColorType : int32_t {
  kCvtColorTypeBGR2GRAY = 0x0000,
  kCvtColorTypeBGRA2GRAY,
  kCvtColorTypeRGB2GRAY,
  kCvtColorTypeRGBA2GRAY,

  // not sopport
  kCvtColorTypeNotSupport,
};

enum InterpType : int32_t {
  kInterpTypeNearst = 0x00,
  kInterpTypeLinear = 0x01,

  // not sopport
  kInterpTypeNotSupport,
};

enum BorderType : int32_t {
  kBorderTypeConstant = 0x00,
  kBorderTypeReflect = 0x01,
  kBorderTypeEdge = 0x02,

  // not sopport
  kBorderTypeNotSupport,
};

template <typename T>
class Point3;

/**
 * @brief
 * Template class for 2D points specified by its coordinates `x` an `y`.
 * An instance of the class is interchangeable with C structures, CvPoint
 * and CvPoint2D32f . There is also a cast operator to convert point
 * coordinates to the specified type. The conversion from floating-point
 * coordinates to integer coordinates is done by rounding. Commonly, the
 * conversion uses this operation for each of the coordinates. Besides the
 * class members listed in the declaration above, the following operations
 * on points are implemented:
 * @code
 *     pt1 = pt2 + pt3;
 *     pt1 = pt2 - pt3;
 *     pt1 = pt2 * a;
 *     pt1 = a * pt2;
 *     pt1 = pt2 / a;
 *     pt1 += pt2;
 *     pt1 -= pt2;
 *     pt1 *= a;
 *     pt1 /= a;
 *     double value = norm(pt); // L2 norm
 *     pt1 == pt2;
 *     pt1 != pt2;
 * @endcode
 * For your convenience, the following type aliases are defined:
 * @code
 *     typedef Point<int> Point2i;
 *     typedef Point2i Point;
 *     typedef Point<float> Point2f;
 *     typedef Point<double> Point2d;
 * @endcode
 * Example:
 * @code
 *     Point2f a(0.3f, 0.f), b(0.f, 0.4f);
 *     Point pt = (a + b)*10.f;
 *     cout << pt.x << ", " << pt.y << endl;
 * @endcode
 */
template <typename T>
class Point {
 public:
  typedef T value_type;

  //! default constructor
  Point();
  Point(T _x, T _y);
  Point(const Point& pt);
  Point(Point&& pt) = default;

  Point(const Size<T>& sz);

  //!
  Point& operator=(const Point& pt);
  Point& operator=(Point&& pt) = default;

  //! conversion to another data type
  template <typename Tp2>
  operator Point<Tp2>() const;

  //! dot product
  T dot(const Point& pt) const;
  //! dot product computed in double-precision arithmetics
  double ddot(const Point& pt) const;
  //! cross-product
  double cross(const Point& pt) const;
  //! checks whether the point is inside the specified rectangle
  bool inside(const Rect<T>& r) const;

  T x;  //!< x coordinate of the point
  T y;  //!< y coordinate of the point
};

typedef Point<int> Point2i;
typedef Point<int64> Point2l;
typedef Point<float> Point2f;
typedef Point<double> Point2d;

/**
 * @brief Template class for 3D points specified by its coordinates `x`, `y`
 * and `z`.
 *
 * An instance of the class is interchangeable with the C structure CvPoint2D32f
 * . Similarly to Point , the coordinates of 3D points can be converted to
 * another type. The vector arithmetic and comparison operations are also
 * supported.
 *
 * The following Point3\<\> aliases are available:
 * @code
 *     typedef Point3<int> Point3i;
 *     typedef Point3<float> Point3f;
 *     typedef Point3<double> Point3d;
 * @endcode
 * @see Point3i, Point3f and Point3d
 */
template <typename T>
class Point3 {
 public:
  typedef T value_type;

  //! default constructor
  Point3();
  Point3(T _x, T _y, T _z);

  Point3(const Point3& pt) = default;
  Point3(Point3&& pt) = default;

  explicit Point3(const Point<T>& pt);

  Point3& operator=(const Point3& pt) = default;
  Point3& operator=(Point3&& pt) = default;

  //! conversion to another data type
  template <typename Tp2>
  operator Point3<Tp2>() const;

  //! dot product
  T dot(const Point3& pt) const;
  //! dot product computed in double-precision arithmetics
  double ddot(const Point3& pt) const;
  //! cross product of the 2 3D points
  Point3 cross(const Point3& pt) const;
  T x;  //!< x coordinate of the 3D point
  T y;  //!< y coordinate of the 3D point
  T z;  //!< z coordinate of the 3D point
};

typedef Point3<int> Point3i;
typedef Point3<float> Point3f;
typedef Point3<double> Point3d;

/**
 * @brief Template class for specifying the size of an image or rectangle.
 * The class includes two members called width and height. The structure can be
 * converted to and from the old OpenCV structures CvSize and CvSize2D32f . The
 * same set of arithmetic and comparison operations as for Point is available.
 *
 * OpenCV defines the following Size\<\> aliases:
 * @code
 *     typedef Size<int> Size2i;
 *     typedef Size2i Size;
 *     typedef Size<float> Size2f;
 * @endcode
 */
template <typename T>
class Size {
 public:
  typedef T value_type;

  //! default constructor
  Size();
  Size(T _width, T _height);
  Size(const Size& sz) = default;
  Size(Size&& sz) = default;

  Size(const Point<T>& pt);

  Size& operator=(const Size& sz) = default;
  Size& operator=(Size&& sz) = default;

  //! the area (width*height)
  T area() const;
  //! aspect ratio (width/height)
  double aspectRatio() const;
  //! true if empty
  bool empty() const;

  //! conversion of another data type.
  template <typename Tp2>
  operator Size<Tp2>() const;

  T width;   //!< the width
  T height;  //!< the height
};

typedef Size<int> Size2i;
typedef Size<int64> Size2l;
typedef Size<float> Size2f;
typedef Size<double> Size2d;

/**
 *
 * @brief Template class for 2D rectangles
 * described by the following parameters:
 * -   Coordinates of the top-left corner. This is a default interpretation of
 * Rect::x and Rect::y in OpenCV. Though, in your algorithms you may count x
 * and y from the bottom-left corner.
 * -   Rectangle width and height.
 *
 * OpenCV typically assumes that the top and left boundary of the rectangle are
 * inclusive, while the right and bottom boundaries are not. For example, the
 * method Rect::contains returns true if
 *
 * \f[x  \leq pt.x < x+width,
 *       y  \leq pt.y < y+height\f]
 *
 * Virtually every loop over an image ROI in OpenCV (where ROI is specified by
 * Rect\<int\> ) is implemented as:
 * @code
 *     for(int y = roi.y; y < roi.y + roi.height; y++)
 *         for(int x = roi.x; x < roi.x + roi.width; x++)
 *         {
 *             // ...
 *         }
 * @endcode
 * In addition to the class members, the following operations on rectangles are
 * implemented:
 * -   \f$\texttt{rect} = \texttt{rect} \pm \texttt{point}\f$ (shifting a
 * rectangle by a certain offset)
 * -   \f$\texttt{rect} = \texttt{rect} \pm \texttt{size}\f$ (expanding or
 * shrinking a rectangle by a certain amount)
 * -   rect += point, rect -= point, rect += size, rect -= size (augmenting
 * operations)
 * -   rect = rect1 & rect2 (rectangle intersection)
 * -   rect = rect1 | rect2 (minimum area rectangle containing rect1 and rect2 )
 * -   rect &= rect1, rect |= rect1 (and the corresponding augmenting
 * operations)
 * -   rect == rect1, rect != rect1 (rectangle comparison)
 *
 * This is an example how the partial ordering on rectangles can be established
 * (rect1 \f$\subseteq\f$ rect2):
 * @code
 *     template<typename T> inline bool
 *     operator <= (const Rect<T>& r1, const Rect<T>& r2)
 *     {
 *         return (r1 & r2) == r1;
 *     }
 * @endcode
 * For your convenience, the Rect\<\> alias is available: Rect
 */
template <typename T>
class Rect {
 public:
  typedef T value_type;

  //! default constructor
  Rect();
  Rect(T _x, T _y, T _width, T _height);
  Rect(const Rect& r) = default;
  Rect(Rect&& r) = default;
  Rect(const Point<T>& org, const Size<T>& sz);
  Rect(const Point<T>& pt1, const Point<T>& pt2);

  Rect& operator=(const Rect& r) = default;
  Rect& operator=(Rect&& r) = default;
  //! the top-left corner
  Point<T> tl() const;
  //! the bottom-right corner
  Point<T> br() const;

  //! size (width, height) of the rectangle
  Size<T> size() const;
  //! area (width*height) of the rectangle
  T area() const;
  //! true if empty
  bool empty() const;

  //! conversion to another data type
  template <typename Tp2>
  operator Rect<Tp2>() const;

  //! checks whether the rectangle contains the point
  bool contains(const Point<T>& pt) const;

  T x;       //!< x coordinate of the top-left corner
  T y;       //!< y coordinate of the top-left corner
  T width;   //!< width of the rectangle
  T height;  //!< height of the rectangle
};

typedef Rect<int> Rect2i;
typedef Rect<float> Rect2f;
typedef Rect<double> Rect2d;

/**
 * @brief Template class specifying a continuous subsequence (slice) of a
 * sequence.
 * The class is used to specify a row or a column span in a matrix ( Mat ) and
 * for many other purposes. Range(a,b) is basically the same as a:b in Matlab or
 * a..b in Python. As in Python, start is an inclusive left boundary of the
 * range and end is an exclusive right boundary of the range. Such a half-opened
 * interval is usually denoted as \f$[start,end)\f$ . The static method
 * Range::all() returns a special variable that means "the whole sequence" or
 * "the whole range", just like " : " in Matlab or " ... " in Python. All the
 * methods and functions in OpenCV that take Range support this special
 * Range::all() value. But, of course, in case of your own custom processing,
 * you will probably have to check and handle it explicitly:
 * @code
 *     void my_function(..., const Range& r, ....)
 *     {
 *         if(r == Range::all()) {
 *             // process all the data
 *         }
 *         else {
 *             // process [r.start, r.end)
 *         }
 *     }
 * @endcode
 */
class Range {
 public:
  Range();
  Range(int _start, int _end);
  int size() const;
  bool empty() const;
  static Range all();

  int start, end;
};

/** @brief
 * Template class for a 4-element vector.
 * Scalar\_ and Scalar can be used just as
 * typical 4-element vectors. The type Scalar is widely used in OpenCV to pass
 * pixel values.
 */
template <typename T>
class Scalar {
 public:
  //! default constructor
  Scalar();
  Scalar(T v0, T v1, T v2 = 0, T v3 = 0);
  Scalar(T v0);

  Scalar(const Scalar& s) = default;
  Scalar(Scalar&& s);

  Scalar& operator=(const Scalar& s) = default;
  Scalar& operator=(Scalar&& s);

  //! returns a scalar with all elements set to v0
  static Scalar<T> all(T v0);

  //! conversion to another data type
  template <typename T2>
  operator Scalar<T2>() const;

  //! per-element product
  Scalar<T> mul(const Scalar<T>& a, double scale = 1) const;

  //! returns (v0, -v1, -v2, -v3)
  Scalar<T> conj() const;

  //! returns true iff v1 == v2 == v3 == 0
  bool isReal() const;

  T val[4];
};

typedef Scalar<double> Scalar;

template <typename T>
inline Point<T>::Point() : x(0), y(0) {}

template <typename T>
inline Point<T>::Point(T _x, T _y) : x(_x), y(_y) {}

inline Point<T>::Point(const Point& pt) : x(pt.x), y(pt.y) {}

template <typename T>
inline Point<T>::Point(const Size<T>& sz) : x(sz.width), y(sz.height) {}

template <typename T>
inline Point<T>& Point<T>::operator=(const Point& pt) {
  x = pt.x;
  y = pt.y;
  return *this;
}

template <typename T>
template <typename Tp2>
inline Point<T>::operator Point<Tp2>() const {
  return Point<Tp2>(saturate_cast<Tp2>(x), saturate_cast<Tp2>(y));
}

template <typename T>
inline T Point<T>::dot(const Point& pt) const {
  return saturate_cast<T>(x * pt.x + y * pt.y);
}

template <typename T>
inline double Point<T>::ddot(const Point& pt) const {
  return (double)x * (double)(pt.x) + (double)y * (double)(pt.y);
}

template <typename T>
inline double Point<T>::cross(const Point& pt) const {
  return (double)x * pt.y - (double)y * pt.x;
}

template <typename T>
inline bool Point<T>::inside(const Rect<T>& r) const {
  return r.contains(*this);
}

template <typename T>
static inline Point<T>& operator+=(Point<T>& a, const Point<T>& b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

template <typename T>
static inline Point<T>& operator-=(Point<T>& a, const Point<T>& b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

template <typename T>
static inline Point<T>& operator*=(Point<T>& a, int b) {
  a.x = saturate_cast<T>(a.x * b);
  a.y = saturate_cast<T>(a.y * b);
  return a;
}

template <typename T>
static inline Point<T>& operator*=(Point<T>& a, float b) {
  a.x = saturate_cast<T>(a.x * b);
  a.y = saturate_cast<T>(a.y * b);
  return a;
}

template <typename T>
static inline Point<T>& operator*=(Point<T>& a, double b) {
  a.x = saturate_cast<T>(a.x * b);
  a.y = saturate_cast<T>(a.y * b);
  return a;
}

template <typename T>
static inline Point<T>& operator/=(Point<T>& a, int b) {
  a.x = saturate_cast<T>(a.x / b);
  a.y = saturate_cast<T>(a.y / b);
  return a;
}

template <typename T>
static inline Point<T>& operator/=(Point<T>& a, float b) {
  a.x = saturate_cast<T>(a.x / b);
  a.y = saturate_cast<T>(a.y / b);
  return a;
}

template <typename T>
static inline Point<T>& operator/=(Point<T>& a, double b) {
  a.x = saturate_cast<T>(a.x / b);
  a.y = saturate_cast<T>(a.y / b);
  return a;
}

template <typename T>
static inline double norm(const Point<T>& pt) {
  return std::sqrt((double)pt.x * pt.x + (double)pt.y * pt.y);
}

template <typename T>
static inline bool operator==(const Point<T>& a, const Point<T>& b) {
  return a.x == b.x && a.y == b.y;
}

template <typename T>
static inline bool operator!=(const Point<T>& a, const Point<T>& b) {
  return a.x != b.x || a.y != b.y;
}

template <typename T>
static inline Point<T> operator+(const Point<T>& a, const Point<T>& b) {
  return Point<T>(saturate_cast<T>(a.x + b.x), saturate_cast<T>(a.y + b.y));
}

template <typename T>
static inline Point<T> operator-(const Point<T>& a, const Point<T>& b) {
  return Point<T>(saturate_cast<T>(a.x - b.x), saturate_cast<T>(a.y - b.y));
}

template <typename T>
static inline Point<T> operator-(const Point<T>& a) {
  return Point<T>(saturate_cast<T>(-a.x), saturate_cast<T>(-a.y));
}

template <typename T>
static inline Point<T> operator*(const Point<T>& a, int b) {
  return Point<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b));
}

template <typename T>
static inline Point<T> operator*(int a, const Point<T>& b) {
  return Point<T>(saturate_cast<T>(b.x * a), saturate_cast<T>(b.y * a));
}

template <typename T>
static inline Point<T> operator*(const Point<T>& a, float b) {
  return Point<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b));
}

template <typename T>
static inline Point<T> operator*(float a, const Point<T>& b) {
  return Point<T>(saturate_cast<T>(b.x * a), saturate_cast<T>(b.y * a));
}

template <typename T>
static inline Point<T> operator*(const Point<T>& a, double b) {
  return Point<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b));
}

template <typename T>
static inline Point<T> operator*(double a, const Point<T>& b) {
  return Point<T>(saturate_cast<T>(b.x * a), saturate_cast<T>(b.y * a));
}

template <typename T>
static inline Point<T> operator/(const Point<T>& a, int b) {
  Point<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename T>
static inline Point<T> operator/(const Point<T>& a, float b) {
  Point<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename T>
static inline Point<T> operator/(const Point<T>& a, double b) {
  Point<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename _AccTp>
static inline _AccTp normL2Sqr(const Point<int>& pt);
template <typename _AccTp>
static inline _AccTp normL2Sqr(const Point<int64>& pt);
template <typename _AccTp>
static inline _AccTp normL2Sqr(const Point<float>& pt);
template <typename _AccTp>
static inline _AccTp normL2Sqr(const Point<double>& pt);

template <>
inline int normL2Sqr<int>(const Point<int>& pt) {
  return pt.dot(pt);
}
template <>
inline int64 normL2Sqr<int64>(const Point<int64>& pt) {
  return pt.dot(pt);
}
template <>
inline float normL2Sqr<float>(const Point<float>& pt) {
  return pt.dot(pt);
}
template <>
inline double normL2Sqr<double>(const Point<int>& pt) {
  return pt.dot(pt);
}

template <>
inline double normL2Sqr<double>(const Point<float>& pt) {
  return pt.ddot(pt);
}
template <>
inline double normL2Sqr<double>(const Point<double>& pt) {
  return pt.ddot(pt);
}

template <typename T>
inline Point3<T>::Point3() : x(0), y(0), z(0) {}

template <typename T>
inline Point3<T>::Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

template <typename T>
inline Point3<T>::Point3(const Point<T>& pt) : x(pt.x), y(pt.y), z(T()) {}

template <typename T>
template <typename Tp2>
inline Point3<T>::operator Point3<Tp2>() const {
  return Point3<Tp2>(saturate_cast<Tp2>(x), saturate_cast<Tp2>(y),
                     saturate_cast<Tp2>(z));
}

template <typename T>
inline T Point3<T>::dot(const Point3& pt) const {
  return saturate_cast<T>(x * pt.x + y * pt.y + z * pt.z);
}

template <typename T>
inline double Point3<T>::ddot(const Point3& pt) const {
  return (double)x * pt.x + (double)y * pt.y + (double)z * pt.z;
}

template <typename T>
inline Point3<T> Point3<T>::cross(const Point3<T>& pt) const {
  return Point3<T>(y * pt.z - z * pt.y, z * pt.x - x * pt.z,
                   x * pt.y - y * pt.x);
}

template <typename T>
static inline Point3<T>& operator+=(Point3<T>& a, const Point3<T>& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

template <typename T>
static inline Point3<T>& operator-=(Point3<T>& a, const Point3<T>& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

template <typename T>
static inline Point3<T>& operator*=(Point3<T>& a, int b) {
  a.x = saturate_cast<T>(a.x * b);
  a.y = saturate_cast<T>(a.y * b);
  a.z = saturate_cast<T>(a.z * b);
  return a;
}

template <typename T>
static inline Point3<T>& operator*=(Point3<T>& a, float b) {
  a.x = saturate_cast<T>(a.x * b);
  a.y = saturate_cast<T>(a.y * b);
  a.z = saturate_cast<T>(a.z * b);
  return a;
}

template <typename T>
static inline Point3<T>& operator*=(Point3<T>& a, double b) {
  a.x = saturate_cast<T>(a.x * b);
  a.y = saturate_cast<T>(a.y * b);
  a.z = saturate_cast<T>(a.z * b);
  return a;
}

template <typename T>
static inline Point3<T>& operator/=(Point3<T>& a, int b) {
  a.x = saturate_cast<T>(a.x / b);
  a.y = saturate_cast<T>(a.y / b);
  a.z = saturate_cast<T>(a.z / b);
  return a;
}

template <typename T>
static inline Point3<T>& operator/=(Point3<T>& a, float b) {
  a.x = saturate_cast<T>(a.x / b);
  a.y = saturate_cast<T>(a.y / b);
  a.z = saturate_cast<T>(a.z / b);
  return a;
}

template <typename T>
static inline Point3<T>& operator/=(Point3<T>& a, double b) {
  a.x = saturate_cast<T>(a.x / b);
  a.y = saturate_cast<T>(a.y / b);
  a.z = saturate_cast<T>(a.z / b);
  return a;
}

template <typename T>
static inline double norm(const Point3<T>& pt) {
  return std::sqrt((double)pt.x * pt.x + (double)pt.y * pt.y +
                   (double)pt.z * pt.z);
}

template <typename T>
static inline bool operator==(const Point3<T>& a, const Point3<T>& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

template <typename T>
static inline bool operator!=(const Point3<T>& a, const Point3<T>& b) {
  return a.x != b.x || a.y != b.y || a.z != b.z;
}

template <typename T>
static inline Point3<T> operator+(const Point3<T>& a, const Point3<T>& b) {
  return Point3<T>(saturate_cast<T>(a.x + b.x), saturate_cast<T>(a.y + b.y),
                   saturate_cast<T>(a.z + b.z));
}

template <typename T>
static inline Point3<T> operator-(const Point3<T>& a, const Point3<T>& b) {
  return Point3<T>(saturate_cast<T>(a.x - b.x), saturate_cast<T>(a.y - b.y),
                   saturate_cast<T>(a.z - b.z));
}

template <typename T>
static inline Point3<T> operator-(const Point3<T>& a) {
  return Point3<T>(saturate_cast<T>(-a.x), saturate_cast<T>(-a.y),
                   saturate_cast<T>(-a.z));
}

template <typename T>
static inline Point3<T> operator*(const Point3<T>& a, int b) {
  return Point3<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b),
                   saturate_cast<T>(a.z * b));
}

template <typename T>
static inline Point3<T> operator*(int a, const Point3<T>& b) {
  return Point3<T>(saturate_cast<T>(b.x * a), saturate_cast<T>(b.y * a),
                   saturate_cast<T>(b.z * a));
}

template <typename T>
static inline Point3<T> operator*(const Point3<T>& a, float b) {
  return Point3<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b),
                   saturate_cast<T>(a.z * b));
}

template <typename T>
static inline Point3<T> operator*(float a, const Point3<T>& b) {
  return Point3<T>(saturate_cast<T>(b.x * a), saturate_cast<T>(b.y * a),
                   saturate_cast<T>(b.z * a));
}

template <typename T>
static inline Point3<T> operator*(const Point3<T>& a, double b) {
  return Point3<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b),
                   saturate_cast<T>(a.z * b));
}

template <typename T>
static inline Point3<T> operator*(double a, const Point3<T>& b) {
  return Point3<T>(saturate_cast<T>(b.x * a), saturate_cast<T>(b.y * a),
                   saturate_cast<T>(b.z * a));
}

template <typename T>
static inline Point3<T> operator/(const Point3<T>& a, int b) {
  Point3<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename T>
static inline Point3<T> operator/(const Point3<T>& a, float b) {
  Point3<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename T>
static inline Point3<T> operator/(const Point3<T>& a, double b) {
  Point3<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename T>
inline Size<T>::Size() : width(0), height(0) {}

template <typename T>
inline Size<T>::Size(T _width, T _height) : width(_width), height(_height) {}

template <typename T>
inline Size<T>::Size(const Point<T>& pt) : width(pt.x), height(pt.y) {}

template <typename T>
template <typename Tp2>
inline Size<T>::operator Size<Tp2>() const {
  return Size<Tp2>(saturate_cast<Tp2>(width), saturate_cast<Tp2>(height));
}

template <typename T>
inline T Size<T>::area() const {
  const T result = width * height;
  return result;
}

template <typename T>
inline double Size<T>::aspectRatio() const {
  return width / static_cast<double>(height);
}

template <typename T>
inline bool Size<T>::empty() const {
  return width <= 0 || height <= 0;
}

template <typename T>
static inline Size<T>& operator*=(Size<T>& a, T b) {
  a.width *= b;
  a.height *= b;
  return a;
}

template <typename T>
static inline Size<T> operator*(const Size<T>& a, T b) {
  Size<T> tmp(a);
  tmp *= b;
  return tmp;
}

template <typename T>
static inline Size<T>& operator/=(Size<T>& a, T b) {
  a.width /= b;
  a.height /= b;
  return a;
}

template <typename T>
static inline Size<T> operator/(const Size<T>& a, T b) {
  Size<T> tmp(a);
  tmp /= b;
  return tmp;
}

template <typename T>
static inline Size<T>& operator+=(Size<T>& a, const Size<T>& b) {
  a.width += b.width;
  a.height += b.height;
  return a;
}

template <typename T>
static inline Size<T> operator+(const Size<T>& a, const Size<T>& b) {
  Size<T> tmp(a);
  tmp += b;
  return tmp;
}

template <typename T>
static inline Size<T>& operator-=(Size<T>& a, const Size<T>& b) {
  a.width -= b.width;
  a.height -= b.height;
  return a;
}

template <typename T>
static inline Size<T> operator-(const Size<T>& a, const Size<T>& b) {
  Size<T> tmp(a);
  tmp -= b;
  return tmp;
}

template <typename T>
static inline bool operator==(const Size<T>& a, const Size<T>& b) {
  return a.width == b.width && a.height == b.height;
}

template <typename T>
static inline bool operator!=(const Size<T>& a, const Size<T>& b) {
  return !(a == b);
}

template <typename T>
inline Rect<T>::Rect() : x(0), y(0), width(0), height(0) {}

template <typename T>
inline Rect<T>::Rect(T _x, T _y, T _width, T _height)
    : x(_x), y(_y), width(_width), height(_height) {}

template <typename T>
inline Rect<T>::Rect(const Point<T>& org, const Size<T>& sz)
    : x(org.x), y(org.y), width(sz.width), height(sz.height) {}

template <typename T>
inline Rect<T>::Rect(const Point<T>& pt1, const Point<T>& pt2) {
  x = std::min(pt1.x, pt2.x);
  y = std::min(pt1.y, pt2.y);
  width = std::max(pt1.x, pt2.x) - x;
  height = std::max(pt1.y, pt2.y) - y;
}

template <typename T>
inline Point<T> Rect<T>::tl() const {
  return Point<T>(x, y);
}

template <typename T>
inline Point<T> Rect<T>::br() const {
  return Point<T>(x + width, y + height);
}

template <typename T>
inline Size<T> Rect<T>::size() const {
  return Size<T>(width, height);
}

template <typename T>
inline T Rect<T>::area() const {
  const T result = width * height;
  return result;
}

template <typename T>
inline bool Rect<T>::empty() const {
  return width <= 0 || height <= 0;
}

template <typename T>
template <typename Tp2>
inline Rect<T>::operator Rect<Tp2>() const {
  return Rect<Tp2>(saturate_cast<Tp2>(x), saturate_cast<Tp2>(y),
                   saturate_cast<Tp2>(width), saturate_cast<Tp2>(height));
}

template <typename T>
inline bool Rect<T>::contains(const Point<T>& pt) const {
  return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
}

template <typename T>
static inline Rect<T>& operator+=(Rect<T>& a, const Point<T>& b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

template <typename T>
static inline Rect<T>& operator-=(Rect<T>& a, const Point<T>& b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

template <typename T>
static inline Rect<T>& operator+=(Rect<T>& a, const Size<T>& b) {
  a.width += b.width;
  a.height += b.height;
  return a;
}

template <typename T>
static inline Rect<T>& operator-=(Rect<T>& a, const Size<T>& b) {
  const T width = a.width - b.width;
  const T height = a.height - b.height;
  a.width = width;
  a.height = height;
  return a;
}

template <typename T>
static inline Rect<T>& operator&=(Rect<T>& a, const Rect<T>& b) {
  if (a.empty() || b.empty()) {
    a = Rect();
    return a;
  }
  const Rect<T>& Rx_min = (a.x < b.x) ? a : b;
  const Rect<T>& Rx_max = (a.x < b.x) ? b : a;
  const Rect<T>& Ry_min = (a.y < b.y) ? a : b;
  const Rect<T>& Ry_max = (a.y < b.y) ? b : a;
  // Looking at the formula below, we will compute Rx_min.width - (Rx_max.x -
  // Rx_min.x) but we want to avoid overflows. Rx_min.width >= 0 and (Rx_max.x -
  // Rx_min.x) >= 0 by definition so the difference does not overflow. The only
  // thing that can overflow is (Rx_max.x - Rx_min.x). And it can only overflow
  // if Rx_min.x < 0. Let us first deal with the following case.
  if ((Rx_min.x < 0 && Rx_min.x + Rx_min.width < Rx_max.x) ||
      (Ry_min.y < 0 && Ry_min.y + Ry_min.height < Ry_max.y)) {
    a = Rect();
    return a;
  }
  // We now know that either Rx_min.x >= 0, or
  // Rx_min.x < 0 && Rx_min.x + Rx_min.width >= Rx_max.x and therefore
  // Rx_min.width >= (Rx_max.x - Rx_min.x) which means (Rx_max.x - Rx_min.x)
  // is inferior to a valid int and therefore does not overflow.
  a.width = std::min(Rx_min.width - (Rx_max.x - Rx_min.x), Rx_max.width);
  a.height = std::min(Ry_min.height - (Ry_max.y - Ry_min.y), Ry_max.height);
  a.x = Rx_max.x;
  a.y = Ry_max.y;
  if (a.empty()) a = Rect();
  return a;
}

template <typename T>
static inline Rect<T>& operator|=(Rect<T>& a, const Rect<T>& b) {
  if (a.empty()) {
    a = b;
  } else if (!b.empty()) {
    T x1 = std::min(a.x, b.x);
    T y1 = std::min(a.y, b.y);
    a.width = std::max(a.x + a.width, b.x + b.width) - x1;
    a.height = std::max(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
  }
  return a;
}

template <typename T>
static inline bool operator==(const Rect<T>& a, const Rect<T>& b) {
  return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template <typename T>
static inline bool operator!=(const Rect<T>& a, const Rect<T>& b) {
  return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

template <typename T>
static inline Rect<T> operator+(const Rect<T>& a, const Point<T>& b) {
  return Rect<T>(a.x + b.x, a.y + b.y, a.width, a.height);
}

template <typename T>
static inline Rect<T> operator-(const Rect<T>& a, const Point<T>& b) {
  return Rect<T>(a.x - b.x, a.y - b.y, a.width, a.height);
}

template <typename T>
static inline Rect<T> operator+(const Rect<T>& a, const Size<T>& b) {
  return Rect<T>(a.x, a.y, a.width + b.width, a.height + b.height);
}

template <typename T>
static inline Rect<T> operator-(const Rect<T>& a, const Size<T>& b) {
  const T width = a.width - b.width;
  const T height = a.height - b.height;
  return Rect<T>(a.x, a.y, width, height);
}

template <typename T>
static inline Rect<T> operator&(const Rect<T>& a, const Rect<T>& b) {
  Rect<T> c = a;
  return c &= b;
}

template <typename T>
static inline Rect<T> operator|(const Rect<T>& a, const Rect<T>& b) {
  Rect<T> c = a;
  return c |= b;
}

/**
 * @brief measure dissimilarity between two sample sets
 *
 * computes the complement of the Jaccard Index as described in
 * <https://en.wikipedia.org/wiki/Jaccard_index>. For rectangles this reduces to
 * computing the intersection over the union.
 */
template <typename T>
static inline double jaccardDistance(const Rect<T>& a, const Rect<T>& b) {
  T Aa = a.area();
  T Ab = b.area();

  if ((Aa + Ab) <= std::numeric_limits<T>::epsilon()) {
    // jaccard_index = 1 -> distance = 0
    return 0.0;
  }

  double Aab = (a & b).area();
  // distance = 1 - jaccard_index
  return 1.0 - Aab / (Aa + Ab - Aab);
}

/** @brief Finds out if there is any intersection between two rectangles
 *
 * mainly useful for language bindings
 * @param rect1 First rectangle
 * @param rect2 Second rectangle
 * @return the area of the intersection
 */
inline double rectangleIntersectionArea(const Rect2d& a, const Rect2d& b) {
  return (a & b).area();
}

inline Range::Range() : start(0), end(0) {}

inline Range::Range(int _start, int _end) : start(_start), end(_end) {}

inline int Range::size() const { return end - start; }

inline bool Range::empty() const { return start == end; }

inline Range Range::all() { return Range(INT_MIN, INT_MAX); }

static inline bool operator==(const Range& r1, const Range& r2) {
  return r1.start == r2.start && r1.end == r2.end;
}

static inline bool operator!=(const Range& r1, const Range& r2) {
  return !(r1 == r2);
}

static inline bool operator!(const Range& r) { return r.start == r.end; }

static inline Range operator&(const Range& r1, const Range& r2) {
  Range r(std::max(r1.start, r2.start), std::min(r1.end, r2.end));
  r.end = std::max(r.end, r.start);
  return r;
}

static inline Range& operator&=(Range& r1, const Range& r2) {
  r1 = r1 & r2;
  return r1;
}

static inline Range operator+(const Range& r1, int delta) {
  return Range(r1.start + delta, r1.end + delta);
}

static inline Range operator+(int delta, const Range& r1) {
  return Range(r1.start + delta, r1.end + delta);
}

static inline Range operator-(const Range& r1, int delta) {
  return r1 + (-delta);
}

template <typename T>
inline Scalar<T>::Scalar() {
  this->val[0] = this->val[1] = this->val[2] = this->val[3] = 0;
}

template <typename T>
inline Scalar<T>::Scalar(T v0, T v1, T v2, T v3) {
  this->val[0] = v0;
  this->val[1] = v1;
  this->val[2] = v2;
  this->val[3] = v3;
}

template <typename T>
inline Scalar<T>::Scalar(Scalar<T>&& s) {
  this->val[0] = std::move(s.val[0]);
  this->val[1] = std::move(s.val[1]);
  this->val[2] = std::move(s.val[2]);
  this->val[3] = std::move(s.val[3]);
}

template <typename T>
inline Scalar<T>& Scalar<T>::operator=(const Scalar<T>& s) {
  this->val[0] = s.val[0];
  this->val[1] = s.val[1];
  this->val[2] = s.val[2];
  this->val[3] = s.val[3];
  return *this;
}

template <typename T>
inline Scalar<T>& Scalar<T>::operator=(Scalar<T>&& s) {
  this->val[0] = std::move(s.val[0]);
  this->val[1] = std::move(s.val[1]);
  this->val[2] = std::move(s.val[2]);
  this->val[3] = std::move(s.val[3]);
  return *this;
}

template <typename T>
inline Scalar<T>::Scalar(T v0) {
  this->val[0] = v0;
  this->val[1] = this->val[2] = this->val[3] = 0;
}

template <typename T>
inline Scalar<T> Scalar<T>::all(T v0) {
  return Scalar<T>(v0, v0, v0, v0);
}

template <typename T>
inline Scalar<T> Scalar<T>::mul(const Scalar<T>& a, double scale) const {
  return Scalar<T>(saturate_cast<T>(this->val[0] * a.val[0] * scale),
                   saturate_cast<T>(this->val[1] * a.val[1] * scale),
                   saturate_cast<T>(this->val[2] * a.val[2] * scale),
                   saturate_cast<T>(this->val[3] * a.val[3] * scale));
}

template <typename T>
inline Scalar<T> Scalar<T>::conj() const {
  return Scalar<T>(
      saturate_cast<T>(this->val[0]), saturate_cast<T>(-this->val[1]),
      saturate_cast<T>(-this->val[2]), saturate_cast<T>(-this->val[3]));
}

template <typename T>
inline bool Scalar<T>::isReal() const {
  return this->val[1] == 0 && this->val[2] == 0 && this->val[3] == 0;
}

template <typename T>
template <typename T2>
inline Scalar<T>::operator Scalar<T2>() const {
  return Scalar<T2>(
      saturate_cast<T2>(this->val[0]), saturate_cast<T2>(this->val[1]),
      saturate_cast<T2>(this->val[2]), saturate_cast<T2>(this->val[3]));
}

template <typename T>
static inline Scalar<T>& operator+=(Scalar<T>& a, const Scalar<T>& b) {
  a.val[0] += b.val[0];
  a.val[1] += b.val[1];
  a.val[2] += b.val[2];
  a.val[3] += b.val[3];
  return a;
}

template <typename T>
static inline Scalar<T>& operator-=(Scalar<T>& a, const Scalar<T>& b) {
  a.val[0] -= b.val[0];
  a.val[1] -= b.val[1];
  a.val[2] -= b.val[2];
  a.val[3] -= b.val[3];
  return a;
}

template <typename T>
static inline Scalar<T>& operator*=(Scalar<T>& a, T v) {
  a.val[0] *= v;
  a.val[1] *= v;
  a.val[2] *= v;
  a.val[3] *= v;
  return a;
}

template <typename T>
static inline bool operator==(const Scalar<T>& a, const Scalar<T>& b) {
  return a.val[0] == b.val[0] && a.val[1] == b.val[1] && a.val[2] == b.val[2] &&
         a.val[3] == b.val[3];
}

template <typename T>
static inline bool operator!=(const Scalar<T>& a, const Scalar<T>& b) {
  return a.val[0] != b.val[0] || a.val[1] != b.val[1] || a.val[2] != b.val[2] ||
         a.val[3] != b.val[3];
}

template <typename T>
static inline Scalar<T> operator+(const Scalar<T>& a, const Scalar<T>& b) {
  return Scalar<T>(a.val[0] + b.val[0], a.val[1] + b.val[1],
                   a.val[2] + b.val[2], a.val[3] + b.val[3]);
}

template <typename T>
static inline Scalar<T> operator-(const Scalar<T>& a, const Scalar<T>& b) {
  return Scalar<T>(saturate_cast<T>(a.val[0] - b.val[0]),
                   saturate_cast<T>(a.val[1] - b.val[1]),
                   saturate_cast<T>(a.val[2] - b.val[2]),
                   saturate_cast<T>(a.val[3] - b.val[3]));
}

template <typename T>
static inline Scalar<T> operator*(const Scalar<T>& a, T alpha) {
  return Scalar<T>(a.val[0] * alpha, a.val[1] * alpha, a.val[2] * alpha,
                   a.val[3] * alpha);
}

template <typename T>
static inline Scalar<T> operator*(T alpha, const Scalar<T>& a) {
  return a * alpha;
}

template <typename T>
static inline Scalar<T> operator-(const Scalar<T>& a) {
  return Scalar<T>(saturate_cast<T>(-a.val[0]), saturate_cast<T>(-a.val[1]),
                   saturate_cast<T>(-a.val[2]), saturate_cast<T>(-a.val[3]));
}

template <typename T>
static inline Scalar<T> operator*(const Scalar<T>& a, const Scalar<T>& b) {
  return Scalar<T>(
      saturate_cast<T>(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]),
      saturate_cast<T>(a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]),
      saturate_cast<T>(a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]),
      saturate_cast<T>(a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]));
}

template <typename T>
static inline Scalar<T>& operator*=(Scalar<T>& a, const Scalar<T>& b) {
  a = a * b;
  return a;
}

template <typename T>
static inline Scalar<T> operator/(const Scalar<T>& a, T alpha) {
  return Scalar<T>(a.val[0] / alpha, a.val[1] / alpha, a.val[2] / alpha,
                   a.val[3] / alpha);
}

template <typename T>
static inline Scalar<float> operator/(const Scalar<float>& a, float alpha) {
  float s = 1 / alpha;
  return Scalar<float>(a.val[0] * s, a.val[1] * s, a.val[2] * s, a.val[3] * s);
}

template <typename T>
static inline Scalar<double> operator/(const Scalar<double>& a, double alpha) {
  double s = 1 / alpha;
  return Scalar<double>(a.val[0] * s, a.val[1] * s, a.val[2] * s, a.val[3] * s);
}

template <typename T>
static inline Scalar<T>& operator/=(Scalar<T>& a, T alpha) {
  a = a / alpha;
  return a;
}

template <typename T>
static inline Scalar<T> operator/(T a, const Scalar<T>& b) {
  T s = a / (b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]);
  return b.conj() * s;
}

template <typename T>
static inline Scalar<T> operator/(const Scalar<T>& a, const Scalar<T>& b) {
  return a * ((T)1 / b);
}

template <typename T>
static inline Scalar<T>& operator/=(Scalar<T>& a, const Scalar<T>& b) {
  a = a / b;
  return a;
}

}  // namespace base
}  // namespace nndeploy

#endif
