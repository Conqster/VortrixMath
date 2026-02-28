#pragma once

#if defined (_MSC_VER)
#define VX_INLINE __forceinline
#else
#define VX_INLINE inline
#endif // defined (_MSVC_VER)

#include <iostream>
#if defined(VX_USE_SSE)
#include <xmmintrin.h>

#endif // defined(USE_SIMD_SSE)

#define VX_ASSERT(expr, ...) \
		do { if(!(expr)) { \
				std::cout << "Assertion Failed (" << #expr << "): \nMessage: " << __VA_ARGS__ << ".\nFile: " << __FILE__ << ".\nLine: " << __LINE__ << ".\n"; \
				__debugbreak(); \
		 } } while (0)


using uint8 = std::uint8_t;
using uint32 = std::uint32_t;




//forward declare
namespace vx {

	enum class Axis : std::uint8_t;

	struct Float3;

	class Vec2;
	class Vec3;
	class Vec4;

	class Quat;

	class Mat44;
}

