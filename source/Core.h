#pragma once

#if defined (_MSC_VER)
#define VX_FORCE_INLINE __forceinline
#else
#define VX_FORCE_INLINE inline
#endif // defined (_MSVC_VER)

#if defined(VX_USE_SSE)
#include <xmmintrin.h>

#include <iostream>

#define VX_ASSERT(expr, ...) \
		do { if(!(expr)) { \
				std::cout << "Assertion Failed (" << #expr << "): \nMessage: " << __VA_ARGS__ << ".\nFile: " << __FILE__ << ".\nLine: " << __LINE__ << ".\n"; \
				__debugbreak(); \
		 } } while (0)



static float Get_m128Lane(const __m128& v, int idx)
{
	switch (idx)
	{
	case 0: return _mm_cvtss_f32(v);
	case 1: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
	case 2: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)));
	case 3: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)));
	default: return _mm_cvtss_f32(v);
	}
}

#endif // defined(USE_SIMD_SSE)



//forward declare
namespace vx {
	class Vec3;
	class Vec4;
}


