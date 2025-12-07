#pragma once

#if defined (_MSC_VER)
#define VX_FORCE_INLINE __forceinline
#else
#define VX_FORCE_INLINE inline
#endif // defined (_MSVC_VER)

#define VX_USE_SSE 1

#if defined(VX_USE_SSE)
#include <xmmintrin.h>
#endif // defined(USE_SIMD_SSE)


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


#include <iostream>


//forward declare
namespace vx {
	class Vec3;
	class Vec4;
}