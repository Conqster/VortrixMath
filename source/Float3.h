#pragma once
#include "Core.h"


namespace vx
{

	struct Float3
	{
		Float3() : x(0), y(0), z(0) {}
		explicit Float3(float v) : x(v), y(v), z(v) {}
		Float3(const Float3& rhs) = default;
		Float3& operator=(const Float3& rhs) = default;
		constexpr Float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z){}

		float& operator[](uint32 i)
		{
			VX_ASSERT(i < 3, "");
			return (&x)[i];
		}

		const float& operator[](uint32 i) const
		{
			VX_ASSERT(i < 3, "");
			return (&x)[i];
		}

		bool operator ==(const Float3& rhs) const
		{
			return x == rhs.x &&
					y == rhs.y &&
					z == rhs.z;
		}

		bool operator !=(const Float3& rhs) const { return !(*this == rhs); }


		VX_INLINE friend std::ostream& operator<<(std::ostream& os, const Float3& v)
		{
			os << "Float3(" << v.x << ", " << v.y << ", " << v.z << ")";
			return os;
		}

		float x, y, z;
	};

	static_assert(std::is_standard_layout_v<Float3>);
	static_assert(sizeof(Float3) == 12);
	static_assert(alignof(Float3) == alignof(float));
	static_assert(std::is_trivially_copyable_v<Float3>);
}