#include "Mat44.h"

std::ostream& vx::operator<<(std::ostream& os, const Mat44& m)
{
    for (const auto& c : m.mCol)
        os << "\n|" << c.X() << " " << c.Y() << " " << c.Z() << " " << c.W() << "|";

    os << '\n';
    return os;
}
