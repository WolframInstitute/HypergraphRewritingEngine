// The initial-condition generators of hgcommon/initial_condition_core.hpp against the outputs of
// the Wolfram Language generators they replace, recorded in reference/initial_condition_cases.txt
// from the paclet at commit a30399a2.
//
// Lattice cases ("RandomizeDirections" -> False) compare the edge list exactly and the
// coordinates to 1e-12. Point-set cases take the points the Wolfram Language generator drew and
// compare the edges the C++ join builds from them, since the two draw from different random
// streams.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "hgcommon/initial_condition_core.hpp"
#include "initial_condition_job.hpp"

namespace {

namespace ic = hgcommon;

struct Case {
    std::vector<uint32_t> edges;
    uint32_t coord_dim = 0;
    std::vector<double> coords;
    uint32_t point_dim = 0;
    std::vector<double> points;
};

std::map<std::string, Case> load_cases() {
    std::map<std::string, Case> out;
    std::ifstream in(std::string(HG_SOURCE_DIR) + "/reference/initial_condition_cases.txt");
    std::string line, name;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "case") { ss >> name; continue; }
        Case& c = out[name];
        if (tag == "edges") {
            size_t k = 0;
            ss >> k;
            c.edges.resize(2 * k);
            for (auto& v : c.edges) ss >> v;
        } else if (tag == "coords" || tag == "points") {
            uint32_t dim = 0;
            size_t k = 0;
            ss >> dim >> k;
            std::vector<double> v(size_t(dim) * k);
            for (auto& x : v) {
                std::string t;
                ss >> t;
                x = std::strtod(t.c_str(), nullptr);
            }
            if (tag == "coords") { c.coord_dim = dim; c.coords = v; }
            else { c.point_dim = dim; c.points = v; }
        }
    }
    return out;
}

const std::map<std::string, Case>& cases() {
    static const std::map<std::string, Case> c = load_cases();
    return c;
}

ic::IcSpec lattice(ic::IcType t) {
    ic::IcSpec s;
    s.type = t;
    s.randomize_directions = false;
    return s;
}

void expect_lattice(const std::string& name, const ic::IcSpec& s) {
    const auto it = cases().find(name);
    ASSERT_NE(it, cases().end()) << name << " is missing from reference/initial_condition_cases.txt";
    const Case& c = it->second;
    const ic::IcGraph g = ic::ic_generate(s);
    EXPECT_TRUE(g.error.empty()) << name << ": " << g.error;
    EXPECT_EQ(g.edges, c.edges) << name;
    ASSERT_EQ(g.coord_dim, c.coord_dim) << name;
    ASSERT_EQ(g.coords.size(), c.coords.size()) << name;
    for (size_t i = 0; i < c.coords.size(); ++i)
        EXPECT_NEAR(g.coords[i], c.coords[i], 1e-12) << name << " coordinate " << i;
}

}  // namespace

TEST(InitialConditions, GridMatchesTheRecordedOutput) {
    ic::IcSpec s = lattice(ic::IcType::Grid);
    s.width = 5; s.height = 4;
    expect_lattice("grid_5x4", s);
    s.diagonals = true;
    expect_lattice("grid_5x4_diagonals", s);
}

TEST(InitialConditions, GridWithHolesMatchesTheRecordedOutput) {
    ic::IcSpec s = lattice(ic::IcType::GridWithHoles);
    s.width = 8; s.height = 7; s.diagonals = true;
    s.holes = {{3, 3, 1.5}, {6, 2, 1.0}};
    expect_lattice("grid_with_holes_8x7", s);
}

TEST(InitialConditions, SurfacesMatchTheRecordedOutput) {
    ic::IcSpec s = lattice(ic::IcType::Cylinder);
    s.resolution = 6; s.height = 4;
    expect_lattice("cylinder_6x4", s);
    s = lattice(ic::IcType::Torus);
    s.resolution = 5;
    expect_lattice("torus_5", s);
    s = lattice(ic::IcType::Sphere);
    s.resolution = 4;
    expect_lattice("sphere_4", s);
    s.resolution = 7;
    expect_lattice("sphere_7", s);
    s = lattice(ic::IcType::KleinBottle);
    s.resolution = 6; s.height = 4;
    expect_lattice("klein_6x4", s);
    s = lattice(ic::IcType::MobiusStrip);
    s.resolution = 6; s.width = 3; s.radius = 2.0;
    expect_lattice("mobius_6x3", s);
}

TEST(InitialConditions, MinkowskiLinksMatchTheRecordedOutput) {
    struct Row { const char* name; uint32_t d; bool reduce; uint32_t cap; };
    for (const Row& r : {Row{"minkowski_60_d1", 1, true, 50}, Row{"minkowski_60_d2", 2, true, 50},
                         Row{"minkowski_60_d2_unreduced_cap3", 2, false, 3}}) {
        const Case& c = cases().at(r.name);
        ASSERT_EQ(c.point_dim, r.d + 1) << r.name;
        ic::IcSpec s;
        s.type = ic::IcType::MinkowskiSprinkling;
        s.spatial_dim = r.d;
        s.transitivity_reduction = r.reduce;
        s.max_edges_per_vertex = r.cap;
        ic::IcGraph g;
        g.spacetime = c.points;
        ic::ic_minkowski_links(s, g);
        EXPECT_EQ(g.edges, c.edges) << r.name;
        EXPECT_GT(g.edges.size(), 0u) << r.name;
    }
}

TEST(InitialConditions, BrillLindquistJoinMatchesTheRecordedOutput) {
    struct Row { const char* name; uint32_t d; bool periodic; };
    for (const Row& r : {Row{"brill_lindquist_150_d2", 2, false},
                         Row{"brill_lindquist_150_d2_periodic", 2, true},
                         Row{"brill_lindquist_150_d3", 3, false},
                         Row{"brill_lindquist_150_d3_periodic", 3, true}}) {
        const Case& c = cases().at(r.name);
        ASSERT_EQ(c.point_dim, r.d) << r.name;
        ic::IcSpec s;
        s.type = ic::IcType::BrillLindquist;
        s.dimension = r.d;
        s.periodic = r.periodic;
        s.mass1 = 3.0; s.mass2 = 2.0; s.separation = 10.0;
        for (auto& b : s.box) { b[0] = -15.0; b[1] = 15.0; }
        s.randomize_directions = false;
        ic::IcGraph g;
        ic::IcRng rng(0);
        ic::ic_brill_lindquist_join(s, c.points, rng, g);
        EXPECT_EQ(g.edges, c.edges) << r.name;
        EXPECT_GT(g.edges.size(), 0u) << r.name;
    }
}

TEST(InitialConditions, PlanarJoinsMatchTheRecordedOutput) {
    struct Row { const char* name; double threshold; };
    for (const Row& r : {Row{"poisson_disk_80", 1.6}, Row{"uniform_random_100", 1.5}}) {
        const Case& c = cases().at(r.name);
        ASSERT_EQ(c.point_dim, 2u) << r.name;
        ic::IcSpec s;
        s.randomize_directions = false;
        ic::IcGraph g;
        ic::IcRng rng(0);
        ic::ic_join_planar(g, rng, s, c.points, r.threshold);
        EXPECT_EQ(g.edges, c.edges) << r.name;
        EXPECT_GT(g.edges.size(), 0u) << r.name;
    }
}

// A seed gives one output; the point-set generators keep their counts and constraints.
TEST(InitialConditions, PointSetsAreSeededAndKeepTheirConstraints) {
    ic::IcSpec s;
    s.type = ic::IcType::PoissonDisk;
    s.n = 80; s.min_distance = 0.8; s.seed = 5;
    const ic::IcGraph a = ic::ic_generate(s), b = ic::ic_generate(s);
    EXPECT_EQ(a.edges, b.edges);
    EXPECT_EQ(a.coords, b.coords);
    for (uint32_t i = 0; i < a.n_vertices; ++i)
        for (uint32_t j = i + 1; j < a.n_vertices; ++j)
            EXPECT_GE(std::hypot(a.coords[2 * i] - a.coords[2 * j], a.coords[2 * i + 1] - a.coords[2 * j + 1]), 0.8);
    s.seed = 6;
    EXPECT_NE(ic::ic_generate(s).coords, a.coords);

    ic::IcSpec bl;
    bl.type = ic::IcType::BrillLindquist;
    bl.n = 300; bl.seed = 3; bl.dimension = 3;
    for (auto& x : bl.box) { x[0] = -15.0; x[1] = 15.0; }
    const ic::IcGraph g = ic::ic_generate(bl);
    EXPECT_EQ(g.n_vertices, 300u);
    const ic::IcBlGeometry geo(bl);
    for (uint32_t i = 0; i < g.n_vertices; ++i) EXPECT_FALSE(geo.inside(&g.coords[3 * size_t(i)]));

    ic::IcSpec m;
    m.type = ic::IcType::MinkowskiSprinkling;
    m.n = 50; m.seed = 1;
    const ic::IcGraph mk = ic::ic_generate(m);
    for (uint32_t i = 1; i < 50; ++i) EXPECT_LE(mk.spacetime[3 * size_t(i - 1)], mk.spacetime[3 * size_t(i)]);
    EXPECT_GT(mk.edges.size(), 0u);
}

// The "InitialCondition" job's spec (paclet_source/initial_condition_job.hpp): integers stand for
// reals, a "Grid" with holes is a "GridWithHoles", an absent key keeps the type's default, and an
// unknown type or key is refused.
TEST(InitialConditions, TheJobSpecReadsEveryKeyAndRefusesUnknownOnes) {
    using wxf::WXFValue;
    using wxf::WXFValueList;
    auto spec_of = [](wxf::WXFValueAssociation a) {
        const WXFValue v(std::move(a));
        wxf::ValueParser p(v);
        return hg::ffi::read_initial_condition_spec(p);
    };
    const ic::IcSpec g = spec_of({{WXFValue("Type"), WXFValue("Grid")},
                                  {WXFValue("Width"), WXFValue(int64_t{7})},
                                  {WXFValue("Holes"), WXFValue(WXFValueList{WXFValue(WXFValueList{
                                       WXFValue(int64_t{3}), WXFValue(int64_t{3}), WXFValue(1.5)})})},
                                  {WXFValue("RandomizeDirections"), WXFValue(wxf::Symbol{"False"})}});
    EXPECT_EQ(g.type, ic::IcType::GridWithHoles);
    EXPECT_EQ(g.width, 7u);
    EXPECT_EQ(g.height, 10u);
    ASSERT_EQ(g.holes.size(), 1u);
    EXPECT_EQ(g.holes[0].r, 1.5);
    EXPECT_FALSE(g.randomize_directions);

    const ic::IcSpec bl = spec_of({{WXFValue("Type"), WXFValue("BrillLindquist")},
                                   {WXFValue("Mass1"), WXFValue(int64_t{2})},
                                   {WXFValue("BoxX"), WXFValue(WXFValueList{WXFValue(int64_t{-5}), WXFValue(int64_t{5})})},
                                   {WXFValue("Seed"), WXFValue(int64_t{9})}});
    EXPECT_EQ(bl.type, ic::IcType::BrillLindquist);
    EXPECT_EQ(bl.mass1, 2.0);
    EXPECT_EQ(bl.box[0][0], -5.0);
    EXPECT_EQ(bl.box[1][0], -15.0);
    EXPECT_EQ(bl.seed, 9u);

    EXPECT_THROW(spec_of({{WXFValue("Type"), WXFValue("Hexagon")}}), std::runtime_error);
    EXPECT_THROW(spec_of({{WXFValue("Type"), WXFValue("Torus")}, {WXFValue("Colour"), WXFValue(int64_t{1})}}),
                 std::runtime_error);
    EXPECT_THROW(spec_of({{WXFValue("Type"), WXFValue("Torus")}, {WXFValue("BoxX"), WXFValue(WXFValueList{WXFValue(1.0)})}}),
                 std::runtime_error);
}
