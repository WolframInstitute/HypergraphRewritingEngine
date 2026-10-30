#pragma once
#include "hgcommon/namespace.hpp"
// GENERATED INITIAL CONDITIONS, one body for the CPU worker, the GPU worker, the paclet library
// and the Python module.
//
// Each generator returns a graph whose vertices are numbered 1..n, its binary edges in a fixed
// order, and the vertices' coordinates. The random draws come from one splitmix64 stream seeded
// by IcSpec::seed, so a spec gives the same edges on every platform: every test that decides an
// edge uses IEEE arithmetic and sqrt, both correctly rounded. Coordinates that go through sin and
// cos can differ in the last bit between C libraries; no edge depends on them.
//
// "RandomizeDirections" reverses each edge with probability 1/2, one draw per edge in edge order.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

enum class IcType : uint32_t {
    Grid, GridWithHoles, Cylinder, Torus, Sphere, KleinBottle, MobiusStrip,
    MinkowskiSprinkling, BrillLindquist, PoissonDisk, UniformRandom
};

struct IcHole { double cx, cy, r; };

// Every parameter of every generator; each generator reads its own. The defaults are the
// paclet's option defaults.
struct IcSpec {
    IcType type = IcType::Grid;
    uint64_t seed = 0;
    bool randomize_directions = true;
    // Grid, GridWithHoles
    uint32_t width = 10, height = 10;
    bool diagonals = false;
    std::vector<IcHole> holes;
    // Cylinder, Torus, Sphere, KleinBottle, MobiusStrip
    uint32_t resolution = 10;
    double radius = 1.0;            // Cylinder, Sphere, KleinBottle 1.0; MobiusStrip 2.0
    double major_radius = 3.0, minor_radius = 1.0;
    // Point sets
    uint32_t n = 500;               // "Density": the number of points
    double box[3][2] = {{0, 10}, {0, 10}, {-15, 15}};
    double edge_threshold = -1;     // < 0: the generator's automatic threshold
    // MinkowskiSprinkling
    uint32_t spatial_dim = 2;
    double time_extent = 10, spatial_extent = 10, lightcone = 1, alexandrov = 5;
    bool transitivity_reduction = true;
    uint32_t max_edges_per_vertex = 50;
    // BrillLindquist
    double mass1 = 3, mass2 = 3, separation = 10;
    uint32_t dimension = 2;
    bool periodic = false;
    // PoissonDisk
    double min_distance = 1;
};

// The type a name selects: the names HGEvolve accepts for a generated initial condition.
inline bool ic_type_from_name(const std::string& name, IcType& out) {
    static const struct { const char* name; IcType type; } kNames[] = {
        {"Grid", IcType::Grid}, {"GridWithHoles", IcType::GridWithHoles},
        {"Cylinder", IcType::Cylinder}, {"Torus", IcType::Torus}, {"Sphere", IcType::Sphere},
        {"Klein", IcType::KleinBottle}, {"KleinBottle", IcType::KleinBottle},
        {"Mobius", IcType::MobiusStrip}, {"MobiusStrip", IcType::MobiusStrip},
        {"Sprinkling", IcType::MinkowskiSprinkling}, {"Minkowski", IcType::MinkowskiSprinkling},
        {"MinkowskiSprinkling", IcType::MinkowskiSprinkling},
        {"BrillLindquist", IcType::BrillLindquist},
        {"Poisson", IcType::PoissonDisk}, {"PoissonDisk", IcType::PoissonDisk},
        {"Uniform", IcType::UniformRandom}, {"UniformRandom", IcType::UniformRandom}};
    for (const auto& k : kNames)
        if (name == k.name) { out = k.type; return true; }
    return false;
}

// The defaults of a type: HGEvolve's option defaults ("GridWidth" 10, "GridHeight" 10,
// "SprinklingDensity" 500, "PoissonMinDistance" 1.0, the "Sprinkling..." and "BrillLindquist..."
// options), a Mobius strip 5 wide with radius 2, the Brill-Lindquist box [-15, 15] per axis and
// the point-set box [0, 10]^2.
inline IcSpec ic_spec_for(IcType t) {
    IcSpec s;
    s.type = t;
    s.n = 500;
    if (t == IcType::MobiusStrip) { s.width = 5; s.radius = 2.0; }
    if (t == IcType::BrillLindquist)
        for (auto& b : s.box) { b[0] = -15.0; b[1] = 15.0; }
    return s;
}

struct IcGraph {
    uint32_t n_vertices = 0;
    std::vector<uint32_t> edges;     // 2 per edge, vertices 1..n_vertices
    uint32_t coord_dim = 0;          // "VertexCoordinates": coord_dim values per vertex
    std::vector<double> coords;
    uint32_t coord3_dim = 0;         // "VertexCoordinates3D": 0 or 3 values per vertex
    std::vector<double> coords3;
    std::vector<double> spacetime;   // MinkowskiSprinkling: {t, x_1..x_d} per vertex, by t
    double threshold = 0;            // the edge threshold used (point sets)
    double dimension_estimate = 0;   // MinkowskiSprinkling
    std::vector<double> horizon_centers;  // BrillLindquist: coord_dim values per horizon, 2 horizons
    std::vector<double> horizon_radii;    // BrillLindquist: 2 values
    std::string error;               // non-empty: the spec is invalid and nothing was generated
};

// One splitmix64 stream: state advances by the golden-ratio increment, each draw is the
// finalizer of the state. uniform() has 53 random bits in [0, 1).
struct IcRng {
    uint64_t s;
    explicit IcRng(uint64_t seed) : s(seed) {}
    uint64_t next() { s += 0x9e3779b97f4a7c15ULL; return splitmix64(s); }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
};

inline const double kIcPi = 3.14159265358979323846;

// Round half to even, as the Wolfram Language's Round.
inline long long ic_round_even(double x) { return static_cast<long long>(std::nearbyint(x)); }

// Round[num / den] half to even, for den > 0 and num >= 0, exactly.
inline uint64_t ic_round_ratio(uint64_t num, uint64_t den) {
    const uint64_t q = num / den, r = num % den;
    if (2 * r > den || (2 * r == den && (q & 1))) return q + 1;
    return q;
}

inline void ic_edge(IcGraph& g, IcRng& rng, bool randomize, uint32_t a, uint32_t b) {
    if (randomize && rng.uniform() < 0.5) std::swap(a, b);
    g.edges.push_back(a);
    g.edges.push_back(b);
}

// Index of the lattice vertex (i, j), 1-based, i outer: the order the paclet numbered them.
inline uint32_t ic_lattice(uint32_t i, uint32_t j, uint32_t rows) { return (i - 1) * rows + j; }

inline IcGraph ic_grid(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t w = s.width, h = s.height;
    g.n_vertices = w * h; g.coord_dim = 2;
    for (uint32_t i = 1; i <= w; ++i)
        for (uint32_t j = 1; j <= h; ++j) { g.coords.push_back(i - 1.0); g.coords.push_back(j - 1.0); }
    for (uint32_t i = 1; i <= w; ++i)
        for (uint32_t j = 1; j <= h; ++j) {
            const uint32_t v = ic_lattice(i, j, h);
            if (i < w) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i + 1, j, h));
            if (j < h) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i, j + 1, h));
            if (s.diagonals) {
                if (i < w && j < h) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i + 1, j + 1, h));
                if (i < w && j > 1) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i + 1, j - 1, h));
            }
        }
    return g;
}

inline IcGraph ic_grid_with_holes(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t w = s.width, h = s.height;
    auto inside = [&](double x, double y) {
        for (const IcHole& o : s.holes)
            if ((x - o.cx) * (x - o.cx) + (y - o.cy) * (y - o.cy) < o.r * o.r) return true;
        return false;
    };
    std::vector<uint32_t> index(static_cast<size_t>(w) * h + 1, 0);   // 0: in a hole
    g.coord_dim = 2;
    for (uint32_t i = 1; i <= w; ++i)
        for (uint32_t j = 1; j <= h; ++j)
            if (!inside(i - 1.0, j - 1.0)) {
                index[ic_lattice(i, j, h)] = ++g.n_vertices;
                g.coords.push_back(i - 1.0); g.coords.push_back(j - 1.0);
            }
    auto join = [&](uint32_t i, uint32_t j, uint32_t i2, uint32_t j2) {
        const uint32_t a = index[ic_lattice(i, j, h)], b = index[ic_lattice(i2, j2, h)];
        if (b == 0) return;
        const double mx = ((i - 1.0) + (i2 - 1.0)) / 2, my = ((j - 1.0) + (j2 - 1.0)) / 2;
        if (!inside(mx, my)) ic_edge(g, rng, s.randomize_directions, a, b);
    };
    for (uint32_t i = 1; i <= w; ++i)
        for (uint32_t j = 1; j <= h; ++j) {
            if (index[ic_lattice(i, j, h)] == 0) continue;
            if (i < w) join(i, j, i + 1, j);
            if (j < h) join(i, j, i, j + 1);
            if (s.diagonals) {
                if (i < w && j < h) join(i, j, i + 1, j + 1);
                if (i < w && j > 1) join(i, j, i + 1, j - 1);
            }
        }
    return g;
}

inline IcGraph ic_cylinder(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t res = s.resolution, h = s.height;
    g.n_vertices = res * h; g.coord_dim = 2; g.coord3_dim = 3;
    for (uint32_t i = 1; i <= res; ++i)
        for (uint32_t j = 1; j <= h; ++j) {
            const double theta = (i - 1) * (2 * kIcPi / res), z = j - 1.0;
            g.coords.insert(g.coords.end(), {theta, z});
            g.coords3.insert(g.coords3.end(), {s.radius * std::cos(theta), s.radius * std::sin(theta), z});
        }
    for (uint32_t i = 1; i <= res; ++i)
        for (uint32_t j = 1; j <= h; ++j) {
            const uint32_t v = ic_lattice(i, j, h);
            ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i % res + 1, j, h));
            if (j < h) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i, j + 1, h));
        }
    return g;
}

inline IcGraph ic_torus(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t res = s.resolution;
    g.n_vertices = res * res; g.coord_dim = 2; g.coord3_dim = 3;
    for (uint32_t i = 1; i <= res; ++i)
        for (uint32_t j = 1; j <= res; ++j) {
            const double theta = (i - 1) * 2 * kIcPi / res, phi = (j - 1) * 2 * kIcPi / res;
            const double rho = s.major_radius + s.minor_radius * std::cos(phi);
            g.coords.insert(g.coords.end(), {theta, phi});
            g.coords3.insert(g.coords3.end(),
                             {rho * std::cos(theta), rho * std::sin(theta), s.minor_radius * std::sin(phi)});
        }
    for (uint32_t i = 1; i <= res; ++i)
        for (uint32_t j = 1; j <= res; ++j) {
            const uint32_t v = ic_lattice(i, j, res);
            ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i % res + 1, j, res));
            ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i, j % res + 1, res));
        }
    return g;
}

// A UV sphere: resolution latitude rings at theta = pi (i - 1/2) / resolution, ring i holding
// max(1, Round[2 resolution sin theta]) vertices. Each vertex joins the next on its ring and the
// vertex of the next ring at the same fraction of a turn, rounded.
inline IcGraph ic_sphere(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t n_lat = s.resolution, n_lon = 2 * s.resolution;
    std::vector<uint32_t> lon(n_lat), off(n_lat, 0);
    g.coord_dim = 2; g.coord3_dim = 3;
    for (uint32_t i = 1; i <= n_lat; ++i) {
        const double theta = kIcPi * (i - 0.5) / n_lat, st = std::sin(theta);
        lon[i - 1] = static_cast<uint32_t>(std::max<long long>(1, ic_round_even(n_lon * st)));
        if (i > 1) off[i - 1] = off[i - 2] + lon[i - 2];
        for (uint32_t j = 1; j <= lon[i - 1]; ++j) {
            const double phi = 2 * kIcPi * (j - 1) / lon[i - 1];
            g.coords.insert(g.coords.end(), {theta, phi});
            g.coords3.insert(g.coords3.end(), {s.radius * st * std::cos(phi), s.radius * st * std::sin(phi),
                                               s.radius * std::cos(theta)});
        }
    }
    g.n_vertices = n_lat ? off[n_lat - 1] + lon[n_lat - 1] : 0;
    for (uint32_t li = 1; li <= n_lat; ++li) {
        const uint32_t cur = lon[li - 1];
        for (uint32_t k = 1; k <= cur; ++k) {
            const uint32_t v = off[li - 1] + k;
            ic_edge(g, rng, s.randomize_directions, v, off[li - 1] + k % cur + 1);
            if (li < n_lat) {
                const uint32_t next = lon[li];
                const uint32_t cj = std::min<uint64_t>(ic_round_ratio(uint64_t(k - 1) * next, cur) + 1, next);
                ic_edge(g, rng, s.randomize_directions, v, off[li] + cj);
            }
        }
    }
    return g;
}

// KleinBottle (rows = height, coordinates {radius theta, z}) and MobiusStrip (rows = width): a
// lattice whose last column joins the first with the row order reversed.
inline IcGraph ic_twisted(const IcSpec& s, bool mobius) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t res = s.resolution, rows = mobius ? s.width : s.height;
    g.n_vertices = res * rows; g.coord_dim = 2; g.coord3_dim = mobius ? 3 : 0;
    for (uint32_t i = 1; i <= res; ++i)
        for (uint32_t j = 1; j <= rows; ++j) {
            const double theta = (i - 1) * 2 * kIcPi / res;
            if (!mobius) {
                g.coords.insert(g.coords.end(), {s.radius * theta, j - 1.0});
                continue;
            }
            const double w = rows > 1 ? (j - 1.0) / (rows - 1) - 0.5 : -0.5;
            const double half = theta / 2;
            g.coords.insert(g.coords.end(), {s.radius * theta, w * s.radius});
            g.coords3.insert(g.coords3.end(), {(s.radius + w * std::cos(half)) * std::cos(theta),
                                               (s.radius + w * std::cos(half)) * std::sin(theta),
                                               w * std::sin(half)});
        }
    for (uint32_t i = 1; i <= res; ++i)
        for (uint32_t j = 1; j <= rows; ++j) {
            const uint32_t v = ic_lattice(i, j, rows);
            if (i < res) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i + 1, j, rows));
            else ic_edge(g, rng, s.randomize_directions, v, ic_lattice(1, rows - j + 1, rows));
            if (j < rows) ic_edge(g, rng, s.randomize_directions, v, ic_lattice(i, j + 1, rows));
        }
    return g;
}

// Candidate pairs (i, j), i < j, of points closer than `t`, for i ascending and j ascending
// within i. Points are binned into cells of side at least t, so a pair is found by scanning the
// 3^d cells around a point: O(n + pairs) for a box at least 3 cells wide per axis. `delta`
// returns the displacement used (periodic or not); `cell_of` bins a coordinate.
template <class Delta>
inline void ic_near_pairs(const std::vector<double>& pts, uint32_t d, const double (*box)[2],
                          bool periodic, double t, Delta delta,
                          std::vector<std::vector<uint32_t>>& out) {
    const uint32_t n = static_cast<uint32_t>(pts.size() / d);
    out.assign(n, {});
    uint32_t cells[3] = {1, 1, 1};
    for (uint32_t a = 0; a < d; ++a) {
        const double len = box[a][1] - box[a][0];
        const double c = t > 0 ? std::floor(len / t) : 1;
        cells[a] = static_cast<uint32_t>(std::max(1.0, std::min(c, 1024.0)));
    }
    auto cell = [&](uint32_t p, uint32_t a) {
        const double len = box[a][1] - box[a][0];
        double f = len > 0 ? (pts[size_t(p) * d + a] - box[a][0]) / len : 0;
        long long c = static_cast<long long>(std::floor(f * cells[a]));
        if (c < 0) c = 0;
        if (c >= cells[a]) c = cells[a] - 1;
        return static_cast<uint32_t>(c);
    };
    const uint32_t total = cells[0] * cells[1] * cells[2];
    std::vector<uint32_t> start(total + 1, 0), order(n);
    std::vector<uint32_t> cid(n);
    for (uint32_t p = 0; p < n; ++p) {
        uint32_t c = 0;
        for (uint32_t a = d; a-- > 0;) c = c * cells[a] + cell(p, a);
        cid[p] = c; ++start[c + 1];
    }
    for (uint32_t c = 0; c < total; ++c) start[c + 1] += start[c];
    {
        std::vector<uint32_t> fill(start.begin(), start.end() - 1);
        for (uint32_t p = 0; p < n; ++p) order[fill[cid[p]]++] = p;
    }
    const double t2 = t * t;
    std::vector<uint32_t> seen_cells;
    for (uint32_t p = 0; p < n; ++p) {
        uint32_t base[3] = {0, 0, 0};
        uint32_t rest = cid[p];
        for (uint32_t a = 0; a < d; ++a) { base[a] = rest % cells[a]; rest /= cells[a]; }
        seen_cells.clear();
        const int span[3] = {cells[0] >= 3 ? 1 : 0, d > 1 && cells[1] >= 3 ? 1 : 0, d > 2 && cells[2] >= 3 ? 1 : 0};
        for (int o2 = -span[2]; o2 <= span[2]; ++o2)
            for (int o1 = -span[1]; o1 <= span[1]; ++o1)
                for (int o0 = -span[0]; o0 <= span[0]; ++o0) {
                    const int off[3] = {o0, o1, o2};
                    uint32_t c = 0;
                    bool ok = true;
                    for (uint32_t a = d; a-- > 0;) {
                        long long x = static_cast<long long>(base[a]) + off[a];
                        if (cells[a] < 3) x = base[a];
                        if (x < 0 || x >= cells[a]) {
                            if (!periodic) { ok = false; break; }
                            x = (x + cells[a]) % cells[a];
                        }
                        c = c * cells[a] + static_cast<uint32_t>(x);
                    }
                    if (!ok) continue;
                    if (std::find(seen_cells.begin(), seen_cells.end(), c) != seen_cells.end()) continue;
                    seen_cells.push_back(c);
                }
        // A box under 3 cells on an axis scans that axis whole.
        std::vector<uint32_t>& row = out[p];
        auto scan_cell = [&](uint32_t c) {
            for (uint32_t k = start[c]; k < start[c + 1]; ++k) {
                const uint32_t q = order[k];
                if (q <= p) continue;
                double dd[3];
                delta(p, q, dd);
                double s2 = 0;
                for (uint32_t a = 0; a < d; ++a) s2 += dd[a] * dd[a];
                if (s2 < t2 * (1 + 1e-9)) row.push_back(q);
            }
        };
        bool whole = false;
        for (uint32_t a = 0; a < d; ++a) whole |= cells[a] < 3;
        if (whole) {
            // Cells along a short axis are not neighbours by offset; take every cell that differs
            // from `base` by at most one on the long axes.
            for (uint32_t c = 0; c < total; ++c) {
                uint32_t r2 = c;
                bool near = true;
                for (uint32_t a = 0; a < d; ++a) {
                    const uint32_t x = r2 % cells[a]; r2 /= cells[a];
                    if (cells[a] < 3) continue;
                    const uint32_t dx = x > base[a] ? x - base[a] : base[a] - x;
                    if (!(dx <= 1 || (periodic && dx == cells[a] - 1))) { near = false; break; }
                }
                if (near) scan_cell(c);
            }
        } else {
            for (uint32_t c : seen_cells) scan_cell(c);
        }
        std::sort(row.begin(), row.end());
    }
}

// The Brill-Lindquist geometry: two horizons on the x axis at +-separation/2 with radii m/2, and
// displacements taken to the nearest periodic image when the box is periodic.
struct IcBlGeometry {
    uint32_t d;
    bool periodic;
    double lo[3], len[3];
    double centers[2][3];
    double radii[2];
    explicit IcBlGeometry(const IcSpec& s)
        : d(s.dimension == 3 ? 3 : 2), periodic(s.periodic),
          centers{{s.separation / 2, 0, 0}, {-s.separation / 2, 0, 0}},
          radii{s.mass1 / 2, s.mass2 / 2} {
        for (uint32_t a = 0; a < 3; ++a) { lo[a] = s.box[a][0]; len[a] = s.box[a][1] - s.box[a][0]; }
    }
    void disp(const double* a, const double* b, double* out) const {
        for (uint32_t k = 0; k < d; ++k) {
            double v = b[k] - a[k];
            if (periodic) v -= len[k] * static_cast<double>(ic_round_even(v / len[k]));
            out[k] = v;
        }
    }
    double dist(const double* a, const double* b) const {
        double dd[3];
        disp(a, b, dd);
        double s2 = 0;
        for (uint32_t k = 0; k < d; ++k) s2 += dd[k] * dd[k];
        return std::sqrt(s2);
    }
    bool inside(const double* p) const {
        return dist(centers[0], p) < radii[0] || dist(centers[1], p) < radii[1];
    }
};

// Joins two of the points `pts` (d per point) closer than the threshold unless their midpoint
// (on the periodic box: the midpoint toward the nearest image, wrapped into the box) is inside a
// horizon. Pairs are taken for i ascending, j ascending within i.
inline void ic_brill_lindquist_join(const IcSpec& s, const std::vector<double>& pts, IcRng& rng,
                                    IcGraph& g) {
    const IcBlGeometry geo(s);
    const uint32_t d = geo.d;
    const uint32_t count = static_cast<uint32_t>(pts.size() / d);
    g.n_vertices = count; g.coord_dim = d; g.coords = pts;
    for (uint32_t h = 0; h < 2; ++h) {
        g.horizon_centers.insert(g.horizon_centers.end(), geo.centers[h], geo.centers[h] + d);
        g.horizon_radii.push_back(geo.radii[h]);
    }
    const double t = s.edge_threshold < 0 ? 2.0 : s.edge_threshold;
    g.threshold = t;
    std::vector<std::vector<uint32_t>> near;
    ic_near_pairs(pts, d, s.box, s.periodic, t,
                  [&](uint32_t a, uint32_t b, double* out) { geo.disp(&pts[size_t(a) * d], &pts[size_t(b) * d], out); },
                  near);
    for (uint32_t i = 0; i < count; ++i)
        for (uint32_t j : near[i]) {
            const double* a = &pts[size_t(i) * d];
            const double* b = &pts[size_t(j) * d];
            double dd[3];
            geo.disp(a, b, dd);
            double s2 = 0;
            for (uint32_t k = 0; k < d; ++k) s2 += dd[k] * dd[k];
            if (!(std::sqrt(s2) < t)) continue;
            double mid[3];
            for (uint32_t k = 0; k < d; ++k) {
                if (!s.periodic) { mid[k] = (a[k] + b[k]) / 2; continue; }
                const double m = a[k] + dd[k] / 2 - geo.lo[k];
                mid[k] = geo.lo[k] + (m - geo.len[k] * std::floor(m / geo.len[k]));
            }
            if (!geo.inside(mid)) ic_edge(g, rng, s.randomize_directions, i + 1, j + 1);
        }
}

// Points drawn one at a time: one coordinate per axis, then, for a point outside both horizons,
// one acceptance draw against psi^(2 d) / max(100, 3^(2 d)), psi = 1 + m1/(2 r1) + m2/(2 r2),
// the volume element of the metric psi^4 delta_ij in d dimensions. At most 10 n max_vol draws.
inline IcGraph ic_brill_lindquist(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    if (s.dimension != 2 && s.dimension != 3) {
        g.error = "\"BrillLindquistDimension\" is " + std::to_string(s.dimension) + "; use 2 or 3";
        return g;
    }
    const IcBlGeometry geo(s);
    const uint32_t d = geo.d;
    double max_vol = 1;
    for (uint32_t k = 0; k < 2 * d; ++k) max_vol *= 3.0;
    max_vol = std::max(100.0, max_vol);
    const uint64_t max_attempts = uint64_t(s.n) * 10 * static_cast<uint64_t>(ic_round_even(max_vol));
    std::vector<double> pts;
    uint32_t count = 0;
    for (uint64_t att = 0; count < s.n && att < max_attempts; ++att) {
        double p[3] = {0, 0, 0};
        for (uint32_t k = 0; k < d; ++k) p[k] = rng.uniform(s.box[k][0], s.box[k][1]);
        const double r1 = geo.dist(geo.centers[0], p), r2 = geo.dist(geo.centers[1], p);
        if (r1 < geo.radii[0] || r2 < geo.radii[1]) continue;
        const double psi = 1 + s.mass1 / (2 * std::max(r1, 0.001)) + s.mass2 / (2 * std::max(r2, 0.001));
        double vol = 1;
        for (uint32_t k = 0; k < 2 * d; ++k) vol *= psi;
        if (rng.uniform() < vol / max_vol) { pts.insert(pts.end(), p, p + d); ++count; }
    }
    ic_brill_lindquist_join(s, pts, rng, g);
    return g;
}

// Points in the box [box[0]] x [box[1]], joined when closer than the threshold.
inline void ic_join_planar(IcGraph& g, IcRng& rng, const IcSpec& s, const std::vector<double>& pts, double t) {
    std::vector<std::vector<uint32_t>> near;
    ic_near_pairs(pts, 2, s.box, false, t,
                  [&](uint32_t a, uint32_t b, double* out) {
                      out[0] = pts[2 * size_t(b)] - pts[2 * size_t(a)];
                      out[1] = pts[2 * size_t(b) + 1] - pts[2 * size_t(a) + 1];
                  },
                  near);
    for (uint32_t i = 0; i < near.size(); ++i)
        for (uint32_t j : near[i]) {
            const double dx = pts[2 * size_t(i)] - pts[2 * size_t(j)];
            const double dy = pts[2 * size_t(i) + 1] - pts[2 * size_t(j) + 1];
            if (std::sqrt(dx * dx + dy * dy) < t) ic_edge(g, rng, s.randomize_directions, i + 1, j + 1);
        }
}

// Dart throwing: up to 100 n candidates, each kept when no kept point is closer than
// min_distance. Kept points are binned by cells of side min_distance.
inline IcGraph ic_poisson_disk(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const double md = s.min_distance, md2 = md * md;
    const double x0 = s.box[0][0], y0 = s.box[1][0];
    const double lx = s.box[0][1] - x0, ly = s.box[1][1] - y0;
    const uint32_t cx = static_cast<uint32_t>(std::max(1.0, std::min(md > 0 ? std::floor(lx / md) : 1, 4096.0)));
    const uint32_t cy = static_cast<uint32_t>(std::max(1.0, std::min(md > 0 ? std::floor(ly / md) : 1, 4096.0)));
    std::vector<std::vector<uint32_t>> bins(size_t(cx) * cy);
    auto bin = [](double v, double o, double l, uint32_t c) {
        long long k = l > 0 ? static_cast<long long>(std::floor((v - o) / l * c)) : 0;
        return static_cast<uint32_t>(std::min<long long>(std::max<long long>(k, 0), c - 1));
    };
    std::vector<double> pts;
    const uint64_t max_attempts = uint64_t(s.n) * 100;
    for (uint64_t att = 0; pts.size() / 2 < s.n && att < max_attempts; ++att) {
        const double x = rng.uniform(s.box[0][0], s.box[0][1]);
        const double y = rng.uniform(s.box[1][0], s.box[1][1]);
        const uint32_t bx = bin(x, x0, lx, cx), by = bin(y, y0, ly, cy);
        bool ok = true;
        for (long long ix = static_cast<long long>(bx) - 1; ok && ix <= static_cast<long long>(bx) + 1; ++ix)
            for (long long iy = static_cast<long long>(by) - 1; ok && iy <= static_cast<long long>(by) + 1; ++iy) {
                if (ix < 0 || iy < 0 || ix >= cx || iy >= cy) continue;
                for (uint32_t p : bins[size_t(ix) * cy + size_t(iy)]) {
                    const double dx = x - pts[2 * size_t(p)], dy = y - pts[2 * size_t(p) + 1];
                    if (dx * dx + dy * dy < md2 && std::sqrt(dx * dx + dy * dy) < md) { ok = false; break; }
                }
            }
        if (!ok) continue;
        bins[size_t(bx) * cy + by].push_back(static_cast<uint32_t>(pts.size() / 2));
        pts.push_back(x); pts.push_back(y);
    }
    g.n_vertices = static_cast<uint32_t>(pts.size() / 2); g.coord_dim = 2; g.coords = pts;
    g.threshold = s.edge_threshold < 0 ? 2.0 * md : s.edge_threshold;
    ic_join_planar(g, rng, s, pts, g.threshold);
    return g;
}

inline IcGraph ic_uniform_random(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    std::vector<double> pts;
    for (uint32_t i = 0; i < s.n; ++i) {
        const double x = rng.uniform(s.box[0][0], s.box[0][1]);
        pts.push_back(x);
        pts.push_back(rng.uniform(s.box[1][0], s.box[1][1]));
    }
    g.n_vertices = s.n; g.coord_dim = 2; g.coords = pts;
    const double area = (s.box[0][1] - s.box[0][0]) * (s.box[1][1] - s.box[1][0]);
    g.threshold = s.edge_threshold < 0 ? (s.n ? std::sqrt(area / s.n) * 1.5 : 0) : s.edge_threshold;
    ic_join_planar(g, rng, s, pts, g.threshold);
    return g;
}

// The causal links of points sorted by time, {t, x_1..x_d} each, in g.spacetime: i precedes j
// when j is inside i's light cone (|dx|^2 < c^2 dt^2) and the proper time is at most the
// Alexandrov cutoff. With the transitivity reduction a link i -> j is dropped when some k with
// i -> k also has k -> j, both in the unreduced relation. Each vertex then keeps its first
// max_edges_per_vertex links (0: all).
inline void ic_minkowski_links(const IcSpec& s, IcGraph& g) {
    const uint32_t d = s.spatial_dim, w = d + 1;
    const uint32_t n = static_cast<uint32_t>(g.spacetime.size() / w);
    const double* q = g.spacetime.data();
    std::vector<std::vector<uint32_t>> links(n);
    const double c2 = s.lightcone * s.lightcone, a2 = s.alexandrov * s.alexandrov;
    for (uint32_t i = 0; i + 1 < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j) {
            const double dt = q[size_t(j) * w] - q[size_t(i) * w];
            if (!(dt > 0)) continue;
            double dx2 = 0;
            for (uint32_t k = 1; k < w; ++k) {
                const double v = q[size_t(j) * w + k] - q[size_t(i) * w + k];
                dx2 += v * v;
            }
            if (dx2 < c2 * (dt * dt) && dt * dt - dx2 / c2 <= a2) links[i].push_back(j);
        }
    if (s.transitivity_reduction) {
        // links[k] for k > i is still unreduced when i is reduced, since every k in links[i] is
        // later in time than i. A per-k membership test is a sorted binary search.
        std::vector<std::vector<uint32_t>> reduced(n);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j : links[i]) {
                bool via = false;
                for (uint32_t k : links[i]) {
                    if (k == j || k >= j) { if (k >= j) break; continue; }
                    if (std::binary_search(links[k].begin(), links[k].end(), j)) { via = true; break; }
                }
                if (!via) reduced[i].push_back(j);
            }
        links.swap(reduced);
    }
    g.n_vertices = n; g.coord_dim = 2;
    for (uint32_t i = 0; i < n; ++i) {
        g.coords.push_back(d > 0 ? q[size_t(i) * w + 1] : 0.0);
        g.coords.push_back(q[size_t(i) * w]);
    }
    std::unordered_set<uint64_t> kept;
    for (uint32_t i = 0; i < n; ++i) {
        const size_t take = s.max_edges_per_vertex > 0 ? std::min<size_t>(links[i].size(), s.max_edges_per_vertex)
                                                       : links[i].size();
        for (size_t t = 0; t < take; ++t) {
            g.edges.push_back(i + 1);
            g.edges.push_back(links[i][t] + 1);
            kept.insert((uint64_t(i) << 32) | links[i][t]);
        }
    }
    // The fraction of related pairs among i <= min(n - 1, 100), i < j <= i + 100: 2 when it is in
    // (0.001, 0.9), else spatial_dim + 1.
    uint64_t pairs = 0, related = 0;
    for (uint32_t i = 0; i < std::min<uint32_t>(n ? n - 1 : 0, 100); ++i)
        for (uint32_t j = i + 1; j <= std::min<uint32_t>(i + 100, n - 1); ++j) {
            ++pairs;
            related += kept.count((uint64_t(i) << 32) | j) ? 1 : 0;
        }
    const double r = pairs ? double(related) / double(pairs) : 0;
    g.dimension_estimate = (r > 0.001 && r < 0.9) ? 2.0 : double(d + 1);
}

// A causal set: n points uniform in [0, T] x [-L/2, L/2]^d (each point's spatial coordinates
// drawn before its time), sorted by time, joined by ic_minkowski_links.
inline IcGraph ic_minkowski(const IcSpec& s) {
    IcGraph g; IcRng rng(s.seed);
    const uint32_t n = s.n, d = s.spatial_dim, w = d + 1;
    std::vector<double> p(size_t(n) * w);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < d; ++k) p[size_t(i) * w + 1 + k] = rng.uniform(-s.spatial_extent / 2, s.spatial_extent / 2);
        p[size_t(i) * w] = rng.uniform(0, s.time_extent);
    }
    std::vector<uint32_t> ord(n);
    for (uint32_t i = 0; i < n; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return p[size_t(a) * w] < p[size_t(b) * w]; });
    g.spacetime.resize(p.size());
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t k = 0; k < w; ++k) g.spacetime[size_t(i) * w + k] = p[size_t(ord[i]) * w + k];
    ic_minkowski_links(s, g);
    return g;
}

inline IcGraph ic_generate(const IcSpec& s) {
    switch (s.type) {
        case IcType::Grid:                return ic_grid(s);
        case IcType::GridWithHoles:       return ic_grid_with_holes(s);
        case IcType::Cylinder:            return ic_cylinder(s);
        case IcType::Torus:               return ic_torus(s);
        case IcType::Sphere:              return ic_sphere(s);
        case IcType::KleinBottle:         return ic_twisted(s, false);
        case IcType::MobiusStrip:         return ic_twisted(s, true);
        case IcType::MinkowskiSprinkling: return ic_minkowski(s);
        case IcType::BrillLindquist:      return ic_brill_lindquist(s);
        case IcType::PoissonDisk:         return ic_poisson_disk(s);
        case IcType::UniformRandom:       return ic_uniform_random(s);
    }
    IcGraph g;
    g.error = "unknown initial-condition type";
    return g;
}

}  // namespace common
}  // namespace HG_NAMESPACE
