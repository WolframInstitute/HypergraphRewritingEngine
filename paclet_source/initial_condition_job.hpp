#pragma once
// THE "InitialCondition" JOB: <|"Op" -> "InitialCondition", "InitialCondition" -> spec|>.
// The spec names a generator of hgcommon/initial_condition_core.hpp and its parameters; the
// reply is the generated graph. HGEvolve and the Python package send this job for a generated
// initial condition and evolve from the edges it returns, so the generator runs in the engine
// for both, on either worker.
//
// Spec keys (absent keys take ic_spec_for(type)'s defaults):
//   "Type"                   string, a name of ic_type_from_name
//   "Seed"                   integer
//   "RandomizeDirections", "Diagonals", "Periodic", "TransitivityReduction"   True | False
//   "Width", "Height", "Resolution", "Density" (points), "SpatialDim", "Dimension",
//   "MaxEdgesPerVertex"      integer
//   "Radius", "MajorRadius", "MinorRadius", "EdgeThreshold", "MinDistance", "TimeExtent",
//   "SpatialExtent", "LightconeAngle", "AlexandrovCutoff", "Mass1", "Mass2", "Separation"   real
//   "BoxX", "BoxY", "BoxZ"   {real, real}
//   "Holes"                  {{x, y, r}, ...}, reals
// A "Grid" with holes is a "GridWithHoles".
//
// Reply: "Edges" {{a, b}, ...} over vertices 1..n, "VertexCount", "VertexCoordinates" {{x, y,
// ...}, ...} in vertex order, and where the generator gives them "VertexCoordinates3D",
// "SpacetimePoints" ({t, x...} per vertex), "EdgeThreshold", "DimensionEstimate", and for
// "BrillLindquist" "HorizonCenters" ({x, y, ...} per horizon) and "HorizonRadii".

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "hgcommon/initial_condition_core.hpp"
#include "wxf.hpp"

namespace HG_NAMESPACE {
namespace ffi {

// The spec association at the parser's cursor. Throws std::runtime_error on an unknown type or
// key, or a value out of range.
template <class Parser>
hgcommon::IcSpec read_initial_condition_spec(Parser& parser) {
    // Two passes over the keys: the type fixes the defaults the other keys override.
    struct Entry { std::string key; size_t at; };
    std::vector<Entry> keys;
    std::string type_name = "Grid";
    const size_t start = parser.position();
    parser.read_association([&](const std::string& key, Parser& vp) {
        if (key == "Type") type_name = vp.template read<std::string>();
        else vp.skip_value();
    });
    hgcommon::IcType type;
    if (!hgcommon::ic_type_from_name(type_name, type))
        throw std::runtime_error("unknown initial condition \"" + type_name + "\"");
    hgcommon::IcSpec s = hgcommon::ic_spec_for(type);
    parser.seek(start);

    // A real may arrive as an integer token, and a list of reals as a list of integers.
    auto number = [](Parser& vp) {
        const size_t at = vp.position();
        try {
            return vp.template read<double>();
        } catch (const std::exception&) {
            vp.seek(at);
            return static_cast<double>(vp.template read<int64_t>());
        }
    };
    // A packed array of reals, else a List read element by element (integers and reals mixed).
    auto reals = [&](Parser& vp) {
        const size_t at = vp.position();
        try {
            return vp.template read<std::vector<double>>();
        } catch (const std::exception&) {
            vp.seek(at);
            std::vector<double> out;
            vp.read_function([&](const std::string& head, size_t n, Parser& args) {
                if (head != "List") throw std::runtime_error("a list of reals was expected");
                for (size_t i = 0; i < n; ++i) out.push_back(number(args));
            });
            return out;
        }
    };
    auto real = [&](Parser& vp, const std::string& key) {
        const double v = number(vp);
        if (!std::isfinite(v)) throw std::runtime_error("\"" + key + "\" is not a finite real");
        return v;
    };
    auto count = [](Parser& vp, const std::string& key) {
        const int64_t v = vp.template read<int64_t>();
        if (v < 0 || v > int64_t{1} << 30) throw std::runtime_error("\"" + key + "\" is out of range");
        return static_cast<uint32_t>(v);
    };
    auto flag = [](Parser& vp, const std::string& key) {
        const std::string v = vp.template read<std::string>();
        if (v != "True" && v != "False") throw std::runtime_error("\"" + key + "\" is not True or False");
        return v == "True";
    };
    auto interval = [&](Parser& vp, const std::string& key, double* out) {
        const std::vector<double> v = reals(vp);
        if (v.size() != 2 || !std::isfinite(v[0]) || !std::isfinite(v[1]) || !(v[0] < v[1]))
            throw std::runtime_error("\"" + key + "\" is not {low, high} with low < high");
        out[0] = v[0];
        out[1] = v[1];
    };
    parser.read_association([&](const std::string& key, Parser& vp) {
        if (key == "Type") vp.skip_value();
        else if (key == "Seed") s.seed = static_cast<uint64_t>(vp.template read<int64_t>());
        else if (key == "RandomizeDirections") s.randomize_directions = flag(vp, key);
        else if (key == "Diagonals") s.diagonals = flag(vp, key);
        else if (key == "Periodic") s.periodic = flag(vp, key);
        else if (key == "TransitivityReduction") s.transitivity_reduction = flag(vp, key);
        else if (key == "Width") s.width = count(vp, key);
        else if (key == "Height") s.height = count(vp, key);
        else if (key == "Resolution") s.resolution = count(vp, key);
        else if (key == "Density") s.n = count(vp, key);
        else if (key == "SpatialDim") s.spatial_dim = count(vp, key);
        else if (key == "Dimension") s.dimension = count(vp, key);
        else if (key == "MaxEdgesPerVertex") s.max_edges_per_vertex = count(vp, key);
        else if (key == "Radius") s.radius = real(vp, key);
        else if (key == "MajorRadius") s.major_radius = real(vp, key);
        else if (key == "MinorRadius") s.minor_radius = real(vp, key);
        else if (key == "EdgeThreshold") s.edge_threshold = real(vp, key);
        else if (key == "MinDistance") s.min_distance = real(vp, key);
        else if (key == "TimeExtent") s.time_extent = real(vp, key);
        else if (key == "SpatialExtent") s.spatial_extent = real(vp, key);
        else if (key == "LightconeAngle") s.lightcone = real(vp, key);
        else if (key == "AlexandrovCutoff") s.alexandrov = real(vp, key);
        else if (key == "Mass1") s.mass1 = real(vp, key);
        else if (key == "Mass2") s.mass2 = real(vp, key);
        else if (key == "Separation") s.separation = real(vp, key);
        else if (key == "BoxX") interval(vp, key, s.box[0]);
        else if (key == "BoxY") interval(vp, key, s.box[1]);
        else if (key == "BoxZ") interval(vp, key, s.box[2]);
        else if (key == "Holes") {
            const size_t at = vp.position();
            std::vector<std::vector<double>> holes;
            try {
                holes = vp.template read<std::vector<std::vector<double>>>();
            } catch (const std::exception&) {
                vp.seek(at);
                vp.read_function([&](const std::string& head, size_t n, Parser& rows) {
                    if (head != "List") throw std::runtime_error("\"Holes\" is a list of {x, y, radius}");
                    for (size_t i = 0; i < n; ++i) holes.push_back(reals(rows));
                });
            }
            for (const auto& h : holes) {
                if (h.size() != 3) throw std::runtime_error("a hole is {x, y, radius}");
                s.holes.push_back({h[0], h[1], h[2]});
            }
        } else {
            throw std::runtime_error("unknown initial condition key \"" + key + "\"");
        }
    });
    if (s.type == hgcommon::IcType::Grid && !s.holes.empty()) s.type = hgcommon::IcType::GridWithHoles;
    return s;
}

// The reply of an "InitialCondition" job.
inline wxf::WXFValueAssociation initial_condition_reply(const hgcommon::IcGraph& g) {
    using wxf::WXFValue;
    using wxf::WXFValueList;
    auto rows = [](const std::vector<double>& v, uint32_t d) {
        WXFValueList out;
        if (d == 0) return out;
        out.reserve(v.size() / d);
        for (size_t i = 0; i + d <= v.size(); i += d) {
            WXFValueList row;
            for (uint32_t k = 0; k < d; ++k) row.push_back(WXFValue(v[i + k]));
            out.push_back(WXFValue(std::move(row)));
        }
        return out;
    };
    WXFValueList edges;
    edges.reserve(g.edges.size() / 2);
    for (size_t i = 0; i + 1 < g.edges.size(); i += 2)
        edges.push_back(WXFValue(WXFValueList{WXFValue(int64_t{g.edges[i]}), WXFValue(int64_t{g.edges[i + 1]})}));
    wxf::WXFValueAssociation out;
    out.push_back({WXFValue("Edges"), WXFValue(std::move(edges))});
    out.push_back({WXFValue("VertexCount"), WXFValue(int64_t{g.n_vertices})});
    out.push_back({WXFValue("VertexCoordinates"), WXFValue(rows(g.coords, g.coord_dim))});
    if (g.coord3_dim)
        out.push_back({WXFValue("VertexCoordinates3D"), WXFValue(rows(g.coords3, g.coord3_dim))});
    if (!g.spacetime.empty())
        out.push_back({WXFValue("SpacetimePoints"),
                       WXFValue(rows(g.spacetime, static_cast<uint32_t>(g.spacetime.size() / (g.n_vertices ? g.n_vertices : 1))))});
    if (g.threshold > 0) out.push_back({WXFValue("EdgeThreshold"), WXFValue(g.threshold)});
    if (g.dimension_estimate > 0)
        out.push_back({WXFValue("DimensionEstimate"), WXFValue(g.dimension_estimate)});
    if (!g.horizon_radii.empty()) {
        out.push_back({WXFValue("HorizonCenters"), WXFValue(rows(g.horizon_centers, g.coord_dim))});
        WXFValueList radii;
        for (double r : g.horizon_radii) radii.push_back(WXFValue(r));
        out.push_back({WXFValue("HorizonRadii"), WXFValue(std::move(radii))});
    }
    return out;
}

}  // namespace ffi
}  // namespace HG_NAMESPACE
