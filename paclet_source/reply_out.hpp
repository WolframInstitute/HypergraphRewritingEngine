#pragma once
#include "hgcommon/namespace.hpp"
//
// The two forms of a CPU job's reply. run_rewriting_core builds every reply through one of them:
//
//   WireReply   the WXF association hg_evolve and the paclet library send, as wxf::Pieces.
//   ArrayReply  the same keys as arrays in memory, for an in-process caller (the Python module):
//               a record list ("States", "Events", "GlobalEdges", "StateBitvectors") as a Table of
//               columns, a relation ("CausalEdges", "BranchialEdges", "BranchialStateEdges") as a
//               row-major integer matrix, and every other key as its wxf::WXFValue.
//
// Both take the same calls in the same order, so the reply's content is computed once:
//
//   value(key, v)                  one key and its value
//   records(key, n, par, item)     n records; item(i, sink) writes record i through a sink with
//                                  key(id), begin(fields), i64, i64_list, edges and vertex_reals
//                                  (graph_marshal.hpp's StreamRecordSink is the wire one)
//   relation(key, rows, names, par) rows in ascending order (hgmarshal::sort_relation_rows)
//   branchial_state_edges(set)     "BranchialStateEdges" and "BranchialStateVertices"
//   global_edges(n, par, item)     "GlobalEdges" and "StateBitvectors": item(i, put) calls
//   state_bitvectors(n, par, item) put(id, count, each) once, where each(f) calls f(x) for the
//                                  `count` integers of item i (an edge's vertices, a state's edges)
//   session_ack(handle)            the whole reply of a Close
//   keep_reduced(values)           the reply to give instead if the reply runs out of memory
//   use_reduced()                  give it

#include "graph_marshal.hpp"
#include "hg_core.hpp"
#include "wxf.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace ffi {

// One column of a record table. `offsets` has one entry per record plus one, starting at 0, for
// every kind but Int.
struct Column {
    enum class Kind { Int, IntLists, Edges, VertexReals };
    Kind kind = Kind::Int;
    // Int: one value per record. IntLists: the lists' values. Edges: the edges' ids.
    // VertexReals: the vertices.
    std::vector<int64_t> ints;
    std::vector<int64_t> offsets;
    // Edges: each edge's vertices, `vertex_offsets` one entry per edge plus one.
    std::vector<int64_t> vertex_offsets;
    std::vector<int64_t> vertices;
    // VertexReals: the value of each vertex in `ints`.
    std::vector<double> reals;
};

// Records as named columns, in the order a record writes its fields.
struct Table {
    size_t rows = 0;
    std::vector<std::pair<std::string, Column>> columns;
};

// A relation: `rows` rows of names.size() integers each, row-major.
struct Relation {
    std::vector<std::string> names;
    size_t rows = 0;
    std::vector<int64_t> cells;
};

// The record sink of ArrayReply: each record a row of `table`. Every record of a table writes the
// same fields in the same order; a record that does not throws std::logic_error.
class TableSink {
public:
    explicit TableSink(Table& table) : t_(table) {}
    void key(int64_t) {}
    void begin(std::size_t fields);
    void i64(const char* key, int64_t v);
    void i64_list(const char* key, const std::vector<int64_t>& v);
    void edges(const char* key, const std::vector<std::pair<int64_t, std::vector<uint32_t>>>& e);
    void vertex_reals(const char* key, const std::vector<std::pair<uint32_t, double>>& kv);

private:
    Column& column_(const char* key, Column::Kind kind);
    Table& t_;
    size_t field_ = 0;
    size_t fields_ = 0;
};

// The tables `parts` made of consecutive records, joined in order.
Table join_tables(std::vector<Table>&& parts);

class ArrayReply {
public:
    static constexpr bool kWire = false;
    // Records per part of a table, each part written by one job of `par`.
    static constexpr size_t kRows = 256;

    std::vector<std::pair<std::string, Table>> tables;
    std::vector<std::pair<std::string, Relation>> relations;
    wxf::WXFValueAssociation values;

    void value(std::string key, wxf::WXFValue v) {
        values.push_back({wxf::WXFValue(std::move(key)), std::move(v)});
    }

    template <class Item>
    void records(const char* key, size_t n, const marshal::ParallelFor& par, Item item) {
        std::vector<Table> parts((n + kRows - 1) / kRows);
        par(parts.size(), [&](size_t c) {
            TableSink sink(parts[c]);
            const size_t end = std::min(n, (c + 1) * kRows);
            for (size_t i = c * kRows; i < end; ++i) item(i, sink);
        });
        tables.emplace_back(key, join_tables(std::move(parts)));
    }

    template <size_t N>
    void relation(const char* key, std::vector<std::array<int64_t, N>> rows,
                  const std::array<const char*, N>& names, const marshal::ParallelFor&) {
        marshal::sort_relation_rows(rows);
        Relation r;
        r.names.assign(names.begin(), names.end());
        r.rows = rows.size();
        r.cells.reserve(rows.size() * N);
        for (const auto& row : rows) r.cells.insert(r.cells.end(), row.begin(), row.end());
        relations.emplace_back(key, std::move(r));
    }

    void branchial_state_edges(const marshal::BranchialStateEdgeSet& set);

    template <class Item>
    void global_edges(size_t n, const marshal::ParallelFor& par, Item item) {
        id_lists("GlobalEdges", "Id", "Vertices", n, par, item);
    }
    template <class Item>
    void state_bitvectors(size_t n, const marshal::ParallelFor& par, Item item) {
        id_lists("StateBitvectors", "State", "Edges", n, par, item);
    }

    void session_ack(uint64_t handle) {
        *this = ArrayReply{};
        value("Session", wxf::WXFValue(static_cast<int64_t>(handle)));
    }
    void keep_reduced(wxf::WXFValueAssociation v) { reduced_ = std::move(v); }
    bool has_reduced() const { return !reduced_.empty(); }
    void use_reduced() {
        wxf::WXFValueAssociation v = std::move(reduced_);
        *this = ArrayReply{};
        values = std::move(v);
    }
    void finish() {}

private:
    // The list buffer of id_lists, one per thread. A thread_local declared inside id_lists's
    // lambda stops MSVC 2022 with an internal compiler error (C1001).
    static std::vector<int64_t>& id_list_buffer() {
        thread_local std::vector<int64_t> list;
        return list;
    }
    // A table of `n` rows: an Int column `id_name` and an IntLists column `list_name`, row i from
    // item(i, put) (see the header). One list buffer per part of the table.
    template <class Item>
    void id_lists(const char* key, const char* id_name, const char* list_name, size_t n,
                  const marshal::ParallelFor& par, Item item) {
        records(key, n, par, [&](size_t i, TableSink& sink) {
            std::vector<int64_t>& list = id_list_buffer();
            item(i, [&](int64_t id, size_t count, auto&& each) {
                list.clear();
                list.reserve(count);
                each([&](int64_t x) { list.push_back(x); });
                sink.begin(2);
                sink.i64(id_name, id);
                sink.i64_list(list_name, list);
            });
        });
    }
    wxf::WXFValueAssociation reduced_;
};

// The WXF reply: the record sections streamed into one buffer ahead of every other entry
// (wxf::association_pieces).
class WireReply {
public:
    static constexpr bool kWire = true;
    void value(std::string key, wxf::WXFValue v) {
        rest_.push_back({wxf::WXFValue(std::move(key)), std::move(v)});
    }

    template <class Item>
    void records(const char* key, size_t n, const marshal::ParallelFor& par, Item item) {
        sections_.write_key(key);
        sections_.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        sections_.write_varint(n);
        marshal::write_items(sections_, n, par, [&](wxf::Writer& w, size_t i) {
            marshal::StreamRecordSink sink(w);
            item(i, sink);
        });
        ++streamed_;
    }

    template <size_t N>
    void relation(const char* key, std::vector<std::array<int64_t, N>> rows,
                  const std::array<const char*, N>& names, const marshal::ParallelFor& par) {
        value(key, marshal::relation_list(std::move(rows), names, par));
    }

    void branchial_state_edges(const marshal::BranchialStateEdgeSet& set) {
        marshal::push_branchial_state_edges(rest_, set);
    }

    template <class Item>
    void global_edges(size_t n, const marshal::ParallelFor& par, Item item) {
        id_lists("GlobalEdges", n, par, item, /*association=*/false);
    }
    template <class Item>
    void state_bitvectors(size_t n, const marshal::ParallelFor& par, Item item) {
        id_lists("StateBitvectors", n, par, item, /*association=*/true);
    }

    void session_ack(uint64_t handle) { done_ = wxf::Pieces::of(marshal::session_ack(handle)); }
    void keep_reduced(wxf::WXFValueAssociation v) {
        wxf::Writer w;
        w.write_header();
        w.write(wxf::WXFValue(std::move(v)));
        reduced_ = w.release_data();
    }
    bool has_reduced() const { return !reduced_.empty(); }
    void use_reduced() { done_ = wxf::Pieces::of(std::move(reduced_)); }
    // A reply made elsewhere (the GPU backend).
    void set_pieces(wxf::Pieces p) { done_ = std::move(p); }
    // The record sections and the entries joined into the reply.
    void finish() {
        done_ = wxf::association_pieces(sections_.release_data(), streamed_, std::move(rest_));
    }

    // The reply's bytes as runs.
    wxf::Pieces take() { return std::move(done_); }
    size_t size() const { return done_.size(); }
    // The size of the streamed record sections, and each other entry, for HG_FFI_PAYLOAD_FILE.
    size_t sections_bytes() const { return sections_.size(); }
    const wxf::WXFValueAssociation& entries() const { return rest_; }

private:
    // GlobalEdges: a list of {id, v1, v2, ...}. StateBitvectors: <|id -> {e1, e2, ...}, ...|>.
    template <class Item>
    void id_lists(const char* key, size_t n, const marshal::ParallelFor& par, Item item,
                  bool association) {
        wxf::Writer w;
        if (association) {
            w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
            w.write_varint(n);
        } else {
            w.write_function("List", n);
        }
        marshal::write_items(w, n, par, [&](wxf::Writer& out, size_t i) {
            item(i, [&](int64_t id, size_t count, auto&& each) {
                if (association) {
                    out.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    out.write(id);
                    out.write_function("List", count);
                } else {
                    out.write_function("List", count + 1u);
                    out.write(id);
                }
                each([&](int64_t x) { out.write(x); });
            });
        });
        value(key, wxf::WXFValue(wxf::Encoded{w.release_data()}));
    }

    wxf::Writer sections_;
    size_t streamed_ = 0;
    wxf::WXFValueAssociation rest_;
    wxf::Pieces done_;
    std::vector<uint8_t> reduced_;
};

}  // namespace ffi
}  // namespace HG_NAMESPACE

// run_rewriting_core's job given as a value (the WXF envelope's association, read through
// wxf::ValueParser) and replied as arrays. Throws std::exception on error, as it does.
HG_NAMESPACE::ffi::ArrayReply run_rewriting_arrays(const HG_NAMESPACE::wxf::WXFValue& job,
                                                  const HostBridge& host);
