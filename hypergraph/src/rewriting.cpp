#include <hypergraph/rewriting.hpp>
#include <sstream>

namespace hypergraph {


namespace debug {

std::string rule_to_string(const RewritingRule& rule) {
    std::ostringstream oss;
    oss << "Rule:\n";
    oss << "  LHS: " << rule.lhs.num_edges() << " edges, ";
    oss << rule.lhs.num_variable_vertices() << " variables, ";
    oss << rule.lhs.num_concrete_vertices() << " concrete vertices\n";
    oss << "  RHS: " << rule.rhs.num_edges() << " edges, ";
    oss << rule.rhs.num_variable_vertices() << " variables, ";
    oss << rule.rhs.num_concrete_vertices() << " concrete vertices\n";
    oss << "  Well-formed: " << (rule.is_well_formed() ? "yes" : "no");
    return oss.str();
}

std::string result_to_string(const RewritingResult& result) {
    std::ostringstream oss;
    oss << "RewritingResult:\n";
    oss << "  Applied: " << (result.applied ? "yes" : "no") << "\n";
    if (result.applied) {
        oss << "  Removed " << result.removed_edges.size() << " edges\n";
        oss << "  Added " << result.added_edges.size() << " edges\n";
        oss << "  Anchor vertex: " << result.anchor_vertex << "\n";
        oss << "  Variable assignments: " << result.variable_assignment.variable_to_concrete.size();
    }
    return oss.str();
}

} // namespace debug

} // namespace hypergraph