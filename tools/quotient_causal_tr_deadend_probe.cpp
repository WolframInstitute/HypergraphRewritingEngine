// Decisive confirmation that project(reduce(raw)) is NOT a function of the quotient
// skeleton. Two raw event DAGs P and Q are built with the SAME (esig,depth) weighted
// multigraph -- i.e. identical canonical events, depths, and edge multiplicities, which
// is everything qc_* / any skeleton reconstruction can see -- yet DIFFERENT transitive
// reductions projected to esig pairs. If the multigraphs match and the reduced sets
// differ, causal-TR is information-theoretically unreconstructable from the skeleton.
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>
using namespace std;

// raw event: id -> (esig, depth)
struct Ev { char esig; int depth; };
using Edge = pair<int,int>;              // (producer raw id, consumer raw id)

// path u->..->v of length >=2 exists? (over the full edge set, skipping the direct edge)
static bool bypassed(int u,int v,const vector<Edge>& E){
    map<int,vector<int>> adj; for(auto&e:E) adj[e.first].push_back(e.second);
    set<int> vis; vector<int> st;
    for(int w: adj[u]) if(w!=v){ st.push_back(w); vis.insert(w); }   // first hop != v
    while(!st.empty()){ int x=st.back(); st.pop_back(); if(x==v) return true;
        for(int y: adj[x]) if(vis.insert(y).second) st.push_back(y); }
    return false;
}

// reduced set projected to (esig,esig) pairs
static set<pair<char,char>> reduce_project(const vector<Edge>& E, const map<int,Ev>& ev){
    set<pair<char,char>> R;
    for(auto&e:E){ if(e.first==e.second) continue;
        if(!bypassed(e.first,e.second,E)) R.insert({ev.at(e.first).esig, ev.at(e.second).esig}); }
    return R;
}

// (esig_p,depth_p,esig_c,depth_c) -> count : everything the skeleton determines
static map<tuple<char,int,char,int>,int> multigraph(const vector<Edge>& E, const map<int,Ev>& ev){
    map<tuple<char,int,char,int>,int> M;
    for(auto&e:E){ const Ev&p=ev.at(e.first),&c=ev.at(e.second);
        M[{p.esig,p.depth,c.esig,c.depth}]++; }
    return M;
}

int main(){
    // ids: a1=1,a2=2 (esig A, depth 0); b1=3,b2=4 (esig B, depth 1); c1=5,c2=6 (esig C, depth 2)
    map<int,Ev> ev = {{1,{'A',0}},{2,{'A',0}},{3,{'B',1}},{4,{'B',1}},{5,{'C',2}},{6,{'C',2}}};

    // P: a1->b1->c1, a2->b2->c2, plus direct a1->c1, a2->c2   (both A->C instances bypassed)
    vector<Edge> P = {{1,3},{3,5},{2,4},{4,6},{1,5},{2,6}};
    // Q: same but b2->c1 instead of b2->c2 (so a2->c2 has NO bypass)
    vector<Edge> Q = {{1,3},{3,5},{2,4},{4,5},{1,5},{2,6}};

    auto MP = multigraph(P,ev), MQ = multigraph(Q,ev);
    auto RP = reduce_project(P,ev), RQ = reduce_project(Q,ev);

    printf("(esig,depth) weighted multigraph  P == Q ?  %s\n", MP==MQ ? "IDENTICAL" : "different");
    printf("  P multigraph: "); for(auto&kv:MP) printf("%c%d->%c%d:%d ",get<0>(kv.first),get<1>(kv.first),get<2>(kv.first),get<3>(kv.first),kv.second); printf("\n");
    printf("  Q multigraph: "); for(auto&kv:MQ) printf("%c%d->%c%d:%d ",get<0>(kv.first),get<1>(kv.first),get<2>(kv.first),get<3>(kv.first),kv.second); printf("\n");
    printf("project(reduce(raw))  P: "); for(auto&p:RP) printf("%c->%c ",p.first,p.second); printf("(%zu edges)\n", RP.size());
    printf("project(reduce(raw))  Q: "); for(auto&p:RQ) printf("%c->%c ",p.first,p.second); printf("(%zu edges)\n", RQ.size());
    printf("\nVERDICT: skeleton identical=%s  reduced sets differ=%s  => %s\n",
           MP==MQ?"yes":"no", RP!=RQ?"yes":"no",
           (MP==MQ && RP!=RQ) ? "DEAD END CONFIRMED (causal-TR not a function of the skeleton)"
                              : "no collision (inconclusive)");
    return 0;
}
