# Graph re-ranking self-ablation on LongMemEval: findings

## Verdict

Turning memsom's associative graph ON as a retrieval re-ranker made answer recall
**worse**, not better, on LongMemEval-S. Clean-arm utility fell from 0.943 to
0.781 (a 16 point drop). Across 192 items the graph broke 31 that plain retrieval
had gotten right and rescued 0 that it had missed. This is a clean, reproducible
negative result. It is bounded to this task and this regime, and it does not
generalize to "the graph is useless" (see "What this does not mean").

## Setup (a fair ablation)

The two arms are identical in every way except one flag, so any difference is
attributable only to the graph:

- Corpus: LongMemEval-S (cleaned), the full-haystack version, ~48 sessions per
  item. 192 of 500 items were used; 308 were skipped honestly because they have
  no marked answer-bearing turn (recall would be undefined).
- Config: max_evidence 20, topk 8, poison rate 0.5, both arms. Poison selection
  is deterministic, so both arms saw identical items and identical poison.
- Arm 1, retrieve: BM25 + vector, graph OFF.
- Arm 2, graph: same retrieval, then re-ranked by the rel_edges graph, hops 2.
- Edges: produced by a blind LLM linker (gpt-4o, temperature 0). It sees ONLY the
  evidence text, never the question, the answer-bearing flags, or the scorer. Its
  function signature is the anti-cheat proof. Mean 11.9 edges per item after the
  prompt was tightened to reduce over-linking.
- Metric: judge OFF. "utility" here is a per-item boolean recall proxy: did the
  answer end up in the retrieved set. The judge (LLM utility grading) was not run
  because the recall proxy already shows the regression cleanly, and a worse
  retrieval can only feed a worse synthesis.

## Headline numbers

```
                    retrieve    graph     delta
clean utility         0.943     0.781    -0.161
poisoned utility      0.943     0.818    -0.125
poisoned ASR          0.354     0.312    -0.042  (graph marginally more robust)

clean item churn (n = 192)
  both correct                 150
  both wrong                    11
  graph BROKE (was right)       31
  graph FIXED (was wrong)        0
  net                          -31
```

Fixed zero is the whole story in one number. The graph never surfaced an answer
that retrieval alone had missed. Every bit of movement it caused was downward.

## Worked example (item 51a45a95)

Question: "Where did I redeem a $5 coupon on coffee creamer?" Answer: Target, in
evidence turn 0. The other 19 turns are unrelated conversations (a Spanish
conquistador, conflict-management tips, women entrepreneurs in Chhattisgarh).

```
RETRIEVE (graph off)            GRAPH (re-ranked, hops 2)
  1. turn 0  <- ANSWER            1. turn 3   (conflict mgmt)
  2. turn 18                      2. turn 4   (conflict mgmt)
  3. turn 14                      3. turn 12  (conflict mgmt)
  4. turn 4                       4. turn 8   (conflict mgmt)
  5. turn 2                       5. turn 5   (conflict mgmt)
  6. turn 1                       6. turn 10  (conflict mgmt)
  7. turn 19                      7. turn 11  (conflict mgmt)
  8. turn 12                      8. turn 7   (conflict mgmt)
  --- topk cutoff (8) ---         --- topk cutoff (8) ---
  ANSWER rank = 1                 ANSWER rank = 15
```

Retrieval ranked the answer first. The graph buried it at 15. The answer node had
exactly one edge, a spurious link the blind linker drew between "coffee creamer
coupon" and a "conflict strategies" turn. Turns 3 through 12 are one dense
conflict-management cluster, every turn linked to its neighbors, and they boosted
each other as a bloc into the entire top 8.

## Why it got worse (the mechanism, in full)

Five effects, compounding. The first two are structural and would bite any strong
retriever; the last three are specific to this corpus and linker.

### 1. Near-ceiling re-ranking is negative-sum

Retrieval was already right on ~94% of items. Re-ranking is just a permutation of
the candidate set. For an already-correct item, the only reachable outcomes are
"still correct" or "now wrong," never "improved." So when the base retriever is
strong, the expected value of any re-ranker is dominated by its downside: there is
almost nothing left to gain and a great deal to lose. `fixed = 0` is that
asymmetry made literal. There were only 11 items retrieval failed, and the graph
rescued none of them, while it had 150 correct items to accidentally damage.

### 2. Neighbor-boosting rewards centrality, not relevance

`retrieve_graph` boosts each node by the sum of its neighbors' retrieval scores,
decayed by hop distance. That is a centrality mechanism, a small PageRank. A node
scores highly if it is well connected to other high-scoring nodes. Centrality and
answer-relevance are different things. The correct answer to a factoid question is
usually a single distinctive statement with low connectivity. Noise, by contrast,
can be highly connected. So the boosting systematically reweights toward
well-connected nodes and away from the lone fact that actually answers the query.

### 3. Conversation structure makes noise dense and answers sparse

In LongMemEval an item is many separate chat sessions. A single session on one
topic (10 turns about conflict management) becomes a dense intra-topic cluster,
every turn linked to its neighbors. The answer is often one short user statement
in a different, brief session. The graph therefore mirrors conversation length,
not relevance: long tangential threads become big self-reinforcing clusters,
while the answer sits alone or nearly alone. Effect 2 then amplifies exactly the
wrong nodes. In the worked example, a 10-turn tangent outvoted a rank-1 answer.

### 4. The blind edges added noise, not signal

The tightened linker still leaned on adjacency, linking each question turn to its
neighboring answer turn. Adjacent turns are already close in retrieval space, so
those links reinforce structure retrieval already has, adding no new ability to
surface a missed answer. Worse, it also drew genuinely wrong links (coffee-creamer
to conflict-strategies). The result: no upside (nothing rescued) and fresh
downside (spurious clusters to amplify). `fixed = 0` again.

### 5. The missing ingredient is curation

This is the deep reason and the one that reconciles the negative result with the
graph's real value. In Matthew's personal memory the edges are curated
`[[wikilinks]]` between durable, meaningful notes; a link means "these relate in a
way that matters." In LongMemEval the edges are an LLM's one-shot guesses over
transient conversation turns, with no curation, no reuse, no human check. The same
neighbor-boosting mechanism that exploits meaningful structure will faithfully
amplify meaningless structure when that is what it is given. The algorithm did not
fail; the edges were noise, and it amplified noise.

## What this does not mean

It does not mean the associative graph is worthless. It means graph re-ranking
over a near-ceiling retriever, on uncurated conversational edges, is a net
negative. Matthew's actual brain is the opposite regime: a dense, deliberately
cross-linked web of durable notes where traversal was already shown to surface
connections that keyword and vector recall miss. Different data, different edge
quality, different result. This benchmark is simply the wrong showcase for the
feature, and it now has the numbers to say so precisely.

## Recommendation

- Report this as a loss. Do not ship graph re-ranking as a default retrieval path
  for short-conversation factoid recall.
- Do not chase the linker for this benchmark. The ceiling caps any possible gain
  at roughly 6 points while the displacement downside stays wide open, and
  `fixed = 0` shows the regime barely rewards association at all. A better linker
  reduces effect 4 but not effects 1 through 3.
- If graph retrieval is ever offered, gate it: only re-rank when base-retriever
  confidence is low (there is headroom), and damp the centrality boost so a dense
  cluster cannot outvote a top-ranked singleton.
- The feature's real evaluation belongs on a corpus with curated, reused link
  structure or genuine multi-hop questions, where edges encode relevance rather
  than conversational adjacency.

## Validity notes

- Judge OFF: the metric is a recall proxy, not LLM-graded answer quality. A judge
  run would very likely show the same or a larger gap (worse retrieval feeds worse
  synthesis), so it was not worth the cost and rate-limit exposure to confirm.
- Reproducible: deterministic poison selection, temperature-0 linker, and a
  content-hashed edge cache (192 items, 288 distinct text-list variants) mean both
  arms rerun to the same numbers.
- The mem0 head-to-head arm was deliberately deferred. The self-ablation stands on
  its own and answers the question that was actually asked.
```
