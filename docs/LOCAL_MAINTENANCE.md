# Local HNSW deletion and vector updates

The managed `hnsw.api` now repairs an existing graph rather than constructing a
replacement graph for every deletion or update. Public method signatures are unchanged.

## Algorithm

1. On the first effective deletion/update, construct a reverse-edge index at every
   layer, a level-to-live-IDs map, and a pool of free ordinals. These are auxiliary
   structures; construction-only indexes leave them unallocated.
2. For each removed node, record both its incoming and outgoing neighbors per
   layer. HNSW edges can be asymmetric: outgoing neighbors alone do not identify
   every node that must lose a reference to the removed node.
3. Remove all incident edges and the node itself. If it was the entry point,
   choose a live node from the highest remaining layer, or nil for an empty index.
4. For each affected boundary node, combine its surviving neighbors with that
   layer's boundary candidates. Rank by the configured construction distance,
   retain at most max(efConstruction, layer degree limit) candidates, then apply
   the existing HNSW diversity heuristic. Upper layers remain bounded by M and
   layer zero by 2M. No self-links or links to missing/lower-level nodes are added.
5. Apply batch removals sequentially to the evolving graph, so adjacent deleted
   nodes are bypassed through repaired links instead of leaving stale references.
   Update reverse edges on every subsequent insertion, pruning and repair.
6. For a vector update, validate and copy the replacement first, repair the old
   neighborhood, then insert the new vector using normal hierarchical HNSW
   construction at the original level and ordinal. Other nodes are not recreated.
7. Refresh the compact search snapshot once per public mutation call.

The incoming/outgoing-neighborhood approach is inspired by graph ANN maintenance,
including [FreshDiskANN's deletion discussion and Algorithm 4](https://arxiv.org/html/2105.09613v1#S4.SS2).
The reference [hnswlib update implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h)
also repairs neighborhoods and reconnects updated vectors. This implementation is
**not** a port of FreshDiskANN: it retains HNSW's hierarchy and pruning heuristic,
uses eager physical deletion rather than tombstones, and has no SSD merge pipeline.
It does not inherit another algorithm's recall-stability guarantees.

## Cost and concurrency

- Initial maintenance-cache construction takes O(N + E) time and memory; recovering
  free ordinal gaps also visits the ordinal high-water range. Thereafter, graph
  repair inspects affected neighborhoods, not every graph node. Hubs and large
  deletion batches can still require substantial work; this is not O(1) deletion.
  Entry-point replacement deterministically sorts the highest surviving level's
  ID bucket; that bucket can be large if all upper-layer nodes have disappeared.
- After activation, reverse edges are maintained during parallel batch insertion
  as well. That increases memory and mutation bookkeeping versus a build-only index.
- Free ordinals bound visited-array growth during fixed-size delete/insert churn.
  They are not tombstones; removed nodes and their vectors leave the mutable graph.
- The immutable compact snapshot is still recreated in O(ND + E) time and temporarily
  needs replacement array storage. This may dominate single-node mutation latency.
  Batch deletion avoids repeated snapshot refreshes; incremental snapshots remain
  future work.
- The managed write lock covers repair, reinsertion and snapshot publication.
  Readers cannot observe half-repaired graphs. Internal raw graph mutation helpers
  require exclusive access from their caller.
- Invalid replacement vectors fail before changing the graph. Unexpected runtime
  failures during in-place mutation have no transactional rollback. The search
  snapshot is invalidated before graph changes, so it cannot silently return deleted
  or outdated nodes afterward.
- Saves retain the current topology and vectors. Auxiliary maintenance structures
  are not serialized; the first mutation after loading reconstructs them, including
  reusable ordinal gaps. No persistence format change is required.

## Validation and limitations

Regression tests cover asymmetric edges, adjacent deletions, all layers, entry-point
removal, empty/singleton indexes, parallel insertions after deletion, slot reuse,
invalid updates, metadata, save/load, concurrent readers, degree limits, exact reverse
edge consistency, and recall after repeated deletions and large vector moves.

No local-pruning heuristic guarantees unchanged recall or global reachability for
every possible graph and mutation sequence. Monitor recall against exact neighbors
on representative queries, particularly after large or adversarial deletion batches;
recreate the index from active source vectors if quality deteriorates.

The reproducible Bible experiment is `examples/bible_benchmark/run_maintenance.sh`.
It compares local operations with the old full-rebuild path from identical starting
topologies. Both include snapshot creation; clone preparation, exact ground truth
and explicit GC are excluded. The first local deletion includes cache initialization.
This is **not** a comparison with hnswlib's constant-time deletion marks, which do
not physically remove nodes and repair their incident edges.
