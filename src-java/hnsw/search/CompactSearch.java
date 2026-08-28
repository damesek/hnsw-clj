package hnsw.search;

import java.util.Arrays;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class CompactSearch {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final SearchResult EMPTY_RESULT =
            new SearchResult(new String[0], new double[0]);

    private final String[] ids;
    private final double[] vectors;
    private final int dimension;
    private final int entryPoint;
    private final int maxLevel;
    private final int[][] offsetsByLevel;
    private final int[][] neighborsByLevel;
    private final ThreadLocal<Workspace> workspaces;

    public CompactSearch(
            String[] ids,
            double[] vectors,
            int dimension,
            int entryPoint,
            int maxLevel,
            int[][] offsetsByLevel,
            int[][] neighborsByLevel) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        if (vectors.length != Math.multiplyExact(ids.length, dimension)) {
            throw new IllegalArgumentException("vector storage size does not match ids and dimension");
        }
        if (ids.length > 0 && (entryPoint < 0 || entryPoint >= ids.length)) {
            throw new IllegalArgumentException("entry point is outside the index");
        }
        if (offsetsByLevel.length != neighborsByLevel.length
                || offsetsByLevel.length != maxLevel + 1) {
            throw new IllegalArgumentException("invalid level storage");
        }
        this.ids = ids;
        this.vectors = vectors;
        this.dimension = dimension;
        this.entryPoint = entryPoint;
        this.maxLevel = maxLevel;
        this.offsetsByLevel = offsetsByLevel;
        this.neighborsByLevel = neighborsByLevel;
        this.workspaces = ThreadLocal.withInitial(() -> new Workspace(ids.length, dimension));
    }

    public SearchResult search(double[] query, int k, int efSearch) {
        if (query.length != dimension) {
            throw new IllegalArgumentException(
                    "query dimension " + query.length + " does not match index dimension " + dimension);
        }
        if (k <= 0 || ids.length == 0) {
            return EMPTY_RESULT;
        }

        int resultCount = Math.min(k, ids.length);
        int ef = Math.min(ids.length, Math.max(resultCount, efSearch));
        Workspace workspace = workspaces.get();
        double[] normalizedQuery = workspace.normalize(query);
        int nearest = entryPoint;
        double nearestDistance = distance(normalizedQuery, nearest);

        for (int level = maxLevel; level > 0; level--) {
            boolean changed;
            do {
                changed = false;
                int[] offsets = offsetsByLevel[level];
                int[] neighbors = neighborsByLevel[level];
                int end = offsets[nearest + 1];
                for (int position = offsets[nearest]; position < end; position++) {
                    int candidate = neighbors[position];
                    double candidateDistance = distance(normalizedQuery, candidate);
                    if (candidateDistance < nearestDistance) {
                        nearest = candidate;
                        nearestDistance = candidateDistance;
                        changed = true;
                    }
                }
            } while (changed);
        }

        workspace.begin(ids.length, ef);
        workspace.visit(nearest);
        workspace.addCandidate(nearest, nearestDistance);
        workspace.addResult(nearest, nearestDistance);
        int[] offsets = offsetsByLevel[0];
        int[] neighbors = neighborsByLevel[0];

        while (workspace.candidateSize > 0) {
            int current = workspace.candidateNodes[0];
            double currentDistance = workspace.candidateDistances[0];
            double worstDistance = workspace.resultDistances[0];
            if (workspace.resultSize >= ef && currentDistance > worstDistance) {
                break;
            }
            workspace.popCandidate();

            int end = offsets[current + 1];
            for (int position = offsets[current]; position < end; position++) {
                int candidate = neighbors[position];
                if (!workspace.visit(candidate)) {
                    continue;
                }
                double candidateDistance = distance(normalizedQuery, candidate);
                worstDistance = workspace.resultDistances[0];
                if (workspace.resultSize < ef || candidateDistance < worstDistance) {
                    workspace.addCandidate(candidate, candidateDistance);
                    workspace.addResult(candidate, candidateDistance);
                    if (workspace.resultSize > ef) {
                        workspace.popResult();
                    }
                }
            }
        }

        while (workspace.resultSize > resultCount) {
            workspace.popResult();
        }
        String[] resultIds = new String[workspace.resultSize];
        double[] resultDistances = new double[workspace.resultSize];
        for (int position = workspace.resultSize - 1; position >= 0; position--) {
            resultIds[position] = ids[workspace.resultNodes[0]];
            resultDistances[position] = workspace.resultDistances[0];
            workspace.popResult();
        }
        return new SearchResult(resultIds, resultDistances);
    }

    private double distance(double[] query, int node) {
        int vectorOffset = node * dimension;
        int upperBound = SPECIES.loopBound(dimension);
        DoubleVector accumulator = DoubleVector.zero(SPECIES);
        int index = 0;
        for (; index < upperBound; index += SPECIES.length()) {
            DoubleVector queryVector = DoubleVector.fromArray(SPECIES, query, index);
            DoubleVector storedVector = DoubleVector.fromArray(
                    SPECIES, vectors, vectorOffset + index);
            accumulator = accumulator.add(queryVector.mul(storedVector));
        }
        double dotProduct = accumulator.reduceLanes(VectorOperators.ADD);
        for (; index < dimension; index++) {
            dotProduct += query[index] * vectors[vectorOffset + index];
        }
        return 1.0 - dotProduct;
    }

    public static final class SearchResult {
        private final String[] ids;
        private final double[] distances;

        private SearchResult(String[] ids, double[] distances) {
            this.ids = ids;
            this.distances = distances;
        }

        public int size() {
            return ids.length;
        }

        public String idAt(int index) {
            return ids[index];
        }

        public double distanceAt(int index) {
            return distances[index];
        }
    }

    private static final class Workspace {
        private double[] normalizedQuery;
        private int[] visited;
        private int generation;
        private int[] candidateNodes;
        private double[] candidateDistances;
        private int candidateSize;
        private int[] resultNodes;
        private double[] resultDistances;
        private int resultSize;

        private Workspace(int nodeCount, int dimension) {
            normalizedQuery = new double[dimension];
            visited = new int[Math.max(1, nodeCount)];
            candidateNodes = new int[64];
            candidateDistances = new double[64];
            resultNodes = new int[64];
            resultDistances = new double[64];
        }

        private double[] normalize(double[] query) {
            if (normalizedQuery.length != query.length) {
                normalizedQuery = new double[query.length];
            }
            double normSquared = 0.0;
            for (double value : query) {
                normSquared += value * value;
            }
            if (normSquared > 0.0) {
                double scale = 1.0 / Math.sqrt(normSquared);
                for (int index = 0; index < query.length; index++) {
                    normalizedQuery[index] = query[index] * scale;
                }
            } else {
                System.arraycopy(query, 0, normalizedQuery, 0, query.length);
            }
            return normalizedQuery;
        }

        private void begin(int nodeCount, int ef) {
            if (visited.length < nodeCount) {
                visited = new int[nodeCount];
            }
            if (generation == Integer.MAX_VALUE) {
                Arrays.fill(visited, 0);
                generation = 1;
            } else {
                generation++;
            }
            candidateSize = 0;
            resultSize = 0;
            ensureCandidateCapacity(Math.max(64, ef * 2));
            ensureResultCapacity(Math.max(64, ef + 1));
        }

        private boolean visit(int node) {
            if (visited[node] == generation) {
                return false;
            }
            visited[node] = generation;
            return true;
        }

        private void ensureCandidateCapacity(int capacity) {
            if (candidateNodes.length < capacity) {
                int newCapacity = Math.max(capacity, candidateNodes.length * 2);
                candidateNodes = Arrays.copyOf(candidateNodes, newCapacity);
                candidateDistances = Arrays.copyOf(candidateDistances, newCapacity);
            }
        }

        private void ensureResultCapacity(int capacity) {
            if (resultNodes.length < capacity) {
                int newCapacity = Math.max(capacity, resultNodes.length * 2);
                resultNodes = Arrays.copyOf(resultNodes, newCapacity);
                resultDistances = Arrays.copyOf(resultDistances, newCapacity);
            }
        }

        private void addCandidate(int node, double distance) {
            ensureCandidateCapacity(candidateSize + 1);
            int position = candidateSize++;
            while (position > 0) {
                int parent = (position - 1) >>> 1;
                if (candidateDistances[parent] <= distance) {
                    break;
                }
                candidateNodes[position] = candidateNodes[parent];
                candidateDistances[position] = candidateDistances[parent];
                position = parent;
            }
            candidateNodes[position] = node;
            candidateDistances[position] = distance;
        }

        private void popCandidate() {
            int last = --candidateSize;
            if (last == 0) {
                return;
            }
            int node = candidateNodes[last];
            double distance = candidateDistances[last];
            int position = 0;
            int half = candidateSize >>> 1;
            while (position < half) {
                int child = (position << 1) + 1;
                int right = child + 1;
                if (right < candidateSize
                        && candidateDistances[right] < candidateDistances[child]) {
                    child = right;
                }
                if (distance <= candidateDistances[child]) {
                    break;
                }
                candidateNodes[position] = candidateNodes[child];
                candidateDistances[position] = candidateDistances[child];
                position = child;
            }
            candidateNodes[position] = node;
            candidateDistances[position] = distance;
        }

        private void addResult(int node, double distance) {
            ensureResultCapacity(resultSize + 1);
            int position = resultSize++;
            while (position > 0) {
                int parent = (position - 1) >>> 1;
                if (resultDistances[parent] >= distance) {
                    break;
                }
                resultNodes[position] = resultNodes[parent];
                resultDistances[position] = resultDistances[parent];
                position = parent;
            }
            resultNodes[position] = node;
            resultDistances[position] = distance;
        }

        private void popResult() {
            int last = --resultSize;
            if (last == 0) {
                return;
            }
            int node = resultNodes[last];
            double distance = resultDistances[last];
            int position = 0;
            int half = resultSize >>> 1;
            while (position < half) {
                int child = (position << 1) + 1;
                int right = child + 1;
                if (right < resultSize && resultDistances[right] > resultDistances[child]) {
                    child = right;
                }
                if (distance >= resultDistances[child]) {
                    break;
                }
                resultNodes[position] = resultNodes[child];
                resultDistances[position] = resultDistances[child];
                position = child;
            }
            resultNodes[position] = node;
            resultDistances[position] = distance;
        }
    }
}
