package hnsw.search;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorDistance {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private VectorDistance() {}

    public static double normalizedCosine(double[] left, double[] right) {
        int length = left.length;
        int upperBound = SPECIES.loopBound(length);
        DoubleVector accumulator = DoubleVector.zero(SPECIES);
        int index = 0;

        for (; index < upperBound; index += SPECIES.length()) {
            DoubleVector leftVector = DoubleVector.fromArray(SPECIES, left, index);
            DoubleVector rightVector = DoubleVector.fromArray(SPECIES, right, index);
            accumulator = accumulator.add(leftVector.mul(rightVector));
        }

        double dotProduct = accumulator.reduceLanes(VectorOperators.ADD);
        for (; index < length; index++) {
            dotProduct += left[index] * right[index];
        }
        return 1.0 - dotProduct;
    }
}
