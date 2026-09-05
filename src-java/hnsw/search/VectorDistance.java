package hnsw.search;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorDistance {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

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

    public static double normalizedCosineFloat(float[] left, float[] right) {
        int length = left.length;
        int lanes = FLOAT_SPECIES.length();
        int stride = lanes * 4;
        int unrolledBound = length - length % stride;
        FloatVector first = FloatVector.zero(FLOAT_SPECIES);
        FloatVector second = FloatVector.zero(FLOAT_SPECIES);
        FloatVector third = FloatVector.zero(FLOAT_SPECIES);
        FloatVector fourth = FloatVector.zero(FLOAT_SPECIES);
        int index = 0;

        for (; index < unrolledBound; index += stride) {
            first = first.add(FloatVector.fromArray(FLOAT_SPECIES, left, index)
                    .mul(FloatVector.fromArray(FLOAT_SPECIES, right, index)));
            second = second.add(FloatVector.fromArray(FLOAT_SPECIES, left, index + lanes)
                    .mul(FloatVector.fromArray(FLOAT_SPECIES, right, index + lanes)));
            third = third.add(FloatVector.fromArray(FLOAT_SPECIES, left, index + lanes * 2)
                    .mul(FloatVector.fromArray(FLOAT_SPECIES, right, index + lanes * 2)));
            fourth = fourth.add(FloatVector.fromArray(FLOAT_SPECIES, left, index + lanes * 3)
                    .mul(FloatVector.fromArray(FLOAT_SPECIES, right, index + lanes * 3)));
        }

        FloatVector accumulator = first.add(second).add(third).add(fourth);
        int upperBound = FLOAT_SPECIES.loopBound(length);
        for (; index < upperBound; index += lanes) {
            accumulator = accumulator.add(FloatVector.fromArray(FLOAT_SPECIES, left, index)
                    .mul(FloatVector.fromArray(FLOAT_SPECIES, right, index)));
        }
        double dotProduct = accumulator.reduceLanes(VectorOperators.ADD);
        for (; index < length; index++) {
            dotProduct += (double) left[index] * right[index];
        }
        return 1.0 - dotProduct;
    }
}
