import java.util.Random;

/**
 * ErgodicSemigroupSimulator
 *
 * A self-contained numerical experiment for the Ornstein-Uhlenbeck
 * stochastic process:
 *
 *      dX_t = -lambda X_t dt + sigma dW_t
 *
 * The program investigates several concepts that appear in the
 * ergodic theory of stochastic differential equations and, at a
 * more advanced level, stochastic partial differential equations.
 *
 * Main concepts:
 *
 *   1. Markov semigroups
 *   2. Monte Carlo approximation
 *   3. Invariant measures
 *   4. Ergodicity
 *   5. Loss of dependence on initial conditions
 *   6. Ergodic time averages
 *   7. Numerical convergence
 *
 * The stochastic differential equation is discretized using the
 * Euler-Maruyama method:
 *
 *      X_(n+1) =
 *          X_n - lambda X_n dt
 *          + sigma sqrt(dt) Z_n
 *
 * where Z_n is a standard normal random variable.
 *
 * No external libraries are required.
 */
public class ErgodicSemigroupSimulator {

    /*
     * A fixed seed makes the experiment reproducible.
     *
     * Changing this number changes the particular random realization,
     * but should not substantially change the statistical conclusions.
     */
    private static final Random RANDOM =
            new Random(20260918L);

    /*
     * The observable used throughout the experiment.
     *
     * In semigroup notation:
     *
     *      f(x) = x^2
     *
     * Studying an observable rather than only the raw trajectory
     * allows us to connect the numerical experiment directly to
     * the Markov semigroup P_t.
     */
    private static double observable(double x) {
        return x * x;
    }

    public static void main(String[] args) {

        /*
         * ============================================================
         * MODEL PARAMETERS
         * ============================================================
         */

        // Dissipation coefficient.
        double lambda = 1.0;

        // Noise intensity.
        double sigma = 2.0;

        /*
         * Numerical time step.
         *
         * Smaller dt generally gives a more accurate Euler-Maruyama
         * approximation, at the cost of more computation.
         */
        double dt = 0.005;

        // Length of each simulation.
        double totalTime = 20.0;

        // Number of Euler-Maruyama time steps.
        int steps = (int) Math.round(totalTime / dt);

        /*
         * Number of independent trajectories used for Monte Carlo
         * estimation of expectations.
         */
        int samples = 100_000;

        /*
         * Several initial conditions are deliberately chosen far
         * apart from one another.
         *
         * If the stochastic system is ergodic, its long-term
         * statistics should eventually become largely independent
         * of these initial conditions.
         */
        double[] initialConditions = {
                -10.0,
                -5.0,
                0.0,
                5.0,
                10.0
        };

        /*
         * ============================================================
         * HEADER
         * ============================================================
         */

        printHeader();

        System.out.println();
        System.out.println("MODEL PARAMETERS");
        System.out.println("----------------");
        System.out.printf("lambda              : %.4f%n", lambda);
        System.out.printf("sigma               : %.4f%n", sigma);
        System.out.printf("time step           : %.5f%n", dt);
        System.out.printf("simulation time     : %.2f%n", totalTime);
        System.out.printf("time steps          : %,d%n", steps);
        System.out.printf("Monte Carlo samples : %,d%n", samples);

        /*
         * ============================================================
         * INVARIANT MEASURE
         * ============================================================
         *
         * For lambda > 0, the Ornstein-Uhlenbeck process possesses
         * a unique Gaussian invariant probability measure.
         *
         * Its variance is:
         *
         *              sigma^2
         *      Var =  -----------
         *              2 lambda
         *
         * Since our observable is f(x)=x^2 and the invariant
         * distribution has mean zero,
         *
         *      integral x^2 dmu(x)
         *
         * is exactly this variance.
         */

        double theoreticalVariance =
                (sigma * sigma) / (2.0 * lambda);

        System.out.println();
        System.out.println("INVARIANT MEASURE");
        System.out.println("-----------------");
        System.out.printf(
                "Theoretical invariant variance : %.8f%n",
                theoreticalVariance
        );

        System.out.println();
        System.out.println(
                "The experiment will attempt to recover this value"
        );
        System.out.println(
                "numerically from stochastic trajectories."
        );

        /*
         * ============================================================
         * EXPERIMENT 1
         *
         * MARKOV SEMIGROUP
         * ============================================================
         *
         * Given a stochastic process X_t^x starting from x, its
         * Markov semigroup acts on an observable f according to
         *
         *      P_t f(x) = E[f(X_t^x)].
         *
         * We approximate this expectation using Monte Carlo:
         *
         *                  N
         *                 ---
         *      P_t f(x) ~= \  f(X_t^(j))
         *                 /   N
         *                 ---
         *                 j=1
         *
         * Here f(x)=x^2.
         */

        System.out.println();
        printSection(
                "EXPERIMENT 1: MARKOV SEMIGROUP"
        );

        System.out.println(
                "Approximating P_t f(x) with f(x)=x^2."
        );
        System.out.println();

        System.out.printf(
                "%16s %20s %20s%n",
                "Initial x",
                "P_t f(x)",
                "Distance from invariant"
        );

        System.out.println(
                "------------------------------------------------------------"
        );

        for (double initial : initialConditions) {

            double semigroupValue =
                    simulateSemigroup(
                            initial,
                            lambda,
                            sigma,
                            dt,
                            steps,
                            samples
                    );

            double distance =
                    Math.abs(
                            semigroupValue
                                    - theoreticalVariance
                    );

            System.out.printf(
                    "%16.2f %20.8f %20.8f%n",
                    initial,
                    semigroupValue,
                    distance
            );
        }

        /*
         * ============================================================
         * EXPERIMENT 2
         *
         * LOSS OF INITIAL-CONDITION MEMORY
         * ============================================================
         *
         * One way to numerically investigate ergodicity is to compare
         *
         *      P_t f(x)
         *
         * for different initial conditions.
         *
         * We use x=-10 and x=10.
         *
         * Define
         *
         *      D(t)
         *        = |P_t f(-10) - P_t f(10)|.
         *
         * If the influence of the initial condition disappears,
         * D(t) should become small at large times.
         */

        System.out.println();
        printSection(
                "EXPERIMENT 2: INITIAL-CONDITION MEMORY"
        );

        double initialA = -10.0;
        double initialB = 10.0;

        double[] checkpoints = {
                0.0,
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                15.0,
                20.0
        };

        System.out.printf(
                "%10s %20s %20s %20s%n",
                "Time",
                "P_t f(-10)",
                "P_t f(10)",
                "Difference"
        );

        System.out.println(
                "--------------------------------------------------------------------------"
        );

        for (double time : checkpoints) {

            int checkpointSteps =
                    (int) Math.round(time / dt);

            double valueA =
                    simulateSemigroup(
                            initialA,
                            lambda,
                            sigma,
                            dt,
                            checkpointSteps,
                            samples
                    );

            double valueB =
                    simulateSemigroup(
                            initialB,
                            lambda,
                            sigma,
                            dt,
                            checkpointSteps,
                            samples
                    );

            double difference =
                    Math.abs(valueA - valueB);

            System.out.printf(
                    "%10.2f %20.8f %20.8f %20.8f%n",
                    time,
                    valueA,
                    valueB,
                    difference
            );
        }

        System.out.println();
        System.out.println(
                "The decreasing discrepancy illustrates the numerical"
        );
        System.out.println(
                "loss of information about the starting state."
        );

        /*
         * ============================================================
         * EXPERIMENT 3
         *
         * ERGODIC TIME AVERAGE
         * ============================================================
         *
         * Ergodicity connects long-time averages along a trajectory
         * with averages against the invariant measure.
         *
         * We investigate
         *
         *             1
         *      A_T = ---
         *             T
         *
         *                 T
         *                 /
         *                 | X_t^2 dt.
         *                 /
         *                 0
         *
         * Numerically:
         *
         *             1
         *      A_T = ---
         *             T
         *
         *             N
         *             ---
         *             \ X_n^2 dt.
         *             ---
         *             n=1
         *
         * The expected long-time value is
         *
         *             sigma^2
         *      ---------------------.
         *             2 lambda
         */

        System.out.println();
        printSection(
                "EXPERIMENT 3: ERGODIC TIME AVERAGE"
        );

        double[] averagingTimes = {
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
                50.0,
                100.0
        };

        System.out.printf(
                "%12s %24s %24s%n",
                "Time T",
                "Time Average",
                "Absolute Error"
        );

        System.out.println(
                "----------------------------------------------------------------"
        );

        for (double averagingTime : averagingTimes) {

            double average =
                    ergodicAverage(
                            5.0,
                            lambda,
                            sigma,
                            dt,
                            averagingTime
                    );

            double error =
                    Math.abs(
                            average
                                    - theoreticalVariance
                    );

            System.out.printf(
                    "%12.2f %24.10f %24.10f%n",
                    averagingTime,
                    average,
                    error
            );
        }

        /*
         * ============================================================
         * EXPERIMENT 4
         *
         * EMPIRICAL STATIONARY DISTRIBUTION
         * ============================================================
         *
         * We now simulate many trajectories for a long time and
         * examine basic statistics of their final states.
         *
         * The theoretical invariant distribution is
         *
         *          sigma^2
         *      N(0, -----------).
         *          2 lambda
         *
         * Therefore:
         *
         *      E[X]    = 0
         *
         *      Var[X]  = sigma^2 / (2 lambda)
         *
         * The experiment estimates both quantities.
         */

        System.out.println();
        printSection(
                "EXPERIMENT 4: EMPIRICAL STATIONARY STATISTICS"
        );

        int stationarySamples = 100_000;

        double[] finalStates =
                generateFinalStates(
                        0.0,
                        lambda,
                        sigma,
                        dt,
                        steps,
                        stationarySamples
                );

        double empiricalMean =
                calculateMean(finalStates);

        double empiricalVariance =
                calculateVariance(
                        finalStates,
                        empiricalMean
                );

        System.out.printf(
                "Empirical mean       : %.10f%n",
                empiricalMean
        );

        System.out.printf(
                "Theoretical mean     : %.10f%n",
                0.0
        );

        System.out.printf(
                "Empirical variance   : %.10f%n",
                empiricalVariance
        );

        System.out.printf(
                "Theoretical variance : %.10f%n",
                theoreticalVariance
        );

        System.out.printf(
                "Variance error       : %.10f%n",
                Math.abs(
                        empiricalVariance
                                - theoreticalVariance
                )
        );

        /*
         * ============================================================
         * EXPERIMENT 5
         *
         * SEMIGROUP CONVERGENCE AT MULTIPLE TIMES
         * ============================================================
         *
         * Rather than only examining the final time, this experiment
         * follows the evolution of P_t f(x) from a single initial
         * state.
         *
         * This gives a numerical picture of convergence toward the
         * invariant expectation.
         */

        System.out.println();
        printSection(
                "EXPERIMENT 5: SEMIGROUP CONVERGENCE"
        );

        double convergenceInitial = 10.0;

        double[] times = {
                0.0,
                0.25,
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
                7.5,
                10.0,
                15.0,
                20.0
        };

        System.out.printf(
                "%10s %22s %22s%n",
                "Time",
                "P_t f(10)",
                "|P_t f - invariant|"
        );

        System.out.println(
                "---------------------------------------------------------------"
        );

        for (double time : times) {

            int numberOfSteps =
                    (int) Math.round(time / dt);

            double value =
                    simulateSemigroup(
                            convergenceInitial,
                            lambda,
                            sigma,
                            dt,
                            numberOfSteps,
                            samples
                    );

            double error =
                    Math.abs(
                            value
                                    - theoreticalVariance
                    );

            System.out.printf(
                    "%10.2f %22.10f %22.10f%n",
                    time,
                    value,
                    error
            );
        }

        /*
         * ============================================================
         * FINAL INTERPRETATION
         * ============================================================
         */

        System.out.println();
        printSection(
                "INTERPRETATION"
        );

        System.out.println(
                "The experiments provide a numerical illustration of"
        );
        System.out.println(
                "several central ideas in stochastic ergodic theory:"
        );

        System.out.println();
        System.out.println(
                "  * Markov semigroup evolution:"
        );
        System.out.println(
                "      P_t f(x) = E[f(X_t^x)]"
        );

        System.out.println();
        System.out.println(
                "  * Invariant statistical behavior:"
        );
        System.out.println(
                "      P_t f(x) -> integral f dmu"
        );

        System.out.println();
        System.out.println(
                "  * Ergodic time averaging:"
        );
        System.out.println(
                "      (1/T) integral_0^T f(X_t)dt"
        );
        System.out.println(
                "      approaches the invariant expectation."
        );

        System.out.println();
        System.out.println(
                "  * Loss of initial-condition memory:"
        );
        System.out.println(
                "      different initial states become statistically"
        );
        System.out.println(
                "      indistinguishable at sufficiently long times."
        );

        System.out.println();
        System.out.println(
                "These numerical observations are not a mathematical"
        );
        System.out.println(
                "proof of ergodicity. They provide computational"
        );
        System.out.println(
                "intuition for concepts that become substantially"
        );
        System.out.println(
                "more difficult in infinite-dimensional stochastic"
        );
        System.out.println(
                "partial differential equations."
        );

        /*
         * ============================================================
         * CONNECTION TO FUTURE SPDE EXPERIMENTS
         * ============================================================
         */

        System.out.println();
        printSection(
                "NEXT STEP: INFINITE-DIMENSIONAL DYNAMICS"
        );

        System.out.println(
                "The natural extension is to replace X_t with a"
        );
        System.out.println(
                "spatial field u(x,t) and study a stochastic heat"
        );
        System.out.println(
                "equation such as:"
        );

        System.out.println();
        System.out.println(
                "      du = (kappa Delta u - alpha u)dt"
        );
        System.out.println(
                "           + sigma dW_t."
        );

        System.out.println();
        System.out.println(
                "After spatial discretization, this produces a"
        );
        System.out.println(
                "high-dimensional stochastic dynamical system."
        );

        System.out.println();
        System.out.println(
                "That model will provide the bridge from this"
        );
        System.out.println(
                "finite-dimensional experiment toward the"
        );
        System.out.println(
                "infinite-dimensional SPDE setting."
        );

        System.out.println();
        System.out.println(
                "============================================================"
        );
        System.out.println(
                "Simulation complete."
        );
        System.out.println(
                "============================================================"
        );
    }

    /**
     * Approximates the Markov semigroup
     *
     *      P_t f(x) = E[f(X_t^x)]
     *
     * using Monte Carlo simulation.
     *
     * The observable is f(x)=x^2.
     */
    private static double simulateSemigroup(
            double initial,
            double lambda,
            double sigma,
            double dt,
            int steps,
            int samples
    ) {

        double sqrtDt =
                Math.sqrt(dt);

        double sum =
                0.0;

        for (int sample = 0;
             sample < samples;
             sample++) {

            double x =
                    initial;

            for (int n = 0;
                 n < steps;
                 n++) {

                /*
                 * Generate the Brownian increment.
                 *
                 * dW ~= sqrt(dt) * Z
                 *
                 * where Z ~ N(0,1).
                 */
                double gaussian =
                        RANDOM.nextGaussian();

                /*
                 * Euler-Maruyama update:
                 *
                 * X_(n+1)
                 * =
                 * X_n
                 * -
                 * lambda X_n dt
                 * +
                 * sigma sqrt(dt) Z.
                 */
                x +=
                        -lambda * x * dt
                        +
                        sigma * sqrtDt * gaussian;
            }

            /*
             * Apply f(x)=x^2.
             */
            sum +=
                    observable(x);
        }

        /*
         * Monte Carlo expectation.
         */
        return sum / samples;
    }

    /**
     * Computes the time average
     *
     *      (1/T) integral_0^T X_t^2 dt
     *
     * using a discrete numerical approximation.
     */
    private static double ergodicAverage(
            double initial,
            double lambda,
            double sigma,
            double dt,
            double totalTime
    ) {

        int steps =
                (int) Math.round(
                        totalTime / dt
                );

        double sqrtDt =
                Math.sqrt(dt);

        double x =
                initial;

        double accumulated =
                0.0;

        for (int n = 0;
             n < steps;
             n++) {

            /*
             * Approximate the integral of x^2.
             */
            accumulated +=
                    observable(x) * dt;

            double gaussian =
                    RANDOM.nextGaussian();

            x +=
                    -lambda * x * dt
                    +
                    sigma * sqrtDt * gaussian;
        }

        return accumulated / totalTime;
    }

    /**
     * Generates the final states of many independent
     * Ornstein-Uhlenbeck trajectories.
     */
    private static double[] generateFinalStates(
            double initial,
            double lambda,
            double sigma,
            double dt,
            int steps,
            int samples
    ) {

        double[] states =
                new double[samples];

        double sqrtDt =
                Math.sqrt(dt);

        for (int sample = 0;
             sample < samples;
             sample++) {

            double x =
                    initial;

            for (int n = 0;
                 n < steps;
                 n++) {

                double gaussian =
                        RANDOM.nextGaussian();

                x +=
                        -lambda * x * dt
                        +
                        sigma * sqrtDt * gaussian;
            }

            states[sample] =
                    x;
        }

        return states;
    }

    /**
     * Computes the empirical mean of an array.
     */
    private static double calculateMean(
            double[] values
    ) {

        double sum =
                0.0;

        for (double value : values) {
            sum += value;
        }

        return sum / values.length;
    }

    /**
     * Computes the empirical variance.
     *
     * The denominator N-1 is used to obtain the usual unbiased
     * sample variance estimator.
     */
    private static double calculateVariance(
            double[] values,
            double mean
    ) {

        double squaredDeviation =
                0.0;

        for (double value : values) {

            double difference =
                    value - mean;

            squaredDeviation +=
                    difference * difference;
        }

        if (values.length <= 1) {
            return 0.0;
        }

        return squaredDeviation /
                (values.length - 1);
    }

    /**
     * Prints the program title.
     */
    private static void printHeader() {

        System.out.println(
                "============================================================"
        );

        System.out.println(
                "           ERGODIC SEMIGROUP SIMULATOR"
        );

        System.out.println(
                "      Stochastic Dynamics Research Experiment"
        );

        System.out.println(
                "============================================================"
        );

        System.out.println();
        System.out.println(
                "Model: Ornstein-Uhlenbeck stochastic process"
        );

        System.out.println(
                "Equation: dX_t = -lambda X_t dt + sigma dW_t"
        );
    }

    /**
     * Prints a formatted section heading.
     */
    private static void printSection(
            String title
    ) {

        System.out.println(
                "============================================================"
        );

        System.out.println(
                title
        );

        System.out.println(
                "============================================================"
        );
    }
}
