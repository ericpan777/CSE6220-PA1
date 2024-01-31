#include <iostream>
#include <iomanip> // For std::setprecision
#include <cstdlib>
#include <ctime>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <number_of_points>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = std::atoi(argv[1]);
    }

    // Broadcast n to all processors
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int pointsPerProc = n / size;
    int leftover = n % size;

    // Processor 0 will handle the leftovers if n is not divisible by size
    if (rank == 0) {
        pointsPerProc += leftover;
    }

    // Seed the random number generator differently for each processor
    std::srand(static_cast<unsigned int>(std::time(nullptr)) + rank);

    // Calculate the range of x for each processor
    double x_min = static_cast<double>(rank) / size;
    double x_max = static_cast<double>(rank + 1) / size;

    int pointsInsideCircle = 0;
    double start = MPI_Wtime(); // Start timing

    for (int i = 0; i < pointsPerProc; ++i) {
        // Scale x coordinate to fit the processor's stripe
        double x = x_min + (x_max - x_min) * (static_cast<double>(std::rand()) / RAND_MAX);
        double y = static_cast<double>(std::rand()) / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            ++pointsInsideCircle;
        }
    }

    int globalPointsInsideCircle;
    MPI_Reduce(&pointsInsideCircle, &globalPointsInsideCircle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime(); // End timing

    if (rank == 0) {
        double piEstimate = 4.0 * globalPointsInsideCircle / n;
        double timeTaken = end - start;
        // Print estimated value of Pi and time taken, comma-separated
        std::cout << std::fixed << std::setprecision(12) << piEstimate << ", " << timeTaken << std::endl;
    }

    MPI_Finalize();
    return 0;
}
