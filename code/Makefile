# Variables
EXECUTABLE1 = pi_calc_1
EXECUTABLE2 = pi_calc_2
NUM_POINTS = 1000000
MAX_N = 6

# Default target
all: run_sequence1 run_sequence2

# Target to run the sequence for pi_calc_1
run_sequence1:
	@for n in $(shell seq 1 $(MAX_N)); do \
		echo "Running $(EXECUTABLE1) with $$n processors"; \
		srun -n $$n ./$(EXECUTABLE1) $(NUM_POINTS) 2>/dev/null | grep -E '^[0-9]+\.[0-9]+, [0-9]+\.[0-9]+'; \
	done

# Target to run the sequence for pi_calc_2
run_sequence2:
	@for n in $(shell seq 1 $(MAX_N)); do \
		echo "Running $(EXECUTABLE2) with $$n processors"; \
		srun -n $$n ./$(EXECUTABLE2) $(NUM_POINTS) 2>/dev/null | grep -E '^[0-9]+\.[0-9]+, [0-9]+\.[0-9]+'; \
	done

# Clean target to clean up any generated files if necessary
clean:
	echo "Clean up!"

.PHONY: all run_sequence1 run_sequence2 clean
