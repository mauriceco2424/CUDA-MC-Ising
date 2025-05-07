#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdio.h>
#include <random>


// ---------------------------
// CONSTANT MEMORY
// ---------------------------
__constant__ float bf[2]; // bf[0] for delta_E=4, bf[1] for delta_E=8

// critical temperature for 2-D square Ising (Onsager)
constexpr float Tc = 2.269185f;

/* return the number of sweeps to skip between measurements
   – long skips only very close to Tc                                     */
__host__ __device__ inline int choose_skip(float T)
{
    const float dT = fabsf(T - Tc);
    if (dT > 0.06f) return 10;     // far from critical region
    else if (dT > 0.03f) return 50; // shoulder
    else                 return 100; // |T−Tc| ≤ 0.03
}



// ---------------------------
// CONFIGURABLE BLOCK/THREAD SETUP
// ---------------------------
#define BLOCK_X 32
#define BLOCK_Y 32
#define REDUCTION_THREADS 256

// ---------------------------
// CUDA KERNELS
// ---------------------------
__global__ void init_kernel(int* spins, curandState_t* states, unsigned long long seed, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < L && idy < L) {
        int index = idy * L + idx;
        curand_init(seed + index, index, 0, &states[index]);
        float r = curand_uniform(&states[index]);
        spins[index] = (r < 0.5f) ? -1 : 1;
    }
}


__global__ void metropolis_kernel(int* spins, curandState_t* states, int L, int sublattice) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < L && idy < L && (idx + idy) % 2 == sublattice) {
        int index = idy * L + idx;
        int s = spins[index];

        int right = (idx + 1) % L;
        int left  = (idx - 1 + L) % L;
        int down  = (idy + 1) % L;
        int up    = (idy - 1 + L) % L;

        int sum_neighbors = spins[idy * L + right] +
                            spins[idy * L + left] +
                            spins[down * L + idx] +
                            spins[up * L + idx];

        int delta_E = 2 * s * sum_neighbors;

        float prob;
        if (delta_E <= 0) {
            prob = 1.0f;
        } else if (delta_E == 4) {
            prob = bf[0];
        } else if (delta_E == 8) {
            prob = bf[1];
        } else {
            prob = 0.0f;
        }

        float r = curand_uniform(&states[index]);
        if (r < prob) {
            spins[index] = -s;
        }
    }
}

__global__ void metropolis_kernel_with_mask(int* spins, curandState_t* states, int L, int sublattice, int* active_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < L && idy < L && (idx + idy) % 2 == sublattice) {
        int index = idy * L + idx;
        if (active_mask[index] == 0) return;  // frozen → do nothing
        int s = spins[index];

        int right = (idx + 1) % L;
        int left = (idx - 1 + L) % L;
        int down = (idy + 1) % L;
        int up = (idy - 1 + L) % L;

        int sum_neighbors = spins[idy * L + right] +
            spins[idy * L + left] +
            spins[down * L + idx] +
            spins[up * L + idx];

        int delta_E = 2 * s * sum_neighbors;

        float prob;
        if (delta_E <= 0) {
            prob = 1.0f;
        }
        else if (delta_E == 4) {
            prob = bf[0];
        }
        else if (delta_E == 8) {
            prob = bf[1];
        }
        else {
            prob = 0.0f;
        }

        float r = curand_uniform(&states[index]);
        if (r < prob) {
            spins[index] = -s;
        }
    }
}

__global__ void reduction_kernel(int* spins, int* partial_sums, int N) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    int sum = 0;
    while (idx < N) {
        sum += spins[idx];
        idx += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void energy_reduction_kernel(int* spins, int* partial_sums, int L, int N) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    int sum_contrib = 0;
    while (idx < N) {
        int i = idx / L;
        int j = idx % L;
        int right = (j + 1) % L;
        int down  = (i + 1) % L;
        int index_right = i * L + right;
        int index_down  = down * L + j;
        sum_contrib += spins[idx] * (spins[index_right] + spins[index_down]);
        idx += gridSize;
    }
    sdata[tid] = sum_contrib;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// metropolis_kernel_with_field.cu
__global__ void metropolis_kernel_with_field(int* spins, curandState_t* states, int L, int sublattice, float h, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < L && idy < L && ((idx + idy) & 1) == sublattice)  // parity test
    {
        int index = idy * L + idx;
        int s = spins[index];          // current spin (+1 / –1)

        // 4-nearest neighbours (periodic boundaries)
        int right = (idx + 1) % L;
        int left = (idx - 1 + L) % L;
        int down = (idy + 1) % L;
        int up = (idy - 1 + L) % L;

        int sum_n =
            spins[idy * L + right] +
            spins[idy * L + left] +
            spins[down * L + idx] +
            spins[up * L + idx];

        // ΔE = 2 s (Σ_n  s_n  +  h)
        float delta_E = 2.0f * s * (float(sum_n) + h);

        float p = 1.0f;                        // Metropolis probability
        if (delta_E > 0.0f)
            p = expf(-beta * delta_E);

        if (curand_uniform(&states[index]) < p)
            spins[index] = -s;                 // accept the flip
    }
}





// ---------------------------
// HELPER MACRO
// ---------------------------
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
}

// ---------------------------
// 3.1 Task 1: Magnetization vs n_sweeps for fixed temperatures
// ---------------------------
void run_magnetization_vs_nsweeps() {
    const int L = 40;
    const int N = L * L;
    const int n_sweeps = 5000;
    const float temperatures[] = { 2.1f, 2.3f, 2.5f };
    const int num_temps = sizeof(temperatures) / sizeof(temperatures[0]);

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
        (L + blockDim.y - 1) / blockDim.y);

    int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;

    auto global_start = std::chrono::high_resolution_clock::now();

    for (int temp_idx = 0; temp_idx < num_temps; temp_idx++) {
        float T = temperatures[temp_idx];
        float beta = 1.0f / T;
        float bf_host[2] = { expf(-4.0f * beta), expf(-8.0f * beta) };
        cudaMemcpyToSymbol(bf, bf_host, 2 * sizeof(float));

        int* spins;
        curandState_t* states;
        int* partial_sums;
        cudaMalloc(&spins, N * sizeof(int));
        cudaMalloc(&states, N * sizeof(curandState_t));
        cudaMalloc(&partial_sums, num_blocks * sizeof(int));
        cudaCheckError();

        unsigned long long seed = 1234 + temp_idx * 100;
        init_kernel <<<gridDim, blockDim >>> (spins, states, seed, L);
        cudaDeviceSynchronize();
        cudaCheckError();

        // --- Save initial spin configuration as snapshot ---
        std::vector<int> host_spins(N);
        cudaMemcpy(host_spins.data(), spins, N * sizeof(int), cudaMemcpyDeviceToHost);


        std::vector<float> magnetizations(n_sweeps);

        auto start = std::chrono::high_resolution_clock::now();

        for (int sweep = 0; sweep < n_sweeps; sweep++) {
            metropolis_kernel <<<gridDim, blockDim >>> (spins, states, L, 0);
            cudaDeviceSynchronize();
            metropolis_kernel <<<gridDim, blockDim >>> (spins, states, L, 1);
            cudaDeviceSynchronize();

            reduction_kernel << <num_blocks, REDUCTION_THREADS, REDUCTION_THREADS * sizeof(int) >> > (spins, partial_sums, N);
            cudaDeviceSynchronize();
            cudaCheckError();
            std::vector<int> host_partial_sums(num_blocks);
            cudaMemcpy(host_partial_sums.data(), partial_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

            int total_sum = 0;
            for (auto val : host_partial_sums) total_sum += val;
            magnetizations[sweep] = static_cast<float>(total_sum) / static_cast<float>(N);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "T = " << T << ": Elapsed time = " << elapsed.count() << " seconds\n";

        char filename[64];
        sprintf(filename, "magnetization_T_%.1f.txt", T);
        std::ofstream outfile(filename);
        outfile << "# Magnetization per spin at T=" << T << "\n";
        for (auto m : magnetizations) outfile << m << "\n";
        outfile.close();

        cudaFree(spins);
        cudaFree(states);
        cudaFree(partial_sums);
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = global_end - global_start;
    std::cout << "Total runtime: " << total_elapsed.count() << " seconds\n";
}

// ---------------------------
// 3.1 Task 2: Energy and magnetization versus temperature
// ---------------------------
void run_E_M_vs_T_plus_snapshot() {
    const int L = 40;
    const int N = L * L;
    const int num_equil_sweeps = 1000;
    const int num_measurements = 500;
    int n_skip = 20;
    int num_snapshots = 5;

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
        (L + blockDim.y - 1) / blockDim.y);

    int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;


    std::vector<float> temperatures;
    for (float T = 1.0f; T <= 4.0f; T += 0.1f) {
        temperatures.push_back(T);
    }

    std::vector<float> avg_energies(temperatures.size());
    std::vector<float> avg_magnetizations(temperatures.size());

    std::vector<size_t> snapshot_indices;
    for (size_t i = 0; i < num_snapshots; i++) {
        snapshot_indices.push_back(i * (temperatures.size() - 1) / (num_snapshots - 1));
    }

    for (size_t temp_idx = 0; temp_idx < temperatures.size(); temp_idx++) {
        float T = temperatures[temp_idx];
        float beta = 1.0f / T;
        float bf_host[2] = { expf(-4.0f * beta), expf(-8.0f * beta) };
        cudaMemcpyToSymbol(bf, bf_host, 2 * sizeof(float));

        int* spins;
        curandState_t* states;
        int* partial_sums;
        cudaMalloc(&spins, N * sizeof(int));
        cudaMalloc(&states, N * sizeof(curandState_t));
        cudaMalloc(&partial_sums, num_blocks * sizeof(int));
        cudaCheckError();

        unsigned long long seed = 1234 + temp_idx * 100;
        init_kernel <<<gridDim, blockDim >>> (spins, states, seed, L);
        cudaDeviceSynchronize();
        cudaCheckError();

        // Equilibration (warmup)
        for (int sweep = 0; sweep < num_equil_sweeps; sweep++) {
            metropolis_kernel <<<gridDim, blockDim >>> (spins, states, L, 0);
            cudaDeviceSynchronize();
            metropolis_kernel <<<gridDim, blockDim >>> (spins, states, L, 1);
            cudaDeviceSynchronize();
        }

        double sum_M = 0.0;
        double sum_E = 0.0;

        // Measurements with skipping
        for (int measurement = 0; measurement < num_measurements; measurement++) {
            // Do n_skip sweeps before each measurement
            for (int skip = 0; skip < n_skip; skip++) {
                metropolis_kernel <<<gridDim, blockDim >>> (spins, states, L, 0);
                cudaDeviceSynchronize();
                metropolis_kernel <<<gridDim, blockDim >>> (spins, states, L, 1);
                cudaDeviceSynchronize();
            }

            // Magnetization
            reduction_kernel << <num_blocks, REDUCTION_THREADS, REDUCTION_THREADS * sizeof(int) >> > (spins, partial_sums, N);
            cudaDeviceSynchronize();
            cudaCheckError();
            std::vector<int> host_partial_sums(num_blocks);
            cudaMemcpy(host_partial_sums.data(), partial_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

            int total_sum = 0;
            for (auto val : host_partial_sums) total_sum += val;
            sum_M += fabs(static_cast<float>(total_sum) / static_cast<float>(N));

            // Energy
            energy_reduction_kernel << <num_blocks, REDUCTION_THREADS, REDUCTION_THREADS * sizeof(int) >> > (spins, partial_sums, L, N);
            cudaDeviceSynchronize();
            cudaCheckError();
            cudaMemcpy(host_partial_sums.data(), partial_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
            int total_bond_sum = 0;
            for (auto val : host_partial_sums) total_bond_sum += val;
            sum_E += -(float)total_bond_sum / static_cast<float>(N);
        }

        avg_magnetizations[temp_idx] = sum_M / num_measurements;
        avg_energies[temp_idx] = sum_E / num_measurements;

        // Save snapshot if needed
        if (std::find(snapshot_indices.begin(), snapshot_indices.end(), temp_idx) != snapshot_indices.end()) {
            std::vector<int> host_spins(N);
            cudaMemcpy(host_spins.data(), spins, N * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaCheckError();

            // Save snapshot to file
            char filename[64];
            sprintf(filename, "snapshot_T_%.2f.txt", T);  
            std::ofstream snapshot_file(filename);
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    snapshot_file << host_spins[y * L + x] << " ";
                }
                snapshot_file << "\n";
            }
            snapshot_file.close();
            std::cout << "Snapshot saved for T=" << T << std::endl;
        }

        cudaFree(spins);
        cudaFree(states);
        cudaFree(partial_sums);
    }

    // Write out energy and magnetization results
    std::ofstream energy_file("energy_vs_temperature.txt");
    std::ofstream magnetization_file("magnetization_vs_temperature.txt");
    energy_file << "# Temperature Energy\n";
    magnetization_file << "# Temperature Magnetization\n";

    for (size_t i = 0; i < temperatures.size(); i++) {
        energy_file << temperatures[i] << " " << avg_energies[i] << "\n";
        magnetization_file << temperatures[i] << " " << avg_magnetizations[i] << "\n";
    }
    energy_file.close();
    magnetization_file.close();
}


// ---------------------------
// 3.1 Task 3: Heat capacity versus temperature near Tc
// ---------------------------
void study_heat_capacity_near_Tc()
{
    // --------------------------------------------------
    // Tunables
    // --------------------------------------------------
    const int num_ensembles = 10;    
    const int n_warmups = 1000;
    const int n_measurements = 500;

    const std::vector<int> system_sizes = { 10,20,30,40,50,60 };

    std::vector<float> temperatures;
    for (float T = 2.1f; T <= 2.5f; T += 0.02f) temperatures.push_back(T);

    // filename encodes simulation parameters
    char fname[128];
    std::sprintf(fname, "C_vs_T_warm_%d_meas_%d_ens_%d.txt",
        n_warmups, n_measurements, num_ensembles);

    std::ofstream out(fname);
    out << "# L  T  <C>  dC\n";

    // --------------------------------------------------
    // Loop over lattice sizes
    // --------------------------------------------------
    for (int L : system_sizes)
    {
        std::cout << "Studying L = " << L << " ..." << std::endl;

        const int N = L * L;
        dim3 blockDim(32, 32);
        dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
            (L + blockDim.y - 1) / blockDim.y);
        const int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;

        // --------------------------------------------------
        // Loop over temperatures
        // --------------------------------------------------
        for (float T : temperatures)
        {
            double ens_sum_C = 0.0;
            double ens_sum_C2 = 0.0;

            // ----------------------------------------------
            // Independent ensembles (replicas)
            // ----------------------------------------------
            for (int e = 0; e < num_ensembles; ++e)
            {
                //--------------------------------------------------------------
                // Allocate & initialise this ensemble
                //--------------------------------------------------------------
                int* spins;            int* partial_sums;
                curandState_t* states;
                cudaMalloc(&spins, N * sizeof(int));
                cudaMalloc(&states, N * sizeof(curandState_t));
                cudaMalloc(&partial_sums, num_blocks * sizeof(int));

                // Unique seed per ensemble/size/T
                const unsigned long long seed = 4321ULL +
                    1'000'000ULL * e +
                    10'000ULL * L +
                    200ULL * static_cast<int>((T - 2.1f) * 50);

                init_kernel << <gridDim, blockDim >> > (spins, states, seed, L);
                cudaDeviceSynchronize();
                cudaCheckError();

                //--------------------------------------------------------------
                // Pre‑compute Boltzmann factors for this T
                //--------------------------------------------------------------
                const float beta = 1.0f / T;
                float bf_host[2] = { expf(-4.0f * beta), expf(-8.0f * beta) };
                cudaMemcpyToSymbol(bf, bf_host, 2 * sizeof(float));

                const int n_skip = choose_skip(T);

                //--------------------------------------------------------------
                // Warm‑up sweeps
                //--------------------------------------------------------------
                for (int sweep = 0; sweep < n_warmups; ++sweep)
                {
                    metropolis_kernel << <gridDim, blockDim >> > (spins, states, L, 0);
                    metropolis_kernel << <gridDim, blockDim >> > (spins, states, L, 1);
                }
                cudaDeviceSynchronize();

                //--------------------------------------------------------------
                // Measurement phase
                //--------------------------------------------------------------
                double sum_E = 0.0;
                double sum_E2 = 0.0;
                std::vector<int> host_partial_sums(num_blocks);

                for (int m = 0; m < n_measurements; ++m)
                {
                    // decorrelate configurations
                    for (int s = 0; s < n_skip; ++s)
                    {
                        metropolis_kernel << <gridDim, blockDim >> > (spins, states, L, 0);
                        metropolis_kernel << <gridDim, blockDim >> > (spins, states, L, 1);
                    }

                    energy_reduction_kernel << <num_blocks, REDUCTION_THREADS,
                        REDUCTION_THREADS * sizeof(int) >> > (spins, partial_sums, L, N);
                    cudaDeviceSynchronize();
                    cudaCheckError();

                    cudaMemcpy(host_partial_sums.data(), partial_sums,
                        num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

                    int total_bond_sum = 0;
                    for (int v : host_partial_sums) total_bond_sum += v;

                    const float E = -static_cast<float>(total_bond_sum) / N;
                    sum_E += E;
                    sum_E2 += E * E;
                }

                //--------------------------------------------------------------
                // Heat capacity for this ensemble
                //--------------------------------------------------------------
                const float avg_E = sum_E / n_measurements;
                const float avg_E2 = sum_E2 / n_measurements;
                const float C = (avg_E2 - avg_E * avg_E) / (T * T);

                ens_sum_C += C;
                ens_sum_C2 += C * C;

                cudaFree(spins);
                cudaFree(states);
                cudaFree(partial_sums);
            } // ensemble loop

            // ----------------------------------------------
            // Ensemble averages and √N error bars
            // ----------------------------------------------
            const double mean_C = ens_sum_C / num_ensembles;
            const double var_C = ens_sum_C2 / num_ensembles - mean_C * mean_C;
            const double err_C = std::sqrt(var_C / num_ensembles);

            out << L << ' ' << T << ' ' << mean_C << ' ' << err_C << "\n";
        } // temperature loop
    }     // size loop

    out.close();
    std::cout << "Heat‑capacity study completed (" << fname << ")" << std::endl;
}

// ---------------------------
// 3.1 Task 4: System with random impurities
// ---------------------------
void study_disordered_system_with_impurities()
{
    //---------------- Tunables -------------------------------------------------
    const int    L = 40;
    const int    N = L * L;
    const int    num_ensembles = 10;      // replicas per (T,p)
    const int    n_warmups = 1000;
    const int    n_measurements = 500;
    const float  dT = 0.02f;
    const int    seed_base = 2025;

    //---------------- Vacuum concentrations ------------------------------------
    const std::vector<float> impurity_conc = { 0.03f, 0.10f, 0.25f };

    std::vector<float> temperatures;
    for (float T = 2.1f; T <= 2.6f; T += dT) temperatures.push_back(T);

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
        (L + blockDim.y - 1) / blockDim.y);
    const int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;

    //---------------------------------------------------------------------------
    for (float p : impurity_conc)
    {
        std::cout << "Running p = " << p << " ...\n";

        //---------------- create & copy site mask -----------------------------
        std::vector<int> h_mask(N, 1);
        std::mt19937 rng(seed_base + int(1e3 * p));
        std::uniform_real_distribution<float> U(0.f, 1.f);
        for (int i = 0; i < N; ++i) if (U(rng) < p) h_mask[i] = 0;

        int* d_mask;
        cudaMalloc(&d_mask, N * sizeof(int));
        cudaMemcpy(d_mask, h_mask.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        //---------------- open output file ------------------------------------
        char fname[128];
        std::sprintf(fname, "obs_vs_T_L_%d_p_%.2f_ens_%d.txt",
            L, p, num_ensembles);
        std::ofstream out(fname);
        out << "# T   <E>  dE   <|M|>  d|M|   <C>  dC\n";

        //---------------- temperature loop ------------------------------------
        for (float T : temperatures)
        {
            const float beta = 1.0f / T;
            float bf_host[2] = { std::exp(-4.f * beta), std::exp(-8.f * beta) };
            cudaMemcpyToSymbol(bf, bf_host, 2 * sizeof(float));

            double ens_sum_E = 0.0, ens_sum_E2 = 0.0;
            double ens_sum_M = 0.0, ens_sum_M2 = 0.0;
            double ens_sum_C = 0.0, ens_sum_C2 = 0.0;

            // -------------- ensemble loop (replicas) ------------------------
            for (int e = 0; e < num_ensembles; ++e)
            {
                // allocate GPU arrays
                int* spins, * partial;
                curandState_t* states;
                cudaMalloc(&spins, N * sizeof(int));
                cudaMalloc(&partial, num_blocks * sizeof(int));
                cudaMalloc(&states, N * sizeof(curandState_t));

                // unique seed per (T,p,ensemble)
                const unsigned long long seed =
                    4321ULL + 1'000'000ULL * e
                    + 10'000ULL * int(p * 100)
                    + 200ULL * int((T - 2.1f) / dT);
                init_kernel << <gridDim, blockDim >> > (spins, states, seed, L);
                cudaDeviceSynchronize();

                const int n_skip = choose_skip(T);

                // -------- warm‑up sweeps ------------------------------------
                for (int w = 0; w < n_warmups; ++w) {
                    metropolis_kernel_with_mask << <gridDim, blockDim >> > (spins, states, L, 0, d_mask);
                    metropolis_kernel_with_mask << <gridDim, blockDim >> > (spins, states, L, 1, d_mask);
                }
                cudaDeviceSynchronize();

                // -------- measurement phase ---------------------------------
                double sum_E = 0.0, sum_E2 = 0.0, sum_M = 0.0, sum_M2 = 0.0;
                std::vector<int> h_partial(num_blocks);

                for (int m = 0; m < n_measurements; ++m)
                {
                    // decorrelate
                    for (int s = 0; s < n_skip; ++s) {
                        metropolis_kernel_with_mask << <gridDim, blockDim >> > (spins, states, L, 0, d_mask);
                        metropolis_kernel_with_mask << <gridDim, blockDim >> > (spins, states, L, 1, d_mask);
                    }

                    // magnetisation
                    reduction_kernel << <num_blocks, REDUCTION_THREADS,
                        REDUCTION_THREADS * sizeof(int) >> >
                        (spins, partial, N);
                    cudaMemcpy(h_partial.data(), partial,
                        num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
                    int totM = 0; for (int v : h_partial) totM += v;
                    float M = static_cast<float>(totM) / N;
                    sum_M += std::fabs(M);
                    sum_M2 += M * M;

                    // energy
                    energy_reduction_kernel << <num_blocks, REDUCTION_THREADS,
                        REDUCTION_THREADS * sizeof(int) >> >
                        (spins, partial, L, N);
                    cudaMemcpy(h_partial.data(), partial,
                        num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
                    int bond = 0; for (int v : h_partial) bond += v;
                    float E = -static_cast<float>(bond) / N;
                    sum_E += E;
                    sum_E2 += E * E;
                }

                // per‑ensemble observables
                const float Ebar = sum_E / n_measurements;
                const float E2bar = sum_E2 / n_measurements;
                const float Mbar = sum_M / n_measurements;
                const float C = (E2bar - Ebar * Ebar) / (T * T);

                ens_sum_E += Ebar;  ens_sum_E2 += Ebar * Ebar;
                ens_sum_M += Mbar;  ens_sum_M2 += Mbar * Mbar;
                ens_sum_C += C;     ens_sum_C2 += C * C;

                cudaFree(spins); cudaFree(states); cudaFree(partial);
            } // ensemble loop

            //---------------- averages & error bars --------------------------
            auto err = [](double S, double S2, int N) {
                double mean = S / N;
                double var = S2 / N - mean * mean;
                return (var > 0.0 ? std::sqrt(var / N) : 0.0);
                };

            const double mean_E = ens_sum_E / num_ensembles;
            const double mean_M = ens_sum_M / num_ensembles;
            const double mean_C = ens_sum_C / num_ensembles;

            out << T << ' '
                << mean_E << ' ' << err(ens_sum_E, ens_sum_E2, num_ensembles) << ' '
                << mean_M << ' ' << err(ens_sum_M, ens_sum_M2, num_ensembles) << ' '
                << mean_C << ' ' << err(ens_sum_C, ens_sum_C2, num_ensembles) << '\n';
        } // temperature loop

        cudaFree(d_mask);
        out.close();
        std::cout << "wrote " << fname << '\n';
    } // p‑loop
}
// -----------------------------------------------------------------------------
// 3.1 Task 5 : Simulated‑annealing with Impurities
// -----------------------------------------------------------------------------
void run_simulated_annealing_with_impurities()
{
    // ───────── Tunables ──────────────────────────────────────────────────────
    const int   L = 40;
    const int   N = L * L;
    const int   n_sweeps_per_T = 1000;
    const int   num_runs = 5;          // independent annealing runs
    const int   seed_base = 1337;

    const float T_start = 3.0f;
    const float T_end = 1.0f;
    const float dT = 0.05f;      // schedule step

    const std::vector<float> impurity_conc = { 0.03f, 0.10f, 0.25f };

    // build temperature schedule (hot → cold)
    std::vector<float> schedule;
    for (float T = T_start; T >= T_end - 1e-6f; T -= dT) schedule.push_back(T);

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
        (L + blockDim.y - 1) / blockDim.y);
    const int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;

    // ───────── impurity loop ────────────────────────────────────────────────
    for (float p : impurity_conc)
    {
        std::cout << "Annealing, p = " << p << "...\n";

        // -------- site‑mask --------------------------------------------------
        std::vector<int> h_mask(N, 1);
        std::mt19937 rng(seed_base + int(1e3 * p));
        std::uniform_real_distribution<float> U(0.f, 1.f);
        for (int i = 0; i < N; ++i) if (U(rng) < p) h_mask[i] = 0;

        int* d_mask;
        cudaMalloc(&d_mask, N * sizeof(int));
        cudaMemcpy(d_mask, h_mask.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        // to remember the best configuration over all runs
        float best_E = 1e30f;
        std::vector<int> best_spins(N);

        // -------- independent runs ------------------------------------------
        for (int run = 0; run < num_runs; ++run)
        {
            std::printf("  run %d/%d\n", run + 1, num_runs);

            // --- allocate one annealing replica -----------------------------
            int* spins, * partial;
            curandState_t* states;
            cudaMalloc(&spins, N * sizeof(int));
            cudaMalloc(&partial, num_blocks * sizeof(int));
            cudaMalloc(&states, N * sizeof(curandState_t));

            const unsigned long long seed =
                seed_base + 10'000ULL * run + 1'000ULL * int(p * 100);
            init_kernel << <gridDim, blockDim >> > (spins, states, seed, L);
            cudaDeviceSynchronize();

            // --- open run‑log file ------------------------------------------
            char fname[128];
            std::sprintf(fname,
                "anneal_L_%d_T%.2f-%.2f_dT%.2f_p%.2f_run%d.txt",
                L, T_start, T_end, dT, p, run + 1);
            std::ofstream log(fname);
            log << "# T   E   M\n";

            std::vector<int> h_partial(num_blocks);

            // --- schedule loop (hot → cold) ---------------------------------
            for (float T : schedule)
            {
                const float beta = 1.0f / T;
                float bf[2] = { std::exp(-4.f * beta), std::exp(-8.f * beta) };
                cudaMemcpyToSymbol(bf, bf, 2 * sizeof(float));

                for (int s = 0; s < n_sweeps_per_T; ++s) {
                    metropolis_kernel_with_mask << <gridDim, blockDim >> > (spins, states, L, 0, d_mask);
                    metropolis_kernel_with_mask << <gridDim, blockDim >> > (spins, states, L, 1, d_mask);
                }
                cudaDeviceSynchronize();

                // ---- measure E and M ---------------------------------------
                reduction_kernel << <num_blocks, REDUCTION_THREADS,
                    REDUCTION_THREADS * sizeof(int) >> >
                    (spins, partial, N);
                cudaMemcpy(h_partial.data(), partial,
                    num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
                int Msum = 0; for (int v : h_partial) Msum += v;
                float M = float(Msum) / N;

                energy_reduction_kernel << <num_blocks, REDUCTION_THREADS,
                    REDUCTION_THREADS * sizeof(int) >> >
                    (spins, partial, L, N);
                cudaMemcpy(h_partial.data(), partial,
                    num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
                int Esum = 0; for (int v : h_partial) Esum += v;
                float E = -float(Esum) / N;

                log << T << ' ' << E << ' ' << M << '\n';

                // store lowest‑E configuration after final temperature
                if (T <= T_end + 1e-6f && E < best_E) {
                    cudaMemcpy(best_spins.data(), spins, N * sizeof(int),
                        cudaMemcpyDeviceToHost);
                    best_E = E;
                }
            } // schedule loop

            log.close();
            cudaFree(spins); cudaFree(states); cudaFree(partial);
        } // run loop

        // -------- snapshot of best configuration ----------------------------
        char snap[128];
        std::sprintf(snap, "snapshot_L_%d_T%.2f-%.2f_dT%.2f_p%.2f.txt",
            L, T_start, T_end, dT, p);
        std::ofstream out(snap);
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) out << best_spins[y * L + x] << ' ';
            out << '\n';
        }
        std::cout << "  saved best configuration (E = "
            << best_E << ") to " << snap << '\n';

        cudaFree(d_mask);
    } // p‑loop
}


// ---------------------------
// 3.2 Task 1: Magnetization vs nsweeps with field 
// ---------------------------
void run_magnetization_vs_nsweeps_with_field() {
    const int L = 40;
    const int N = L * L;
    const int n_sweeps = 1000;
    const int snapshot_interval = 200;
    const float T = 2.3f;
    const float h = 0.01f;

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x, (L + blockDim.y - 1) / blockDim.y);
    int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;

    float beta = 1.0f / T;

    int* spins;
    curandState_t* states;
    int* partial_sums;

    cudaMalloc(&spins, N * sizeof(int));
    cudaMalloc(&states, N * sizeof(curandState_t));
    cudaMalloc(&partial_sums, num_blocks * sizeof(int));
    cudaCheckError();

    // Init spins
    unsigned long long seed = 7000;
    init_kernel << <gridDim, blockDim >> > (spins, states, seed, L);
    cudaDeviceSynchronize();
    cudaCheckError();

    std::vector<float> magnetizations(n_sweeps);

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        metropolis_kernel_with_field << <gridDim, blockDim >> > (
            spins, states, L, 0, h, beta);
        cudaDeviceSynchronize();
        metropolis_kernel_with_field << <gridDim, blockDim >> > (
            spins, states, L, 1, h, beta);
        cudaDeviceSynchronize();

        // Compute magnetization
        reduction_kernel << <num_blocks, REDUCTION_THREADS, REDUCTION_THREADS * sizeof(int) >> > (
            spins, partial_sums, N);
        cudaDeviceSynchronize();
        cudaCheckError();

        std::vector<int> host_partial_sums(num_blocks);
        cudaMemcpy(host_partial_sums.data(), partial_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        int total_sum = 0;
        for (auto val : host_partial_sums) total_sum += val;
        magnetizations[sweep] = static_cast<float>(total_sum) / N;

        // Save snapshot
        if (sweep % snapshot_interval == 0) {
            std::vector<int> host_spins(N);
            cudaMemcpy(host_spins.data(), spins, N * sizeof(int), cudaMemcpyDeviceToHost);

            char filename[64];
            sprintf(filename, "snapshot_T_2.3_h_0.01_sweep_%04d.txt", sweep);
            std::ofstream snapshot_file(filename);
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    snapshot_file << host_spins[y * L + x] << " ";
                }
                snapshot_file << "\n";
            }
            snapshot_file.close();
        }
    }

    // Save magnetization data
    std::ofstream outfile("magnetization_vs_nsweeps_T_2.3_h_0.01.txt");
    outfile << "# Sweep\tMagnetization\n";
    for (int i = 0; i < n_sweeps; i++) {
        outfile << i << "\t" << magnetizations[i] << "\n";
    }
    outfile.close();

    cudaFree(spins);
    cudaFree(states);
    cudaFree(partial_sums);

    std::cout << "Finished magnetization-vs-sweeps simulation at T=2.3, h=0.01.\n";
}

// ---------------------------
// 3.2 Task 2: Susceptibilty and magnetic field vs temperature for different fields 
// ---------------------------
void run_susceptibility_vs_temperature_for_different_H()
{
    const int L = 40;
    const int N = L * L;
    const int num_equil_sweeps = 1000;
    const int num_measurements = 500;

    const int num_ensembles = 10;        

    float fields[] = { 0.01f, 0.02f, 0.03f, 0.04f };
    const int num_fields = sizeof(fields) / sizeof(fields[0]);

    std::vector<float> temperatures;
    for (float T = 2.2f; T <= 2.8f; T += 0.02f) temperatures.push_back(T);

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
        (L + blockDim.y - 1) / blockDim.y);
    int num_blocks = (N + REDUCTION_THREADS - 1) / REDUCTION_THREADS;

    for (int f = 0; f < num_fields; ++f)
    {
        float h = fields[f];

        char out_name[64];
        sprintf(out_name, "chi_vs_T_h_%.2f.txt", h);
        std::ofstream out(out_name);
        out << "# T   <|M|>  d<|M|>   <chi>  d<chi>\n";

        for (size_t t = 0; t < temperatures.size(); ++t)
        {
            float T = temperatures[t];
            float beta = 1.0f / T;

            //--------------------------------------------------
            // ensemble accumulators
            //--------------------------------------------------
            double ens_sum_Mabs = 0.0, ens_sum_Mabs2 = 0.0;
            double ens_sum_chi = 0.0, ens_sum_chi2 = 0.0;

            for (int e = 0; e < num_ensembles; ++e)
            {
                //--------------------------------------------------
                // --- allocate & initialise one ensemble ----------
                //--------------------------------------------------
                int* spins;            int* partial_sums;
                curandState_t* states;
                cudaMalloc(&spins, N * sizeof(int));
                cudaMalloc(&states, N * sizeof(curandState_t));
                cudaMalloc(&partial_sums, num_blocks * sizeof(int));

                unsigned long long seed = 1234ULL
                    + 1000ULL * f      // pick field
                    + 100ULL * t      // temperature
                    + 10ULL * e;     // ensemble #
                init_kernel << <gridDim, blockDim >> > (spins, states, seed, L);
                cudaDeviceSynchronize();

                //--------------------------------------------------
                // --- equilibration ------------------------------
                //--------------------------------------------------
                for (int s = 0;s < num_equil_sweeps;++s) {
                    metropolis_kernel_with_field << <gridDim, blockDim >> > (
                        spins, states, L, 0, h, beta);
                    metropolis_kernel_with_field << <gridDim, blockDim >> > (
                        spins, states, L, 1, h, beta);
                }

                //--------------------------------------------------
                // --- measurements -------------------------------
                //--------------------------------------------------
                double sum_M = 0.0, sum_M2 = 0.0;
             
                // Adaptive skipping

                int n_skip;
                if (h <= 0.012f) {                // ≈ the 0.01-field (allow tiny round-off)
                    n_skip = choose_skip(T);      // long, adaptive skips
                }
                else {
                    n_skip = 20;                  // short skip for larger fields
                }


                for (int m = 0;m < num_measurements;++m) {
                    for (int s = 0;s < n_skip;++s) {
                        metropolis_kernel_with_field << <gridDim, blockDim >> > (
                            spins, states, L, 0, h, beta);
                        metropolis_kernel_with_field << <gridDim, blockDim >> > (
                            spins, states, L, 1, h, beta);
                    }

                    reduction_kernel << <num_blocks, REDUCTION_THREADS,
                        REDUCTION_THREADS * sizeof(int) >> > (
                            spins, partial_sums, N);
                    cudaDeviceSynchronize();

                    std::vector<int> host_partial(num_blocks);
                    cudaMemcpy(host_partial.data(), partial_sums,
                        num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

                    int total = 0; for (int v : host_partial) total += v;
                    float M = float(total) / N;

                    sum_M += M;
                    sum_M2 += M * M;
                }

                //----------------------------------------------------------------
                // ensemble-level observables for this run
                //----------------------------------------------------------------
                double Mbar = sum_M / num_measurements;
                double M2bar = sum_M2 / num_measurements;
                double chi = (M2bar - Mbar * Mbar) / T;

                ens_sum_Mabs += fabs(Mbar);
                ens_sum_Mabs2 += Mbar * Mbar;      // used for error of |M|
                ens_sum_chi += chi;
                ens_sum_chi2 += chi * chi;

                cudaFree(spins); cudaFree(states); cudaFree(partial_sums);
            } // end ensemble loop

            //--------------------------------------------------------------------
            // final averages and √N error bars
            //--------------------------------------------------------------------
            double mean_Mabs = ens_sum_Mabs / num_ensembles;
            double var_Mabs = ens_sum_Mabs2 / num_ensembles - mean_Mabs * mean_Mabs;
            double err_Mabs = sqrt(var_Mabs / num_ensembles);

            double mean_chi = ens_sum_chi / num_ensembles;
            double var_chi = ens_sum_chi2 / num_ensembles - mean_chi * mean_chi;
            double err_chi = sqrt(var_chi / num_ensembles);

            out << T << ' ' << mean_Mabs << ' '
                << err_Mabs << ' ' << mean_chi << ' ' << err_chi << '\n';
        }
        out.close();
        std::cout << "wrote " << out_name << " with " << num_ensembles
            << " ensembles per point\n";
    }
}





// ---------------------------
// MAIN FUNCTION
// ---------------------------
int main() {
    std::cout << "Choose Simulation:\n";
    std::cout << "(1) 3.1 Task 1: Magnetization vs n_sweeps\n";
    std::cout << "(2) 3.1 Task 2: Energy and magnetization vs temperature plus snapshots\n";
    std::cout << "(3) 3.1 Task 3: Heat capacity versus temperature near Tc for various system sizes\n";
    std::cout << "(4) 3.1 Task 4: Energy, magnetization, and heat capacity vs T with impurities\n";
    std::cout << "(5) 3.1 Task 5: Simulated Annealing with Impurities\n";
    std::cout << "(6) 3.2 Task 1: Magnetization vs sweeps at fixed T=2.3, h=0.01 with snapshots\n";
    std::cout << "(7) 3.2 Task 2: Susceptibilty and magnetic field vs temperature for different fields\n";
    std::cout << "Enter choice (1-7): ";
    int choice;
    std::cin >> choice;

    if (choice == 1) {
        run_magnetization_vs_nsweeps();
    }
    else if (choice == 2) {
        run_E_M_vs_T_plus_snapshot();
    }
    else if (choice == 3) {
        study_heat_capacity_near_Tc();
    }
    else if (choice == 4) {
        study_disordered_system_with_impurities();
    }
    else if (choice == 5) {
        run_simulated_annealing_with_impurities();
    }
    else if (choice == 6) {
        run_magnetization_vs_nsweeps_with_field();
    }
    else if (choice == 7) {
        run_susceptibility_vs_temperature_for_different_H();
    }
    else {
        std::cout << "Invalid choice!" << std::endl;
    }

    // Wait for user before exiting
    std::cout << "Press Enter to exit...";
    std::cin.ignore(); // flush leftover newline
    std::cin.get();    // wait for actual Enter

    return 0;
}
