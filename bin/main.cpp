#include <chrono>
#include <iostream>
#include <omp.h>
#include <string>

#ifdef USE_SUITESPARSE
#include "suitesparse/amd.h"
#endif

#include "paramd/paramd.h"

// Display help information
static void show_help()
{
  std::cout << "Usage: paramd [OPTIONS]\n"
            << "Options:\n"
            << "  -h, --help                               Show this help message\n"
            << "  --matrix <path>                          Path to the input matrix in mtx format (required)\n"
            << "  --algo <paramd, amd (if available)>      Algorithm to benchmark (default: paramd) \n"
            << "  --mult <double>                          Multiplicative relaxation factor (default: 1.1)\n"
            << "  --lim <double>                           Limitation factor (default: 8192), will be divided by #threads\n"
            << "  --mem <double>                           Extra memory factor (default: 1.5)\n"
            << "  --seed <64-bit integer>                  Seed used to randomly permute the matrix (default: 1)\n"
            << "  --breakdown                              Enable time breakdown (default: false)\n"
            << "  --stat                                   Enable statistics (default: false)\n"
            << "  --sym                                    Specify sym if the matrix is known to be symmetric (default: false)\n"
            << "  --hierarchical                           Enable hierarchical decomposition (default: false)\n"
            << "  --partition-threshold <integer>          Threshold for matrix partitioning (default: 10000)\n"
            << "  --recursion-depth <integer>              Maximum recursion depth (default: 10)\n"
            << "  --balance-factor <double>                Balance factor for partitioning (0.5 = balanced) (default: 0.5)\n"
            << "  --use-metis                             Use METIS for partitioning (default if available)\n"
            << "  --no-metis                              Don't use METIS for partitioning\n";
}

int main(int argc, char *argv[])
{
  paramd::config config;
  std::string matrix = "", algo = "paramd";

  // Parse arguments
  for (int i = 1; i < argc; i++)
  {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help")
    {
      show_help();
      return 0;
    }
    else if (arg == "--matrix" && i + 1 < argc)
    {
      matrix = std::string(argv[++i]);
    }
    else if (arg == "--algo" && i + 1 < argc)
    {
      algo = std::string(argv[++i]);
      if (algo != "amd" && algo != "paramd")
      {
        std::cerr << algo << " is not in the algo list!" << std::endl;
        show_help();
        exit(1);
      }
    }
    else if (arg == "--mult" && i + 1 < argc)
    {
      config.mult = std::stod(argv[++i]);
    }
    else if (arg == "--lim" && i + 1 < argc)
    {
      config.lim = std::stod(argv[++i]);
    }
    else if (arg == "--mem" && i + 1 < argc)
    {
      config.mem = std::stod(argv[++i]);
    }
    else if (arg == "--seed" && i + 1 < argc)
    {
      config.seed = std::stoull(argv[++i]);
    }
    else if (arg == "--breakdown")
    {
      config.breakdown = true;
    }
    else if (arg == "--stat")
    {
      config.stat = true;
    }
    else if (arg == "--sym")
    {
      config.sym = true;
    }
    else if (arg == "--hierarchical")
    {
      config.hierarchical = true;
    }
    else if (arg == "--partition-threshold" && i + 1 < argc)
    {
      config.partition_threshold = std::stoi(argv[++i]);
    }
    else if (arg == "--recursion-depth" && i + 1 < argc)
    {
      config.max_recursion_depth = std::stoi(argv[++i]);
    }
    else if (arg == "--balance-factor" && i + 1 < argc)
    {
      config.balance_factor = std::stod(argv[++i]);
    }
    else if (arg == "--use-metis")
    {
      config.use_metis = true;
    }
    else if (arg == "--no-metis")
    {
      config.use_metis = false;
    }
    else
    {
      std::cerr << "Unknown option: " << arg << std::endl;
      show_help();
      exit(1);
    }
  }

  if (matrix == "")
  {
    std::cerr << "Please specify the input matrix!" << std::endl;
    show_help();
    exit(1);
  }

  std::cout << "Reading matrix market file: " << matrix << std::endl;
  paramd::matrix A(matrix);

  std::cout << "Applying random inverse permutation with seed: " << config.seed << std::endl;
  std::vector<paramd::vtype> rand_iperm = paramd::random_inverse_permutation(A.n, config.seed);
  A.apply_inverse_permutation(rand_iperm);
  std::vector<paramd::vtype> perm(A.n);

  std::cout << "Ordering using algorithm: " << algo << std::endl;

  uint64_t fill = 0;
  double start = 0, end = 0;

  if (algo == "paramd")
  {
    start = omp_get_wtime();
    fill = paramd::paramd(A.n, A.rowptr.data(), A.colidx.data(), perm.data(), config);
    end = omp_get_wtime();
  }
  else
  {
#ifdef USE_SUITESPARSE
    double info[AMD_INFO];
    start = omp_get_wtime();
    amd_order(A.n, A.rowptr.data(), A.colidx.data(), perm.data(), NULL, info);
    end = omp_get_wtime();
    if (info[AMD_STATUS] != AMD_OK)
    {
      std::cout << "AMD returns errors. Fall back to 64-bit mode" << std::endl;
      int64_t n64 = A.n;
      std::vector<int64_t> rowptr64(A.rowptr.size());
      std::vector<int64_t> colidx64(A.colidx.size());
      std::vector<int64_t> perm64(perm.size());
#pragma omp parallel for
      for (size_t i = 0; i < rowptr64.size(); ++i)
        rowptr64[i] = A.rowptr[i];
#pragma omp parallel for
      for (size_t i = 0; i < colidx64.size(); ++i)
        colidx64[i] = A.colidx[i];
      std::cout << "Ordering using algorithm: amd_l" << std::endl;
      start = omp_get_wtime();
      amd_l_order(n64, rowptr64.data(), colidx64.data(), perm64.data(), NULL, info);
      end = omp_get_wtime();
      if (info[AMD_STATUS] != AMD_OK)
      {
        std::cerr << "AMD returns error code: " << info[AMD_STATUS] << std::endl;
        exit(1);
      }
#pragma omp parallel for
      for (size_t i = 0; i < A.n; ++i)
        perm[i] = perm64[i];
    }
    fill = uint64_t(info[AMD_LNZ]);
#else
    std::cerr << "AMD not found. Please specify the AMD installation path via CMake!" << std::endl;
    exit(1);
#endif
  }

  std::cout << "#Fill = " << fill << std::endl;
  std::cout << "Ordering Time = " << end - start << " seconds" << std::endl;
  std::cout << "Reverting random inverse permutation with seed: " << config.seed << std::endl;
  std::vector<paramd::vtype> rand_perm = paramd::inverse_permutation(rand_iperm);
  perm = paramd::apply_function(perm, rand_perm);
  std::string out_perm_fp = matrix + ".perm";
  std::cout << "Writing permutation to: " << out_perm_fp << std::endl;
  paramd::write_perm(out_perm_fp, perm);

  return 0;
}