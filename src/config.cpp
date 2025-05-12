#include <iostream>
#include <omp.h>
#include "paramd/paramd.h"
#include <metis.h> // Include METIS header for partitioning
#define USE_METIS 1

namespace paramd
{
  // Default constructor for config
  // In config.cpp
  config::config()
  {
    // Existing initialization
    mult = 1.1;
    lim = 8192;
    mem = 1.5;
    seed = 1;
    breakdown = false;
    stat = false;
    sym = false;
    hierarchical = false;
    partition_threshold = 10000;
    max_recursion_depth = 10;
    balance_factor = 0.5;

    // METIS options initialization
    use_metis = true; // Default to using METIS if available

#ifdef USE_METIS
    METIS_SetDefaultOptions(metis_options);
    metis_options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // We want vertex separators
    metis_options[METIS_OPTION_NUMBERING] = 0;               // 0-based numbering
#endif
  }

  // Update print method too
  void config::print() const
  {
    // Existing print code
    std::cout << "Hierarchical decomposition enabled: " << (hierarchical ? "Yes" : "No") << "\n";
    if (hierarchical)
    {
      std::cout << "Partition threshold: " << partition_threshold << "\n";
      std::cout << "Maximum recursion depth: " << max_recursion_depth << "\n";
      std::cout << "Balance factor: " << balance_factor << "\n";

#ifdef USE_METIS
      std::cout << "Using METIS for partitioning: " << (use_metis ? "Yes" : "No") << "\n";
#else
      std::cout << "METIS support not available\n";
#endif
    }
  }
} // end of namespace paramd
