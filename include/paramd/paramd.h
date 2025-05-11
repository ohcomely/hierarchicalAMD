#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace paramd
{
  typedef int32_t vtype;
  typedef int32_t etype;

  // config used for paramd
  struct config
  {
    double mult, lim, mem;
    uint64_t seed;
    bool breakdown, stat, sym;

    // New parameters for hierarchical decomposition
    bool hierarchical;         // Enable hierarchical decomposition
    vtype partition_threshold; // Threshold for partitioning
    int max_recursion_depth;   // Maximum recursion depth
    double balance_factor;     // Balance factor for partitioning (0.5 = balanced)

    // Default constructor for config
    config();

    // Print config
    void print() const;
  };

  // matrix storage
  struct matrix
  {
    vtype n;
    etype nnz;
    std::vector<etype> rowptr;
    std::vector<vtype> colidx;

    // Construct a matrix from fp
    matrix(std::string fp);

    // Apply iperm symmetrically to the matrix
    void apply_inverse_permutation(const std::vector<vtype> &iperm);
  };

  // Generate a length n random inverse permutation from seed
  std::vector<vtype> random_inverse_permutation(const vtype n, const uint64_t seed);

  // Get the inverse permutation of a permutation
  std::vector<vtype> inverse_permutation(const std::vector<vtype> &perm);

  // Apply function f to a vector src
  std::vector<vtype> apply_function(const std::vector<vtype> &src, const std::vector<vtype> &f);

  // Write permutation to fp
  void write_perm(std::string fp, const std::vector<vtype> &perm);

  // Parallel Approximate Minimum Degree Ordering Algorithm
  uint64_t paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config);
} // end of namespace paramd