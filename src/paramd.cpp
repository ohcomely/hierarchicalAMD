#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <omp.h>
#include "paramd/paramd.h"
#include <random>
#include <vector>
#include <queue>

namespace paramd
{
  // Internal timer for ParAMD
  struct timer
  {
    bool on;
    double start;
    std::map<std::string, double> count;
    timer(bool on) : on(on), start(omp_get_wtime()) {}
    void time(const std::string &str)
    {
      if (on)
      {
        if (omp_in_parallel())
        {
#pragma omp barrier
#pragma omp master
          {
            double end = omp_get_wtime();
            count[str] += end - start;
            start = end;
          }
#pragma omp barrier
        }
        else
        {
          double end = omp_get_wtime();
          count[str] += end - start;
          start = end;
        }
      }
    }

    void print() const
    {
      if (on)
      {
        for (const auto [v, s] : count)
          std::cout << v << ": " << s << " seconds\n";
      }
    }
  };

  // Original AMD function (will be called by hierarchical_paramd)
  uint64_t standard_paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config);

  // Hierarchical AMD function
  uint64_t hierarchical_paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config, int recursion_level);

  // Graph partitioning function
  void partition_graph(const vtype n, const vtype *rowptr, const etype *colidx,
                       std::vector<vtype> &part1_nodes, std::vector<vtype> &part2_nodes,
                       std::vector<vtype> &separator_nodes, double balance_factor);

  // Function to combine orderings from subgraphs
  void combine_orderings(const vtype n, vtype *perm,
                         const std::vector<vtype> &part1_nodes, const std::vector<vtype> &part1_perm,
                         const std::vector<vtype> &part2_nodes, const std::vector<vtype> &part2_perm,
                         const std::vector<vtype> &separator_nodes, const std::vector<vtype> &separator_perm);

  vtype find_peripheral_node(const vtype n, const vtype *rowptr, const etype *colidx);

  // Sequential Approximate Degree Lists
  struct approximate_degree_lists
  {
    vtype n, min_deg, cand_end;
    vtype *base_, *head, *next, *prev, *loc, *cand;

    void init(const vtype &n_ = 0)
    {
      n = n_;
      min_deg = n;
      base_ = (vtype *)std::malloc(sizeof(vtype) * 5 * n);
      std::memset(base_, -1, sizeof(vtype) * 4 * n);
      head = base_;
      next = base_ + n;
      prev = base_ + n * 2;
      loc = base_ + n * 3;
      cand = base_ + n * 4;
    }

    void finalize() { std::free(base_); }

    void remove(const vtype deg, const vtype index)
    {
      vtype prv = prev[index], nxt = next[index];
      if (nxt != -1)
        prev[nxt] = prv;
      if (prv != -1)
        next[prv] = nxt;
      else
        head[deg] = nxt;
      loc[index] = -1;
    }

    void insert(const vtype deg, const vtype index)
    {
      if (loc[index] != -1)
        remove(loc[index], index);
      if (head[deg] != -1)
        prev[head[deg]] = index;
      next[index] = head[deg];
      prev[index] = -1;
      head[deg] = index;
      loc[index] = deg;
      min_deg = min_deg > deg ? deg : min_deg;
    }

    vtype get_min_deg(const size_t tid, vtype *affinity)
    {
      while (min_deg < n)
      {
        for (vtype index = head[min_deg]; index != -1;)
        {
          vtype nxt = next[index];
          if (affinity[index] != tid)
            remove(min_deg, index);
          index = nxt;
        }
        if (head[min_deg] != -1)
          break;
        ++min_deg;
      }
      return min_deg;
    }

    void traverse(const size_t tid, const vtype from, vtype to, vtype *affinity, const vtype lim)
    {
      cand_end = 0;
      to = to >= n ? n - 1 : to;
      for (vtype deg = from; deg <= to; ++deg)
      {
        for (vtype index = head[deg]; index != -1;)
        {
          vtype nxt = next[index];
          if (affinity[index] != tid)
            remove(deg, index);
          else
          {
            cand[cand_end++] = index;
            if (cand_end >= lim)
              break;
          }
          index = nxt;
        }
        if (cand_end >= lim)
          break;
      }
    }
  };

  // Concurrent Approximate Degree Lists
  struct concurrent_approximate_degree_lists
  {
    vtype n;
    approximate_degree_lists *deglists;
    vtype *affinity;
    concurrent_approximate_degree_lists(const vtype &n = 0) : n(n)
    {
      deglists = (approximate_degree_lists *)std::malloc(sizeof(approximate_degree_lists) * omp_get_max_threads());
      affinity = (vtype *)std::malloc(sizeof(vtype) * n);
#pragma omp parallel for
      for (vtype i = 0; i < omp_get_max_threads(); ++i)
        deglists[i].init(n);
    }

    ~concurrent_approximate_degree_lists()
    {
#pragma omp parallel for
      for (vtype i = 0; i < omp_get_max_threads(); ++i)
        deglists[i].finalize();
      std::free(affinity);
      std::free(deglists);
    }

    void traverse(vtype &min_deg, vtype &num_candidates, vtype *candidates, const double mult, const double lim)
    {
      const size_t tid = omp_get_thread_num();
#pragma omp master
      {
        min_deg = n;
        num_candidates = 0;
      }
#pragma omp barrier
      vtype local_min_deg = deglists[tid].get_min_deg(tid, affinity);
#pragma omp atomic compare
      min_deg = min_deg > local_min_deg ? local_min_deg : min_deg;
#pragma omp barrier
      deglists[tid].traverse(tid, min_deg, mult * min_deg, affinity, lim / omp_get_num_threads());
      vtype base = 0;
#pragma omp atomic capture
      {
        base = num_candidates;
        num_candidates += deglists[tid].cand_end;
      }
      std::memcpy(candidates + base, deglists[tid].cand, sizeof(vtype) * deglists[tid].cand_end);
#pragma omp barrier
    }

    void insert(const vtype deg, const vtype index)
    {
      const size_t tid = omp_get_thread_num();
      deglists[tid].insert(deg, index);
      affinity[index] = tid;
    }

    void remove(const vtype deg, const vtype index)
    {
      affinity[index] = -1;
    }
  };

  // Hash Lists
  struct hashlists
  {
    vtype n;
    vtype *base_, *head, *next, *hash;

    hashlists(const vtype &n = 0) : n(n)
    {
      base_ = (vtype *)std::malloc(sizeof(vtype) * n * 3);
      std::memset(base_, -1, sizeof(vtype) * 3 * n);
      head = base_;
      next = base_ + n;
      hash = base_ + n * 2;
    }

    ~hashlists() { std::free(base_); }

    void insert(vtype hsh, vtype index)
    {
      next[index] = head[hsh];
      head[hsh] = index;
      hash[index] = hsh;
    }

    bool empty(vtype hsh) const { return head[hsh] == -1; }

    bool is_tail(vtype index) const { return next[index] == -1; }

    vtype get_hash(vtype index) const { return hash[index]; }

    vtype pop(vtype hsh)
    {
      vtype index = head[hsh];
      head[hsh] = next[head[hsh]];
      return index;
    }

    vtype get_next(vtype index) const { return next[index]; }

    void remove(vtype index, vtype prev_index)
    {
      if (prev_index != -1)
      {
        next[prev_index] = next[index];
      }
      else
      {
        head[hash[index]] = next[index];
      }
    }

    vtype get_nil() const { return -1; }
  };

  // Clear timestamp
  void clear_stp(vtype *stp, const vtype n, vtype &tstp, const vtype tlim)
  {
    if (tstp < 2 || tstp >= tlim)
    {
      for (vtype i = 0; i < n; ++i)
        if (stp[i] != 0)
          stp[i] = 1;
      tstp = 2;
    }
  }

  void combine_orderings(const vtype n, vtype *perm,
                         const std::vector<vtype> &part1_nodes, const std::vector<vtype> &part1_perm,
                         const std::vector<vtype> &part2_nodes, const std::vector<vtype> &part2_perm,
                         const std::vector<vtype> &separator_nodes, const std::vector<vtype> &separator_perm)
  {
    // Create a mapping from original indices to positions in the final ordering
    std::vector<vtype> order_mapping(n);

    // First order part1
    vtype current_pos = 0;
    for (vtype i = 0; i < part1_perm.size(); i++)
    {
      vtype original_node = part1_nodes[part1_perm[i]];
      perm[current_pos] = original_node;
      order_mapping[original_node] = current_pos;
      current_pos++;
    }

    // Next order part2
    for (vtype i = 0; i < part2_perm.size(); i++)
    {
      vtype original_node = part2_nodes[part2_perm[i]];
      perm[current_pos] = original_node;
      order_mapping[original_node] = current_pos;
      current_pos++;
    }

    // Finally order separator
    for (vtype i = 0; i < separator_perm.size(); i++)
    {
      vtype original_node = separator_nodes[separator_perm[i]];
      perm[current_pos] = original_node;
      order_mapping[original_node] = current_pos;
      current_pos++;
    }
  }

  // Extract a subgraph based on a subset of nodes
  void extract_subgraph(const vtype n, const vtype *rowptr, const etype *colidx,
                        const std::vector<vtype> &nodes,
                        std::vector<etype> &sub_rowptr, std::vector<vtype> &sub_colidx)
  {
    // Create mapping from original indices to new indices
    std::vector<vtype> node_mapping(n, -1);
    for (vtype i = 0; i < nodes.size(); i++)
    {
      node_mapping[nodes[i]] = i;
    }

    // Initialize rowptr
    sub_rowptr.resize(nodes.size() + 1);
    sub_rowptr[0] = 0;

    // First pass: count edges per node
    for (vtype i = 0; i < nodes.size(); i++)
    {
      vtype orig_node = nodes[i];
      vtype edge_count = 0;

      for (etype j = rowptr[orig_node]; j < rowptr[orig_node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (node_mapping[neighbor] != -1)
        {
          edge_count++;
        }
      }

      sub_rowptr[i + 1] = sub_rowptr[i] + edge_count;
    }

    // Allocate space for column indices
    sub_colidx.resize(sub_rowptr.back());

    // Second pass: fill column indices
    for (vtype i = 0; i < nodes.size(); i++)
    {
      vtype orig_node = nodes[i];
      etype pos = sub_rowptr[i];

      for (etype j = rowptr[orig_node]; j < rowptr[orig_node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (node_mapping[neighbor] != -1)
        {
          sub_colidx[pos++] = node_mapping[neighbor];
        }
      }
    }
  }

  // Extract separator subgraph with connections to both parts
  // Extract separator subgraph with connections to both parts
  void extract_separator_subgraph(const vtype n, const vtype *rowptr, const etype *colidx,
                                  const std::vector<vtype> &separator_nodes,
                                  const std::vector<vtype> &part1_nodes,
                                  const std::vector<vtype> &part2_nodes,
                                  std::vector<etype> &sep_rowptr,
                                  std::vector<vtype> &sep_colidx)
  {
    // Create mappings from original indices to new indices
    std::vector<vtype> node_mapping(n, -1);
    for (vtype i = 0; i < separator_nodes.size(); i++)
    {
      node_mapping[separator_nodes[i]] = i;
    }

    // Create sets for faster lookup
    std::vector<bool> in_part1(n, false);
    std::vector<bool> in_part2(n, false);

    for (vtype node : part1_nodes)
    {
      in_part1[node] = true;
    }

    for (vtype node : part2_nodes)
    {
      in_part2[node] = true;
    }

    // Initialize rowptr
    sep_rowptr.resize(separator_nodes.size() + 1);
    sep_rowptr[0] = 0;

    // First pass: count edges per node
    for (vtype i = 0; i < separator_nodes.size(); i++)
    {
      vtype orig_node = separator_nodes[i];
      vtype edge_count = 0;

      for (etype j = rowptr[orig_node]; j < rowptr[orig_node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (node_mapping[neighbor] != -1)
        {
          // Edge to another separator node
          edge_count++;
        }
      }

      sep_rowptr[i + 1] = sep_rowptr[i] + edge_count;
    }

    // Allocate space for column indices
    sep_colidx.resize(sep_rowptr.back());

    // Second pass: fill column indices
    for (vtype i = 0; i < separator_nodes.size(); i++)
    {
      vtype orig_node = separator_nodes[i];
      etype pos = sep_rowptr[i];

      for (etype j = rowptr[orig_node]; j < rowptr[orig_node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (node_mapping[neighbor] != -1)
        {
          // Edge to another separator node
          sep_colidx[pos++] = node_mapping[neighbor];
        }
      }
    }
  }

  void partition_graph(const vtype n, const vtype *rowptr, const etype *colidx,
                       std::vector<vtype> &part1_nodes, std::vector<vtype> &part2_nodes,
                       std::vector<vtype> &separator_nodes, double balance_factor)
  {
    // Clear output vectors
    part1_nodes.clear();
    part2_nodes.clear();
    separator_nodes.clear();

    // Find a good starting node (pick a peripheral node)
    vtype start_node = find_peripheral_node(n, rowptr, colidx);

    // Run BFS from start_node
    std::vector<vtype> level_set;
    std::vector<vtype> node_level(n, -1);
    std::vector<bool> visited(n, false);

    std::queue<vtype> queue;
    queue.push(start_node);
    visited[start_node] = true;
    node_level[start_node] = 0;

    vtype max_level = 0;

    while (!queue.empty())
    {
      vtype node = queue.front();
      queue.pop();

      max_level = std::max(max_level, node_level[node]);

      for (etype j = rowptr[node]; j < rowptr[node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (!visited[neighbor])
        {
          visited[neighbor] = true;
          node_level[neighbor] = node_level[node] + 1;
          queue.push(neighbor);
        }
      }
    }

    // Find split level based on balance factor
    vtype target_size = static_cast<vtype>(n * balance_factor);
    std::vector<vtype> level_sizes(max_level + 1, 0);

    for (vtype i = 0; i < n; i++)
    {
      if (node_level[i] != -1)
      {
        level_sizes[node_level[i]]++;
      }
    }

    vtype cumulative_size = 0;
    vtype split_level = 0;

    for (vtype level = 0; level <= max_level; level++)
    {
      cumulative_size += level_sizes[level];
      if (cumulative_size >= target_size)
      {
        split_level = level;
        break;
      }
    }

    // Identify separator nodes (nodes at split_level with connections to split_level+1)
    std::vector<bool> is_separator(n, false);

    for (vtype i = 0; i < n; i++)
    {
      if (node_level[i] == split_level)
      {
        bool has_higher_neighbor = false;

        for (etype j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
          vtype neighbor = colidx[j];
          if (node_level[neighbor] > split_level)
          {
            has_higher_neighbor = true;
            break;
          }
        }

        if (has_higher_neighbor)
        {
          is_separator[i] = true;
        }
      }
    }

    // Assign nodes to parts
    for (vtype i = 0; i < n; i++)
    {
      if (node_level[i] == -1)
      {
        // Disconnected nodes go to part1
        part1_nodes.push_back(i);
      }
      else if (is_separator[i])
      {
        separator_nodes.push_back(i);
      }
      else if (node_level[i] <= split_level)
      {
        part1_nodes.push_back(i);
      }
      else
      {
        part2_nodes.push_back(i);
      }
    }

    // Return if partitioning is unsuccessful
    if (part1_nodes.empty() || part2_nodes.empty())
    {
      // Clear outputs and put all nodes in part1
      part1_nodes.clear();
      part2_nodes.clear();
      separator_nodes.clear();

      for (vtype i = 0; i < n; i++)
      {
        part1_nodes.push_back(i);
      }
    }
  }

  // Find a peripheral node using a double BFS
  vtype find_peripheral_node(const vtype n, const vtype *rowptr, const etype *colidx)
  {
    // Start from a random node
    vtype start = rand() % n;

    // First BFS to find a distant node
    std::vector<bool> visited(n, false);
    std::queue<vtype> queue;

    queue.push(start);
    visited[start] = true;

    vtype last_node = start;

    while (!queue.empty())
    {
      vtype node = queue.front();
      queue.pop();
      last_node = node;

      for (etype j = rowptr[node]; j < rowptr[node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (!visited[neighbor])
        {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }

    // Second BFS from the distant node
    std::fill(visited.begin(), visited.end(), false);

    queue.push(last_node);
    visited[last_node] = true;

    vtype peripheral_node = last_node;

    while (!queue.empty())
    {
      vtype node = queue.front();
      queue.pop();
      peripheral_node = node;

      for (etype j = rowptr[node]; j < rowptr[node + 1]; j++)
      {
        vtype neighbor = colidx[j];
        if (!visited[neighbor])
        {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }

    return peripheral_node;
  }

  // Order separator with constraints from parts
  uint64_t order_separator_with_constraints(const vtype n, const vtype *rowptr, const etype *colidx,
                                            vtype *perm, const config &config,
                                            const std::vector<vtype> &part1_nodes,
                                            const std::vector<vtype> &part1_perm,
                                            const std::vector<vtype> &part2_nodes,
                                            const std::vector<vtype> &part2_perm)
  {
    // Implement a constrained AMD ordering for the separator
    // Consider connections to ordered parts when selecting pivots
    // ...

    // For prototype, can use standard AMD and post-process
    return standard_paramd(n, rowptr, colidx, perm, config);
  }

  // Estimate interface cost between partitions
  uint64_t estimate_interface_cost(const std::vector<vtype> &part1_nodes,
                                   const std::vector<vtype> &part2_nodes,
                                   const std::vector<vtype> &separator_nodes)
  {
    // Estimate the fill-in cost of the interfaces between partitions
    // This is a heuristic function based on the size of the separator
    // and its connections

    return separator_nodes.size() * (part1_nodes.size() + part2_nodes.size()) / 100;
  }

  // A + AT
  void symmetrize(const vtype n, etype &free_start, const vtype *rowptr, const etype *colidx,
                  etype *&symrowptr, vtype *&neighborhood, const double mem, const bool sym)
  {
    if (sym)
    {
      symrowptr = (etype *)std::malloc(sizeof(etype) * (n + 1));
      symrowptr[0] = 0;
#pragma omp parallel for
      for (vtype i = 0; i < n; ++i)
      {
        bool diag = false;
        for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
          if (i == colidx[j])
          {
            diag = true;
            break;
          }
        symrowptr[i + 1] = rowptr[i + 1] - rowptr[i] - diag;
      }
      for (vtype i = 0; i < n; ++i)
        symrowptr[i + 1] += symrowptr[i];
      free_start = symrowptr[n];
      neighborhood = (vtype *)std::malloc(sizeof(vtype) * free_start * (1 + mem));
#pragma omp parallel for
      for (vtype i = 0; i < n; ++i)
      {
        vtype idx = 0;
        for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
          if (i != colidx[j])
            neighborhood[symrowptr[i] + (idx++)] = colidx[j];
      }
      return;
    }

    etype *newrowptr = (etype *)std::malloc(sizeof(etype) * (n + 1));
    symrowptr = (etype *)std::malloc(sizeof(etype) * (n + 1));
    vtype *newcolidx = (vtype *)std::malloc(sizeof(vtype) * rowptr[n] * 2);
#pragma omp parallel for
    for (vtype i = 0; i <= n; ++i)
      newrowptr[i] = 0;
    const vtype stride = (n + 15) / 16 * 16;
    vtype *cnt_ = (vtype *)std::malloc(sizeof(vtype) * stride * omp_get_max_threads());
#pragma omp parallel
    {
      const size_t tid = omp_get_thread_num();
      vtype *cnt = cnt_ + tid * stride;
      for (vtype i = 0; i < n; ++i)
        cnt[i] = 0;
      vtype l = n * tid / omp_get_num_threads();
      vtype r = n * (tid + 1) / omp_get_num_threads();
      for (vtype i = l; i < r; ++i)
        for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
          if (i != colidx[j])
          {
            ++cnt[i];
            ++cnt[colidx[j]];
          }
      for (vtype i = 0; i < n; ++i)
      {
        vtype add = cnt[i];
        if (add != 0)
        {
#pragma omp atomic capture
          {
            cnt[i] = newrowptr[i + 1];
            newrowptr[i + 1] += add;
          }
        }
      }
#pragma omp barrier
#pragma omp master
      {
        for (vtype i = 0; i < n; ++i)
        {
          newrowptr[i + 1] += newrowptr[i];
        }
      }
#pragma omp barrier
      for (vtype i = l; i < r; ++i)
        for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
          if (i != colidx[j])
          {
            newcolidx[(cnt[i]++) + newrowptr[i]] = colidx[j];
            newcolidx[(cnt[colidx[j]]++) + newrowptr[colidx[j]]] = i; // bottleneck
          }
#pragma omp barrier
      for (vtype i = l; i < r; ++i)
      {
        vtype unique = 0, stp = -i - 1;
        for (etype j = newrowptr[i]; j < newrowptr[i + 1]; ++j)
        {
          if (cnt[newcolidx[j]] != stp)
          {
            cnt[newcolidx[j]] = stp;
            newcolidx[newrowptr[i] + (unique++)] = newcolidx[j];
          }
        }
        symrowptr[i + 1] = unique;
      }
#pragma omp barrier
#pragma omp master
      {
        symrowptr[0] = 0;
        for (vtype i = 0; i < n; ++i)
        {
          symrowptr[i + 1] += symrowptr[i];
        }
        free_start = symrowptr[n];
        neighborhood = (vtype *)std::malloc(sizeof(vtype) * free_start * (1 + mem));
      }
#pragma omp barrier
      for (vtype i = l; i < r; ++i)
      {
        for (etype j = symrowptr[i]; j < symrowptr[i + 1]; ++j)
        {
          neighborhood[j] = newcolidx[newrowptr[i] + j - symrowptr[i]];
        }
      }
    }
    std::free(cnt_);
    std::free(newrowptr);
    std::free(newcolidx);
  }

  uint64_t paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config)
  {
    if (config.hierarchical && n > config.partition_threshold)
    {
      return hierarchical_paramd(n, rowptr, colidx, perm, config, 0);
    }
    else
    {
      return standard_paramd(n, rowptr, colidx, perm, config);
    }
  }

  // Parallel Approximate Minimum Degree Ordering Algorithm
  uint64_t standard_paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config)
  {
#define EMPTY (-1)
#define FLIP(x) (-(x) - 2)
#define UNFLIP(x) ((x < EMPTY) ? FLIP(x) : (x))
    struct Node
    {
      etype neighborhood_ptr;

      vtype stp;

      vtype neighborhood_len;
      vtype deg;

      vtype supernode_size;
      vtype num_hyperedge;

      vtype edge_head;
      vtype edge_next;
      vtype order;

      vtype central_pivot;

      uint64_t luby, luby_min;
      uint32_t valid;
      char pad[4];
    };

    constexpr double alpha = 10.0;
    constexpr bool aggressive = true;
    constexpr uint64_t INF = 1ULL << 63;

    timer timer(config.breakdown);
    etype free_start = 0;
    etype *symrowptr = nullptr;
    vtype *neighborhood = nullptr;
    symmetrize(n, free_start, rowptr, colidx, symrowptr, neighborhood, config.mem, config.sym);

    timer.time("A + AT");

    vtype *const iperm = (vtype *)std::malloc(n * sizeof(vtype));
    vtype *const stk = (vtype *)std::malloc(n * sizeof(vtype));
    vtype *const inv_rank = (vtype *)std::malloc(n * sizeof(vtype));
    vtype *const candidates = (vtype *)std::malloc(n * sizeof(vtype));
    Node *const s = (Node *)std::malloc(n * sizeof(Node));

    vtype num_dense = 0, num_eliminated = 0;
    const vtype dense_threshold = std::min(n, std::max(16, vtype(alpha < 0 ? n - 2 : alpha * std::sqrt(n))));

    concurrent_approximate_degree_lists deglists(n);
    vtype min_deg = 1;

#pragma omp parallel reduction(+ : num_eliminated, num_dense)
    {
#pragma omp for
      for (vtype i = 0; i < n; ++i)
      {
        s[i].stp = 1;
        s[i].supernode_size = 1;
        s[i].neighborhood_ptr = symrowptr[i];
        s[i].num_hyperedge = 0;
        s[i].deg = s[i].neighborhood_len = symrowptr[i + 1] - symrowptr[i];
        s[i].edge_head = s[i].edge_next = s[i].order = EMPTY;
        s[i].central_pivot = EMPTY;

        iperm[i] = 0;
        stk[i] = EMPTY;
        inv_rank[i] = EMPTY;

        if (s[i].deg == 0)
        {
          s[i].num_hyperedge = FLIP(1);
          ++num_eliminated;
          s[i].neighborhood_ptr = EMPTY;
          s[i].stp = 0;
        }
        else if (s[i].deg > dense_threshold)
        {
          ++num_dense;
          ++num_eliminated;
          s[i].neighborhood_ptr = EMPTY;
          s[i].stp = 0;
          s[i].supernode_size = 0;
          s[i].num_hyperedge = EMPTY;
        }
        else
        {
          deglists.insert(s[i].deg, i);
        }
      }
    }

    uint64_t lnz = uint64_t(num_dense) * (num_dense - 1) / 2;
    vtype num_candidates = 0;

    std::vector<vtype> size_profile;
    vtype size_profile_total = 0;

#pragma omp parallel
    {
      hashlists hashlists(n);
      vtype *const private_ = (vtype *)std::malloc(n * sizeof(vtype) * 3);
      vtype *const private_stp = private_;
      vtype *const workspace = private_ + n;
      vtype *const private_cand = private_ + n * 2;
      std::fill(private_stp, private_stp + n, 1);
      vtype num_private_cand = 0, workspace_end = 0, round = 1;
      vtype private_tstp = 2, private_t_max_step = 0;
      const vtype private_tlim = std::numeric_limits<vtype>::max() - n;
      uint64_t private_lnz = 0;
      std::mt19937 gen(omp_get_thread_num());
      std::uniform_int_distribution<uint64_t> dis(0, n - 1);
      timer.time("Other");
      while (num_eliminated < n)
      {
#pragma omp barrier
        deglists.traverse(min_deg, num_candidates, candidates, config.mult, config.lim);
        {
          num_private_cand = 0;
          ++round;
#pragma omp for
          for (vtype i = 0; i < num_candidates; ++i)
          {
            const vtype cand = candidates[i];
            private_cand[num_private_cand++] = cand;
            s[cand].luby = dis(gen) << 32 | cand;
            s[cand].stp = round;
            s[cand].valid = 1;
          }
          for (vtype i = 0; i < num_private_cand; ++i)
          {
            const vtype cand = private_cand[i];
            uint32_t &valid = s[cand].valid;
            s[cand].luby_min = INF;
            const etype hyper_start = s[cand].neighborhood_ptr;
            const etype hyper_end = hyper_start + s[cand].num_hyperedge;
            for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr)
            {
              const vtype hyper = neighborhood[hyper_ptr];
              const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
              for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr)
              {
                const vtype nei = neighborhood[nei_ptr];
                if (s[nei].stp != 0)
                {
                  if (s[nei].stp == round && s[nei].luby < s[cand].luby)
                  {
                    valid = false;
                  }
                  else if (s[nei].luby_min != INF)
                  {
                    s[nei].luby_min = INF;
                  }
                }
              }
            }
            if (!valid)
              continue;
            const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
            for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr)
            {
              const vtype nei = neighborhood[nei_ptr];
              if (s[nei].stp != 0)
              {
                if (s[nei].stp == round && s[nei].luby < s[cand].luby)
                {
                  valid = false;
                }
                else if (s[nei].luby_min != INF)
                {
                  s[nei].luby_min = INF;
                }
              }
            }
          }
#pragma omp barrier
          for (vtype i = 0; i < num_private_cand; ++i)
          {
            const vtype cand = private_cand[i];
            const uint64_t luby_cand = s[cand].luby;
            uint32_t &valid = s[cand].valid;
            if (!valid)
              continue;
#pragma omp atomic compare
            s[cand].luby_min = s[cand].luby_min > luby_cand ? luby_cand : s[cand].luby_min;
            if (s[cand].luby_min != luby_cand)
              valid = false;
            if (!valid)
              continue;
            const etype hyper_start = s[cand].neighborhood_ptr;
            const etype hyper_end = hyper_start + s[cand].num_hyperedge;
            for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr)
            {
              const vtype hyper = neighborhood[hyper_ptr];
              const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
              for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr)
              {
                const vtype nei = neighborhood[nei_ptr];
                if (s[nei].stp != 0)
                {
#pragma omp atomic compare
                  s[nei].luby_min = s[nei].luby_min > luby_cand ? luby_cand : s[nei].luby_min;
                  if (s[nei].luby_min != luby_cand)
                    valid = false;
                }
              }
            }
            if (!valid)
              continue;
            const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
            for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr)
            {
              const vtype nei = neighborhood[nei_ptr];
              if (s[nei].stp != 0)
              {
#pragma omp atomic compare
                s[nei].luby_min = s[nei].luby_min > luby_cand ? luby_cand : s[nei].luby_min;
                if (s[nei].luby_min != luby_cand)
                  valid = false;
              }
            }
          }
#pragma omp barrier
          vtype resize = 0;
          for (vtype i = 0; i < num_private_cand; ++i)
          {
            const vtype cand = private_cand[i];
            const uint64_t luby_cand = s[cand].luby;
            bool valid = s[cand].luby_min == luby_cand && s[cand].valid;
            if (!valid)
              continue;
            const etype hyper_start = s[cand].neighborhood_ptr;
            const etype hyper_end = hyper_start + s[cand].num_hyperedge;
            for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr)
            {
              const vtype hyper = neighborhood[hyper_ptr];
              const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
              for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr)
              {
                const vtype nei = neighborhood[nei_ptr];
                if (s[nei].stp != 0 && s[nei].luby_min != luby_cand)
                {
                  valid = false;
                }
              }
            }
            if (!valid)
              continue;
            const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
            for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr)
            {
              const vtype nei = neighborhood[nei_ptr];
              if (s[nei].stp != 0 && s[nei].luby_min != luby_cand)
              {
                valid = false;
              }
            }
            if (valid)
            {
              private_cand[resize++] = cand;
            }
          }
          num_private_cand = resize;
        }
        timer.time("Distance-2 Independent Sets");

        if (config.stat)
        {
#pragma omp master
          {
            size_profile_total = 0;
          }
#pragma omp barrier
#pragma omp atomic
          size_profile_total += num_private_cand;
#pragma omp barrier
#pragma omp master
          {
            size_profile.emplace_back(size_profile_total);
          }
        }

        vtype private_num_eliminated = 0;
        workspace_end = 0;
        for (vtype cand_idx = 0; cand_idx < num_private_cand; ++cand_idx)
        {
          const vtype pivot = private_cand[cand_idx];
          vtype npiv = s[pivot].supernode_size;
          private_num_eliminated += npiv;
          s[pivot].central_pivot = pivot;
          s[pivot].supernode_size = -npiv;

          vtype pivot_deg = 0;
          etype new_nei_start = workspace_end;
          deglists.remove(s[pivot].deg, pivot);
          const etype hyper_start = s[pivot].neighborhood_ptr, hyper_end = hyper_start + s[pivot].num_hyperedge;
          for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr)
          {
            const vtype hyper = neighborhood[hyper_ptr];
            const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
            for (etype nei_ptr = nei_start; nei_ptr < nei_end; ++nei_ptr)
            {
              const vtype nei = neighborhood[nei_ptr];
              const vtype nei_size = s[nei].supernode_size;
              if (nei_size > 0)
              {
                pivot_deg += nei_size;
                s[nei].supernode_size = -nei_size;
                workspace[workspace_end++] = nei;
                s[nei].central_pivot = pivot;
                deglists.remove(s[nei].deg, nei);
              }
            }
            s[hyper].neighborhood_ptr = FLIP(pivot);
            s[hyper].stp = 0;
          }
          const etype super_end = s[pivot].neighborhood_ptr + s[pivot].neighborhood_len;
          for (etype nei_ptr = hyper_end; nei_ptr < super_end; ++nei_ptr)
          {
            const vtype nei = neighborhood[nei_ptr];
            const vtype nei_size = s[nei].supernode_size;
            if (nei_size > 0)
            {
              pivot_deg += nei_size;
              s[nei].supernode_size = -nei_size;
              workspace[workspace_end++] = nei;
              s[nei].central_pivot = pivot;
              deglists.remove(s[nei].deg, nei);
            }
          }
          etype new_nei_end = workspace_end;

          s[pivot].neighborhood_ptr = new_nei_start;
          s[pivot].num_hyperedge = FLIP(npiv + pivot_deg);
          clear_stp(private_stp, n, private_tstp, private_tlim);
          private_t_max_step = 0;

          for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
          {
            const vtype nei = workspace[nei_ptr];
            const etype hyper_start = s[nei].neighborhood_ptr, hyper_end = s[nei].neighborhood_ptr + s[nei].num_hyperedge;
            for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr)
            {
              const vtype hyper = neighborhood[hyper_ptr];
              if (private_stp[hyper] >= private_tstp)
              {
                private_stp[hyper] += s[nei].supernode_size;
              }
              else if (s[hyper].stp != 0)
              {
                private_stp[hyper] = s[hyper].deg + private_tstp + s[nei].supernode_size;
                private_t_max_step = std::max(private_t_max_step, s[hyper].deg);
              }
            }
          }

          for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
          {
            const vtype nei = workspace[nei_ptr];
            const etype hyper_start = s[nei].neighborhood_ptr, hyper_end = s[nei].neighborhood_ptr + s[nei].num_hyperedge;
            etype new_end = hyper_start;
            vtype nei_deg = 0;
            vtype hash = 0;
            for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr)
            {
              const vtype hyper = neighborhood[hyper_ptr];
              if (s[hyper].stp != 0)
              {
                const vtype external_deg = private_stp[hyper] - private_tstp;
                if (!aggressive || external_deg > 0)
                {
                  nei_deg += external_deg;
                  neighborhood[new_end++] = hyper;
                  hash += hyper;
                }
                else
                {
                  s[hyper].neighborhood_ptr = FLIP(pivot);
                  s[hyper].stp = 0;
                }
              }
            }
            s[nei].num_hyperedge = new_end - hyper_start + 1;
            const etype super_end = hyper_start + s[nei].neighborhood_len;
            const etype super_start = new_end;
            for (etype super_ptr = hyper_end; super_ptr < super_end; ++super_ptr)
            {
              const vtype super = neighborhood[super_ptr];
              if (s[super].supernode_size != 0 && s[super].central_pivot != pivot)
              {
                nei_deg += std::abs(s[super].supernode_size);
                neighborhood[new_end++] = super;
                hash += super;
              }
            }

            if (s[nei].num_hyperedge == 1 && new_end == super_start)
            {
              s[nei].neighborhood_ptr = FLIP(pivot);
              const vtype nei_size = -s[nei].supernode_size;
              pivot_deg -= nei_size;
              private_num_eliminated += nei_size;
              npiv += nei_size;
              s[nei].num_hyperedge = EMPTY;
              s[nei].supernode_size = 0;
            }
            else
            {
              s[nei].deg = std::min(s[nei].deg, nei_deg);
              neighborhood[new_end] = neighborhood[super_start];
              neighborhood[super_start] = neighborhood[hyper_start];
              neighborhood[hyper_start] = pivot;
              s[nei].neighborhood_len = new_end - hyper_start + 1;
              hash = (hash % n);
              if (hash < 0)
                hash += n;
              hashlists.insert(hash, nei);
            }
          }

          s[pivot].deg = pivot_deg;
          s[pivot].supernode_size = npiv;
          private_tstp += private_t_max_step;
          clear_stp(private_stp, n, private_tstp, private_tlim);

          for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
          {
            const vtype nei = workspace[nei_ptr];
            if (s[nei].supernode_size < 0)
            {
              const vtype hash = hashlists.get_hash(nei);
              while (!hashlists.empty(hash))
              {
                const vtype nei_i = hashlists.pop(hash);
                const vtype nei_len_i = s[nei_i].neighborhood_len;
                const vtype nhe_i = s[nei_i].num_hyperedge;
                if (hashlists.empty(hash))
                  break;
                for (etype nei_i_ptr = s[nei_i].neighborhood_ptr + 1; nei_i_ptr < s[nei_i].neighborhood_ptr + nei_len_i; ++nei_i_ptr)
                {
                  private_stp[neighborhood[nei_i_ptr]] = private_tstp;
                }
                vtype nei_j = nei_i, prev_nei_j = hashlists.get_nil();
                while (!hashlists.is_tail(nei_j))
                {
                  nei_j = hashlists.get_next(nei_j);
                  const vtype nei_len_j = s[nei_j].neighborhood_len;
                  const vtype nhe_j = s[nei_j].num_hyperedge;
                  bool same = (nei_len_i == nei_len_j) && (nhe_i == nhe_j);
                  for (etype nei_j_ptr = s[nei_j].neighborhood_ptr + 1; nei_j_ptr < s[nei_j].neighborhood_ptr + nei_len_j; ++nei_j_ptr)
                  {
                    same &= (private_stp[neighborhood[nei_j_ptr]] == private_tstp);
                  }
                  if (same)
                  {
                    s[nei_j].neighborhood_ptr = FLIP(nei_i);
                    s[nei_i].supernode_size += s[nei_j].supernode_size;
                    s[nei_j].supernode_size = 0;
                    s[nei_j].num_hyperedge = EMPTY;
                    hashlists.remove(nei_j, prev_nei_j);
                  }
                  else
                  {
                    prev_nei_j = nei_j;
                  }
                }
                ++private_tstp;
              }
            }
          }

          etype final_nei_end = new_nei_start;
          for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
          {
            const vtype nei = workspace[nei_ptr];
            const vtype nei_size = -s[nei].supernode_size;
            if (nei_size > 0)
            {
              s[nei].supernode_size = nei_size;
              s[nei].deg = std::min(s[nei].deg + pivot_deg - nei_size, n - num_eliminated - nei_size);
              workspace[final_nei_end++] = nei;
              deglists.insert(s[nei].deg, nei);
            }
          }
          s[pivot].neighborhood_len = final_nei_end - new_nei_start;
          workspace_end = final_nei_end;
          private_lnz += uint64_t(s[pivot].supernode_size) * (s[pivot].deg + num_dense) + uint64_t(s[pivot].supernode_size) * (s[pivot].supernode_size - 1) / 2;
          if (s[pivot].neighborhood_len == 0)
          {
            s[pivot].neighborhood_ptr = EMPTY;
            s[pivot].stp = 0;
          }
        }

        etype base = 0;
#pragma omp atomic capture
        {
          base = free_start;
          free_start += workspace_end;
        }
        std::copy(workspace, workspace + workspace_end, neighborhood + base);

        for (vtype cand_idx = 0; cand_idx < num_private_cand; ++cand_idx)
        {
          const vtype pivot = private_cand[cand_idx];
          if (s[pivot].neighborhood_len != 0)
            s[pivot].neighborhood_ptr += base;
        }

#pragma omp atomic
        num_eliminated += private_num_eliminated;
#pragma omp barrier
        timer.time("Core");
      }
#pragma omp atomic
      lnz += private_lnz;
      std::free(private_);
    }

#pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
    {
      s[i].neighborhood_ptr = FLIP(s[i].neighborhood_ptr);
      s[i].num_hyperedge = FLIP(s[i].num_hyperedge);
    }

#pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
    {
      if (s[i].supernode_size == 0)
      {
        vtype p = s[i].neighborhood_ptr;
        if (p != EMPTY)
        {
          while (s[p].supernode_size == 0)
            p = s[p].neighborhood_ptr;
          const vtype hyper = p;
          for (vtype ptr = i; ptr != hyper;)
          {
            p = s[ptr].neighborhood_ptr;
            s[ptr].neighborhood_ptr = hyper;
            ptr = p;
          }
        }
      }
    }

#pragma omp parallel for
    for (vtype i = n - 1; i >= 0; --i)
    {
      if (s[i].supernode_size > 0)
      {
        vtype p = s[i].neighborhood_ptr;
        if (p != EMPTY)
        {
#pragma omp atomic capture
          {
            s[i].edge_next = s[p].edge_head;
            s[p].edge_head = i;
          }
        }
      }
    }

#pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
    {
      if (s[i].supernode_size > 0 && s[i].edge_head != EMPTY)
      {
        vtype prev = EMPTY, mxsz = EMPTY, mxprev = EMPTY, mxptr = EMPTY;
        for (vtype j = s[i].edge_head; j != EMPTY; j = s[j].edge_next)
        {
          if (s[j].num_hyperedge >= mxsz)
          {
            mxsz = s[j].num_hyperedge;
            mxprev = prev;
            mxptr = j;
          }
          prev = j;
        }

        if (s[mxptr].edge_next != EMPTY)
        {
          if (mxprev == EMPTY)
            s[i].edge_head = s[mxptr].edge_next;
          else
            s[mxprev].edge_next = s[mxptr].edge_next;
          s[mxptr].edge_next = EMPTY;
          s[prev].edge_next = mxptr;
        }
      }
    }

    vtype stk_head = 0, k = 0;

    for (vtype i = 0; i < n; ++i)
    {
      if (s[i].neighborhood_ptr == EMPTY && s[i].supernode_size > 0)
      {
        stk[stk_head = 0] = i;
        while (stk_head != EMPTY)
        {
          vtype cur = stk[stk_head];
          if (s[cur].edge_head != EMPTY)
          {
            for (vtype j = s[cur].edge_head; j != EMPTY; j = s[j].edge_next)
            {
              ++stk_head;
            }
            for (vtype j = s[cur].edge_head, h = stk_head; j != EMPTY; j = s[j].edge_next)
            {
              stk[h--] = j;
            }
            s[cur].edge_head = EMPTY;
          }
          else
          {
            --stk_head;
            s[cur].order = k++;
          }
        }
      }
    }

#pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
    {
      if (s[i].order != EMPTY)
      {
        inv_rank[s[i].order] = i;
      }
    }

    num_eliminated = 0;

    for (vtype i = 0; i < n; ++i)
    {
      vtype hyper = inv_rank[i];
      if (hyper == EMPTY)
        break;
      iperm[hyper] = num_eliminated;
      num_eliminated += s[hyper].supernode_size;
    }

#pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
    {
      if (s[i].supernode_size == 0)
      {
        if (s[i].neighborhood_ptr != EMPTY)
        {
#pragma omp atomic capture
          {
            iperm[i] = iperm[s[i].neighborhood_ptr];
            ++iperm[s[i].neighborhood_ptr];
          }
        }
        else
        {
#pragma omp atomic capture
          {
            iperm[i] = num_eliminated;
            ++num_eliminated;
          }
        }
      }
    }

#pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
    {
      perm[iperm[i]] = i;
    }

    std::free(s);
    std::free(candidates);
    std::free(inv_rank);
    std::free(stk);
    std::free(iperm);
    std::free(neighborhood);
    std::free(symrowptr);

    timer.time("Other");
    timer.print();

    if (config.stat)
    {
      std::cout << "Size of distance-2 independent sets: [";
      for (auto x : size_profile)
        std::cout << x << ", ";
      std::cout << "]\n";
    }

    return lnz;

#undef UNFLIP
#undef FLIP
#undef EMPTY
  }

  uint64_t hierarchical_paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config_p, int recursion_level)
  {
    // Base case: if matrix is small enough or maximum recursion depth reached
    if (n <= config_p.partition_threshold || recursion_level >= config_p.max_recursion_depth)
    {
      return standard_paramd(n, rowptr, colidx, perm, config_p);
    }

    // Step 1: Partition the graph
    std::vector<vtype> part1_nodes, part2_nodes, separator_nodes;
    partition_graph(n, rowptr, colidx, part1_nodes, part2_nodes, separator_nodes, config_p.balance_factor);

    // If partitioning failed or produced unbalanced partitions, fall back to standard algorithm
    if (part1_nodes.empty() || part2_nodes.empty() ||
        part1_nodes.size() < n * 0.2 || part2_nodes.size() < n * 0.2)
    {
      return standard_paramd(n, rowptr, colidx, perm, config_p);
    }

    // Step 2: Create subgraphs
    std::vector<etype> part1_rowptr, part2_rowptr, separator_rowptr;
    std::vector<vtype> part1_colidx, part2_colidx, separator_colidx;

    // Extract subgraphs (implementation details omitted for brevity)
    extract_subgraph(n, rowptr, colidx, part1_nodes, part1_rowptr, part1_colidx);
    extract_subgraph(n, rowptr, colidx, part2_nodes, part2_rowptr, part2_colidx);

    // Extract separator with connections to both parts
    extract_separator_subgraph(n, rowptr, colidx, separator_nodes, part1_nodes, part2_nodes,
                               separator_rowptr, separator_colidx);

    // Step 3: Recursively compute ordering for each part
    std::vector<vtype> part1_perm(part1_nodes.size()), part2_perm(part2_nodes.size());

    uint64_t fill1 = 0, fill2 = 0, fillSep = 0;

// Process parts in parallel
#pragma omp parallel sections
    {
#pragma omp section
      {
        fill1 = hierarchical_paramd(part1_nodes.size(), part1_rowptr.data(), part1_colidx.data(),
                                    part1_perm.data(), config_p, recursion_level + 1);
      }

#pragma omp section
      {
        fill2 = hierarchical_paramd(part2_nodes.size(), part2_rowptr.data(), part2_colidx.data(),
                                    part2_perm.data(), config_p, recursion_level + 1);
      }
    }

    // Step 4: Order separator with constraints from part1 and part2
    std::vector<vtype> separator_perm(separator_nodes.size());
    config sep_config = config_p;
    sep_config.hierarchical = false;

    // Add constraints from ordered parts to separator ordering
    fillSep = order_separator_with_constraints(separator_nodes.size(), separator_rowptr.data(),
                                               separator_colidx.data(), separator_perm.data(),
                                               sep_config, part1_nodes, part1_perm,
                                               part2_nodes, part2_perm);

    // Step 5: Combine the orderings
    combine_orderings(n, perm, part1_nodes, part1_perm, part2_nodes, part2_perm,
                      separator_nodes, separator_perm);

    // Return estimated fill-in (sum of parts plus interface cost)
    uint64_t interface_cost = estimate_interface_cost(part1_nodes, part2_nodes, separator_nodes);
    return fill1 + fill2 + fillSep + interface_cost;
  }

} // end of namespace paramd