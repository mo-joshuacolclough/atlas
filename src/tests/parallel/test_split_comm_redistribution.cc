/*
 * (C) Crown Copyright 2025 Met Office.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */


#include <chrono>
#include <thread>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string_view>
#include <string>

#include "atlas/array.h"
#include "atlas/array/ArrayView.h"
#include "atlas/array/MakeView.h"
#include "atlas/field.h"
#include "atlas/field/for_each.h"
#include "atlas/functionspace.h"
#include "atlas/grid.h"
#include "atlas/library/config.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/parallel/mpi/mpi.h"
#include "atlas/redistribution/Redistribution.h"

#include "eckit/mpi/Comm.h"

#include "tests/AtlasTestEnvironment.h"


namespace atlas {
namespace test {

namespace colour_management {
    // 1:3 ratio.
    static constexpr float ratio = 0.25;

    int get_colour(const int world_rank = atlas::mpi::comm("world").rank(),
                   const int world_size = atlas::mpi::comm("world").size()) {
        if (static_cast<float>(world_rank) / static_cast<float>(world_size) < ratio) {
            return 0;
        } else {
            return 1;
        }
    }

    // Method to work out what world rank offset this colour is.
    size_t colour_rank_offset(const atlas::mpi::Comm& comm) {
        int rank_offset = 0;
        // If this instance is the root of the current comm, add the world rank to the offset.
        if (comm.rank() == 0) {
            rank_offset = atlas::mpi::comm("world").rank();
        }

        // Comm sum. Aim here is that only the root rank has a value, so it will be: offset + 0 + 0 + ...
        comm.allReduceInPlace(rank_offset, eckit::mpi::sum());
        return rank_offset;
    }
}  // namespace colour_management


std::ostream& worldlog() {
    // |R<rank> C<colour>| ==>
    std::cout << "|R" << atlas::mpi::comm("world").rank() << " C" << colour_management::get_colour() << "| ==> ";
    return std::cout;
}


struct AtlasSplitCommEnvironment : public AtlasTestEnvironment {
    static constexpr std::string_view s_split_comm_name = "split_comm";

    AtlasSplitCommEnvironment(int argc, char* argv[]): AtlasTestEnvironment(argc, argv) {
        // Split world communicator
        atlas::mpi::comm().split(colour_management::get_colour(), std::string(s_split_comm_name));
    }
};

const atlas::mpi::Comm& split_comm() { return atlas::mpi::comm(AtlasSplitCommEnvironment::s_split_comm_name); }

//-----------------------------------------------------------------------------

CASE("test_split_comm_redistribution") {
    // 4 PEs         Colour
    // [0, 1, 2]  :  0
    // [3]        :  1
  
    eckit::mpi::setCommDefault(AtlasSplitCommEnvironment::s_split_comm_name);

    // === LOCAL GROUP ===
    if (colour_management::get_colour() == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
  
    const std::string grid_name = "O8";
    const auto grid = Grid(grid_name);
  
    grid::Partitioner partition_scheme("equal_area");
    grid::Distribution comm_dist = partition_scheme.partition(grid);
  
    // Partitioned either on 1 PE or 3 PEs depending on group.
    functionspace::StructuredColumns comm_fspace(grid, comm_dist);
    Field comm_field = comm_fspace.createField<float>(atlas::option::name("comm_field"));

    field::for_each_value(comm_field, [&](float& x) {
        x = static_cast<float>(colour_management::get_colour());
    });

    /*
    if (atlas::mpi::comm().rank() == 0) {
      const gidx_t g_size = comm_dist.size();
      for (gidx_t ij = 0; ij < g_size; ++ij) {
        worldlog() << ij << " ==> " << comm_dist.partition(ij) << std::endl;
      }
    }
    */
  
    // === GLOBAL GROUP ===
    eckit::mpi::setCommDefault("world");
  
    const size_t rankInColour = split_comm().rank();
  
    // -- Convert comm distribution to global ranks.
    const idx_t npts = comm_dist.nb_pts()[rankInColour];
    worldlog() << "Partition size = " << npts << std::endl;
  
    const gidx_t part_offset = [&]() {
        gidx_t offset = 0;
        for (size_t r = 1; r < rankInColour + 1; ++r) {
            offset += static_cast<gidx_t>(comm_dist.nb_pts()[r - 1]);
        }
        return offset;
    }();
  
    const size_t colour_rank_offset = colour_management::colour_rank_offset(split_comm());
  
    const gidx_t num_grid_points = comm_dist.size();
    std::vector<int> partitioning(num_grid_points, -1);
    // Convert
    for (gidx_t global_idx = 0; global_idx < num_grid_points; ++global_idx) {
        partitioning[global_idx] = colour_rank_offset + comm_dist.partition(global_idx);
    }
  
    const size_t world_size = atlas::mpi::comm("world").size();
    grid::Distribution global_dist(world_size, num_grid_points, partitioning.data());
    worldlog() << "GLOBAL SIZE == " << global_dist.size() << std::endl;

    if (colour_management::get_colour() == 1 && split_comm().rank() == 0) {
      const gidx_t g_size = global_dist.size();
      for (gidx_t ij = 0; ij < g_size; ++ij) {
        worldlog() << "GLOBAL " << ij << " ==> " << global_dist.partition(ij) << std::endl;
      }
    }

    //functionspace::StructuredColumns global_fspace(grid, global_dist);
    
    // (global in terms of comm distribution)
    //Field global_field = comm_fspace.createField<float>(atlas::option::name("global_field"));

}

//-----------------------------------------------------------------------------

}  // namespace test
}  // namespace atlas

int main(int argc, char** argv) {
    return atlas::test::run<atlas::test::AtlasSplitCommEnvironment>(argc, argv);
}
