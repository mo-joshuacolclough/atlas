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

const atlas::mpi::Comm& parent_comm() { return atlas::mpi::comm("world"); }
const atlas::mpi::Comm& split_comm() { return atlas::mpi::comm(AtlasSplitCommEnvironment::s_split_comm_name); }

//-----------------------------------------------------------------------------

CASE("test_split_comm_redistribution") {
    // 4 PEs         Colour
    // [0, 1, 2]  :  0
    // [3]        :  1
  
    /*
     * Start on the parent communicator, with distribution `D`.
     * Then, split comm and create a distribution `d`. Then, convert
     * `d` to be a global distribution `D_d`. This takes a field on a world comm
     * and distributes it only to a sub comm with distribution `d`.
     *
     * `D --D_d--> d`
    */

    eckit::mpi::setCommDefault(parent_comm().name());

    // === LOCAL GROUP ===
    if (colour_management::get_colour() == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
  
    const std::string grid_name = "O8";
    const auto grid = Grid(grid_name);
    grid::Partitioner partition_scheme("equal_area");
  
    // D
    grid::Distribution parent_dist = partition_scheme.partition(grid);
  
    // Partitioned either on 1 PE or 3 PEs depending on group.
    functionspace::StructuredColumns parent_fspace(grid, parent_dist);
    Field parent_field = parent_fspace.createField<int>(atlas::option::name("comm_field"));

    field::for_each_value(parent_field, [&](int& x) {
        x = colour_management::get_colour() + 1;
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
    eckit::mpi::setCommDefault(split_comm().name());

    // d
    grid::Distribution comm_dist = partition_scheme.partition(grid);
    EXPECT(comm_dist.size() == parent_dist.size());

    // Create D_d : Distribution of `d` on a parent comm distribution `D`
    std::vector<int> parent_to_comm_dist_data(parent_dist.size(), -1);

    // Convert the local colour rank to parent ranks.
    const size_t rank_offset = colour_management::colour_rank_offset(split_comm());

    for (gidx_t global_idx = 0; global_idx < comm_dist.size(); ++global_idx) {
      parent_to_comm_dist_data[global_idx] = rank_offset + comm_dist.partition(global_idx);
    }

    // D_d
    const int numPartitions = parent_comm().size();
    const int globalSize = parent_to_comm_dist_data.size();
    grid::Distribution D_d(numPartitions, globalSize, parent_to_comm_dist_data.data());
}

//-----------------------------------------------------------------------------

}  // namespace test
}  // namespace atlas

int main(int argc, char** argv) {
    return atlas::test::run<atlas::test::AtlasSplitCommEnvironment>(argc, argv);
}
