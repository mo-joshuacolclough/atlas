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
#include "atlas/interpolation/Interpolation.h"
#include "atlas/library/config.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/output/Gmsh.h"
#include "atlas/parallel/mpi/mpi.h"
#include "atlas/redistribution/Redistribution.h"
#include "atlas/util/Config.h"
#include "atlas/util/function/VortexRollup.h"

#include "eckit/mpi/Comm.h"

#include "tests/AtlasTestEnvironment.h"


using atlas::util::Config;

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

/*
Field distributeToSubComms(const Field& parent_field,
                           const mpi::Comm& parent_comm = parent_comm(),
                           const mpi::Comm& split_comm = split_comm()) {

    grid::Partitioner comm_partition_scheme("equal_bands", split_comm.size());
    grid::Distribution parent_dist = comm_partition_scheme.partition(grid);
    Mesh comm_mesh = StructuredMeshGenerator().generate(grid, comm_dist);

    // Create functionspace with split comm
    eckit::mpi::setCommDefault(split_comm().name());
    functionspace::NodeColumns comm_fspace(comm_mesh);

    Field distributed;

}
*/

double rms(const Field& f, const mpi::Comm& comm) {
    const auto view = array::make_view<const double, 1>(f);
    const auto ghost = array::make_view<const int, 1>(f.functionspace().ghost());

    double sum_sqr = 0;
    for (idx_t ij = 0; ij < view.shape(0); ++ij) {
      if (ghost(ij) == 0) {
        sum_sqr += view(ij) * view(ij);
      }
    }

    idx_t size = view.shape(0);
    comm.allReduceInPlace(size, eckit::mpi::sum());
    comm.allReduceInPlace(sum_sqr, eckit::mpi::sum());

    return std::sqrt( sum_sqr / static_cast<double>(size) );
}

double min(const Field& f, const mpi::Comm& comm) {
    const auto view = array::make_view<const double, 1>(f);
    const auto ghost = array::make_view<const int, 1>(f.functionspace().ghost());

    double min = std::numeric_limits<double>::max();

    for (idx_t ij = 0; ij < view.shape(0); ++ij) {
      if (ghost(ij) == 0) {
        if (view(ij) < min) { min = view(ij); }
      }
    }
    
    comm.allReduceInPlace(min, eckit::mpi::min());
    return min;
}

double max(const Field& f, const mpi::Comm& comm) {
    const auto view = array::make_view<const double, 1>(f);
    const auto ghost = array::make_view<const int, 1>(f.functionspace().ghost());

    double max = -std::numeric_limits<double>::max();

    for (idx_t ij = 0; ij < view.shape(0); ++ij) {
      if (ghost(ij) == 0) {
        if (view(ij) > max) { max = view(ij); }
      }
    }
    
    comm.allReduceInPlace(max, eckit::mpi::max());
    return max;
}

void compareFields(const Field& f1, const Field& f2, const mpi::Comm& print_comm) {
  const mpi::Comm& comm1 = eckit::mpi::comm(f1.functionspace().mpi_comm());
  const mpi::Comm& comm2 = eckit::mpi::comm(f2.functionspace().mpi_comm());

  double min1 = min(f1, comm1);
  double min2 = min(f2, comm2);
  double max1 = max(f1, comm1);
  double max2 = max(f2, comm2);
  double rms1 = rms(f1, comm1);
  double rms2 = rms(f2, comm2);

  if (print_comm.rank() == 0) {
    worldlog() << "=== COMPARE ===" << std::endl
               << "| " << f1.name() << " | " << f2.name() << std::endl
               << "| Min " << min1 << "        " << min2 << std::endl
               << "| Max " << max1 << "        " << max2 << std::endl
               << "| RMS " << rms1 << "        " << rms2 << std::endl;
  }

  EXPECT(min1 == min2);
  EXPECT(max1 == max2);
  EXPECT(rms1 == rms2);
}



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

    worldlog () << "======== INIT. GROUP SIZE = " << split_comm().size() << " ========" << std::endl;

    eckit::mpi::setCommDefault(parent_comm().name());

    // === PARENT GROUP ===
    const std::string grid_name = "O16";
    const auto grid = Grid(grid_name);
    grid::Partitioner parent_partition_scheme("equal_bands", parent_comm().size());
    worldlog() << "Parent NB partitions = " << parent_partition_scheme.nb_partitions() << std::endl;
  
    // D
    grid::Distribution parent_dist = parent_partition_scheme.partition(grid);
    functionspace::StructuredColumns parent_fspace(grid, parent_dist);

    Field parent_field = parent_fspace.createField<double>(atlas::option::name("parent_field"));

    auto parent_view = array::make_view<double, 1>(parent_field);
    auto lonlat = array::make_view<double, 2>(parent_fspace.lonlat());

    for (idx_t ij = 0; ij < parent_view.shape(0); ++ij) {
        parent_view(ij) = util::function::vortex_rollup(lonlat(ij, LON), lonlat(ij, LAT), 0.5 / 2);
    }

    // DUMMY
    compareFields(parent_field, parent_field, parent_comm());

    // === SPLIT GROUP ===

    // d
    grid::Partitioner comm_partition_scheme("equal_bands", split_comm().size());
    grid::Distribution comm_dist = comm_partition_scheme.partition(grid);
    EXPECT(comm_dist.size() == parent_dist.size());

    functionspace::StructuredColumns comm_fspace(grid, comm_dist);
 
    // Gather on a global field on split comm rank 0.
    // TODO: Check - is this just gathering from the split comm..? Not all data.
    //
    // 1: Gather on world.
    // 2: Gather on world, copy to split.
    // 3: Gather on world, copy to split, scatter on split.
    // (and inverses)
    //
    // Copy is to copy from parent comm Fields to Fields with split comm.
    //
    // Check data is the same: SHA hash, min/max & rms. Or use IO somehow.
    // WARNING: Careful of halo - SHA includes halo. Min/max/rms with owned only.
    //          Own SHA function on owned?
    //
    // Or dump to txt. Global index as main index to sort.
    //
    eckit::mpi::setCommDefault(split_comm().name());
    Field global_field = comm_fspace.createField<double>(option::name("global_subcomm_field") | 
                                                         option::global(0));

    eckit::mpi::setCommDefault(parent_comm().name());
    comm_fspace.gather(parent_field, global_field);

    // Scatter on split comm.
    eckit::mpi::setCommDefault(split_comm().name());
    Field comm_field = comm_fspace.createField<double>(option::name("comm_field"));

    comm_fspace.scatter(global_field, comm_field);

    compareFields(comm_field, parent_field, split_comm());

    /*{
        //Mesh mesh = StructuredMeshGenerator().generate(grid, comm_dist);
        output::Gmsh gmsh(grid_name + "_group_" + std::to_string(colour_management::get_colour()) + "_grid.msh");
        //gmsh.write(mesh);
        gmsh.write(comm_field, comm_fspace);
    }*/

    // TODO: Go back again -> back to parent fspace.
    //       Situation of two functions -> one to take you to sub comm, one that takes you back again.
    //       Ensure original data is the same. Values the same, lon lats same order. Could take SHA hashes.
    //
    //       Level of tests - build up.
    

}


//-----------------------------------------------------------------------------

}  // namespace test
}  // namespace atlas

int main(int argc, char** argv) {
    return atlas::test::run<atlas::test::AtlasSplitCommEnvironment>(argc, argv);
}
