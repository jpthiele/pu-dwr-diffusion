/**
 * @file Grid_DWR.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
/*                                                                            */
/*  This file is part of pu-dwr-diffusion                                     */
/*                                                                            */
/*  pu-dwr-diffusion is free software: you can redistribute it and/or modify  */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  pu-dwr-diffusion is distributed in the hope that it will be useful,       */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with pu-dwr-diffusion. If not, see <http://www.gnu.org/licenses/>.  */


// PROJECT includes
#include <diffusion/grid/Grid_DWR.tpl.hh>
#include <diffusion/grid/TriaGenerator.tpl.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_q.h>

// C++ includes
#include <cmath>
#include <limits>

namespace diffusion {

template<int dim, int spacedim>
Grid_DWR<dim,spacedim>::
~Grid_DWR() {
    // clear all dof handlers
    auto slab(slabs.begin());
    auto ends(slabs.end());

    for (; slab != ends; ++slab) {
        Assert(slab->low->dof.use_count(), dealii::ExcNotInitialized());
        Assert(slab->high->dof.use_count(), dealii::ExcNotInitialized());

        slab->low->dof->clear();
        slab->high->dof->clear();
    }
}


template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
initialize_slabs(
    const unsigned int &p_primal,
    const unsigned int &q_dual,
    const double &t0,
    const double &T,
    const double &tau_n) {

    Assert(
        slabs.empty(),
        dealii::ExcMessage(
            "Internal Error: slabs must be empty when calling this function"
        )
    );

    // determine initial time intervals
    unsigned int numoftimeintervals;
    numoftimeintervals = static_cast<unsigned int>(std::floor(
        (T-t0)/tau_n
    ));
    if (std::abs((numoftimeintervals*tau_n)-(T-t0))
            >= std::numeric_limits< double >::epsilon()*T) {
        numoftimeintervals += 1;
    }

    // init spatial "grids" of each slab
    for (unsigned int i{1}; i<= numoftimeintervals; ++i) {
        slabs.emplace_back();
        auto &slab = slabs.back();
        ////////////////////
        // common components
        //
        slab.tria = std::make_shared< dealii::Triangulation<dim> >(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement
            )
        );

        /////////////////////////
        // low order grid components
        //
        slab.low->dof = std::make_shared< dealii::DoFHandler<dim> > (
            *slab.tria
        );

        slab.low->fe = std::make_shared< dealii::FE_Q<dim> > (
            p_primal
        );

        slab.low->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

        slab.low->mapping = std::make_shared< dealii::MappingQ<dim> > (
            p_primal
        );

        ///////////////////////
        // high order grid components
        //
        slab.high->dof = std::make_shared< dealii::DoFHandler<dim> > (
            *slab.tria
        );

        slab.high->fe = std::make_shared< dealii::FE_Q<dim> > (
            q_dual
        );

        slab.high->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

        slab.high->mapping = std::make_shared< dealii::MappingQ<dim> > (
            q_dual
        );

        /////////////////////////
        // pu grid components
        //
        slab.pu->dof = std::make_shared< dealii::DoFHandler<dim> > (
            *slab.tria
        );

        slab.pu->fe = std::make_shared< dealii::FE_Q<dim> > (
            1
        );

        slab.pu->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

        slab.pu->mapping = std::make_shared< dealii::MappingQ<dim> > (
            1
        );
    }

    // init temporal "grids" of each slab
    {
        unsigned int n{1};
        for (auto &slab : slabs) {
            slab.t_m = (n-1)*tau_n+t0;
            slab.t_n = n*tau_n + t0;
            ++n;

            slab.refine_in_time=false;
        }

        auto &last_slab = slabs.back();
        if ( std::abs(last_slab.t_n - T) >=
            std::numeric_limits< double >::epsilon()*T) {
            last_slab.t_n = T;
        }
    }
}


template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
refine_slab_in_time(
    typename DTM::types::spacetime::dwr::slabs<dim>::iterator slab)
{
#ifdef DEBUG
    // check if iterator slab is in the container slabs of this object
    {
        auto _slab{slabs.begin()};
        auto _ends{slabs.end()};
        bool check{false};
        for ( ; _slab != _ends; ++_slab ) {
            if (slab == _slab) {
                check=true;
                break;
            }
        }
        Assert(
            check,
            dealii::ExcMessage("your given iterator slab to be refined could not be found in this->slabs object")
        );
    }
#endif

    // emplace a new slab element in front of the iterator
    slabs.emplace(
        slab
    );
	
    // init new slab ("space-time" tria)
    std::prev(slab)->t_m=slab->t_m;
    std::prev(slab)->t_n=slab->t_m + slab->tau_n()/2.;
    slab->t_m=std::prev(slab)->t_n;

    std::prev(slab)->refine_in_time=false;

    std::prev(slab)->tria = std::make_shared< dealii::Triangulation<dim> > (
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement
        )
    );
    std::prev(slab)->tria->copy_triangulation(*slab->tria);

    // init low order grid components of new slab
    std::prev(slab)->low->dof = std::make_shared< dealii::DoFHandler<dim> > (
        *std::prev(slab)->tria
    );

    std::prev(slab)->low->fe = std::make_shared< dealii::FE_Q<dim> > (
        slab->low->fe->degree
    );

    std::prev(slab)->low->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

    std::prev(slab)->low->mapping = std::make_shared< dealii::MappingQ<dim> > (
        std::prev(slab)->low->fe->degree
    );

    // init high order grid components of new slab
    std::prev(slab)->high->dof = std::make_shared< dealii::DoFHandler<dim> > (
        *std::prev(slab)->tria
    );

    std::prev(slab)->high->fe = std::make_shared< dealii::FE_Q<dim> > (
        slab->high->fe->degree
    );

    std::prev(slab)->high->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

    std::prev(slab)->high->mapping = std::make_shared< dealii::MappingQ<dim> > (
        std::prev(slab)->high->fe->degree
    );

    // init pu grid components of new slab
    std::prev(slab)->pu->dof = std::make_shared< dealii::DoFHandler<dim> > (
        *std::prev(slab)->tria
    );

    std::prev(slab)->pu->fe = std::make_shared< dealii::FE_Q<dim> > (
        1
    );

    std::prev(slab)->pu->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

    std::prev(slab)->pu->mapping = std::make_shared< dealii::MappingQ<dim> > (
        1
    );
}


/// Generate tria on each slab.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
generate() {
    auto slab(this->slabs.begin());
    auto ends(this->slabs.end());

    for (; slab != ends; ++slab) {
        diffusion::TriaGenerator<dim> tria_generator;
        tria_generator.generate(
            TriaGenerator,
            TriaGenerator_Options,
            slab->tria
        );
    }
}


/// Global refinement.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
refine_global(const unsigned int n) {
    auto slab(slabs.begin());
    auto ends(slabs.end());

    for (; slab != ends; ++slab) {
        slab->tria->refine_global(n);
    }
}


/// Set boundary indicators
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
set_boundary_indicators() {
	// base class does not implement this function
	Assert(false, dealii::ExcNotImplemented());
}


template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
distribute() {
    // Distribute the degrees of freedom (dofs)
    DTM::pout
            << "grid: distribute the degrees of freedom (dofs) on slabs"
            << std::endl;
    unsigned int total_cells = 0;
    unsigned int total_dofs_low = 0;
    unsigned int total_dofs_high = 0;
    unsigned int total_dofs_pu = 0;

    auto slab(slabs.begin());
    auto ends(slabs.end());

    unsigned int n{1};
    for (; slab != ends; ++slab) {
        ////////////////////////////////////////////////////////////////////////////
        // distribute low order dofs, create constraints and sparsity pattern sp
        {
            Assert(slab->low->dof.use_count(), dealii::ExcNotInitialized());
            Assert(slab->low->fe.use_count(), dealii::ExcNotInitialized());
            slab->low->dof->distribute_dofs(*(slab->low->fe));

            DTM::pout
                << "grid: low order mesh n_dofs on I_" << n
                << " = " << slab->low->dof->n_dofs()
                << " , on global active cells #K = "
                << slab->tria->n_global_active_cells()
                << " , having tau_" << n << " = " << slab->tau_n()
                << std::endl;

            total_dofs_low += slab->low->dof->n_dofs();
            total_cells += slab->tria->n_global_active_cells();
            // setup constraints (e.g. hanging nodes)
            Assert(slab->low->constraints.use_count(), dealii::ExcNotInitialized());
            slab->low->constraints->clear();
            slab->low->constraints->reinit();

            Assert(slab->low->dof.use_count(), dealii::ExcNotInitialized());
            dealii::DoFTools::make_hanging_node_constraints(
                *slab->low->dof,
                *slab->low->constraints
            );

            slab->low->constraints->close();

        }
		
        ////////////////////////////////////////////////////////////////////////////
        // distribute high order dofs, create constraints and sparsity pattern sp
        {
            Assert(slab->high->dof.use_count(), dealii::ExcNotInitialized());
            Assert(slab->high->fe.use_count(), dealii::ExcNotInitialized());
            slab->high->dof->distribute_dofs(*(slab->high->fe));

            DTM::pout
                << "grid: high order mesh   n_dofs on I_" << n
                << " = " << slab->high->dof->n_dofs()
                << std::endl;

            total_dofs_high += slab->high->dof->n_dofs();

            // setup constraints (e.g. hanging nodes)
            Assert(slab->high->constraints.use_count(), dealii::ExcNotInitialized());
            slab->high->constraints->clear();
            slab->high->constraints->reinit();

            Assert(slab->high->dof.use_count(), dealii::ExcNotInitialized());
            dealii::DoFTools::make_hanging_node_constraints(
                *slab->high->dof,
                *slab->high->constraints
            );

            slab->high->constraints->close();
        }

        ////////////////////////////////////////////////////////////////////////////
        // distribute low order dofs, create constraints and sparsity pattern sp
        {
            Assert(slab->pu->dof.use_count(), dealii::ExcNotInitialized());
            Assert(slab->pu->fe.use_count(), dealii::ExcNotInitialized());
            slab->pu->dof->distribute_dofs(*(slab->pu->fe));

            DTM::pout
                << "grid: PU mesh n_dofs on I_" << n
                << " = " << slab->pu->dof->n_dofs()
                << std::endl;

            total_dofs_pu += slab->pu->dof->n_dofs();

            // setup constraints (e.g. hanging nodes)
            Assert(slab->pu->constraints.use_count(), dealii::ExcNotInitialized());
            slab->pu->constraints->clear();
            slab->pu->constraints->reinit();

            Assert(slab->pu->dof.use_count(), dealii::ExcNotInitialized());
            dealii::DoFTools::make_hanging_node_constraints(
                *slab->pu->dof,
                *slab->pu->constraints
            );

            slab->pu->constraints->close();

        }

        ++n;
    } // end for-loop slab
    DTM::pout << "total number of space time cells/DoFs:"
              << "\t cells = " << total_cells
              << "\t low = " << total_dofs_low
              << "\t high = " << total_dofs_high
              << "\t pu = " << total_dofs_pu
              << std::endl << std::endl;
}

} // namespaces

#include "Grid_DWR.inst.in"
