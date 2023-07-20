/**
 * @file L2_Je_global_Mean_Assembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser                      */
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
#include <diffusion/assembler/L2_Je_global_Mean_Assembly.tpl.hh>

namespace diffusion {
namespace Assemble {
namespace L2 {
namespace Je_global_Mean {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
Je_global_MeanAssembly<dim>::Je_global_MeanAssembly(
    const dealii::FiniteElement<dim> &fe,
    const dealii::Mapping<dim> &mapping,
    const dealii::Quadrature<dim> &quad,
    const dealii::UpdateFlags &uflags) :
    fe_values(mapping, fe, quad, uflags),
    phi(fe.dofs_per_cell),
    JxW(0),
    q(0),
    k(0),
    i(0){
}


/// (Struct-) Copy constructor.
template<int dim>
Je_global_MeanAssembly<dim>::Je_global_MeanAssembly(
    const Je_global_MeanAssembly &scratch) :
    fe_values(
        scratch.fe_values.get_mapping(),
        scratch.fe_values.get_fe(),
        scratch.fe_values.get_quadrature(),
        scratch.fe_values.get_update_flags()),
    phi(scratch.phi),
    JxW(scratch.JxW),
    q(scratch.q),
    k(scratch.k),
    i(scratch.i) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
Je_global_MeanAssembly<dim>::Je_global_MeanAssembly(
    const dealii::FiniteElement<dim> &fe) :
    vi_Jei_vector(fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
Je_global_MeanAssembly<dim>::Je_global_MeanAssembly(
    const Je_global_MeanAssembly& copydata) :
    vi_Jei_vector(copydata.vi_Jei_vector),
    local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::Mapping<dim> > mapping,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    dof(dof),
    fe(fe),
    mapping(mapping),
    constraints(constraints) {
    // init update flags
    uflags =
        dealii::update_quadrature_points |
        dealii::update_values |
        dealii::update_JxW_values;
}


template<int dim>
void Assembler<dim>::assemble(
    std::shared_ptr< dealii::Vector<double> > _Je,
    const double ,
    const unsigned int q,
    const bool quadrature_points_auto_mode)
{
    // init
    Je = _Je;
    Assert( Je.use_count(), dealii::ExcNotInitialized() );
    *Je = 0.;

    const dealii::QGauss<dim> quad{
        (quadrature_points_auto_mode ? (fe->tensor_degree()+3) : q)
    };

    if (!quad.size()) {
        return;
    }

    typedef
        dealii::FilteredIterator<const typename dealii::DoFHandler<dim>::active_cell_iterator>
        CellFilter;

    // Using WorkStream to assemble.
    dealii::WorkStream::
    run(
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), dof->begin_active()
        ),
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), dof->end()
        ),
        std::bind (
            &Assembler<dim>::local_assemble_cell,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &Assembler<dim>::copy_local_to_global_cell,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::Je_global_MeanAssembly<dim> (*fe, *mapping, quad, uflags),
        Assembly::CopyData::Je_global_MeanAssembly<dim> (*fe)
    );
}


template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::Je_global_MeanAssembly<dim> &scratch,
    Assembly::CopyData::Je_global_MeanAssembly<dim> &copydata)
{
    // reinit fe_values and init local to global dof mapping
    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(copydata.local_dof_indices);

    // initialize local vector
    copydata.vi_Jei_vector = 0;

    // assemble cell terms
    for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points;
         ++scratch.q) {
        scratch.JxW = scratch.fe_values.JxW(scratch.q);

        // prefetch data
        for (scratch.k=0; scratch.k < scratch.fe_values.get_fe().dofs_per_cell;
             ++scratch.k) {
            scratch.phi[scratch.k] =
                scratch.fe_values.shape_value(
                    scratch.k,
                    scratch.q
                );
        }

        // assemble
        for (scratch.i=0; scratch.i < scratch.fe_values.get_fe().dofs_per_cell;
             ++scratch.i) {
            copydata.vi_Jei_vector[scratch.i] += (
                scratch.phi[scratch.i] *
                scratch.JxW
            );
        } // for i
    } // for q
}


template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::Je_global_MeanAssembly<dim> &copydata)
{
    constraints->distribute_local_to_global(
        copydata.vi_Jei_vector, copydata.local_dof_indices, *Je
    );
}


}}}}

#include "L2_Je_global_Mean_Assembly.inst.in"
