/**
 * @file slabs.tpl.hh
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


#ifndef __slabs_hh
#define __slabs_hh

// DEAL.II includes
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

namespace DTM {
namespace types {
namespace spacetime {
namespace dwr {

template <int dim>
/// FEInfo: collects all information to
struct FEInfo {
    std::shared_ptr< dealii::DoFHandler<dim> > dof;
    std::shared_ptr< dealii::FiniteElement<dim> > fe;
    std::shared_ptr< dealii::Mapping<dim> >  mapping;
    std::shared_ptr< dealii::AffineConstraints<double> > constraints;
};

/// slab: collects data structures and functions of a space-time slab for dwr
template <int dim>
struct s_slab {
    /// deal.II Triangulation<dim> for \f$ \Omega_h \f$ on \f$ I_n \f$.
    std::shared_ptr< dealii::Triangulation<dim> > tria;

    double t_m; ///< left endpoint of \f$ I_n=(t_m, t_n) \f$
    double t_n; ///< right endpoint of \f$ I_n=(t_m, t_n) \f$

    // additional data for slab
    std::shared_ptr < FEInfo<dim>> low  = std::make_shared<FEInfo<dim>>();
    std::shared_ptr < FEInfo<dim>> high = std::make_shared<FEInfo<dim>>();
    std::shared_ptr < FEInfo<dim>> pu = std::make_shared<FEInfo<dim>>();

    struct  {
        std::shared_ptr < FEInfo<dim>> fe_info;
        std::shared_ptr< dealii::SparsityPattern > sp;
    } primal;


    struct  {
        std::shared_ptr < FEInfo<dim>> fe_info;
        std::shared_ptr< dealii::SparsityPattern > sp;
    } dual;

    bool refine_in_time; // flag for marking: refinement in time of this slab
    bool coarsen_in_time; // flag for marking: coarsen in time of this slab
	
    // additional member functions

    /// get \f$ \tau_n = t_n - t_m \f$ of this slab
    double tau_n() const { return (t_n-t_m); };

    void set_refine_in_time_flag() { refine_in_time=true; };
    void clear_refine_in_time_flag() { refine_in_time=false; };

    void set_coarsen_in_time_flag() { coarsen_in_time=true; };
    void clear_coarsen_in_time_flag() { coarsen_in_time=false; };
};

template <int dim>
using slab = struct s_slab<dim>;

template <int dim>
using slabs = std::list< slab<dim> >;

}}}}

#endif
