/**
 * @file L2_Je_global_Mean_Assembly.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @brief Purpose: Assemble J(e)(v) = (v,e)_Omega = (v, u_E(t) - I(u_kh(t)))_Omega
 * NOTE: I(u_kh(t)) is the interpolated primal function on the dual solution space
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

#ifndef __L2_Je_global_Mean_Assembly_tpl_hh
#define __L2_Je_global_Mean_Assembly_tpl_hh

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <iterator>
#include <functional>
#include <memory>
#include <vector>

namespace diffusion {
namespace Assemble {
namespace L2 {
namespace Je_global_Mean {

namespace Assembly {
namespace Scratch {

template<int dim>
struct Je_global_MeanAssembly {
    Je_global_MeanAssembly(
        const dealii::FiniteElement<dim> &fe,
        const dealii::Mapping<dim> &mapping,
        const dealii::Quadrature<dim> &quad,
        const dealii::UpdateFlags &uflags
    );

    Je_global_MeanAssembly(const Je_global_MeanAssembly &scratch);

    dealii::FEValues<dim> fe_values;
    std::vector<double>   phi;
    double                JxW;

    // other
    unsigned int q;
    unsigned int k;
    unsigned int i;
};

} // namespace Scratch
namespace CopyData {

template<int dim>
struct Je_global_MeanAssembly {
    Je_global_MeanAssembly(const dealii::FiniteElement<dim> &fe);
    Je_global_MeanAssembly(const Je_global_MeanAssembly &copydata);

    dealii::Vector<double> vi_Jei_vector;
    std::vector<unsigned int> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

/**
 * The goal functional assembly is given by 
 * \f[
 * \boldsymbol J^{n,\iota} = ( j_{i} )_{i}\,,\quad
 * j_{i} = \displaystyle \sum_{K \in \mathcal{T}_h}
 * \displaystyle \sum_{i=1}^{N_u}
 * \displaystyle \int_K
 * \varphi^{i}(\boldsymbol x)\, j(\boldsymbol x, t_{n,\iota})
 * \,\text{d} \boldsymbol x\,, \quad
 * \text{with } 1 \leq i \leq N_u\,,
 * \f]
 * where  \f$ N_u \f$ denotes the degrees of freedom in space for a single 
 * temporal degree of freedem of the fully discrete solution 
 * \f$ u_{\tau, h}^{\text{dG}} \f$. Here, 
 * \f$ j=\frac{u-u_{\tau h}^{\text{dG}}}{\|u-u_{\tau h}^{\text{dG}}\|_{\mathcal{Q}_c}}  \f$
 * aims to control the space-time \f$ L^2 \f$-error on a contol volume
 * \f$ \mathcal{Q}_c:=\Omega_c \times I_c \f$, with \f$ \Omega_c \subseteq \Omega \f$
 * and \f$ I_c \subseteq I \f$.
 */
template<int dim>
class Assembler {
public:
    Assembler(
        std::shared_ptr< dealii::DoFHandler<dim> > dof,
        std::shared_ptr< dealii::FiniteElement<dim> > fe,
        std::shared_ptr< dealii::Mapping<dim> > mapping,
        std::shared_ptr< dealii::AffineConstraints<double> > constraints
    );

    ~Assembler() = default;

    /** Assemble vector.
     *  If @param n_quadrature_points = 0 is given,
     *  the dynamic default fe.tensor_degree()+1 will be used.
     */
    void assemble(
        std::shared_ptr< dealii::Vector<double> > Je,
        const double time,
        const unsigned int n_quadrature_points = 0,
        const bool quadrature_points_auto_mode = true
    );
	
protected:
	void local_assemble_cell(
            const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::Je_global_MeanAssembly<dim> &scratch,
            Assembly::CopyData::Je_global_MeanAssembly<dim> &copydata
	);

	void copy_local_to_global_cell(
            const Assembly::CopyData::Je_global_MeanAssembly<dim> &copydata
	);

private:
	std::shared_ptr< dealii::DoFHandler<dim> > dof;
	std::shared_ptr< dealii::FiniteElement<dim> > fe;
	std::shared_ptr< dealii::Mapping<dim> > mapping;
	std::shared_ptr< dealii::AffineConstraints<double> > constraints;
	dealii::UpdateFlags uflags;
	
	std::shared_ptr< dealii::Vector<double> > Je;
	
};

}}}} // namespaces

#endif
