/**
 * @file GeneralPUDoFErrorEstimator.tpl.hh
 *
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
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

#ifndef __GeneralPUDoFErrorEstimator_tpl_hh
#define __GeneralPUDoFErrorEstimator_tpl_hh

// PROJECT includes
#include <diffusion/grid/Grid_DWR.tpl.hh>
#include <diffusion/parameters/ParameterSet.hh>
#include <diffusion/ErrorEstimator/EstimatorArguments.tpl.hh>
// DTM++ includes
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <vector>

namespace diffusion {
namespace dwr {

namespace estimator {

namespace Assembly {

namespace Scratch {

/// Struct for scratch on local error estimate on cell
template<int dim>
struct PUDoFErrorEstimateOnCell {
    PUDoFErrorEstimateOnCell(
        const dealii::DoFHandler<dim>    &dof_dual,
        const dealii::DoFHandler<dim>    &dof_pu,
        const dealii::FiniteElement<dim> &fe_dual,
        const dealii::FiniteElement<dim> &fe_pu,
        const dealii::Mapping<dim> &mapping,
        const dealii::Quadrature<dim> &quad_space,
        const dealii::UpdateFlags &uflags
    );
	
    PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &scratch);

    // data structures of current cell
    dealii::FEValues<dim>               fe_values_dual;
    dealii::FEValues<dim>               fe_values_pu;


    const dealii::DoFHandler<dim> &dof_dual;
    const dealii::DoFHandler<dim> &dof_pu;

    std::vector< dealii::types::global_dof_index > local_dof_indices_dual;

    // shape fun scratch:
    std::vector<double>                 phi;
    std::vector<dealii::Tensor<1,dim> > grad_phi;

    //partition of unity shape functions
    std::vector<double>                 chi;
    std::vector<dealii::Tensor<1,dim> > grad_chi;

    // local dof scratch:
    std::vector<double> 		local_u_kh_m;
    std::vector<double>  	        local_u_kh_n;

    std::vector<double>                 local_u_k_n;
    std::vector<double>                 local_u_k_np1;

    std::vector<double>                 local_z_kh_n;
    std::vector<double>                 local_z_kh_np1;

    std::vector<double>                 local_z_k_n;
    std::vector<double>                 local_z_k_m;

    // function eval scratch:
    double value_epsilon;

    double val_u_j;

    double val_dw_j; //dual weight value
    double val_u_kh_jump_j;

    double val_pw_j;
    double val_z_kh_jump_j;

    dealii::Tensor<1,dim> grad_dw_j; //dual weight gradient
    dealii::Tensor<1,dim> grad_pw_j; //dual weight gradient

    dealii::Tensor<1,dim> grad_u_kh_j;
    dealii::Tensor<1,dim> grad_z_kh_j;

    double f_quad_k; //quadrature part of f
    double f_quad_k_minus; //quadrature part of f
    double f_quad_h; //quadrature part of f
    double f_quad_h_minus; //quadrature part of f

    double u_diff_tm;
    double u_diff_t0;
    double u_diff_tn;
    // other:
    double JxW;
    double tq;

    unsigned int q_x;
    unsigned int q_t;
    unsigned int d;
    unsigned int j;
};

} // namespace Scratch

namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct PUDoFErrorEstimateOnCell{
    PUDoFErrorEstimateOnCell(const dealii::FiniteElement<dim> &ge);
    PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &copydata);

    dealii::Vector<double> local_eta_h_minus_vector;
    dealii::Vector<double> local_eta_h_vector;
    dealii::Vector<double> local_eta_k_minus_vector;
    dealii::Vector<double> local_eta_k_vector;
    std::vector< dealii::types::global_dof_index > local_dof_indices_pu;
};


} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

/**
 * Implements the computation of the node-wise a posteriori error estimator \f$ \eta_i \f$
 * with a partition of unity localization.
 * This is achieved by multiplying the dual weights \f$ \bar{z} = z_{kh}-i_{kh}z_{kh} \f$ with 
 * a node wise partition of unity \f$ \chi_i\in V_{PU}, i=1,\dots,N_{PU} \f$. 
 * The simplest choice is \f$ V_{PU}= Q_1(\Omega\times (0,T)) \f$.
 * Given the variational formulation \f$ A(u,\phi )= f(\phi ) \f$ our primal error
 * estimator reads
 * \f[
 * \eta_i = F(\bar{z}\chi_i) - A(u_{kh},\bar{z}\chi_i) 
 * \f]
 * From this we obtain cell-wise estimators \f$ \eta_K \f$ by 
 * \f[
 * \eta_K = \sum\limits_{i\in K} \eta_i
 * \f]
 * Since we operate on tensor product cells we can split the partition of
 * unity into temporal and spatial components \f$ \chi(x,t) = \tau(t)\xi(x) \f$. We will denote our underlying
 * spatial cell grid as \f$ K_x \f$ and our temporal interval by \f$ I_n \f$.
 * Then, we can expand our cell wise error as
 * \f[
 * \eta_K = \sum\limits_{j\in K_x}\sum\limits_{k\in I_m} 
 *             F(\bar{z}\xi_j\tau_k) - A(u_{kh},\bar{z}\xi_j\tau_j)
 * \f]
 * Adding suitable quadrature formulas in space and time we obtain four sums.
 * These can be reordered so that the two innermost sums are the spatial components, 
 * which can be seen as functions of the temporal quadrature point \f$ q_t \f$.
 * That way our space-time error contributions can be seen as a sum over spatial PU
 * estimators weighted by our temporal PU. 
 * Note: In contrast to the classical estimator we work in a variational setting.
 * Since we don't do partial integration to obtain a strong form of the estimator
 * we do not get any of the explicit face terms in our estimator.
 */
template<int dim>
class PUDoFErrorEstimator {
public:
    PUDoFErrorEstimator() = default;
    virtual ~PUDoFErrorEstimator() = default;
    virtual void set_functions (
        std::shared_ptr< dealii::Function<dim> > density,
        std::shared_ptr< dealii::Function<dim> > epsilon,
        std::shared_ptr< dealii::Function<dim> > f,
        std::shared_ptr< dealii::Function<dim> > u_0,
        std::shared_ptr< dealii::Function<dim> > u_ex
    );

    virtual void set_parameters(
        double L2_error,
        std::string goal_type
    );


    virtual void estimate_primal_split(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr< Arguments> args,
        std::shared_ptr< dealii::Vector<double> > _eta_h,
        std::shared_ptr< dealii::Vector<double> > _eta_k
    );

    virtual void estimate_primal_split_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr< Arguments> args,
        std::shared_ptr< dealii::Vector<double> > _eta_h_minus, //eta_h for m-1 cG1 dof
        std::shared_ptr< dealii::Vector<double> > _eta_h,
        std::shared_ptr< dealii::Vector<double> > _eta_k_minus, //eta_k for m-1 cG1 dof
        std::shared_ptr< dealii::Vector<double> > _eta_k
    );
    virtual void estimate_dual_split(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr< Arguments> args,
        std::shared_ptr< dealii::Vector<double> > _eta_h,
        std::shared_ptr< dealii::Vector<double> > _eta_k
    );

    virtual void estimate_primal_joint(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr< Arguments> args,
        std::shared_ptr< dealii::Vector<double> > _eta_h,
        std::shared_ptr< dealii::Vector<double> > _eta_k
    );

protected:

    ////////////////////////////////////////////////////////////////////////////
    // assemble local functions:
    //

    virtual void assemble_split_primal_error_on_cell(
        const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void assemble_split_primal_cG1_error_on_cell(
        const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void assemble_split_adjoint_error_on_cell(
        const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void assemble_joint_primal_error_on_cell(
        const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void copy_local_error(
        const Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void copy_local_cG1_error(
        const Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    ////////////////////////////////////////////////////////////////////////////
    // internal data structures:
    //

    struct {
        std::shared_ptr< Arguments> args;
    } primal;

    struct {
        struct {
            std::shared_ptr< DTM::types::storage_data_vectors<1> > z;
        } storage;
    } dual;

    struct {
        std::shared_ptr<dealii::AffineConstraints<double>> constraints;
    }pu;

    struct {
        std::shared_ptr<dealii::Vector<double>> x_h_minus;
        std::shared_ptr<dealii::Vector<double>> x_h;
        std::shared_ptr<dealii::Vector<double>> x_k_minus;
        std::shared_ptr<dealii::Vector<double>> x_k;
    } error_estimator;

    struct {
        std::shared_ptr< dealii::Function<dim> > density;
        std::shared_ptr< dealii::Function<dim> > epsilon;
        std::shared_ptr< dealii::Function<dim> > f;
        std::shared_ptr< dealii::Function<dim> > u_0;
        std::shared_ptr< dealii::Function<dim> > u_ex;
    } function;

    // parameter set
    std::shared_ptr< diffusion::dwr::ParameterSet > parameter_set;

    std::string goal_type;
    double L2_error;

    double tau_n;
    double tm;
    double t1;
    double t0;
    double t2;
    double tn;


};

}}} // namespace

#endif
