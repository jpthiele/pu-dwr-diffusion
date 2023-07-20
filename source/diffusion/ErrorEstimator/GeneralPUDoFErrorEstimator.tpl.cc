/**
 * @file GeneralPUDoFErrorEstimator.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
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
#include <diffusion/ErrorEstimator/GeneralPUDoFErrorEstimator.tpl.hh>
#include <diffusion/ErrorEstimator/EstimatorArguments.tpl.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// C++ includes

namespace diffusion {
namespace dwr {

namespace estimator {

namespace Assembly {

namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(
    const dealii::DoFHandler<dim> &dof_dual,
    const dealii::DoFHandler<dim> &dof_pu,
    const dealii::FiniteElement<dim> &fe_dual,
    const dealii::FiniteElement<dim> &fe_pu,
    const dealii::Mapping<dim> &mapping,
    const dealii::Quadrature<dim> &quad_space,
    const dealii::UpdateFlags &uflags) :
    fe_values_dual(mapping, fe_dual, quad_space, uflags),
    fe_values_pu(mapping,fe_pu,quad_space,uflags),
    dof_dual(dof_dual),
    dof_pu(dof_pu),
    local_dof_indices_dual(fe_dual.dofs_per_cell),
    phi(fe_dual.dofs_per_cell),
    grad_phi(fe_dual.dofs_per_cell),
    chi(fe_pu.dofs_per_cell),
    grad_chi(fe_pu.dofs_per_cell),
    local_u_kh_m(fe_dual.dofs_per_cell),
    local_u_kh_n(fe_dual.dofs_per_cell),
    local_u_k_n(fe_dual.dofs_per_cell),
    local_u_k_np1(fe_dual.dofs_per_cell),
    local_z_kh_n(fe_dual.dofs_per_cell),
    local_z_kh_np1(fe_dual.dofs_per_cell),
    local_z_k_n(fe_dual.dofs_per_cell),
    local_z_k_m(fe_dual.dofs_per_cell),
    value_epsilon(0),
    val_u_j(0),
    val_dw_j(0),
    val_u_kh_jump_j(0),
    val_pw_j(0),
    val_z_kh_jump_j(0),
    f_quad_k(0),
    f_quad_k_minus(0),
    f_quad_h(0),
    f_quad_h_minus(0),
    u_diff_tm(0),
    u_diff_t0(0),
    u_diff_tn(0),
    JxW(0),
    tq(0),
    q_x(0),
    q_t(0),
    d(0),
    j(0){
}


/// (Struct-) Copy constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &scratch) :
    fe_values_dual(
        scratch.fe_values_dual.get_mapping(),
        scratch.fe_values_dual.get_fe(),
        scratch.fe_values_dual.get_quadrature(),
        scratch.fe_values_dual.get_update_flags()),
    fe_values_pu(
        scratch.fe_values_pu.get_mapping(),
        scratch.fe_values_pu.get_fe(),
        scratch.fe_values_pu.get_quadrature(),
        scratch.fe_values_pu.get_update_flags()),
    dof_dual(scratch.dof_dual),
    dof_pu(scratch.dof_pu),
    local_dof_indices_dual(scratch.local_dof_indices_dual),
    phi(scratch.phi),
    grad_phi(scratch.grad_phi),
    chi(scratch.chi),
    grad_chi(scratch.grad_chi),
    local_u_kh_m(scratch.local_u_kh_m),
    local_u_kh_n(scratch.local_u_kh_n),
    local_u_k_n(scratch.local_u_k_n),
    local_u_k_np1(scratch.local_u_k_np1),
    local_z_kh_n(scratch.local_z_kh_n),
    local_z_kh_np1(scratch.local_z_kh_np1),
    local_z_k_n(scratch.local_z_k_n),
    local_z_k_m(scratch.local_z_k_m),
    value_epsilon(scratch.value_epsilon),
    val_u_j(scratch.val_u_j),
    val_dw_j(scratch.val_dw_j),
    val_u_kh_jump_j(scratch.val_u_kh_jump_j),
    val_pw_j(scratch.val_pw_j),
    val_z_kh_jump_j(scratch.val_z_kh_jump_j),
    grad_dw_j(scratch.grad_dw_j),
    grad_pw_j(scratch.grad_pw_j),
    grad_u_kh_j(scratch.grad_u_kh_j),
    grad_z_kh_j(scratch.grad_z_kh_j),
    f_quad_k(scratch.f_quad_k),
    f_quad_k_minus(scratch.f_quad_k_minus),
    f_quad_h(scratch.f_quad_h),
    f_quad_h_minus(scratch.f_quad_h_minus),
    u_diff_tm(scratch.u_diff_tm),
    u_diff_t0(scratch.u_diff_t0),
    u_diff_tn(scratch.u_diff_tn),
    JxW(scratch.JxW),
    tq(scratch.tq),
    q_x(scratch.q_x),
    q_t(scratch.q_t),
    d(scratch.d),
    j(scratch.j) {
}


}

namespace CopyData {
/// (Struct-) Constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(
    const dealii::FiniteElement<dim> &fe):
    local_eta_h_minus_vector(fe.dofs_per_cell),
    local_eta_h_vector(fe.dofs_per_cell),
    local_eta_k_minus_vector(fe.dofs_per_cell),
    local_eta_k_vector(fe.dofs_per_cell),
    local_dof_indices_pu(fe.dofs_per_cell){
}

/// (Struct-) Copy constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &copydata) :
    local_eta_h_minus_vector(copydata.local_eta_h_minus_vector),
    local_eta_h_vector(copydata.local_eta_h_vector),
    local_eta_k_minus_vector(copydata.local_eta_k_minus_vector),
    local_eta_k_vector(copydata.local_eta_k_vector),
    local_dof_indices_pu(copydata.local_dof_indices_pu){
}

} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
PUDoFErrorEstimator<dim>::
set_functions(
    std::shared_ptr< dealii::Function<dim> > _density,
    std::shared_ptr< dealii::Function<dim> > _epsilon,
    std::shared_ptr< dealii::Function<dim> > _f,
    std::shared_ptr< dealii::Function<dim> > _u_0,
    std::shared_ptr< dealii::Function<dim> > _u_ex
) {

    Assert(_density.use_count(), dealii::ExcNotInitialized());
    function.density = _density;

    Assert(_epsilon.use_count(), dealii::ExcNotInitialized());
    function.epsilon = _epsilon;

    Assert(_f.use_count(), dealii::ExcNotInitialized());
    function.f = _f;

    Assert(_u_0.use_count(), dealii::ExcNotInitialized());
    function.u_0 = _u_0;

    Assert(_u_ex.use_count(), dealii::ExcNotInitialized());
    function.u_ex = _u_ex;
}

template<int dim>
void
PUDoFErrorEstimator<dim>::
set_parameters(
    double _L2_error,
    std::string _goal_type
) {
    L2_error = _L2_error;
    goal_type = _goal_type;
}


template<int dim>
void
PUDoFErrorEstimator<dim>::
estimate_primal_split(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< Arguments> args,
    std::shared_ptr< dealii::Vector<double> > _eta_h,
    std::shared_ptr< dealii::Vector<double> > _eta_k
) {
	
	
    Assert(args.use_count(), dealii::ExcNotInitialized());
    primal.args = args;
    ////////////////////////////////////////////////////////////////////////////
    // do estimate errors over \Omega x slab loop
    //

    // local time variables
    tm = slab->t_m;
    t0 = tm + slab->tau_n()/2.;
    tn = slab->t_n;
                
    DTM::pout << "evaluating primal error on ( " << tm << ", " << tn << ")"
                    << std::endl;

    pu.constraints = slab->pu->constraints;
    // local tau_n (used in the local assembly functions internally)
    tau_n = slab->tau_n();

    error_estimator.x_h = _eta_h;
    error_estimator.x_k = _eta_k;


    // assemble slab problem
    dealii::QGauss<dim> quad_cell_space(slab->high->fe->tensor_degree()+2);

    dealii::WorkStream::run(
        slab->tria->begin_active(),
        slab->tria->end(),
        std::bind (
            &PUDoFErrorEstimator<dim>::assemble_split_primal_error_on_cell,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::copy_local_error,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> (
            *slab->high->dof,
            *slab->pu->dof,
            *slab->high->fe,
            *slab->pu->fe,
            *slab->high->mapping,
            quad_cell_space,
            //
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values),
            Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> (*slab->pu->fe)
    );

}

template<int dim>
void
PUDoFErrorEstimator<dim>::
estimate_primal_split_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< Arguments> args,
    std::shared_ptr< dealii::Vector<double> > _eta_h_minus,
    std::shared_ptr< dealii::Vector<double> > _eta_h,
    std::shared_ptr< dealii::Vector<double> > _eta_k_minus,
    std::shared_ptr< dealii::Vector<double> > _eta_k
) {


    Assert(args.use_count(), dealii::ExcNotInitialized());
    primal.args = args;
    ////////////////////////////////////////////////////////////////////////////
    // do estimate errors over \Omega x slab loop
    //

    // local time variables
    tm = slab->t_m;
    t1 = tm +    slab->tau_n()/4.;
    t0 = tm +    slab->tau_n()/2.;
    t2 = tm + 3.*slab->tau_n()/4.;
    tn = slab->t_n;

    DTM::pout << "evaluating primal error on ( " << tm << ", " << tn << ")"
                    << std::endl;

    pu.constraints = slab->pu->constraints;
    // local tau_n (used in the local assembly functions internally)
    tau_n = slab->tau_n();

    error_estimator.x_h_minus = _eta_h_minus;
    error_estimator.x_h = _eta_h;
    error_estimator.x_k_minus = _eta_k_minus;
    error_estimator.x_k = _eta_k;


    // assemble slab problem
    dealii::QGauss<dim> quad_cell_space(slab->high->fe->tensor_degree()+2);

    dealii::WorkStream::run(
        slab->tria->begin_active(),
        slab->tria->end(),
        std::bind (
            &PUDoFErrorEstimator<dim>::assemble_split_primal_cG1_error_on_cell,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::copy_local_cG1_error,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> (
            *slab->high->dof,
            *slab->pu->dof,
            *slab->high->fe,
            *slab->pu->fe,
            *slab->high->mapping,
            quad_cell_space,
            //
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values),
            Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> (*slab->pu->fe)
    );

}

template<int dim>
void
PUDoFErrorEstimator<dim>::
estimate_dual_split(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< Arguments> args,
    std::shared_ptr< dealii::Vector<double> > _eta_h,
    std::shared_ptr< dealii::Vector<double> > _eta_k
) {


    Assert(args.use_count(), dealii::ExcNotInitialized());
    primal.args = args;
    ////////////////////////////////////////////////////////////////////////////
    // do estimate errors over \Omega x slab loop
    //

    // local time variables
    tm = slab->t_m;
    t0 = tm + slab->tau_n()/2.;
    tn = slab->t_n;

    DTM::pout << "evaluating adjoint error on ( " << tm << ", " << tn << ")"
                    << std::endl;

    pu.constraints = slab->pu->constraints;
    // local tau_n (used in the local assembly functions internally)
    tau_n = slab->tau_n();

    error_estimator.x_h = _eta_h;
    error_estimator.x_k = _eta_k;


    // assemble slab problem
    dealii::QGauss<dim> quad_cell_space(slab->high->fe->tensor_degree()+2);

    dealii::WorkStream::run(
        slab->tria->begin_active(),
        slab->tria->end(),
        std::bind (
            &PUDoFErrorEstimator<dim>::assemble_split_adjoint_error_on_cell,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::copy_local_error,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> (
            *slab->high->dof,
            *slab->pu->dof,
            *slab->high->fe,
            *slab->pu->fe,
            *slab->high->mapping,
            quad_cell_space,
            //
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values),
            Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> (*slab->pu->fe)
    );

}

template<int dim>
void
PUDoFErrorEstimator<dim>::
estimate_primal_joint(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< Arguments> args,
    std::shared_ptr< dealii::Vector<double> > _eta_h,
    std::shared_ptr< dealii::Vector<double> > _eta_k
) {

    Assert(args.use_count(), dealii::ExcNotInitialized());
    primal.args = args;
    ////////////////////////////////////////////////////////////////////////////
    // do estimate errors over \Omega x slab loop
    //

    // local time variables
    tm = slab->t_m;
    t0 = tm + slab->tau_n()/2.;
    tn = slab->t_n;

    DTM::pout << "evaluating error on ( " << tm << ", " << tn << ")"
                    << std::endl;

    pu.constraints = slab->pu->constraints;
    // local tau_n (used in the local assembly functions internally)
    tau_n = slab->tau_n();

    error_estimator.x_h = _eta_h;
    error_estimator.x_k = _eta_k;


    // assemble slab problem
    dealii::QGauss<dim> quad_cell_space(slab->high->fe->tensor_degree()+2);

    dealii::WorkStream::run(
        slab->tria->begin_active(),
        slab->tria->end(),
        std::bind (
            &PUDoFErrorEstimator<dim>::assemble_joint_primal_error_on_cell,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::copy_local_error,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> (
            *slab->high->dof,
            *slab->pu->dof,
            *slab->high->fe,
            *slab->pu->fe,
            *slab->high->mapping,
            quad_cell_space,
            //
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values),
            Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> (*slab->pu->fe)
    );

    *error_estimator.x_h *= 0.5;
    *error_estimator.x_k = *error_estimator.x_h;


}
////////////////////////////////////////////////////////////////////////////////
//
//


template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_split_primal_error_on_cell(
    const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
    Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata)
{

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.dof_dual);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&tria_cell->get_triangulation(),
                                                                   tria_cell->level(),
                                                                   tria_cell->index(),
                                                                   &scratch.dof_pu);


    // reinit scratch and data to current cell
    scratch.fe_values_dual.reinit(cell_dual);
    scratch.fe_values_pu.reinit(cell_pu);
    // fetch local dof data
    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
        (*primal.args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
        (*primal.args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_n[scratch.j] =
        (*primal.args->tn.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_m[scratch.j] =
        (*primal.args->tm.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_m[scratch.j] =
        (*primal.args->tm.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);

    //assemble PU
    for (scratch.q_x = 0; scratch.q_x < scratch.fe_values_pu.n_quadrature_points; ++scratch.q_x)
    {
        scratch.JxW = scratch.fe_values_pu.JxW(scratch.q_x);

        //shape values for dual basis
        for (scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
             ++scratch.j)
        {
        scratch.phi[scratch.j] =
              scratch.fe_values_dual.shape_value_component(scratch.j,scratch.q_x,0);

        scratch.grad_phi[scratch.j] =
              scratch.fe_values_dual.shape_grad(scratch.j,scratch.q_x);
        }

        //shape values for spatial partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
                         ++ scratch.j )
        {
            scratch.chi[scratch.j] =
                  scratch.fe_values_pu.shape_value(scratch.j,scratch.q_x);

            scratch.grad_chi[scratch.j] =
                  scratch.fe_values_pu.shape_grad(scratch.j,scratch.q_x);
        }

        //get values of given functions
        scratch.value_epsilon = function.epsilon->value(scratch.fe_values_dual.quadrature_point(scratch.q_x),0);

        scratch.f_quad_k = 0;
        scratch.f_quad_h = 0;

        //Simpsons rule for quadrature of rhs
        //For temporal we put the factor of -(1-t) into f_quad
        {
            // tm
            function.f->set_time(tm);
            scratch.f_quad_k -= 1.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_h += 1.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            // t0
            function.f->set_time(t0);
            scratch.f_quad_k -= 4.0/6.0*0.5*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_h += 4.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            // tn
            function.f->set_time(tn);

            scratch.f_quad_h += 1.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);
        }


        //First argument is the same for both error parts
        scratch.grad_u_kh_j = 0.;
        scratch.val_u_kh_jump_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.grad_u_kh_j +=
                (scratch.local_u_kh_n[scratch.j] * scratch.grad_phi[scratch.j]);

            scratch.val_u_kh_jump_j +=
                (scratch.local_u_kh_n[scratch.j] - scratch.local_u_kh_m[scratch.j])
                *scratch.phi[scratch.j];
        }


        //calculating PU for temporal error over I_n
        scratch.val_dw_j = 0.;
        scratch.grad_dw_j = 0.;

        //for temporal error dual weight is
        // dw(t) = (t-1)*(z_k^n-z_k^m) for t in [0,1]
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.val_dw_j +=
                (scratch.local_z_k_n[scratch.j]-scratch.local_z_k_m[scratch.j])
                * scratch.phi[scratch.j];

            scratch.grad_dw_j +=
                (scratch.local_z_k_n[scratch.j]-scratch.local_z_k_m[scratch.j])
                * scratch.grad_phi[scratch.j];
        }



        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // (f, z-zk chi) with midpoint
            copydata.local_eta_k_vector[scratch.j]+=
                scratch.f_quad_k
                * scratch.val_dw_j * scratch.chi[scratch.j] * scratch.JxW *tau_n;

            // (eps * grad u, grad( z-zk chi) ) [A(u,z-zk chi)]
            copydata.local_eta_k_vector[scratch.j] -= (-0.5) *
                (scratch.value_epsilon * scratch.grad_u_kh_j)
                *( scratch.grad_dw_j*scratch.chi[scratch.j]
                  +scratch.val_dw_j*scratch.grad_chi[scratch.j])
                * scratch.JxW * tau_n ;

            // jump term at t_m
            copydata.local_eta_k_vector[scratch.j] -= (-1.0)*
                scratch.val_u_kh_jump_j * scratch.val_dw_j * scratch.chi[scratch.j]* scratch.JxW;
        }


        //calculating PU for spatial error over I_n
        scratch.val_dw_j = 0.;
        scratch.grad_dw_j = 0.;


        //for temporal error dual weight is
        // dw(t) = (z_k^n-z_kh^n) for t in [0,1]
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.val_dw_j +=
                (
                   scratch.local_z_k_n[scratch.j]
                   -scratch.local_z_kh_n[scratch.j]
                ) * scratch.phi[scratch.j];

            scratch.grad_dw_j +=
                (scratch.local_z_k_n[scratch.j]-scratch.local_z_kh_n[scratch.j])
                * scratch.grad_phi[scratch.j];
        }

        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // (f, z-zk chi) with midpoint
            copydata.local_eta_h_vector[scratch.j]+=
                scratch.f_quad_h *
                scratch.val_dw_j *
                scratch.chi[scratch.j]
                * scratch.JxW
                * tau_n;


            // (eps * grad u, grad( z-zk chi) ) [A(u,z-zk chi)]
            copydata.local_eta_h_vector[scratch.j] -=
                scratch.value_epsilon *
                    scratch.grad_u_kh_j
                *( scratch.grad_dw_j*
                   scratch.chi[scratch.j]
                   +
                   scratch.val_dw_j
                   * scratch.grad_chi[scratch.j]
                )
                * scratch.JxW * tau_n ;

            // jump term at t_m
            copydata.local_eta_h_vector[scratch.j] -=
                scratch.val_u_kh_jump_j *
                scratch.val_dw_j *
                scratch.chi[scratch.j]* scratch.JxW;
        }

    } // end quadrature iteration
}


template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_split_primal_cG1_error_on_cell(
    const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
    Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata)
{

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.dof_dual);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&tria_cell->get_triangulation(),
                                                                   tria_cell->level(),
                                                                   tria_cell->index(),
                                                                   &scratch.dof_pu);


    // reinit scratch and data to current cell
    scratch.fe_values_dual.reinit(cell_dual);
    scratch.fe_values_pu.reinit(cell_pu);
    // fetch local dof data
    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
            (*primal.args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
            (*primal.args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_n[scratch.j] =
            (*primal.args->tn.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_m[scratch.j] =
            (*primal.args->tm.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_m[scratch.j] =
            (*primal.args->tm.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    // initialize copydata
    copydata.local_eta_h_minus_vector = 0.;
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_minus_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);

    //assemble PU
    for (scratch.q_x = 0; scratch.q_x < scratch.fe_values_pu.n_quadrature_points; ++scratch.q_x)
    {
        scratch.JxW = scratch.fe_values_pu.JxW(scratch.q_x);

        //shape values for dual basis
        for (scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
             ++scratch.j)
        {
        scratch.phi[scratch.j] =
              scratch.fe_values_dual.shape_value_component(scratch.j,scratch.q_x,0);

        scratch.grad_phi[scratch.j] =
              scratch.fe_values_dual.shape_grad(scratch.j,scratch.q_x);
        }

        //shape values for spatial partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
              ++ scratch.j )
        {
            scratch.chi[scratch.j] =
                scratch.fe_values_pu.shape_value(scratch.j,scratch.q_x);

            scratch.grad_chi[scratch.j] =
                scratch.fe_values_pu.shape_grad(scratch.j,scratch.q_x);
        }

        //get values of given functions
        scratch.value_epsilon = function.epsilon->value(scratch.fe_values_dual.quadrature_point(scratch.q_x),0);

        scratch.f_quad_k = 0;
        scratch.f_quad_k_minus = 0;
        scratch.f_quad_h = 0;
        scratch.f_quad_h_minus = 0;

        //Simpsons rule for quadrature of spatial rhs
        //Milne rule for quadrature of temporal rhs
        //For temporal we put the factor of -(1-t)chi_k(t) into f_quad
        //chi_k^{m-1}(t) = 1-t
        //chi_k^m(t) = t
        {
            // tm aka t = 0
            function.f->set_time(tm);
            scratch.f_quad_k_minus -= 7.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_h_minus +=
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            // t1 aka t = 1/4
            function.f->set_time(t1);
            scratch.f_quad_k_minus -= 18.0 *
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_k -= 6.0 *
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            // t0 aka t = 1/2
            function.f->set_time(t0);
            scratch.f_quad_k_minus -= 3.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_k -= 3.0 *
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_h_minus += 2.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_h += 2.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            // t2 aka t = 3/4
            function.f->set_time(t1);
            scratch.f_quad_k_minus -= 2.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_k -= 6.0 *
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            // tn aka t = 1
            function.f->set_time(tn);

            scratch.f_quad_h +=
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);
        }
        scratch.f_quad_h/=6.0;
        scratch.f_quad_h_minus/=6.0;
        scratch.f_quad_k/=90.;
        scratch.f_quad_k_minus/=90.;

        //First argument is the same for both error parts
        scratch.grad_u_kh_j = 0.;
        scratch.val_u_kh_jump_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.grad_u_kh_j +=
                (scratch.local_u_kh_n[scratch.j] * scratch.grad_phi[scratch.j]);

            scratch.val_u_kh_jump_j +=
                (scratch.local_u_kh_n[scratch.j] - scratch.local_u_kh_m[scratch.j])
                *scratch.phi[scratch.j];
        }


        //calculating PU for temporal error over I_n
        scratch.val_dw_j = 0.;
        scratch.grad_dw_j = 0.;

        //for temporal error dual weight is
        // dw(t) = (t-1)*(z_k^n-z_k^m) for t in [0,1]
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
          scratch.val_dw_j +=
              (scratch.local_z_k_n[scratch.j]-scratch.local_z_k_m[scratch.j])
                  * scratch.phi[scratch.j];

          scratch.grad_dw_j +=
              (scratch.local_z_k_n[scratch.j]-scratch.local_z_k_m[scratch.j])
                  * scratch.grad_phi[scratch.j];
        }



        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // (f, z-zk chi) with midpoint
            copydata.local_eta_k_minus_vector[scratch.j]+=
                scratch.f_quad_k_minus
                  * scratch.val_dw_j * scratch.chi[scratch.j] * scratch.JxW *tau_n;

            copydata.local_eta_k_vector[scratch.j]+=
                scratch.f_quad_k
                  * scratch.val_dw_j * scratch.chi[scratch.j] * scratch.JxW *tau_n;

            // (eps * grad u, grad( z-zk chi) ) [A(u,z-zk chi)]
            copydata.local_eta_k_vector[scratch.j] -= (-2.0/6.0) *
                (scratch.value_epsilon * scratch.grad_u_kh_j)
                *( scratch.grad_dw_j*scratch.chi[scratch.j]
                  +scratch.val_dw_j*scratch.grad_chi[scratch.j])
                * scratch.JxW * tau_n ;


            copydata.local_eta_k_vector[scratch.j] -= (-1.0/6.0) *
                (scratch.value_epsilon * scratch.grad_u_kh_j)
                *( scratch.grad_dw_j*scratch.chi[scratch.j]
                  +scratch.val_dw_j*scratch.grad_chi[scratch.j])
                * scratch.JxW * tau_n ;

            // jump term at t_m
            copydata.local_eta_k_minus_vector[scratch.j] -= (-1.0)*
                scratch.val_u_kh_jump_j * scratch.val_dw_j * scratch.chi[scratch.j]* scratch.JxW;
        }


        //calculating PU for spatial error over I_n
        scratch.val_dw_j = 0.;
        scratch.grad_dw_j = 0.;


        //for spatial error dual weight is
        // dw(t) = (z_k^n-z_kh^n) for t in [0,1]
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
          scratch.val_dw_j +=
              (
                              scratch.local_z_k_n[scratch.j]
                              -scratch.local_z_kh_n[scratch.j]
               ) * scratch.phi[scratch.j];

          scratch.grad_dw_j +=
              (scratch.local_z_k_n[scratch.j]-scratch.local_z_kh_n[scratch.j])
                      * scratch.grad_phi[scratch.j];
        }

        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // (f, z-zk chi) with trapezoidal rule
            copydata.local_eta_h_minus_vector[scratch.j]+=
                scratch.f_quad_h_minus *
                scratch.val_dw_j *
                scratch.chi[scratch.j]
                * scratch.JxW
                * tau_n;

            copydata.local_eta_h_vector[scratch.j]+=
                scratch.f_quad_h *
                scratch.val_dw_j *
                scratch.chi[scratch.j]
                * scratch.JxW
                * tau_n;

            // (eps * grad u, grad( z-zk chi) ) [A(u,z-zk chi)]
            copydata.local_eta_h_minus_vector[scratch.j] -=
                scratch.value_epsilon *
                                scratch.grad_u_kh_j
                *( scratch.grad_dw_j*
                scratch.chi[scratch.j]
                                 +
                scratch.val_dw_j
                                *scratch.grad_chi[scratch.j]
                )
                  *scratch.JxW * 0.5* tau_n ;

            copydata.local_eta_h_vector[scratch.j] -=
                scratch.value_epsilon *
                                scratch.grad_u_kh_j
                *( scratch.grad_dw_j*
                scratch.chi[scratch.j]
                                 +
                scratch.val_dw_j
                                *scratch.grad_chi[scratch.j]
                )
                  *scratch.JxW * 0.5* tau_n ;

            // jump term at t_m
            copydata.local_eta_h_minus_vector[scratch.j] -=
                scratch.val_u_kh_jump_j *
                scratch.val_dw_j *
                scratch.chi[scratch.j]* scratch.JxW;
        }

    } // end quadrature iteration
}

template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_split_adjoint_error_on_cell(
    const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
    Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata)
{

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.dof_dual);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.dof_pu);


    // reinit scratch and data to current cell
    scratch.fe_values_dual.reinit(cell_dual);
    scratch.fe_values_pu.reinit(cell_pu);
    // fetch local dof data
    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
        (*primal.args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_u_k_n[scratch.j] =
        (*primal.args->tn.u_k)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
        (*primal.args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_k_np1[scratch.j] =
        (*primal.args->tnp1.u_k)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_u_kh_m[scratch.j] =
        (*primal.args->tm.u_kh)[scratch.local_dof_indices_dual[scratch.j]];

        scratch.local_z_kh_np1[scratch.j] =
        (*primal.args->tnp1.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);

    //assemble PU
    for (scratch.q_x = 0; scratch.q_x < scratch.fe_values_pu.n_quadrature_points; ++scratch.q_x)
    {
        scratch.JxW = scratch.fe_values_pu.JxW(scratch.q_x);

        //shape values for dual basis
        for (scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
             ++scratch.j)
        {
        scratch.phi[scratch.j] =
            scratch.fe_values_dual.shape_value_component(scratch.j,scratch.q_x,0);

        scratch.grad_phi[scratch.j] =
            scratch.fe_values_dual.shape_grad(scratch.j,scratch.q_x);
        }

        //shape values for spatial partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
              ++ scratch.j )
        {
            scratch.chi[scratch.j] =
                scratch.fe_values_pu.shape_value(scratch.j,scratch.q_x);

            scratch.grad_chi[scratch.j] =
                scratch.fe_values_pu.shape_grad(scratch.j,scratch.q_x);
        }

        //get values of given functions
        scratch.value_epsilon = function.epsilon->value(scratch.fe_values_dual.quadrature_point(scratch.q_x),0);


        //First argument is the same for both error parts
        scratch.grad_z_kh_j = 0.;
        scratch.val_z_kh_jump_j = 0.;
        scratch.val_u_kh_jump_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.grad_z_kh_j +=
                (scratch.local_z_kh_n[scratch.j] * scratch.grad_phi[scratch.j]);

            scratch.val_z_kh_jump_j +=
                (scratch.local_z_kh_n[scratch.j]-scratch.local_z_kh_np1[scratch.j])*
                scratch.phi[scratch.j];

            //using this as storage for u_kh for the L2 error functional
            scratch.val_u_kh_jump_j +=
                scratch.local_u_kh_n[scratch.j]
                *scratch.phi[scratch.j];
        }

        if (goal_type.compare("L2L2")==0){
            //Calculating u_{ex}-u_{kh} for Simpson's rule
            function.u_ex->set_time(tm);
            scratch.u_diff_tm = (function.u_ex->value(scratch.fe_values_dual.quadrature_point(scratch.q_x),0)
                -scratch.val_u_kh_jump_j)/(6.0*L2_error);

            function.u_ex->set_time(t0);
            scratch.u_diff_t0 = 4.0*(function.u_ex->value(scratch.fe_values_dual.quadrature_point(scratch.q_x),0)
                -scratch.val_u_kh_jump_j)/(6.0*L2_error);

            function.u_ex->set_time(tn);
            scratch.u_diff_tn = (function.u_ex->value(scratch.fe_values_dual.quadrature_point(scratch.q_x),0)
                -scratch.val_u_kh_jump_j)/(6.0*L2_error);

        }
        //calculating PU for temporal error over I_n
        scratch.val_pw_j = 0.;
        scratch.grad_pw_j = 0.;

        // for temporal error primal weight is
        // pw(t) = (1-t)*(u_k^m-u_k^n) for t in [0,1]
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
          scratch.val_pw_j +=
              (scratch.local_u_k_np1[scratch.j] - scratch.local_u_k_n[scratch.j])
                  * scratch.phi[scratch.j];

          scratch.grad_pw_j +=
              (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j])
                  * scratch.grad_phi[scratch.j];
        }


        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // J'(u_kh)(pw*chi)
            if (goal_type.compare("L2L2")==0){
                copydata.local_eta_k_vector[scratch.j]+= tau_n*scratch.JxW*(
                    //tm / t = 0
                    scratch.u_diff_tn+
                    //t0 / t= 1/2
                    scratch.u_diff_t0*0.5
                    //tn / t= 1 -> 0*pw in time
                )*scratch.val_pw_j*scratch.chi[scratch.j];
            } else if (goal_type.compare("mean")==0){
                copydata.local_eta_k_vector[scratch.j]+=
                    0.5*
                    scratch.val_pw_j*scratch.chi[scratch.j]
                    * scratch.JxW *tau_n;
            }
            // (eps * grad (u-uk chi), grad z [A(u-uk chi,z)]
            copydata.local_eta_k_vector[scratch.j] -= 0.5 *
                (scratch.value_epsilon * scratch.grad_z_kh_j)
                *( scratch.grad_pw_j*scratch.chi[scratch.j]
                  +scratch.val_pw_j*scratch.grad_chi[scratch.j])
                * scratch.JxW * tau_n ;

            // jump term at t_n
            copydata.local_eta_k_vector[scratch.j] -=
                scratch.val_z_kh_jump_j * scratch.val_pw_j * scratch.chi[scratch.j]* scratch.JxW;
        }

        //calculating PU for spatial error over I_n
        scratch.val_pw_j = 0.;
        scratch.grad_pw_j = 0.;

        //for spatial error dual weight is
        // dw(t) = (u_k^n-u_kh^n) for t in [0,1]
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.val_pw_j +=
                ( scratch.local_u_k_n[scratch.j]
                  - scratch.local_u_kh_n[scratch.j]
                )* scratch.phi[scratch.j];

            scratch.grad_pw_j +=
                ( scratch.local_u_k_n[scratch.j]
                  - scratch.local_u_kh_n[scratch.j]
                )* scratch.grad_phi[scratch.j];
        }

        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // J'(u_kh)(pw*chi)
            if (goal_type.compare("L2L2")==0){
                copydata.local_eta_h_vector[scratch.j]+= tau_n*scratch.JxW*(
                    scratch.u_diff_tm +scratch.u_diff_t0+scratch.u_diff_tn
                )*scratch.val_pw_j*scratch.chi[scratch.j];
            } else if (goal_type.compare("mean")==0){
                copydata.local_eta_h_vector[scratch.j]+=
                    scratch.val_pw_j*
                    scratch.chi[scratch.j]
                    *scratch.JxW
                    *tau_n;
            }

            // (eps * grad z, grad( u-uk chi) ) [A(z,u-uk chi)]
            copydata.local_eta_h_vector[scratch.j] -=
                scratch.value_epsilon *
                scratch.grad_z_kh_j
                *( scratch.grad_pw_j*
                scratch.chi[scratch.j]
                 +
                scratch.val_pw_j
                *scratch.grad_chi[scratch.j]
                )
                  *scratch.JxW * tau_n ;

            // jump term at t_nc
            copydata.local_eta_h_vector[scratch.j] -=
                scratch.val_z_kh_jump_j *
                scratch.val_pw_j *
                scratch.chi[scratch.j]* scratch.JxW;
        }

    } // end quadrature iteration
}


template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_joint_primal_error_on_cell(
    const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
    Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata)
{

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.dof_dual);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&tria_cell->get_triangulation(),
                                                                   tria_cell->level(),
                                                                   tria_cell->index(),
                                                                   &scratch.dof_pu);

    // reinit scratch and data to current cell
    scratch.fe_values_dual.reinit(cell_dual);
    scratch.fe_values_pu.reinit(cell_pu);
    // fetch local dof data
    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
            (*primal.args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
            (*primal.args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_n[scratch.j] =
            (*primal.args->tn.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }


    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_m[scratch.j] =
            (*primal.args->tm.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_m[scratch.j] =
            (*primal.args->tm.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu->get_dof_indices(copydata.local_dof_indices_pu);


    //assemble PU
    for (scratch.q_x = 0; scratch.q_x < scratch.fe_values_pu.n_quadrature_points; ++scratch.q_x)
    {
    	scratch.JxW = scratch.fe_values_pu.JxW(scratch.q_x);

    	//shape values for dual basis
    	for (scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
             ++scratch.j)
    	{
            scratch.phi[scratch.j] =
                scratch.fe_values_dual.shape_value_component(scratch.j,scratch.q_x,0);

            scratch.grad_phi[scratch.j] =
                scratch.fe_values_dual.shape_grad(scratch.j,scratch.q_x);
    	}
    	//shape values for spatial partition of Unity
    	for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
              ++ scratch.j )
    	{
            scratch.chi[scratch.j] =
                scratch.fe_values_pu.shape_value_component(scratch.j,scratch.q_x,0);

            scratch.grad_chi[scratch.j] =
                scratch.fe_values_pu.shape_grad(scratch.j,scratch.q_x);

    	}


    	scratch.value_epsilon = function.epsilon->value(scratch.fe_values_pu.quadrature_point(scratch.q_x),0);

    	scratch.f_quad_k = 0;
    	scratch.f_quad_h = 0;

    	//Simpsons rule for quadrature of rhs
    	//using k and h for the part multiplied with the
    	//left and right dual weight respectively
    	// meaning dw at midpoint = (dw_n + dw_m)*1/2
    	{
            //left
            function.f->set_time(tm);
            scratch.f_quad_k +=1.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            //right
            function.f->set_time(tn);
            scratch.f_quad_h +=1.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            //midpoint t0  0.5*dw_m and dw_n respectively
            function.f->set_time(t0);
            scratch.f_quad_k +=2.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);

            scratch.f_quad_h +=2.0/6.0*
                function.f->value(scratch.fe_values_dual.quadrature_point(scratch.q_x), 0);
    	}

    	//First argument
    	scratch.grad_u_kh_j = 0.;
    	scratch.val_u_kh_jump_j = 0.;

    	for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
    	{
            scratch.grad_u_kh_j +=
                (scratch.local_u_kh_n[scratch.j] * scratch.grad_phi[scratch.j]);

            scratch.val_u_kh_jump_j +=
                (scratch.local_u_kh_n[scratch.j] - scratch.local_u_kh_m[scratch.j])
                    *scratch.phi[scratch.j];
    	}

    	//Calculating PU for dw_m i.e. rhs, 1/2 Laplace and jump term
        scratch.val_dw_j = 0.;
        scratch.grad_dw_j = 0.;

        //for tm dual weight is
        // dw(tm) = (z_k^m-z_kh^n)
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
          scratch.val_dw_j +=
              (scratch.local_z_k_m[scratch.j]-scratch.local_z_kh_n[scratch.j])
                  * scratch.phi[scratch.j];

          scratch.grad_dw_j +=
              (scratch.local_z_k_m[scratch.j]-scratch.local_z_kh_n[scratch.j])
                  * scratch.grad_phi[scratch.j];
        }


        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // (f, z-zk chi) with midpoint
            copydata.local_eta_h_vector[scratch.j]+=
                scratch.f_quad_k
                  * scratch.val_dw_j * scratch.chi[scratch.j] * scratch.JxW *tau_n;

            // (eps * grad u, grad( z-zk chi) ) [A(u,z-zk chi)]
            copydata.local_eta_h_vector[scratch.j] -= 0.5 *
                (scratch.value_epsilon * scratch.grad_u_kh_j)
                *( scratch.grad_dw_j*scratch.chi[scratch.j]
                  +scratch.val_dw_j*scratch.grad_chi[scratch.j])
                * scratch.JxW * tau_n ;

            // jump term at t_m
            copydata.local_eta_h_vector[scratch.j] -=
              scratch.val_u_kh_jump_j * scratch.val_dw_j * scratch.chi[scratch.j]* scratch.JxW;
        }

    	//Calculating PU for dw_n i.e. rhs and  1/2 Laplace
        scratch.val_dw_j = 0.;
        scratch.grad_dw_j = 0.;

        //for tn dual weight is
        // dw(tn) = (z_k^n-z_kh^n)
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ; scratch.j++)
        {
            scratch.val_dw_j +=
                (scratch.local_z_k_n[scratch.j]-scratch.local_z_kh_n[scratch.j])
                      * scratch.phi[scratch.j];

            scratch.grad_dw_j +=
                (scratch.local_z_k_n[scratch.j]-scratch.local_z_kh_n[scratch.j])
                      * scratch.grad_phi[scratch.j];
        }


        for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell; scratch.j++ )
        {
            // (f, z-zk chi) with midpoint
            copydata.local_eta_h_vector[scratch.j]+=
                scratch.f_quad_h
                   * scratch.val_dw_j * scratch.chi[scratch.j] * scratch.JxW *tau_n;

            // (eps * grad u, grad( z-zk chi) ) [A(u,z-zk chi)]
            copydata.local_eta_h_vector[scratch.j] -= 0.5 *
                (scratch.value_epsilon * scratch.grad_u_kh_j)
                *( scratch.grad_dw_j*scratch.chi[scratch.j]
                  +scratch.val_dw_j*scratch.grad_chi[scratch.j])
                * scratch.JxW * tau_n ;
        }

    } // end quadrature iteration

}

template<int dim>
void
PUDoFErrorEstimator<dim>::copy_local_error(
    const Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata) {
    pu.constraints -> distribute_local_to_global(
        copydata.local_eta_h_vector,
        copydata.local_dof_indices_pu,
        *error_estimator.x_h
    );
    pu.constraints -> distribute_local_to_global(
        copydata.local_eta_k_vector,
        copydata.local_dof_indices_pu,
        *error_estimator.x_k
    );

}

template<int dim>
void
PUDoFErrorEstimator<dim>::copy_local_cG1_error(
    const Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata) {
    pu.constraints -> distribute_local_to_global(
        copydata.local_eta_h_minus_vector,
        copydata.local_dof_indices_pu,
        *error_estimator.x_h_minus
    );
    pu.constraints -> distribute_local_to_global(
        copydata.local_eta_h_vector,
        copydata.local_dof_indices_pu,
        *error_estimator.x_h
    );
    pu.constraints -> distribute_local_to_global(
        copydata.local_eta_k_minus_vector,
        copydata.local_dof_indices_pu,
        *error_estimator.x_k_minus
    );
    pu.constraints -> distribute_local_to_global(
        copydata.local_eta_k_vector,
        copydata.local_dof_indices_pu,
        *error_estimator.x_k
    );

}
}}} // namespace

#include "GeneralPUDoFErrorEstimator.inst.in"
