/**
 * @file Diffusion_DWR__cGp_dG0__cGq_cG1.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @brief Diffusion/DWR Problem with primal solver: cG(p)-dG(0) and dual solver: cG(q)-cG(1)
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

#ifndef __Diffusion_DWR__cGp_dG0__cGq_cG1_tpl_hh
#define __Diffusion_DWR__cGp_dG0__cGq_cG1_tpl_hh

// PROJECT includes
#include <diffusion/parameters/ParameterSet.hh>
#include <diffusion/grid/Grid_DWR.tpl.hh>
#include <diffusion/ErrorEstimator/ErrorEstimators.hh>
#include <diffusion/ErrorEstimator/EstimatorArguments.tpl.hh>
#include <diffusion/types/error_functional.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>
#include <DTM++/base/Problem.hh>
#include <DTM++/io/DataOutput.tpl.hh>
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <algorithm>
#include <list>
#include <iterator>

namespace diffusion {

template<int dim>
class Diffusion_PU_DWR : public DTM::Problem {
public:
    Diffusion_PU_DWR() = default;
    virtual ~Diffusion_PU_DWR() = default;

    virtual void set_input_parameters(
        std::shared_ptr< dealii::ParameterHandler > parameter_handler
    );

    virtual void run();

protected:
    std::shared_ptr< diffusion::dwr::ParameterSet > parameter_set;

    std::shared_ptr< diffusion::Grid_DWR<dim,1> > grid;
    virtual void init_grid();

    struct {
        std::shared_ptr< dealii::Function<dim> > density;
        std::shared_ptr< dealii::Function<dim> > epsilon;
        std::shared_ptr< dealii::Function<dim> > f;

        std::shared_ptr< dealii::Function<dim> > u_D;
        std::shared_ptr< dealii::Function<dim> > u_0;

        std::shared_ptr< dealii::Function<dim> > u_E;
    } function;

    virtual void init_functions();

    ////////////////////////////////////////////////////////////////////////////
    // primal problem:
    //

    /// primal: data structures for forward time marching
    struct {
        // storage container
        struct {
            /// primal solution dof list
            std::shared_ptr< DTM::types::storage_data_vectors<1> > u;
        } storage;

        /// temporary storage for primal solution u at \f$ t_m \f$
        std::shared_ptr< dealii::Vector<double> > um; // dof on t_m

        /// temporary storage for primal solution u at \f$ t_n \f$
        std::shared_ptr< dealii::Vector<double> > un; // dof on t_n

        /// temporary storage for primal mass matrix
        std::shared_ptr< dealii::SparseMatrix<double> > M;

        /// temporary storage for primal stiffness matrix
        std::shared_ptr< dealii::SparseMatrix<double> > A;

        /// temporary storage for primal right hand side assembly
        std::shared_ptr< dealii::Vector<double> > f0;

        /// temporary storage for primal system matrix
        std::shared_ptr< dealii::SparseMatrix<double> > K;

        /// temporary storage for primal system right hand side
        std::shared_ptr< dealii::Vector<double> > b;


        std::shared_ptr< dealii::Vector<double> > tmp;
        // Data Output
        std::shared_ptr< DTM::DataOutput<dim> > data_output;
        int    data_output_dwr_loop;
        double data_output_time_value;
        double data_output_trigger;
        bool   data_output_trigger_type_fixed;
    } primal;

    virtual void primal_reinit_storage();

    virtual void primal_setup_slab(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u
    );

    virtual void primal_assemble_system(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );

    virtual void primal_assemble_rhs(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const double t0
    );

    virtual void primal_solve_slab_problem(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u,
        const double t0
    );
	
    /// do the forward time marching process of the primal problem
    virtual void primal_do_forward_TMS();

    /// evaluate solution dof vector u^primal(t) on primal solution space
    virtual void primal_get_u_t_on_slab(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u,
        const double &t,
        std::shared_ptr< dealii::Vector<double> > &u_result
    );

    // post-processing functions for L2(L2) error
    double primal_error;
    bool mean_ref;
    virtual void primal_init_error_computations();
    virtual void primal_do_error_L2(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u
    );
    virtual void primal_do_error_mean(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u
    );
    virtual void primal_finish_error_computations();

    // post-processing functions for data output
    virtual void primal_init_data_output();

    virtual void primal_do_data_output_on_slab(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u,
        const unsigned int dwr_loop,
        const bool dG_initial_value
    );

    virtual void primal_do_data_output(
        const unsigned int dwr_loop,
        bool last
    );
	
    ////////////////////////////////////////////////////////////////////////////
    // dual problem:
    //

    /// dual: data structures for backward time marching and error estimation
    struct {
        // storage container
        struct {
            /// dual solution dof list
            std::shared_ptr< DTM::types::storage_data_vectors<2> > z_cG1;
            std::shared_ptr< DTM::types::storage_data_vectors<1> > z_dG0;
        } storage;

        /// temporary storage for dual solution z at \f$ t_m \f$
        std::shared_ptr< dealii::Vector<double> > zm;

        /// temporary storage for dual mass matrix
        std::shared_ptr< dealii::SparseMatrix<double> > M;

        /// temporary storage for dual stiffness matrix
        std::shared_ptr< dealii::SparseMatrix<double> > A;

        /// temporary storage for dual solution z on \f$ \hat t_0 \f$
        std::shared_ptr< dealii::Vector<double> > u0;

        /// temporary storage for dual solution z on \f$ \hat t_1 \f$
        std::shared_ptr< dealii::Vector<double> > u1;

        /// temporary storage for dual solution z on \f$ \hat t_2 \f$
        std::shared_ptr< dealii::Vector<double> > u2;

        /// temporary storage for dual right hand side assembly
        std::shared_ptr< dealii::Vector<double> > Je0;

        /// temporary storage for dual right hand side assembly
        std::shared_ptr< dealii::Vector<double> > Je1;

        /// temporary storage for dual right hand side assembly
        std::shared_ptr< dealii::Vector<double> > Je2;

        /// temporary storage for dual system matrix
        std::shared_ptr< dealii::SparseMatrix<double> > K;

        /// temporary storage for dual system right hand side
        std::shared_ptr< dealii::Vector<double> > b;

        // Data Output
        std::shared_ptr< DTM::DataOutput<dim> > data_output;
        int    data_output_dwr_loop;
        double data_output_time_value;
        double data_output_trigger;
        bool   data_output_trigger_type_fixed;
    } dual;

    virtual void dual_reinit_storage_cG1();
    virtual void dual_reinit_storage_dG0();

    virtual void dual_setup_slab_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<2>::iterator &z,
        bool mesh_interpolate
    );

    virtual void dual_setup_slab_dG0(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &z,
        bool mesh_interpolate
    );

    virtual void dual_assemble_system(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );
    virtual void dual_construct_system_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );
    virtual void dual_construct_system_dG0(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );
    virtual void dual_assemble_rhs(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u,
        const unsigned int &n,
        const double &t0,
        const double &t1
    );
    virtual void dual_construct_rhs_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<2>::iterator &z
    );

    virtual void dual_construct_rhs_dG0(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &z
    );

    virtual void dual_solve_slab_problem_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<2>::iterator &z
    );

    virtual void dual_solve_slab_problem_dG0(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &z
    );
    /// do the backward time marching process of the dual problem
    virtual void dual_do_backward_TMS_cG1();
    virtual void dual_do_backward_TMS_dG0();

    /// evaluate solution dof vector z^dual(t) on dual solution space
    virtual void dual_get_z_t_on_slab(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<2>::iterator &z,
        const double &t,
        std::shared_ptr< dealii::Vector<double> > &dual_z_result
    );

    // post-processing functions for data output
    virtual void dual_init_data_output();

    virtual void dual_do_data_output_on_slab_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<2>::iterator &z,
        const unsigned int dwr_loop
    );

    virtual void dual_do_data_output_on_slab_dG0(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &z,
        const unsigned int dwr_loop
    );

    virtual void dual_do_data_output_cG1(
        const unsigned int dwr_loop,
        bool last
    );

    virtual void dual_do_data_output_dG0(
        const unsigned int dwr_loop,
        bool last
    );
	
    ////////////////////////////////////////////////////////////////////////////
    // error estimation and space-time grid adaption
    //

    struct {
        struct {
            /// error indicator \f$ \eta_{I_n} \f$  list
            struct {
                std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_h;
                std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_h_minus;
                std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_k;
                std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_k_minus;
            } primal;
            struct {
                std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_h;
                std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_k;
            } adjoint;
        } storage;

        /// error estimator
        std::shared_ptr< diffusion::dwr::estimator::PUDoFErrorEstimator<dim>> pu_dof;
    } error_estimator;

    virtual void eta_reinit_storage();


    virtual void eta_interpolate_slab_dG0(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u,
        const typename DTM::types::storage_data_vectors<1>::iterator &z,
        std::shared_ptr<diffusion::dwr::estimator::Arguments> args,
        bool initial_slab,
        bool last_slab
    );

    virtual void eta_interpolate_slab_cG1(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_vectors<1>::iterator &u,
        const typename DTM::types::storage_data_vectors<2>::iterator &z,
        std::shared_ptr<diffusion::dwr::estimator::Arguments> args,
        bool initial_slab,
        bool last_slab
    );
    virtual void compute_pu_dof_error_indicators();
    virtual void compute_effectivity_index();

    virtual void refine_and_coarsen_space_time_grid();
    virtual void refine_and_coarsen_space_time_grid_global();
    virtual void refine_and_coarsen_space_time_grid_joint_dof();
    virtual void refine_and_coarsen_space_time_grid_sv_dof();
	
    ////////////////////////////////////////////////////////////////////////////
    // other
    //

    unsigned int setw_value_dwr_loops;

    // Convergence Table
    dealii::ConvergenceTable convergence_table;
    virtual void write_convergence_table_to_tex_file();
};

} // namespace

#endif
