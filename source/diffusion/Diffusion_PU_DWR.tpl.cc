/**
 * @file Diffusion_DWR__cGp_dG0__cGq_cG1.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @brief Diffusion/PU DWR Problem with primal solver: cG(p)-dG(0) and dual solver: cG(q)-dG(0) or cG(q)-cG(1)
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
#include <diffusion/Diffusion_PU_DWR.tpl.hh>

#include <diffusion/grid/Grid_DWR_Selector.tpl.hh>

#include <diffusion/Density/Density_Selector.tpl.hh>
#include <diffusion/Permeability/Permeability_Selector.tpl.hh>
#include <diffusion/Force/Force_Selector.tpl.hh>

#include <diffusion/InitialValue/InitialValue_Selector.tpl.hh>
#include <diffusion/DirichletBoundary/DirichletBoundary_Selector.tpl.hh>

#include <diffusion/ExactSolution/ExactSolution_Selector.tpl.hh>

#include <diffusion/types/boundary_id.hh>

#include <diffusion/assembler/L2_MassAssembly.tpl.hh>
#include <diffusion/assembler/L2_LaplaceAssembly.tpl.hh>

#include <diffusion/assembler/L2_ForceConstrainedAssembly.tpl.hh>
template <int dim>
using ForceAssembler = diffusion::Assemble::L2::ForceConstrained::Assembler<dim>;

#include <diffusion/assembler/L2_Je_global_L2L2_Assembly.tpl.hh>
#include <diffusion/assembler/L2_Je_global_Mean_Assembly.tpl.hh>

// DEAL.II includes
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace diffusion {

template<int dim>
void
Diffusion_PU_DWR<dim>::
set_input_parameters(
    std::shared_ptr< dealii::ParameterHandler > parameter_handler) {
    Assert(parameter_handler.use_count(), dealii::ExcNotInitialized());

    parameter_set = std::make_shared< diffusion::dwr::ParameterSet > (
        parameter_handler
    );
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
run() {

    mean_ref = true;

    // check primal time discretisation
    if ((parameter_set->fe.primal.time_type.compare("dG") == 0) &&
        (parameter_set->fe.primal.r == 0)) {
        DTM::pout
            << "primal time discretisation = dG(0)-Q_Gauss_Lobatto"
            << std::endl;
    }
    else {
        AssertThrow(
            false,
            dealii::ExcMessage("primal time discretisation unknown")
        );
    }
	
    // check dual time discretisation
    if ((parameter_set->fe.dual.time_type.compare("cG") == 0) &&
        (parameter_set->fe.dual.s == 1)) {
        DTM::pout
            << "dual time discretisation = cG(1)-Q_Gauss_Lobatto"
            << std::endl;
    }
    else if ((parameter_set->fe.dual.time_type.compare("dG") == 0 ) &&
             (parameter_set->fe.dual.s == 0)) {
        DTM::pout
            << "dual time discretization = dG(0)-Q_Gauss_Lobatto"
            << std::endl;
    }
    else {
        AssertThrow(
            false,
            dealii::ExcMessage("dual time discretisation unknown")
        );
    }
	
    // determine setw value for dwr loop number of data output filename
    setw_value_dwr_loops = static_cast<unsigned int>(
        std::floor(std::log10(parameter_set->dwr.loops))+1
    );

    init_grid();
    init_functions();
	
    ////////////////////////////////////////////////////////////////////////////
    // DWR loop
    //

    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl;
	
    ////////////////////////////////////////////////////////////////////////////
    // setup solver/reduction control for outer dwr loop
    std::shared_ptr< dealii::ReductionControl > solver_control_dwr;
    solver_control_dwr = std::make_shared< dealii::ReductionControl >();
	
    if (!parameter_set->dwr.solver_control.in_use) {
        solver_control_dwr->set_max_steps(parameter_set->dwr.loops);
        solver_control_dwr->set_tolerance(0.);
        solver_control_dwr->set_reduction(0.);

        DTM::pout
            << std::endl
            << "dwr loops (fixed number) = " << solver_control_dwr->max_steps()
            << std::endl << std::endl;
    }
    else {
        solver_control_dwr->set_max_steps(
            parameter_set->dwr.solver_control.max_iterations
        );

        solver_control_dwr->set_tolerance(
            parameter_set->dwr.solver_control.tolerance
        );

        solver_control_dwr->set_reduction(
            parameter_set->dwr.solver_control.reduction_mode ?
            parameter_set->dwr.solver_control.reduction :
            parameter_set->dwr.solver_control.tolerance
        );
    }
	
    DTM::pout
        << std::endl
        << "dwr tolerance = " << solver_control_dwr->tolerance() << std::endl
        << "dwr reduction = " << solver_control_dwr->reduction() << std::endl
        << "dwr max. iterations = " << solver_control_dwr->max_steps() << std::endl
        << std::endl;
	
    dealii::SolverControl::State dwr_loop_state{dealii::SolverControl::State::iterate};
    solver_control_dwr->set_max_steps(solver_control_dwr->max_steps()-1);
    unsigned int dwr_loop{solver_control_dwr->last_step()+1};
    do {
        if (dwr_loop > 0) {
            // do space-time mesh refinements and coarsenings
            refine_and_coarsen_space_time_grid();
        }

        DTM::pout
            << "***************************************************************"
            << "*****************" << std::endl
            << "dwr loop = " << dwr_loop << std::endl;

        convergence_table.add_value("DWR-loop", dwr_loop+1);

        grid->set_boundary_indicators();
        grid->distribute();

        // primal problem:
        primal_reinit_storage();
        primal_init_data_output();
        primal_do_forward_TMS();
        primal_do_data_output(dwr_loop,false);

        // check if dwr has converged
        dwr_loop_state = solver_control_dwr->check(
            dwr_loop,
            primal_error // convergence criterium here
        );

        if (dwr_loop_state == dealii::SolverControl::State::iterate) {
            DTM::pout << "state iterate = true" << std::endl;
        }
        else {
            DTM::pout << "state iterate = false" << std::endl;
        }

        // dual problem cG1
        if ((parameter_set->fe.dual.time_type.compare("cG") == 0) &&
            (parameter_set->fe.dual.s == 1) ){
            dual_reinit_storage_cG1();
            dual_init_data_output();
            dual_do_backward_TMS_cG1();
            dual_do_data_output_cG1(dwr_loop,false);
        }
        // dual problem dG0
        else if ( (parameter_set->fe.dual.time_type.compare("dG") == 0) &&
                  (parameter_set->fe.dual.s == 0) ){
            dual_reinit_storage_dG0();
            dual_init_data_output();
            dual_do_backward_TMS_dG0();
            dual_do_data_output_dG0(dwr_loop,false);
        }

        // error estimation
        eta_reinit_storage();

        DTM::pout << "estimating with DoF-wise partition of unity" << std::endl;
        compute_pu_dof_error_indicators();

        compute_effectivity_index();

        std::cout << "dwr loop = " << dwr_loop << " ... (done)" << std::endl;
    } while((dwr_loop_state == dealii::SolverControl::State::iterate) && ++dwr_loop);

    // data output of the last (final) dwr loop solution
    if (dwr_loop_state == dealii::SolverControl::State::success) {
        primal_do_data_output(dwr_loop,true);
        if ((parameter_set->fe.dual.time_type.compare("cG") == 0) &&
            (parameter_set->fe.dual.s == 1) ){
            dual_do_data_output_cG1(dwr_loop,true);
        }
        else if ( (parameter_set->fe.dual.time_type.compare("dG") == 0) &&
                  (parameter_set->fe.dual.s == 0) ){
            dual_do_data_output_dG0(dwr_loop,true);
        }
    }

    write_convergence_table_to_tex_file();
}


////////////////////////////////////////////////////////////////////////////////
// protected member functions (internal use only)
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
init_grid() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
    ////////////////////////////////////////////////////////////////////////////
    // init grid from input parameter file spec.
    //
    {
        diffusion::grid::Selector<dim> selector;
        selector.create_grid(
            parameter_set->Grid_Class,
            parameter_set->Grid_Class_Options,
            parameter_set->TriaGenerator,
            parameter_set->TriaGenerator_Options,
            grid
        );

        Assert(grid.use_count(), dealii::ExcNotInitialized());
    }
	
    ////////////////////////////////////////////////////////////////////////////
    // initialize slabs of grid
    //

    Assert((parameter_set->fe.primal.p), dealii::ExcInvalidState());
    Assert(
        (parameter_set->fe.primal.p < parameter_set->fe.dual.q),
        dealii::ExcInvalidState()
    );
	
    Assert((parameter_set->t0 >= 0), dealii::ExcInvalidState());
    Assert((parameter_set->t0 < parameter_set->T), dealii::ExcInvalidState());
    Assert((parameter_set->tau_n > 0), dealii::ExcInvalidState());
	
    Assert(grid.use_count(), dealii::ExcNotInitialized());
    grid->initialize_slabs(
        parameter_set->fe.primal.p,
        parameter_set->fe.dual.q,
        parameter_set->t0,
        parameter_set->T,
        parameter_set->tau_n
    );
	
    grid->generate();

    grid->refine_global(
        parameter_set->global_refinement
    );

    DTM::pout
        << "grid: number of slabs = " << grid->slabs.size()
        << std::endl;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
init_functions() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    // density function density:
    {
        diffusion::density::Selector<dim> selector;
        selector.create_function(
            parameter_set->density_function,
            parameter_set->density_options,
            function.density
        );

        Assert(function.density.use_count(), dealii::ExcNotInitialized());
    }
	
    // permeability function epsilon:
    {
        diffusion::permeability::Selector<dim> selector;
        selector.create_function(
            parameter_set->epsilon_function,
            parameter_set->epsilon_options,
            function.epsilon
        );

        Assert(function.epsilon.use_count(), dealii::ExcNotInitialized());
    }
	
    // force function f:
    {
        diffusion::force::Selector<dim> selector;
        selector.create_function(
            parameter_set->force_function,
            parameter_set->force_options,
            function.f
        );

        Assert(function.f.use_count(), dealii::ExcNotInitialized());
    }
	
    // initial value function u_0:
    {
        diffusion::initial_value::Selector<dim> selector;
        selector.create_function(
            parameter_set->initial_value_u0_function,
            parameter_set->initial_value_u0_options,
            function.u_0
        );

        Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
    }
	
    // dirichlet boundary function u_D:
    {
        diffusion::dirichlet_boundary::Selector<dim> selector;
        selector.create_function(
            parameter_set->dirichlet_boundary_u_D_function,
            parameter_set->dirichlet_boundary_u_D_options,
            function.u_D
        );

        Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
    }
	
    // exact solution function u_E (if any)
    {
        diffusion::exact_solution::Selector<dim> selector;
        selector.create_function(
            parameter_set->exact_solution_function,
            parameter_set->exact_solution_options,
            function.u_E
        );

        Assert(function.u_E.use_count(), dealii::ExcNotInitialized());
    }
}


////////////////////////////////////////////////////////////////////////////////
// primal problem
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_reinit_storage() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * primal space: time dG(0) method (having 1 independent solution)
    //       * primal solution dof vectors: u
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    DTM::pout << N << std::endl;
    primal.storage.u = std::make_shared< DTM::types::storage_data_vectors<1> > ();
    primal.storage.u->resize(N);
	
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *primal.storage.u) {
            if (parameter_set->fe.primal.high_order){
                slab->primal.fe_info = 	slab->high;
            } else{
                slab->primal.fe_info = 	slab->low;
            }
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->primal.fe_info->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with n_dofs components:
                element.x[j]->reinit(
                    slab->primal.fe_info->dof->n_dofs()
                );
            }
            ++slab;
        }
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_setup_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator&
) {
    primal.um = std::make_shared< dealii::Vector<double> >();
    Assert(slab->primal.fe_info.use_count(), dealii::ExcNotInitialized());
    primal.um->reinit( slab->primal.fe_info->dof->n_dofs() );

    Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
    function.u_0->set_time(slab->t_m);

    Assert((slab != grid->slabs.end()), dealii::ExcInternalError());
    Assert(slab->primal.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
    Assert(slab->primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
    Assert(primal.um.use_count(), dealii::ExcNotInitialized());

    {
        // create sparsity pattern
        slab->primal.sp = std::make_shared<dealii::SparsityPattern>();

        Assert(slab->primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
        dealii::DynamicSparsityPattern dsp(
            slab->primal.fe_info->dof->n_dofs(), slab->primal.fe_info->dof->n_dofs()
        );

        Assert(
            slab->primal.fe_info->constraints.use_count(),
            dealii::ExcNotInitialized()
        );

        dealii::DoFTools::make_sparsity_pattern(
            *slab->primal.fe_info->dof,
            dsp,
            *slab->primal.fe_info->constraints,
            true //false // keep constrained dofs?
        );

        Assert(slab->primal.sp.use_count(), dealii::ExcNotInitialized());
        slab->primal.sp->copy_from(dsp);
    }

    if ( slab  == grid->slabs.begin()){
        dealii::VectorTools::interpolate(
            *slab->primal.fe_info->mapping,
            *slab->primal.fe_info->dof,
            *function.u_0,
            *primal.um
        );
        // NOTE: after the first dwr-loop the initial triangulation could have
        //       hanging nodes. Therefore,
        // distribute hanging node constraints to make the result continuous again:
        slab->primal.fe_info->constraints->distribute(*primal.um);
    }else{
        Assert(primal.un.use_count(), dealii::ExcNotInitialized());
        primal.um = std::make_shared< dealii::Vector<double> >();
        primal.um->reinit( slab->primal.fe_info->dof->n_dofs() );

        // for n > 1 interpolate between two (different) spatial meshes
        // the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n-1}:
            *std::prev(slab)->primal.fe_info->dof,
            *primal.un,
            // solution on I_n:
            *slab->primal.fe_info->dof,
            *slab->primal.fe_info->constraints,
            *primal.um
        );
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_assemble_system(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
    // ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
    primal.M = std::make_shared< dealii::SparseMatrix<double> > ();
    primal.M->reinit(*slab->primal.sp);

    *primal.M = 0;
    {
        diffusion::Assemble::L2::Mass::
        Assembler<dim> assemble_mass(
            primal.M,
            slab->primal.fe_info->dof,
            slab->primal.fe_info->fe,
            slab->primal.fe_info->mapping,
            slab->primal.fe_info->constraints
        );

        Assert(function.density.use_count(), dealii::ExcNotInitialized());
        assemble_mass.set_density(function.density);

        DTM::pout << "pu-dwr-diffusion: assemble mass matrix...";
        assemble_mass.assemble();
        DTM::pout << " (done)" << std::endl;
    }
	
    // ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
    primal.A = std::make_shared< dealii::SparseMatrix<double> > ();
    primal.A->reinit(*slab->primal.sp);

    *primal.A = 0;
    {
        diffusion::Assemble::L2::Laplace::
        Assembler<dim> assemble_stiffness_cell_terms (
            primal.A,
            slab->primal.fe_info->dof,
            slab->primal.fe_info->fe,
            slab->primal.fe_info->mapping,
            slab->primal.fe_info->constraints
        );

        Assert(function.epsilon.use_count(), dealii::ExcNotInitialized());
        assemble_stiffness_cell_terms.set_epsilon_function(function.epsilon);

        DTM::pout << "pu-dwr-diffusion: assemble cell stiffness matrix...";
        assemble_stiffness_cell_terms.assemble();
        DTM::pout << " (done)" << std::endl;
    }
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_assemble_rhs(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const double t0
) {
    // Force assembly
    primal.f0 = std::make_shared< dealii::Vector<double> > ();
    primal.f0->reinit( slab->primal.fe_info->dof->n_dofs() );




    auto assemble_f0 = std::make_shared< ForceAssembler<dim> > (
        primal.f0,
        slab->primal.fe_info->dof,
        slab->primal.fe_info->fe,
        slab->primal.fe_info->mapping,
        slab->primal.fe_info->constraints
    );
	
    Assert(function.f.use_count(), dealii::ExcNotInitialized());
    assemble_f0->set_function(function.f);

    DTM::pout << "pu-dwr-diffusion: assemble force f0...";
    assemble_f0->assemble(
        t0,
        parameter_set->force_assembler_n_quadrature_points
    );
    DTM::pout << " (done)" << std::endl;

}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_solve_slab_problem(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u,
    const double t0
) {
    ////////////////////////////////////////////////////////////////////////////
    // construct system matrix K = M + tau A
    //

    DTM::pout << "pu-dwr-diffusion: construct system matrix K = M + tau A...";

    primal.K = std::make_shared< dealii::SparseMatrix<double> > ();
    primal.K->reinit(*slab->primal.sp);

    *primal.K = 0;
    primal.K->add(slab->tau_n(), *primal.A);
    primal.K->add(1.0, *primal.M);

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // construct system right hand side vector b = M um + tau_n f0 + tau_n u_N0
    //

    DTM::pout << "pu-dwr-diffusion: construct linear system rhs vector...";

    primal.b = std::make_shared< dealii::Vector<double> > ();
    primal.b->reinit( slab->primal.fe_info->dof->n_dofs() );

    Assert(primal.M.use_count(), dealii::ExcNotInitialized());
    Assert(primal.um.use_count(), dealii::ExcNotInitialized());
    primal.M->vmult(*primal.b, *primal.um);

    primal.b->add(slab->tau_n(), *primal.f0);

    DTM::pout << " (done)" << std::endl;
	
    ////////////////////////////////////////////////////////////////////////////
    // apply inhomogeneous Dirichlet boundary values
    //

    DTM::pout << "pu-dwr-diffusion: dealii::MatrixTools::apply_boundary_values...";
    std::map<dealii::types::global_dof_index, double> boundary_values;

    Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
    function.u_D->set_time(t0);
    dealii::VectorTools::interpolate_boundary_values(
        *slab->primal.fe_info->dof,
        static_cast< dealii::types::boundary_id > (
                diffusion::types::boundary_id::Dirichlet
        ),
        *function.u_D,
        boundary_values
    );

    dealii::MatrixTools::apply_boundary_values(
        boundary_values,
        *primal.K,
        *u->x[0],
        *primal.b
    );
	
    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // condense hanging nodes in system matrix, if any
    //

    DTM::pout << "pu-dwr-diffusion: slab->low.constraints->condense(*primal.K)...";
    slab->primal.fe_info->constraints->condense(*primal.K);

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // solve linear system with direct solver
    //

    DTM::pout << "pu-dwr-diffusion: setup direct lss and solve...";

    dealii::SparseDirectUMFPACK iA;
    iA.initialize(*primal.K);
    iA.vmult(*u->x[0], *primal.b);

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // distribute hanging nodes constraints on solution
    //

    DTM::pout << "pu-dwr-diffusion: primal.constraints->distribute...";
    slab->primal.fe_info->constraints->distribute(
            *u->x[0]
    );
    DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_do_forward_TMS() {
    ////////////////////////////////////////////////////////////////////////////
    // prepare time marching scheme (TMS) loop
    //

    ////////////////////////////////////////////////////////////////////////////
    // grid: init slab iterator to first space-time slab: Omega x I_1
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = grid->slabs.begin();

    ////////////////////////////////////////////////////////////////////////////
    // storage: init iterators to storage_data_vectors
    //          corresponding to first space-time slab: Omega x I_1
    //
	
    Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
    Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
    auto u = primal.storage.u->begin();

    // init error computations (for global L2(L2) goal functional)
    primal_init_error_computations();

    ////////////////////////////////////////////////////////////////////////////
    // do TMS loop
    //
	
    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << "primal: solving forward TMS problem..." << std::endl
        << std::endl;

    unsigned int n{1};
    while (slab != grid->slabs.end()) {
        // local time variables: \f$ t0 \in I_n = (t_m, t_n) \f$
        const double tm = slab->t_m;

        double t0 = slab->t_n;

        const double tn = slab->t_n;

        DTM::pout
                << "primal: solving problem on "
                << "I_" << n << " = (" << tm << ", " << tn << ") "
                << std::endl;

        primal_setup_slab(slab,u);
        // assemble slab problem
        primal_assemble_system(slab);

        primal_assemble_rhs(slab,t0);

        // solve slab problem (i.e. apply boundary values and solve for u0)
        primal_solve_slab_problem(slab,u,t0);

        ////////////////////////////////////////////////////////////////////////
        // do postprocessings on the solution
        //

        if(parameter_set->dwr.goal.type.compare("L2L2") == 0){
            // do error computations ( for global L2(L2) goal )
            primal_do_error_L2(slab,u);
        }
        else if ( parameter_set->dwr.goal.type.compare("mean") == 0){
            // do error computations ( for solution mean )
            primal_do_error_mean(slab,u);
        }


        // evaluate solution u(t_n)
        primal.un = std::make_shared< dealii::Vector<double> >();
        primal.un->reinit( slab->primal.fe_info->dof->n_dofs() );

        double zeta0 = 1.; // zeta0( t_n ) = 1. for dG(0)
        *primal.un = 0;
        primal.un->add(zeta0, *u->x[0]);

        ////////////////////////////////////////////////////////////////////////
        // prepare next I_n slab problem:
        //

        ++n;
        ++slab;
        ++u;

        ////////////////////////////////////////////////////////////////////////
        // allow garbage collector to clean up memory
        //

        primal.M = nullptr;
        primal.A = nullptr;

        primal.f0 = nullptr;

        primal.K = nullptr;
        primal.b = nullptr;

        DTM::pout << std::endl;
    }
	
    DTM::pout
        << "primal: forward TMS problem done" << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << std::endl;
	
    ////////////////////////////////////////////////////////////////////////////
    // allow garbage collector to clean up memory
    //

    primal.um = nullptr;
    primal.un = nullptr;
	
    ////////////////////////////////////////////////////////////////////////////
    // finish error computation ( for global L2(L2) goal functional )
    //

    primal_finish_error_computations();
    DTM::pout
        << "*******************************************************************"
        << "*************" << std::endl;

    if (parameter_set->dwr.goal.type.compare("L2L2") == 0){
        DTM::pout << "primal: || u - u_kh ||_L2(L2) = " << primal_error;
    }
    else{
        DTM::pout << "primal: mean average error " << primal_error;
    }
    DTM::pout << std::endl << std::endl;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_get_u_t_on_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u,
    const double &t,
    std::shared_ptr< dealii::Vector<double> > &u_result
) {
    Assert( (t > slab->t_m), dealii::ExcInvalidState() );
    Assert( (t <= slab->t_n), dealii::ExcInvalidState() );

    u_result = std::make_shared< dealii::Vector<double> > ();
    u_result->reinit(
        slab->primal.fe_info->dof->n_dofs()
    );

    // get time _t on reference time interval I_hat = (0,1)
    [[maybe_unused]] const double _t{ (t - slab->t_m) / slab->tau_n() };

    // trial basis functions evaluation on reference interval
    const double zeta0{1.};

    u_result->equ(zeta0, *u->x[0]);
}


////////////////////////////////////////////////////////////////////////////////
// primal: L2(L2) error computation
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_init_error_computations() {
    primal_error = 0;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_do_error_L2(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u) {
    ////////////////////////////////////////////////////////////////////////////
    // compute L^2 in time error
    //

    // prepare L2(Omega) norm
    double norm_sqr{-1};

    dealii::QGauss<dim> quad_cell(
        slab->primal.fe_info->fe->tensor_degree()+1
    );

    dealii::Vector<double> difference_per_cell(
        slab->primal.fe_info->dof->n_dofs()
    );

    // prepare L2 in time norm
    double zeta0;
    double _t;

    // create quadrature for time integration of L2 in time integral on slab
    // m - number of Gauss points to be evaluated for \int_{I_n} ||err||^2_L2 dt
    [[maybe_unused]]unsigned int m(1);
    dealii::QGauss<1> quad_int_In(m);

    std::vector< dealii::Point<1> > tq(quad_int_In.get_points());
    std::vector< double > w(quad_int_In.get_weights());

    // L_2 norm
    _t = 1;
    function.u_E->set_time(_t * slab->tau_n() + slab->t_m);

    zeta0 = 1.;

    ////////////////////////////////////////////////////////////////////////
    dealii::Vector<double> u_trigger;
    u_trigger.reinit(
        slab->primal.fe_info->dof->n_dofs()
    );
    if (parameter_set->fe.primal.high_order){
        auto tmp_low = std::make_shared< dealii::Vector<double> > ();
        tmp_low->reinit(slab->high->dof->n_dofs()) ;

        dealii::FETools::interpolate(
            *slab->high->dof,
            *u->x[0],
            *slab->low->dof,
            *slab->low->constraints,
            *tmp_low
        );

        dealii::FETools::interpolate(
            *slab->low->dof,
            *tmp_low,
            *slab->high->dof,
            *slab->high->constraints,
            u_trigger
        );
        tmp_low = nullptr;
    }else {
        // evalute space-time solution
        u_trigger.equ(zeta0, *u->x[0]);
    }
    ////////////////////////////////////////////////////////////////////////
    // u:
    difference_per_cell = 0;

    dealii::VectorTools::integrate_difference(
        *slab->primal.fe_info->dof,
        u_trigger,
        *function.u_E,
        difference_per_cell,
        quad_cell,
        dealii::VectorTools::L2_norm
    );

    norm_sqr = difference_per_cell.norm_sqr();

    primal_error += /* w[q] * */ norm_sqr * slab->tau_n();
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_do_error_mean(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u) {

    dealii::QGauss<dim> quad_cell(
        slab->primal.fe_info->fe->tensor_degree()+4
    );

    primal_error += slab->tau_n() * dealii::VectorTools::compute_mean_value(
                    *slab->primal.fe_info->mapping,
                    *slab->primal.fe_info->dof,
                    quad_cell,
                    *u->x[0],
                    0);

}
template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_finish_error_computations() {
    if (parameter_set->dwr.goal.type.compare("L2L2") == 0){
        primal_error = std::sqrt(primal_error);
        DTM::pout << "primal_L2_L2_error_u = " << primal_error << std::endl;
    }
    else if ( parameter_set->dwr.goal.type.compare("mean") == 0 ) {
        primal_error -= parameter_set->dwr.reference.mean;
        primal_error = std::abs(primal_error);
        DTM::pout << std::setprecision(9) << "primal error for mean = " << primal_error << std::endl;
    }

}


////////////////////////////////////////////////////////////////////////////////
// primal data output
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_init_data_output() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    // set up which dwr loop(s) are allowed to make data output:
    if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
        return;
    }
	
    // may output data: initialise (mode: all, last or specific dwr loop)
    DTM::pout
        << "primal solution data output: patches = "
        << parameter_set->data_output.primal.patches
        << std::endl;

    std::vector<std::string> data_field_names;
    data_field_names.push_back("u");

    std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    primal.data_output = std::make_shared< DTM::DataOutput<dim> >();
    primal.data_output->set_data_field_names(data_field_names);
    primal.data_output->set_data_component_interpretation_field(dci_field);

    primal.data_output->set_data_output_patches(
        2//parameter_set->data_output.primal.patches
    );
	
    // check if we use a fixed trigger interval, or, do output once on a I_n
    if ( !parameter_set->data_output.primal.trigger_type.compare("fixed") ) {
        primal.data_output_trigger_type_fixed = true;
    }
    else {
        primal.data_output_trigger_type_fixed = false;
    }
	
    // only for fixed
    primal.data_output_trigger = parameter_set->data_output.primal.trigger;

    if (primal.data_output_trigger_type_fixed) {
        DTM::pout
            << "primal solution data output: using fixed mode with trigger = "
            << primal.data_output_trigger
            << std::endl;
    }
    else {
        DTM::pout
            << "primal solution data output: using I_n mode (trigger adapts to I_n automatically)"
            << std::endl;
    }

    primal.data_output_time_value = parameter_set->t0;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_do_data_output_on_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u,
    const unsigned int dwr_loop,
    const bool dG_initial_value)
{
    if (primal.data_output_trigger <= 0) return;

    primal.data_output->set_DoF_data(
        slab->primal.fe_info->dof
    );

    auto u_trigger = std::make_shared< dealii::Vector<double> > ();
    u_trigger->reinit(
        slab->primal.fe_info->dof->n_dofs()
    );

    std::ostringstream filename;
    filename
        << "solution-dwr_loop-"
        << std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;

    double &t{primal.data_output_time_value};
	
    if (dG_initial_value) {
        // NOTE: for dG-in-time discretisations the initial value function
        //       does not belong to the set of dof's. Thus, we need a special
        //       implementation here to output "primal.um".

        u_trigger->equ(1., *u->x[0]); // NOTE: this must be primal.um!

        primal.data_output->write_data(
            filename.str(),
            u_trigger,
            t
        );

        t += primal.data_output_trigger;
    }
    else {
        // adapt trigger value for I_n output mode
        if (!primal.data_output_trigger_type_fixed) {
            primal.data_output_trigger = slab->tau_n();
            primal.data_output_time_value = slab->t_n;
        }

        for ( ; t <= slab->t_n; t += primal.data_output_trigger) {
            [[maybe_unused]] const double _t{ (t - slab->t_m) / slab->tau_n() };

            const double zeta0{1.};

            // evalute space-time solution
            u_trigger->equ(zeta0, *u->x[0]);

            primal.data_output->write_data(
                filename.str(),
                u_trigger,
                t
            );
        }
    }
	
    // check if data for t=T was written
    if (std::next(slab) == grid->slabs.end()) {
    if (primal.data_output_trigger_type_fixed) {
        const double overshoot_tol{
            std::min(slab->tau_n(), primal.data_output_trigger) * 1e-7
        };

        if ((t > slab->t_n) && (std::abs(t - slab->t_n) < overshoot_tol)) {
            // overshoot of time variable; manually set to t = T and do data output
            t = slab->t_n;

            [[maybe_unused]] const double _t{ (t - slab->t_m) / slab->tau_n() };

            const double zeta0{1.};

            // evalute space-time solution
            u_trigger->equ(zeta0, *u->x[0]);

            primal.data_output->write_data(
                filename.str(),
                u_trigger,
                t
            );
        }
    }}
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
primal_do_data_output(
    const unsigned int dwr_loop,
    bool last
) {
    // set up which dwr loop(s) are allowed to make data output:
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
        return;
    }

    if (!parameter_set->data_output.primal.dwr_loop.compare("last")) {
        // output only the last (final) dwr loop
        if (last) {
            primal.data_output_dwr_loop = dwr_loop;
        }
        else {
            return;
        }
    }
    else {
        if (!parameter_set->data_output.primal.dwr_loop.compare("all")) {
            // output all dwr loops
            if (!last) {
                primal.data_output_dwr_loop = dwr_loop;
            }
            else {
                return;
            }
        }
        else {
            // output on a specific dwr loop
            if (!last) {
                primal.data_output_dwr_loop =
                    std::stoi(parameter_set->data_output.primal.dwr_loop)-1;
            }
            else {
                return;
            }
        }
    }
	
    if (primal.data_output_dwr_loop < 0)
        return;

    if ( static_cast<unsigned int>(primal.data_output_dwr_loop) != dwr_loop )
        return;

    DTM::pout
        << "primal solution data output: dwr loop = "
        << primal.data_output_dwr_loop
        << std::endl;
	
    primal.data_output_time_value = parameter_set->t0;

    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = grid->slabs.begin();
    auto u = primal.storage.u->begin();
	
    // primal: dG: additionally output interpolated u_0(t0)
    {
        // n == 1: initial value function u_0
        primal.um = std::make_shared< dealii::Vector<double> >();
        primal.um->reinit( slab->primal.fe_info->dof->n_dofs() );

        primal.un = std::make_shared< dealii::Vector<double> >();
        primal.un->reinit( slab->primal.fe_info->dof->n_dofs() );

        function.u_0->set_time(slab->t_m);

        Assert(primal.um.use_count(), dealii::ExcNotInitialized());
        dealii::VectorTools::interpolate(
            *slab->primal.fe_info->mapping,
            *slab->primal.fe_info->dof,
            *function.u_0,
            *primal.um
        );
        slab->primal.fe_info->constraints->distribute(*primal.um);

        // output "initial value solution" at initial time t0
        Assert(primal.un.use_count(), dealii::ExcNotInitialized());
        *primal.un = *u->x[0];

        *u->x[0] = *primal.um;
        primal_do_data_output_on_slab(slab,u,dwr_loop,true);

        *u->x[0] = *primal.un;
    }
	
    while (slab != grid->slabs.end()) {
        primal_do_data_output_on_slab(slab,u,dwr_loop,false);

        ++slab;
        ++u;
    }
}


////////////////////////////////////////////////////////////////////////////////
// dual problem
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_reinit_storage_cG1() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * dual space: time cG(1) method (having 2 independent solutions)
    //       * dual solution dof vectors: z
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    dual.storage.z_cG1 = std::make_shared< DTM::types::storage_data_vectors<2> > ();
    dual.storage.z_cG1->resize(N);

    {
        auto slab = grid->slabs.begin();
        for (auto &element : *dual.storage.z_cG1) {
            if (parameter_set->fe.dual.high_order){
                slab->dual.fe_info = slab->high;
            } else{
                slab->dual.fe_info = slab->low;
            }
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->dual.fe_info->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->dual.fe_info->dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with n_dofs components:
                element.x[j]->reinit(
                    slab->dual.fe_info->dof->n_dofs()
                );
            }
            ++slab;
        }
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_reinit_storage_dG0() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * dual space: time cG(1) method (having 2 independent solutions)
    //       * dual solution dof vectors: z
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    dual.storage.z_dG0 = std::make_shared< DTM::types::storage_data_vectors<1> > ();
    dual.storage.z_dG0->resize(N);

    {
        auto slab = grid->slabs.begin();
        for (auto &element : *dual.storage.z_dG0) {
            if (parameter_set->fe.dual.high_order){
                slab->dual.fe_info = slab->high;
            } else{
                slab->dual.fe_info = slab->low;
            }
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->dual.fe_info->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->dual.fe_info->dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with n_dofs components:
                element.x[j]->reinit(
                    slab->dual.fe_info->dof->n_dofs()
                );
            }
            ++slab;
        }
    }
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_setup_slab_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<2>::iterator &z,
    bool mesh_interpolate
){

    Assert(slab->dual.fe_info.use_count(), dealii::ExcNotInitialized());
    dual.b = std::make_shared< dealii::Vector<double> > ();
    dual.b->reinit( slab->dual.fe_info->dof->n_dofs() );

    // create sparsity pattern
    {
        slab->dual.sp = std::make_shared<dealii::SparsityPattern>();

        dealii::DynamicSparsityPattern dsp(
            slab->dual.fe_info->dof->n_dofs(), slab->dual.fe_info->dof->n_dofs()
        );

        Assert(slab->high->constraints.use_count(), dealii::ExcNotInitialized());
        dealii::DoFTools::make_sparsity_pattern(
            *slab->dual.fe_info->dof,
            dsp,
            *slab->dual.fe_info->constraints,
            true //false // keep constrained dofs?
        );

        Assert(slab->dual.sp.use_count(), dealii::ExcNotInitialized());
        slab->dual.sp->copy_from(dsp);
    }

    dual.K = std::make_shared< dealii::SparseMatrix<double> > ();
    dual.K->reinit(*slab->dual.sp);

    if ( mesh_interpolate) {
        // for 0 < n < N interpolate between two (different) spatial meshes
        // the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}

        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n+1}:
            *std::next(slab)->dual.fe_info->dof,
            *std::next(z)->x[0],
            // solution on I_n:
            *slab->dual.fe_info->dof,
            *slab->dual.fe_info->constraints,
            *z->x[1]
        );
        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n+1}:
            *std::next(slab)->dual.fe_info->dof,
            *dual.zm,
            // solution on I_n:
            *slab->dual.fe_info->dof,
            *slab->dual.fe_info->constraints,
            *z->x[1]
        );
        dual.zm = nullptr;
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_setup_slab_dG0(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &z,
    bool mesh_interpolate
){

    Assert(slab->dual.fe_info.use_count(), dealii::ExcNotInitialized());
    dual.b = std::make_shared< dealii::Vector<double> > ();
    dual.b->reinit( slab->dual.fe_info->dof->n_dofs() );

    dual.zm = std::make_shared<dealii::Vector<double>>();
    dual.zm->reinit(slab->dual.fe_info->dof->n_dofs());
    *dual.zm = 0;

    // create sparsity pattern
    {
        slab->dual.sp = std::make_shared<dealii::SparsityPattern>();

        dealii::DynamicSparsityPattern dsp(
            slab->dual.fe_info->dof->n_dofs(), slab->dual.fe_info->dof->n_dofs()
        );

        Assert(slab->dual.fe_info->constraints.use_count(), dealii::ExcNotInitialized());
        dealii::DoFTools::make_sparsity_pattern(
            *slab->dual.fe_info->dof,
            dsp,
            *slab->dual.fe_info->constraints,
            true //false // keep constrained dofs?
        );

        Assert(slab->dual.sp.use_count(), dealii::ExcNotInitialized());
        slab->dual.sp->copy_from(dsp);
    }

    dual.K = std::make_shared< dealii::SparseMatrix<double> > ();
    dual.K->reinit(*slab->dual.sp);

    if ( mesh_interpolate) {
        // for 0 < n < N interpolate between two (different) spatial meshes
        // the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}
            dealii::VectorTools::interpolate_to_different_mesh(
                // solution on I_{n+1}:
                *std::next(slab)->dual.fe_info->dof,
                *std::next(z)->x[0],
                // solution on I_n:
                *slab->dual.fe_info->dof,
                *slab->dual.fe_info->constraints,
                *dual.zm
            );
    }
}
template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_assemble_system(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
    // ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
    dual.M = std::make_shared< dealii::SparseMatrix<double> > ();
    dual.M->reinit(*slab->dual.sp);

    *dual.M = 0;
    {
        diffusion::Assemble::L2::Mass::
        Assembler<dim> assemble_mass(
            dual.M,
            slab->dual.fe_info->dof,
            slab->dual.fe_info->fe,
            slab->dual.fe_info->mapping,
            slab->dual.fe_info->constraints
        );

        Assert(function.density.use_count(), dealii::ExcNotInitialized());
        assemble_mass.set_density(function.density);

        DTM::pout << "pu-dwr-diffusion: assemble mass matrix...";
        assemble_mass.assemble();
        DTM::pout << " (done)" << std::endl;
    }
	
    // ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
    dual.A = std::make_shared< dealii::SparseMatrix<double> > ();
    dual.A->reinit(*slab->dual.sp);

    *dual.A = 0;
    {
        diffusion::Assemble::L2::Laplace::
        Assembler<dim> assemble_stiffness_cell_terms (
            dual.A,
            slab->dual.fe_info->dof,
            slab->dual.fe_info->fe,
            slab->dual.fe_info->mapping,
            slab->dual.fe_info->constraints
        );

        Assert(function.epsilon.use_count(), dealii::ExcNotInitialized());
        assemble_stiffness_cell_terms.set_epsilon_function(function.epsilon);

        DTM::pout << "pu-dwr-diffusion: assemble cell stiffness matrix...";
        assemble_stiffness_cell_terms.assemble();
        DTM::pout << " (done)" << std::endl;
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_construct_system_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab){
    ////////////////////////////////////////////////////////////////////////////
    // construct system matrix K
    //


    *dual.K = 0;

    // construct cG(1)-Q_GL(2) system matrix K = M + tau/2 A
    DTM::pout << "pu-dwr-diffusion: construct system matrix K = M + tau/2 A...";
    dual.K->add(slab->tau_n()/2., *dual.A);
    dual.K->add(1., *dual.M);

    DTM::pout << " (done)" << std::endl;
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_construct_system_dG0(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab){
    ////////////////////////////////////////////////////////////////////////////
    // construct system matrix K
    //
    *dual.K = 0;

    // construct cG(1)-Q_GL(2) system matrix K = M + tau/2 A
    DTM::pout << "pu-dwr-diffusion: construct system matrix K = M + tau A...";
    dual.K->add(slab->tau_n(), *dual.A);
    dual.K->add(1., *dual.M);

    DTM::pout << " (done)" << std::endl;

}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_assemble_rhs(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u,
    const unsigned int &n [[maybe_unused]],
    const double &t0,
    const double &t1
) {

    dual.Je0 = std::make_shared< dealii::Vector<double> > ();
    dual.Je0->reinit( slab->dual.fe_info->dof->n_dofs() );
    dual.Je1 = std::make_shared< dealii::Vector<double> > ();
    dual.Je1->reinit( slab->dual.fe_info->dof->n_dofs() );
    dual.Je2 = std::make_shared< dealii::Vector<double> > ();
    dual.Je2->reinit( slab->dual.fe_info->dof->n_dofs() );
    if ( parameter_set->dwr.goal.type.compare("mean") == 0 ){
        auto assemble_Je = std::make_shared<
            diffusion::Assemble::L2::Je_global_Mean::Assembler<dim> > (
                slab->dual.fe_info->dof,
                slab->dual.fe_info->fe,
                slab->dual.fe_info->mapping,
                slab->dual.fe_info->constraints
            );

        assemble_Je->assemble(
            dual.Je0,
            t0,
            5,   // n_q_points: 0 -> q+1 in auto mode
            false // auto mode
        );

        dual.Je1->equ(1.0,*dual.Je0);
        dual.Je2->equ(1.0,*dual.Je0);
    }
    else if (parameter_set->dwr.goal.type.compare("L2L2") == 0) {

        ////////////////////////////////////////////////////////////////////////////
        // NOTE: this is only for a global L2(L2) goal functional
        //

        Assert(function.u_E.use_count(), dealii::ExcNotInitialized());

        // init assembler:
        auto assemble_Je = std::make_shared<diffusion::Assemble::L2::Je_global_L2L2::Assembler<dim> > (
            slab->dual.fe_info->dof,
            slab->dual.fe_info->fe,
            slab->dual.fe_info->mapping,
            slab->dual.fe_info->constraints
        );

        // interpolate primal solution u_h(t1) to dual solution space
        dual.u1 = std::make_shared< dealii::Vector<double> > ();
        dual.u1->reinit( slab->dual.fe_info->dof->n_dofs() );

        if (parameter_set->fe.primal.high_order){
            auto tmp_low = std::make_shared< dealii::Vector<double> > ();
            tmp_low->reinit(slab->high->dof->n_dofs()) ;

            dealii::FETools::interpolate(
                *slab->high->dof,
                *u->x[0],
                *slab->low->dof,
                *slab->low->constraints,
                *tmp_low
            );

            dealii::FETools::interpolate(
                *slab->low->dof,
                *tmp_low,
                *slab->high->dof,
                *slab->high->constraints,
                *dual.u1
            );
            tmp_low = nullptr;
        }
        else {
            dealii::FETools::interpolate(
                *slab->primal.fe_info->dof,
                *u->x[0],
                *slab->dual.fe_info->dof,
                *slab->dual.fe_info->constraints,
                *dual.u1
            );
        }

        // assemble J(v)(e) = (v,e)
        DTM::pout << "pu-dwr-diffusion: assemble Je0...";

        assemble_Je->assemble(
            dual.Je0,
            t0,
            function.u_E,
            dual.u1,
            0,   // n_q_points: 0 -> q+1 in auto mode
            true // auto mode
        );



        *dual.Je0 *= 1./primal_error;
        DTM::pout << " (done)" << std::endl;

        ///////////////////////////////
        // u_h(t1) = u_h(t1)|_{I_{n}}
        //

        // assemble J(v)(e) = (v,e)
        DTM::pout << "pu-dwr-diffusion: assemble Je1...";

        assemble_Je->assemble(
            dual.Je1,
            t1,
            function.u_E,
            dual.u1,
            0,   // n_q_points: 0 -> q+1 in auto mode
            true // auto mode
        );
        *dual.Je1 *= 1./primal_error;
        DTM::pout << " (done)" << std::endl;
        DTM::pout << "pu-dwr-diffusion: assemble Je2...";

        double t2 = (t0+t1)/2;
        assemble_Je->assemble(
            dual.Je2,
            t2,
            function.u_E,
            dual.u1,
            0,   // n_q_points: 0 -> q+1 in auto mode
            true // auto mode
        );
        *dual.Je2 *= 1./primal_error;
        DTM::pout << " (done)" << std::endl;


        dual.u1 = nullptr;
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_construct_rhs_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<2>::iterator &z)
{
    ////////////////////////////////////////////////////////////////////////////
    // construct system right hand side vector
    // b = tau_n/2. * ( Je^0 + Je^1 ) + (M - tau_n/2 A) z^1
    //
    *dual.b = 0.;

    DTM::pout
        << "pu-dwr-diffusion: construct linear system rhs vector "
        << "b = (M - tau/2 A) z^1 + tau/2 * ( Je^0 + Je^1 ) ...";


    dual.A->vmult(*dual.b, *z->x[1]);
    *dual.b *= -slab->tau_n()/2.;

    dual.M->vmult_add(*dual.b, *z->x[1]);

    dual.b->add(3.*slab->tau_n()/3., *dual.Je2 );

    DTM::pout << " (done)" << std::endl;
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_construct_rhs_dG0(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator& ){
    ////////////////////////////////////////////////////////////////////////////
    // construct system right hand side vector
    // b = tau_n/2. * ( Je^0 + Je^1 ) + (M - tau_n/2 A) z^1
    //
    *dual.b = 0.;

    DTM::pout
        << "pu-dwr-diffusion: construct linear system rhs vector "
        << "b = (M z^1 + tau* ( Je^0 + Je^1 ) ...";

    dual.M->vmult_add(*dual.b, *dual.zm);

    dual.b->add(slab->tau_n()/6., *dual.Je0 );
    dual.b->add(slab->tau_n()/6., *dual.Je1 );
    dual.b->add(2.*slab->tau_n()/3., *dual.Je2 );

    DTM::pout << " (done)" << std::endl;
}
template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_solve_slab_problem_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<2>::iterator &z
) {
	
	

    ////////////////////////////////////////////////////////////////////////////
    // apply homogeneous Dirichlet boundary values
    //

    DTM::pout << "pu-dwr-diffusion: dealii::MatrixTools::apply_boundary_values...";
    std::map<dealii::types::global_dof_index, double> boundary_values;

    dealii::VectorTools::interpolate_boundary_values(
        *slab->dual.fe_info->dof,
        static_cast< dealii::types::boundary_id > (
                diffusion::types::boundary_id::Dirichlet
        ),
        dealii::ZeroFunction<dim>(1),
        boundary_values
    );

    dealii::MatrixTools::apply_boundary_values(
        boundary_values,
        *dual.K,
        *z->x[0],
        *dual.b
    );

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // condense hanging nodes in system matrix, if any
    //

    slab->dual.fe_info->constraints->condense(*dual.K);

    ////////////////////////////////////////////////////////////////////////////
    // solve linear system with direct solver
    //

    DTM::pout << "pu-dwr-diffusion: setup direct lss and solve...";

    dealii::SparseDirectUMFPACK iA;
    iA.initialize(*dual.K);
    iA.vmult(*z->x[0], *dual.b);

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // distribute hanging nodes constraints on solution
    //

    DTM::pout << "pu-dwr-diffusion: dual.constraints->distribute...";
    slab->dual.fe_info->constraints->distribute(
        *z->x[0]
    );
    DTM::pout << " (done)" << std::endl;
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_solve_slab_problem_dG0(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &z
) {



    ////////////////////////////////////////////////////////////////////////////
    // apply homogeneous Dirichlet boundary values
    //

    DTM::pout << "pu-dwr-diffusion: dealii::MatrixTools::apply_boundary_values...";
    std::map<dealii::types::global_dof_index, double> boundary_values;

    dealii::VectorTools::interpolate_boundary_values(
        *slab->dual.fe_info->dof,
        static_cast< dealii::types::boundary_id > (
            diffusion::types::boundary_id::Dirichlet
        ),
        dealii::ZeroFunction<dim>(1),
        boundary_values
    );

    dealii::MatrixTools::apply_boundary_values(
        boundary_values,
        *dual.K,
        *z->x[0],
        *dual.b
    );

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // condense hanging nodes in system matrix, if any
    //

    slab->dual.fe_info->constraints->condense(*dual.K);

    ////////////////////////////////////////////////////////////////////////////
    // solve linear system with direct solver
    //

    DTM::pout << "pu-dwr-diffusion: setup direct lss and solve...";
    dealii::SparseDirectUMFPACK iA;
    iA.initialize(*dual.K);
    iA.vmult(*z->x[0], *dual.b);

    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // distribute hanging nodes constraints on solution
    //

    DTM::pout << "pu-dwr-diffusion: dual.constraints->distribute...";
    slab->dual.fe_info->constraints->distribute(
        *z->x[0]
    );
    DTM::pout << " (done)" << std::endl;
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_do_backward_TMS_cG1() {
    ////////////////////////////////////////////////////////////////////////////
    // prepare time marching scheme (TMS) loop
    //

    ////////////////////////////////////////////////////////////////////////////
    // grid: init slab iterator to last space-time slab: Omega x I_N
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());

    ////////////////////////////////////////////////////////////////////////////
    // storage: init iterators to storage_data_vectors
    //          corresponding to last space-time slab: Omega x I_N
    //
	
    Assert(dual.storage.z_cG1.use_count(), dealii::ExcNotInitialized());
    Assert(dual.storage.z_cG1->size(), dealii::ExcNotInitialized());
    auto z = std::prev(dual.storage.z_cG1->end());

    Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
    Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
    auto u = std::prev(primal.storage.u->end());
	
    ////////////////////////////////////////////////////////////////////////////
    // final condition z_{\tau,h}(T)
    //

    // NOTE: for goal functional || u - u_{\tau,h} ||_L2(L2) -> z(T) = 0
    *z->x[1] = 0;
	
    ////////////////////////////////////////////////////////////////////////////
    // do TMS loop
    //

    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << "dual: solving backward TMS problem..." << std::endl
        << std::endl;
	
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    unsigned int n{N};

    while (n) {
        // local time variables: \f$ t0, t1 \in I_n = (t_m, t_n) \f$
        const double tm = slab->t_m;
        double t0{0};

        t0 = slab->t_m;

        const double t1 = slab->t_n;
        const double tn = slab->t_n;

        DTM::pout
            << "dual: solving problem on "
            << "I_" << n << " = (" << tm << ", " << tn << ") "
            << std::endl;

        dual_setup_slab_cG1(slab,z,(n<N));


        // assemble slab problem
        dual_assemble_system(slab);
            dual_assemble_rhs(slab,u,n,t0,t1);

        //Calculate zN by Backward Euler with tau/2
        if ( n == N )
        {
            //construct system Matrix (identical to CN because of tau/2)
            dual_construct_system_cG1(slab);
            //construct rhs
            *dual.b = 0;
            dual.b->equ(0.5*slab->tau_n(),*dual.Je1);

            //solve for z(1) = zN
            dual_solve_slab_problem_cG1(slab,z);
            z->x[1]->equ(1.0,*z->x[0]);
            *z->x[0] = 0;
        }
        dual_construct_system_cG1(slab);
        dual_construct_rhs_cG1(slab,z);
        // solve slab problem (i.e. apply boundary values and solve for z0)
        dual_solve_slab_problem_cG1(slab,z);

        ////////////////////////////////////////////////////////////////////////
        // prepare next I_n slab problem:
        //

        // Evaluate dual solution z(tm) (at left time-point) and save as dual.zm,
        // which is used to be interpolated to the "next" grid at tm.
        dual.zm = std::make_shared< dealii::Vector<double> > ();
        dual.zm->reinit( slab->dual.fe_info->dof->n_dofs() );
        *dual.zm = 0.;
        dual_get_z_t_on_slab(slab, z, tm, dual.zm);

        --n;
        --slab;
        --u;
        --z;

        ////////////////////////////////////////////////////////////////////////
        // allow garbage collector to clean up memory
        //

        dual.M = nullptr;
        dual.A = nullptr;

        dual.Je0 = nullptr;
        dual.Je1 = nullptr;
        dual.Je2 = nullptr;

        dual.K = nullptr;
        dual.b = nullptr;

        DTM::pout << std::endl;
    }

    DTM::pout
        << "dual: forward TMS problem done" << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // allow garbage collector to clean up memory
    //

    if (dual.zm.use_count())
        dual.zm = nullptr;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_do_backward_TMS_dG0() {
    ////////////////////////////////////////////////////////////////////////////
    // prepare time marching scheme (TMS) loop
    //

    ////////////////////////////////////////////////////////////////////////////
    // grid: init slab iterator to last space-time slab: Omega x I_N
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());

    ////////////////////////////////////////////////////////////////////////////
    // storage: init iterators to storage_data_vectors
    //          corresponding to last space-time slab: Omega x I_N
    //

    Assert(dual.storage.z_dG0.use_count(), dealii::ExcNotInitialized());
    Assert(dual.storage.z_dG0->size(), dealii::ExcNotInitialized());
    auto z = std::prev(dual.storage.z_dG0->end());

    Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
    Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
    auto u = std::prev(primal.storage.u->end());

    ////////////////////////////////////////////////////////////////////////////
    // do TMS loop
    //

    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << "dual: solving backward TMS problem..." << std::endl
        << std::endl;

    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    unsigned int n{N};

    while (n) {
        // local time variables: \f$ t0, t1 \in I_n = (t_m, t_n) \f$
        const double tm = slab->t_m;
        double t0{0};

        t0 = slab->t_m;

        const double t1 = slab->t_n;
        const double tn = slab->t_n;

        DTM::pout
            << "dual: solving problem on "
            << "I_" << n << " = (" << tm << ", " << tn << ") "
            << std::endl;


        dual_setup_slab_dG0(slab,z,(n<N));

        // assemble slab problem
        dual_assemble_system(slab);
        dual_assemble_rhs(slab,u,n,t0,t1);

        dual_construct_system_dG0(slab);
        dual_construct_rhs_dG0(slab,z);
        // solve slab problem (i.e. apply boundary values and solve for z0)
        dual_solve_slab_problem_dG0(slab,z);

        ////////////////////////////////////////////////////////////////////////
        // prepare next I_n slab problem:
        //

        --n;
        --slab;
        --u;
        --z;

        ////////////////////////////////////////////////////////////////////////
        // allow garbage collector to clean up memory
        //

        dual.M = nullptr;
        dual.A = nullptr;

        dual.Je0 = nullptr;
        dual.Je1 = nullptr;
        dual.Je2 = nullptr;

        dual.K = nullptr;
        dual.b = nullptr;

        dual.zm = nullptr;
        DTM::pout << std::endl;
    }

    DTM::pout
        << "dual: forward TMS problem done" << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // allow garbage collector to clean up memory
    //

    if (dual.zm.use_count())
        dual.zm = nullptr;
}
template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_get_z_t_on_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<2>::iterator &z,
    const double &t,
    std::shared_ptr< dealii::Vector<double> > &z_result
) {
    Assert( (t >= slab->t_m), dealii::ExcInvalidState() );
    Assert( (t <= slab->t_n), dealii::ExcInvalidState() );

    z_result = std::make_shared< dealii::Vector<double> > ();
    z_result->reinit(
            slab->dual.fe_info->dof->n_dofs()
    );

    // get time _t on reference time interval I_hat = (0,1)
    const double _t{ (t - slab->t_m) / slab->tau_n() };

    // trial basis functions evaluation on reference interval
    double xi0 = 1.-_t;
    double xi1 = _t;

    z_result->equ(xi0, *z->x[0]);
    z_result->add(xi1, *z->x[1]);
}


////////////////////////////////////////////////////////////////////////////////
// dual data output
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_init_data_output() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    // set up which dwr loop(s) are allowed to make data output:
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    // may output data: initialise (mode: all, last or specific dwr loop)
    DTM::pout
        << "dual solution data output: patches = "
        << parameter_set->data_output.dual.patches
        << std::endl;
	
    std::vector<std::string> data_field_names;
    data_field_names.push_back("z");

    std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    dual.data_output = std::make_shared< DTM::DataOutput<dim> >();
    dual.data_output->set_data_field_names(data_field_names);
    dual.data_output->set_data_component_interpretation_field(dci_field);

    dual.data_output->set_data_output_patches(
        2//parameter_set->data_output.dual.patches
    );

    // check if we use a fixed trigger interval, or, do output once on a I_n
    if ( !parameter_set->data_output.dual.trigger_type.compare("fixed") ) {
        dual.data_output_trigger_type_fixed = true;
    }
    else {
        dual.data_output_trigger_type_fixed = false;
    }

    // only for fixed
    dual.data_output_trigger = parameter_set->data_output.dual.trigger;

    if (dual.data_output_trigger_type_fixed) {
        DTM::pout
            << "dual solution data output: using fixed mode with trigger = "
            << dual.data_output_trigger
            << std::endl;
    }
    else {
        DTM::pout
            << "dual solution data output: using I_n mode (trigger adapts to I_n automatically)"
            << std::endl;
    }

    dual.data_output_time_value = parameter_set->T;
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_do_data_output_on_slab_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<2>::iterator &z,
    const unsigned int dwr_loop) {
    if (dual.data_output_trigger <= 0) return;

    // adapt trigger value for I_n output mode
    if (!dual.data_output_trigger_type_fixed) {
        dual.data_output_trigger = slab->tau_n();

        if (slab == std::prev(grid->slabs.end())) {
            dual.data_output_time_value = slab->t_n;
        }
        else {
            dual.data_output_time_value = slab->t_m;
        }
    }
	
    dual.data_output->set_DoF_data(
        slab->dual.fe_info->dof
    );
	
    auto z_trigger = std::make_shared< dealii::Vector<double> > ();
    z_trigger->reinit(
        slab->dual.fe_info->dof->n_dofs()
    );
	
    std::ostringstream filename;
    filename
        << "dual-dwr_loop-"
        << std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;

    double &t{dual.data_output_time_value};

    for ( ; t >= slab->t_m; t -= dual.data_output_trigger) {
        const double _t{ (t - slab->t_m) / slab->tau_n() };

        // trial basis functions evaluation on reference interval
        double xi0 = 1.-_t;
        double xi1 = _t;

        // evalute space-time solution
        z_trigger->equ(xi0, *z->x[0]);
        z_trigger->add(xi1, *z->x[1]);

        dual.data_output->write_data(
            filename.str(),
            z_trigger,
            t
        );
    }
	
    // check if data for t=0 (t_0) was written
    if (slab == grid->slabs.begin()) {
    if (dual.data_output_trigger_type_fixed) {
        const double overshoot_tol{
            std::min(slab->tau_n(), dual.data_output_trigger) * 1e-7
        };


        if ((t < slab->t_m) && (std::abs(t - slab->t_m) < overshoot_tol)) {
            // undershoot of time variable; manually set t = 0 and do data output
            t = slab->t_m;

            const double _t{ (t - slab->t_m) / slab->tau_n() };

            // trial basis functions evaluation on reference interval
            double xi0 = 1.-_t;
            double xi1 = _t;

            // evalute space-time solution
            z_trigger->equ(xi0, *z->x[0]);
            z_trigger->add(xi1, *z->x[1]);

            dual.data_output->write_data(
                filename.str(),
                z_trigger,
                t
            );
        }
    }}
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_do_data_output_on_slab_dG0(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &z,
    const unsigned int dwr_loop) {
    if (dual.data_output_trigger <= 0) return;

    // adapt trigger value for I_n output mode
    if (!dual.data_output_trigger_type_fixed) {
        dual.data_output_trigger = slab->tau_n();

        if (slab == std::prev(grid->slabs.end())) {
            dual.data_output_time_value = slab->t_n;
        }
        else {
            dual.data_output_time_value = slab->t_m;
        }
    }

    dual.data_output->set_DoF_data(
        slab->dual.fe_info->dof
    );

    auto z_trigger = std::make_shared< dealii::Vector<double> > ();
    z_trigger->reinit(
        slab->dual.fe_info->dof->n_dofs()
    );

    std::ostringstream filename;
    filename
        << "dual-dwr_loop-"
        << std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;

    double &t{dual.data_output_time_value};

    for ( ; t >= slab->t_m; t -= dual.data_output_trigger) {
        // evalute space-time solution
        z_trigger->equ(1.0, *z->x[0]);

        dual.data_output->write_data(
            filename.str(),
            z_trigger,
            t
        );
    }

    // check if data for t=0 (t_0) was written
    if (slab == grid->slabs.begin()) {
    if (dual.data_output_trigger_type_fixed) {
        const double overshoot_tol{
            std::min(slab->tau_n(), dual.data_output_trigger) * 1e-7
        };


        if ((t < slab->t_m) && (std::abs(t - slab->t_m) < overshoot_tol)) {
            // undershoot of time variable; manually set t = 0 and do data output
            t = slab->t_m;

            // evalute space-time solution
            z_trigger->equ(1.0, *z->x[0]);

            dual.data_output->write_data(
                filename.str(),
                z_trigger,
                t
            );
        }
    }}
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_do_data_output_cG1(
    const unsigned int dwr_loop,
    bool last
) {
    // set up which dwr loop(s) are allowed to make data output:
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    if (!parameter_set->data_output.dual.dwr_loop.compare("last")) {
        // output only the last (final) dwr loop
        if (last) {
            dual.data_output_dwr_loop = dwr_loop;
        }
        else {
            return;
        }
    }
    else {
        if (!parameter_set->data_output.dual.dwr_loop.compare("all")) {
            // output all dwr loops
            if (!last) {
                dual.data_output_dwr_loop = dwr_loop;
            }
            else {
                return;
            }
        }
        else {
            // output on a specific dwr loop
            if (!last) {
                dual.data_output_dwr_loop =
                    std::stoi(parameter_set->data_output.dual.dwr_loop)-1;
            }
            else {
                return;
            }
        }

    }

    if (dual.data_output_dwr_loop < 0)
        return;

    if ( static_cast<unsigned int>(dual.data_output_dwr_loop) != dwr_loop )
        return;

    DTM::pout
        << "dual solution data output: dwr loop = "
        << dual.data_output_dwr_loop
        << std::endl;

    dual.data_output_time_value = parameter_set->T;

    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());
    auto z = std::prev(dual.storage.z_cG1->end());

    unsigned int n{static_cast<unsigned int>(grid->slabs.size())};
    while (n) {
        dual_do_data_output_on_slab_cG1(slab,z,dwr_loop);

        --n;
        --slab;
        --z;
    }
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
dual_do_data_output_dG0(
    const unsigned int dwr_loop,
    bool last
) {
    // set up which dwr loop(s) are allowed to make data output:
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    if (!parameter_set->data_output.dual.dwr_loop.compare("last")) {
        // output only the last (final) dwr loop
        if (last) {
            dual.data_output_dwr_loop = dwr_loop;
        }
        else {
            return;
        }
    }
    else {
        if (!parameter_set->data_output.dual.dwr_loop.compare("all")) {
            // output all dwr loops
            if (!last) {
                dual.data_output_dwr_loop = dwr_loop;
            }
            else {
                return;
            }
        }
        else {
            // output on a specific dwr loop
            if (!last) {
                dual.data_output_dwr_loop =
                    std::stoi(parameter_set->data_output.dual.dwr_loop)-1;
            }
            else {
                return;
            }
        }
    }

    if (dual.data_output_dwr_loop < 0)
        return;

    if ( static_cast<unsigned int>(dual.data_output_dwr_loop) != dwr_loop )
        return;

    DTM::pout
        << "dual solution data output: dwr loop = "
        << dual.data_output_dwr_loop
        << std::endl;

    dual.data_output_time_value = parameter_set->T;

    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());
    auto z = std::prev(dual.storage.z_dG0->end());

    unsigned int n{static_cast<unsigned int>(grid->slabs.size())};
    while (n) {
        dual_do_data_output_on_slab_dG0(slab,z,dwr_loop);

        --n;
        --slab;
        --z;
    }
}
////////////////////////////////////////////////////////////////////////////////
// error estimation
//

template<int dim>
void
Diffusion_PU_DWR<dim>::
eta_reinit_storage() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * error indicators \f$ \eta \f$ (one per slab)
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    error_estimator.storage.primal.eta_h = std::make_shared< DTM::types::storage_data_vectors<1> > ();
    error_estimator.storage.primal.eta_h->resize(N);

    error_estimator.storage.primal.eta_h_minus = std::make_shared< DTM::types::storage_data_vectors<1> > ();
    error_estimator.storage.primal.eta_h_minus->resize(N);

    error_estimator.storage.primal.eta_k = std::make_shared<DTM::types::storage_data_vectors<1> > ();
    error_estimator.storage.primal.eta_k->resize(N);

    error_estimator.storage.primal.eta_k_minus = std::make_shared<DTM::types::storage_data_vectors<1> > ();
    error_estimator.storage.primal.eta_k_minus->resize(N);

    error_estimator.storage.adjoint.eta_h = std::make_shared< DTM::types::storage_data_vectors<1> > ();
    error_estimator.storage.adjoint.eta_h->resize(N);

    error_estimator.storage.adjoint.eta_k = std::make_shared<DTM::types::storage_data_vectors<1> > ();
    error_estimator.storage.adjoint.eta_k->resize(N);
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.primal.eta_h) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with
                // #PU-Dofs i.e.
                //
                element.x[j]->reinit(
                    slab->pu->dof->n_dofs()
                );
            }
            ++slab;
        }
    }

    if( parameter_set-> dwr.use_cG1 ){
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.primal.eta_h_minus) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with
                // #PU-Dofs i.e.
                //
                element.x[j]->reinit(
                    slab->pu->dof->n_dofs()
                );
            }
            ++slab;
        }
    }
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.primal.eta_k) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with
                // #PU-Dofs i.e.
                //
                element.x[j]->reinit(
                    slab->pu->dof->n_dofs()
                );
            }
            ++slab;
        }
    }

    if( parameter_set-> dwr.use_cG1 ){
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.primal.eta_k_minus) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with
                // #PU-Dofs i.e.
                //
                element.x[j]->reinit(
                    slab->pu->dof->n_dofs()
                );
            }
            ++slab;
        }
    }
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.adjoint.eta_h) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with
                // #PU-Dofs i.e.
                //
                element.x[j]->reinit(
                    slab->pu->dof->n_dofs()
                );
            }
            ++slab;
        }
    }

    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.adjoint.eta_k) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::Vector<double> > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->low.dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with
                // #PU-Dofs i.e.
                //
                element.x[j]->reinit(
                    slab->pu->dof->n_dofs()
                );
                Assert(
                    slab->tria->n_global_active_cells(),
                    dealii::ExcMessage("Error: slab->tria->n_global_active_cells() == 0")
                );

                // initialise dealii::Vector<double> with
                //   slab->tria->n_global_active_cells
                // components:
                element.x[j]->reinit(
                    slab->tria->n_global_active_cells()
                );
            }
            ++slab;
        }
    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
eta_interpolate_slab_dG0(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u,
    const typename DTM::types::storage_data_vectors<1>::iterator &z,
    std::shared_ptr<diffusion::dwr::estimator::Arguments> args,
    bool initial_slab,
    bool last_slab
){

    args->tm.u_kh = std::make_shared<dealii::Vector<double>>();
    args->tm.u_kh -> reinit(slab->high->dof->n_dofs());

    args->tm.z_k = std::make_shared<dealii::Vector<double>>();
    args->tm.z_k -> reinit(slab->high->dof->n_dofs());

    args->tn.u_k = std::make_shared<dealii::Vector<double>>();
    args->tn.u_k -> reinit(slab->high->dof->n_dofs());

    args->tn.u_kh = std::make_shared<dealii::Vector<double>>();
    args->tn.u_kh -> reinit(slab->high->dof->n_dofs());

    args->tn.z_k = std::make_shared<dealii::Vector<double>>();
    args->tn.z_k -> reinit(slab->high->dof->n_dofs());

    args->tn.z_kh = std::make_shared<dealii::Vector<double>>();
    args->tn.z_kh -> reinit(slab->high->dof->n_dofs());

    args->tnp1.z_kh = std::make_shared<dealii::Vector<double>>();
    args->tnp1.z_kh -> reinit(slab->high->dof->n_dofs());

    args->tnp1.u_k =  std::make_shared<dealii::Vector<double>>();
    args->tnp1.u_k -> reinit(slab->high->dof->n_dofs());

    auto low_tmp = std::make_shared<dealii::Vector<double> >();
    low_tmp ->reinit(slab->low->dof->n_dofs());

    auto high_tmp = std::make_shared<dealii::Vector<double> >();
    high_tmp ->reinit(slab->high->dof->n_dofs());

    if (parameter_set->fe.primal.high_order) {
        // do equal high order

        if ( initial_slab ){
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->high->mapping,
                *slab->high->dof,
                *function.u_0,
                *high_tmp
            );
            slab->high->constraints->distribute(*high_tmp);

            //Z_0 = Z_1
            *args->tm.z_k= *z->x[0];
        }
        else {
            //interpolate z onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->high->dof, *std::prev(z)->x[0],
                /* I_n     */ *slab->high->dof, *slab->high->constraints, *args->tm.z_k
            );

            //interpolate u onto new mesh (note that tm_u_kh is just temporary storage here)
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->high->dof, *std::prev(u)->x[0],
                /* I_n     */ *slab->high->dof, *slab->high->constraints, *high_tmp
            );
        }

        //restrict u to primal space
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof, *high_tmp,
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        //now low_tmp holds u_m in low space. Interpolate to high space for tm.u_kh
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.u_kh
        );

        *low_tmp =0;

        *args->tn.u_k = *u->x[0];

        //calculate restricted primal solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *u->x[0],
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_kh
        );

        *low_tmp =0;

        *args->tn.z_k = *z->x[0];

        //calculate restricted dual solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *z->x[0],
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_kh
        );
        *low_tmp =0;

        if (last_slab) {
            *args->tnp1.z_kh = 0;
            *args->tnp1.u_k = *u->x[0];
        }
        else {
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::next(slab)->high->dof, *std::next(z)->x[0],
                /* I_n     */ *slab->high->dof, *slab->high->constraints, *high_tmp
            );
            dealii::FETools::interpolate(
                /*high  */ *slab->high->dof,  *high_tmp,
                /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
            *high_tmp = 0;
            dealii::FETools::interpolate(
                /*low  */ *slab->low->dof,  *low_tmp,
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.z_kh
            );

            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::next(slab)->high->dof, *std::next(u)->x[0],
                /* I_n     */ *slab->high->dof, *slab->high->constraints, *args->tnp1.u_k
            );
        }
    } else if (!parameter_set->fe.dual.high_order)
    {
        // do equal low order
        if (initial_slab){
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->mapping,
                *slab->low->dof,
                *function.u_0,
                *low_tmp
            );
            slab->low->constraints->distribute(*low_tmp);

            //Z_0 = Z_1
            dealii::FETools::extrapolate(
                /*low  */ *slab->low->dof,  *z->x[0],
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.z_k
            );
        }
        else {
            //interpolate z onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->low->dof, *std::prev(z)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );

            //extrapolate
            dealii::FETools::extrapolate(
                /*low  */ *slab->low->dof,  *low_tmp,
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.z_k
            );

            *low_tmp = 0;
            //interpolate u onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->low->dof, *std::prev(u)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
        }

        //now low_tmp holds u_m in low space. Interpolate to high space for tm.u_kh
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.u_kh
        );

        //interpolate current primal solution to high space
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *u->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_kh
        );

        //extrapolate current primal solution to high space
        dealii::FETools::extrapolate(
            /*low  */ *slab->low->dof,  *u->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_k
        );

        //interpolate current dual solution to high space
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *z->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_kh
        );

        //extrapolate current dual solution to high space
        dealii::FETools::extrapolate(
            /*low  */ *slab->low->dof,  *z->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_k
        );

        if ( last_slab) {
            *args->tnp1.z_kh = 0;

            dealii::FETools::extrapolate(
                /*low  */ *slab->low->dof,  *u->x[0],
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.u_k
            );
        }
        else {
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::next(slab)->low->dof, *std::next(z)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );

            //interpolate current dual solution to high space
            dealii::FETools::interpolate(
                /*low  */ *slab->low->dof,  *low_tmp,
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.z_kh
            );
            *low_tmp = 0;
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::next(slab)->low->dof, *std::next(u)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );

            dealii::FETools::extrapolate(
                /*low  */ *slab->low->dof,  *low_tmp,
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.u_k
            );
        }
    } else{
        // do mixed order
        if ( initial_slab ){
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->mapping,
                *slab->low->dof,
                *function.u_0,
                *low_tmp
            );
            slab->low->constraints->distribute(*low_tmp);

            //Z_0 = Z_1
            *args->tm.z_k= *z->x[0];
        } else {
            //interpolate z onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->high->dof, *std::prev(z)->x[0],
                /* I_n     */ *slab->high->dof, *slab->high->constraints, *args->tm.z_k
            );

            //interpolate u onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->low->dof, *std::prev(u)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
        }

        //now low_tmp holds u_m in low space. Interpolate to high space for tm.u_kh
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.u_kh
        );


        //interpolate current primal solution to high space
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *u->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_kh
        );

        dealii::FETools::extrapolate(
            /*low  */ *slab->low->dof,  *u->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_k
        );


        *low_tmp =0;
        *args->tn.z_k = *z->x[0];
        //calculate restricted dual solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *z->x[0],
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_kh
        );

        if (last_slab) {
            *args->tnp1.z_kh = 0;

            dealii::FETools::extrapolate(
                /*low  */ *slab->low->dof,  *u->x[0],
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.u_k
            );
        }
        else {
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::next(slab)->high->dof, *std::next(z)->x[0],
                /* I_n     */ *slab->high->dof, *slab->high->constraints, *high_tmp
            );
            dealii::FETools::interpolate(
                /*high  */ *slab->high->dof,  *high_tmp,
                /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
            *args->tnp1.z_kh = 0;
            dealii::FETools::interpolate(
                /*low  */ *slab->low->dof,  *low_tmp,
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.z_kh
            );
            *low_tmp = 0;
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::next(slab)->low->dof, *std::next(u)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );

            dealii::FETools::extrapolate(
                /*low  */ *slab->low->dof,  *low_tmp,
                /*high */ *slab->high->dof, *slab->high->constraints, *args->tnp1.u_k
            );
        }
    }
}


template<int dim>
void
Diffusion_PU_DWR<dim>::
eta_interpolate_slab_cG1(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_vectors<1>::iterator &u,
    const typename DTM::types::storage_data_vectors<2>::iterator &z,
    std::shared_ptr<diffusion::dwr::estimator::Arguments> args,
    bool initial_slab,
    bool
){

    args->tm.u_kh = std::make_shared<dealii::Vector<double>>();
    args->tm.u_kh -> reinit(slab->high->dof->n_dofs());

    args->tn.u_kh = std::make_shared<dealii::Vector<double>>();
    args->tn.u_kh -> reinit(slab->high->dof->n_dofs());

    args->tm.z_k = std::make_shared<dealii::Vector<double>>();
    args->tm.z_k -> reinit(slab->high->dof->n_dofs());

    args->tn.z_k = std::make_shared<dealii::Vector<double>>();
    args->tn.z_k -> reinit(slab->high->dof->n_dofs());

    args->tn.z_kh = std::make_shared<dealii::Vector<double>>();
    args->tn.z_kh -> reinit(slab->high->dof->n_dofs());

    auto low_tmp = std::make_shared<dealii::Vector<double> >();
    low_tmp ->reinit(slab->low->dof->n_dofs());

    if (parameter_set->fe.primal.high_order) {
        // do equal high order
        *args->tm.z_k= *z->x[0];
        *args->tn.z_k= *z->x[1];

        if ( initial_slab ){
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->mapping,
                *slab->low->dof,
                *function.u_0,
                *low_tmp
            );
            slab->low->constraints->distribute(*low_tmp);

        } else {
            //interpolate u onto new mesh (note that tm_u_kh is just temporary storage here)
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->high->dof, *std::prev(u)->x[0],
                /* I_n     */ *slab->high->dof, *slab->low->constraints, *args->tm.u_kh
            );
            //restrict u to primal space
            dealii::FETools::interpolate(
                /*high  */ *slab->high->dof, *args->tm.u_kh,
                /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
            *args->tm.u_kh = 0;
        }
        //now low_tmp holds u_m in low space. Interpolate to high space for tm.u_kh
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.u_kh
        );

        *low_tmp =0;
        //calculate restricted primal solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *u->x[0],
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_kh
        );

        *low_tmp =0;
        //calculate restricted dual solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *z->x[1],
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_kh
        );

    } else if (!parameter_set->fe.dual.high_order)
    {
        // do equal low order
        if (initial_slab){
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->mapping,
                *slab->low->dof,
                *function.u_0,
                *low_tmp
            );
            slab->low->constraints->distribute(*low_tmp);
        }
        else {
            //interpolate u onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->low->dof, *std::prev(u)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
        }

        //now low_tmp holds u_m in low space. Interpolate to high space for tm.u_kh
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.u_kh
        );

        //interpolate current primal solution to high space
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *u->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_kh
        );

        //interpolate current dual solution to high space
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *z->x[1],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_kh
        );

        //extrapolate current dual solution to high space
        dealii::FETools::extrapolate(
            /*low  */ *slab->low->dof,  *z->x[1],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_k
        );

        //extrapolate current dual solution to high space
        dealii::FETools::extrapolate(
            /*low  */ *slab->low->dof,  *z->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.z_k
        );
    } else{
        // do mixed order
        *args->tm.z_k= *z->x[0];
        *args->tn.z_k= *z->x[1];


        if ( initial_slab ){
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->mapping,
                *slab->low->dof,
                *function.u_0,
                *low_tmp
            );
            slab->low->constraints->distribute(*low_tmp);

        } else {
            //interpolate u onto new mesh
            dealii::VectorTools::interpolate_to_different_mesh(
                /* I_{n-1} */ *std::prev(slab)->low->dof, *std::prev(u)->x[0],
                /* I_n     */ *slab->low->dof, *slab->low->constraints, *low_tmp
            );
        }

        //now low_tmp holds u_m in low space. Interpolate to high space for tm.u_kh
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tm.u_kh
        );

        //interpolate current primal solution to high space
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *u->x[0],
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.u_kh
        );

        *low_tmp =0;
        //calculate restricted dual solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *z->x[1],
            /*low */ *slab->low->dof, *slab->low->constraints, *low_tmp
        );
        dealii::FETools::interpolate(
            /*low  */ *slab->low->dof,  *low_tmp,
            /*high */ *slab->high->dof, *slab->high->constraints, *args->tn.z_kh
        );

    }
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
compute_pu_dof_error_indicators() {

    error_estimator.pu_dof = std::make_shared<diffusion::dwr::estimator::PUDoFErrorEstimator<dim> > ();

    error_estimator.pu_dof->set_functions (
        function.density,
        function.epsilon,
        function.f,
        function.u_0,
        function.u_E
    );

    error_estimator.pu_dof->set_parameters(
        primal_error,
        parameter_set->dwr.goal.type
    );


    auto eta_h_p_minus = error_estimator.storage.primal.eta_h->begin();
    auto eta_h_p = error_estimator.storage.primal.eta_h->begin();
    auto eta_k_p_minus = error_estimator.storage.primal.eta_k->begin();
    auto eta_k_p = error_estimator.storage.primal.eta_k->begin();

    auto eta_h_a = error_estimator.storage.adjoint.eta_h->begin();
    auto eta_k_a = error_estimator.storage.adjoint.eta_k->begin();

    if ((parameter_set->fe.dual.time_type.compare("cG") == 0) &&
        (parameter_set->fe.dual.s == 1) ){
        auto slab= grid->slabs.begin();

        auto u = primal.storage.u->begin();
        auto z = dual.storage.z_cG1->begin();

        while (slab!= grid->slabs.end()){
            auto args = std::make_shared<diffusion::dwr::estimator::Arguments>();

            eta_interpolate_slab_cG1(slab,
                 u,
                 z,
                 args,
                 (slab == grid->slabs.begin()),
                 (slab == std::prev(grid->slabs.end()))
             );

            if ( parameter_set->dwr.estimator_type.compare("split")==0 ){
                error_estimator.pu_dof->estimate_primal_split(
                    slab,
                    args,
                    eta_h_p->x[0],
                    eta_k_p->x[0]
                );
            } else {
                error_estimator.pu_dof->estimate_primal_joint(
                    slab,
                    args,
                    eta_h_p->x[0],
                    eta_k_p->x[0]
                );
            }


            ////////////////////////////////////////////////////////////////////////
            // prepare next I_n slab problem:
            //

            ++slab;
            ++u; ++z; ++eta_h_p; ++eta_k_p; ++eta_h_a; ++eta_k_a;
        }

    }
    else if ( (parameter_set->fe.dual.time_type.compare("dG") == 0) &&
              (parameter_set->fe.dual.s == 0) ){

        auto slab= grid->slabs.begin();

        auto u = primal.storage.u->begin();
        auto z = dual.storage.z_dG0->begin();

        while (slab!= grid->slabs.end()){
            auto args = std::make_shared<diffusion::dwr::estimator::Arguments>();

            eta_interpolate_slab_dG0(slab,
                u,
                z,
                args,
                (slab == grid->slabs.begin()),
                (slab == std::prev(grid->slabs.end()))
            );

            if ( parameter_set->dwr.estimator_type.compare("split")==0 ){
                if( parameter_set-> dwr.use_cG1 ){
                    error_estimator.pu_dof->estimate_primal_split_cG1(
                        slab,
                        args,
                        eta_h_p_minus->x[0],
                        eta_h_p->x[0],
                        eta_k_p_minus->x[0],
                        eta_k_p->x[0]
                    );
                }
                else {
                    error_estimator.pu_dof->estimate_primal_split(
                        slab,
                        args,
                        eta_h_p->x[0],
                        eta_k_p->x[0]
                    );

                    error_estimator.pu_dof->estimate_dual_split(
                        slab,
                        args,
                        eta_h_a->x[0],
                        eta_k_a->x[0]
                    );
                }
            } else {
                    error_estimator.pu_dof->estimate_primal_joint(
                        slab,
                        args,
                        eta_h_p->x[0],
                        eta_k_p->x[0]
                    );
            }


            ////////////////////////////////////////////////////////////////////////
            // prepare next I_n slab problem:
            //

            ++slab;
            ++u; ++z;
            ++eta_h_p_minus; ++eta_k_p_minus;
            ++eta_h_p; ++eta_k_p;
            ++eta_h_a; ++eta_k_a;
        }

    }
}
template<int dim>
void
Diffusion_PU_DWR<dim>::
compute_effectivity_index() {
    double eta_k_primal{0.};
    for ( const auto &eta_it : *error_estimator.storage.primal.eta_k ) {
        double tmp = std::accumulate(eta_it.x[0]->begin(), eta_it.x[0]->end(), 0.);
        eta_k_primal += tmp;
    }
    if ( parameter_set->dwr.use_cG1){
        for ( const auto &eta_it : *error_estimator.storage.primal.eta_k_minus ) {
            double tmp = std::accumulate(eta_it.x[0]->begin(), eta_it.x[0]->end(), 0.);
            eta_k_primal += tmp;
        }
    }

    double eta_k_adjoint{0.};
    for ( const auto &eta_it : *error_estimator.storage.adjoint.eta_k ) {
        double tmp = std::accumulate(eta_it.x[0]->begin(), eta_it.x[0]->end(), 0.);
        eta_k_adjoint += tmp;
    }
	
    double eta_h_primal{0.};
    double eta_h_adjoint{0.};
    unsigned int K_max{0};
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    auto eta_it_p{error_estimator.storage.primal.eta_h->begin()};
    auto eta_it_p_minus{error_estimator.storage.primal.eta_h_minus->begin()};
    auto eta_it_a{error_estimator.storage.adjoint.eta_h->begin()};

    auto slab{grid->slabs.begin()};
    auto ends{grid->slabs.end()};

    for ( ; slab != ends; ++ slab, ++eta_it_p, ++eta_it_p_minus, ++eta_it_a ) {
        double tmp_p = std::accumulate(eta_it_p->x[0]->begin(), eta_it_p->x[0]->end(), 0.);

        if (parameter_set->dwr.use_cG1){
            tmp_p += std::accumulate(eta_it_p_minus->x[0]->begin(), eta_it_p_minus->x[0]->end(), 0.);
        }
        eta_h_primal += parameter_set->T/(N*slab->tau_n())* tmp_p;

        double tmp_a = std::accumulate(eta_it_a->x[0]->begin(), eta_it_a->x[0]->end(), 0.);
        eta_h_adjoint += parameter_set->T/(N*slab->tau_n())* tmp_a;

        K_max = (K_max > slab->tria->n_global_active_cells()) ? K_max : slab->tria->n_global_active_cells();
    }

    double eta_k = 0.5*(eta_k_primal+eta_k_adjoint);
    if ( parameter_set->dwr.use_cG1){
        eta_k = eta_k_primal;
    }
    double eta_h = 0.5*(eta_h_primal+eta_h_adjoint);
    if ( parameter_set->dwr.use_cG1){
        eta_h = eta_h_primal;
    }
    const double eta_primal{std::abs(eta_k_primal)+std::abs(eta_h_primal)};
    const double eta_adjoint{std::abs(eta_k_adjoint)+std::abs(eta_h_adjoint)};
    double eta{std::abs(eta_k)+std::abs(eta_h)};

    const double I_eff_primal{(eta_primal/primal_error)};
    const double I_eff_adjoint{(eta_adjoint/primal_error)};
    const double I_eff{(eta/primal_error)};

    DTM::pout << "\neta_k = " << eta_k
              << "\neta_k_primal = " << eta_k_primal
              << "\neta_k_adjoint = " << eta_k_adjoint
              << "\neta_h = " << eta_h
              << "\neta_h_primal = " << eta_h_primal
              << "\neta_h_adjoint = " << eta_h_adjoint
              << "\neta   = " << eta
              << "\neta_primal   = " << eta_primal
              << "\neta_adjoint   = " << eta_adjoint
              << "\nprimal_error = " << primal_error
              << "\nI_eff = " << I_eff
              << "\nI_eff_primal = " << I_eff_primal
              << "\nI_eff_adjoint = " << I_eff_adjoint << std::endl;
	
    // push local variables to convergence_table to avoid additional costs later.
    convergence_table.add_value("N_max", N);
    convergence_table.add_value("K_max", K_max);
    convergence_table.add_value("primal_error", primal_error);
    convergence_table.add_value("eta_k", std::abs(eta_k));
    convergence_table.add_value("eta_h", std::abs(eta_h));
    convergence_table.add_value("eta", eta);
    convergence_table.add_value("I_eff", I_eff);
}

template<int dim>
void
Diffusion_PU_DWR<dim>::
refine_and_coarsen_space_time_grid() {
    // see which refinement type is needed
    if (  ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare("global")==0 ) &&
          ( parameter_set->dwr.refine_and_coarsen.time.strategy.compare("global")==0 ) )
    {
        refine_and_coarsen_space_time_grid_global();
    } else {
        refine_and_coarsen_space_time_grid_sv_dof(); //DoF based refinement
    }
}

template <int dim>
void
Diffusion_PU_DWR<dim>::
refine_and_coarsen_space_time_grid_global() {

    unsigned int K_max{0};
    auto slab{grid->slabs.begin()};
    auto ends{grid->slabs.end()};
    for ( unsigned int n{0} ; slab!= ends; ++slab, ++n)
    {
        DTM::pout << "\tn = " << n << std::endl;

        const auto n_active_cells_on_slab{slab->tria->n_global_active_cells()};
        DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
        K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;

        slab->tria->refine_global(1);
        grid->refine_slab_in_time(slab);
    }
    DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
}

template <int dim>
void
Diffusion_PU_DWR<dim>::
refine_and_coarsen_space_time_grid_joint_dof() {
    Assert(
        error_estimator.storage.primal.eta_h->size()==grid->slabs.size(),
        dealii::ExcInternalError()
    );

    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    std::vector<double> eta(N);

    // 1st loop: compute eta^n on I_n for n=1..N
    {
        auto eta_it{error_estimator.storage.primal.eta_h->begin()};
        for (unsigned n{0}; n < N; ++n, ++eta_it) {
            Assert(
                (eta_it != error_estimator.storage.primal.eta_h->end()),
                dealii::ExcInternalError()
            );

            eta[n] = std::accumulate(
                eta_it->x[0]->begin(),
                eta_it->x[0]->end(),
                0.
            );
        }
    }

    // 2nd loop: mark for time refinement
    {
        if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare(
            "global") == 0) {
            // global refinement in time (marks all I_n for refinement)
            auto slab{grid->slabs.begin()};
            auto ends{grid->slabs.end()};
            for (unsigned int n{0} ; slab != ends; ++slab, ++n) {
                Assert((n < N), dealii::ExcInternalError());
                slab->set_refine_in_time_flag();
            }
        }
        else {
            Assert(
                ((parameter_set->dwr.refine_and_coarsen.time.top_fraction >= 0.) &&
                (parameter_set->dwr.refine_and_coarsen.time.top_fraction <= 1.)),
                dealii::ExcMessage(
                    "parameter_set->dwr.refine_and_coarsen.time.top_fraction "
                    "must be in [0,1]"
                )
            );

            if (parameter_set->dwr.refine_and_coarsen.time.top_fraction > 0.) {
                std::vector<double> eta_sorted(eta);
                std::sort(eta_sorted.begin(), eta_sorted.end(),std::greater<double>());

                double threshold = 0.;
                //do Doerfler marking
                if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_fraction") == 0)
                {
                    double D_goal = std::accumulate(
                        eta.begin(),
                        eta.end(),
                        0.
                    ) * parameter_set->dwr.refine_and_coarsen.time.top_fraction;

                    double D_sum = 0.;
                    for (unsigned int n{0} ; n < N ; n++){
                        D_sum += eta_sorted[n];
                        if (D_sum >= D_goal){
                            threshold = eta_sorted[n];
                            n = N;
                        }
                    }

                } else if ( parameter_set -> dwr.refine_and_coarsen.time.strategy.compare("fixed_number" ) == 0){

                    // check if index for eta_criterium_for_mark_time_refinement is valid
                    Assert(
                        ( static_cast<int>(std::ceil(static_cast<double>(N)
                            * parameter_set->dwr.refine_and_coarsen.time.top_fraction)) ) >= 0,
                        dealii::ExcInternalError()
                    );

                    unsigned int index_for_mark_time_refinement {
                        static_cast<unsigned int> (
                            static_cast<int>(std::ceil(
                                static_cast<double>(N)
                                * parameter_set->dwr.refine_and_coarsen.time.top_fraction
                            ))
                        )
                    };

                    threshold = eta_sorted[ index_for_mark_time_refinement < N ?
                                                        index_for_mark_time_refinement : N-1 ] ;
                } else {
                    AssertThrow(
                        false,
                        dealii::ExcMessage(
                            "parameter_set->dwr.refine_and_coarsen.time.strategy unknown"
                        )
                    );
                }

                auto slab{grid->slabs.begin()};
                auto ends{grid->slabs.end()};
                for (unsigned int n{0} ; slab != ends; ++slab, ++n) {
                    Assert((n < N), dealii::ExcInternalError());

                    if (eta[n] >= threshold) {
                        slab->set_refine_in_time_flag();
                    }
                }
            }
        }
    }

    // 3rd loop execute_coarsening_and_refinement
    {
        unsigned int K_max{0};
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};
        auto eta_it{error_estimator.storage.primal.eta_h->begin()};
        for (unsigned int n{0} ; slab != ends; ++slab, ++eta_it, ++n) {
            Assert((n < N), dealii::ExcInternalError());

            Assert(
                    (eta_it != error_estimator.storage.primal.eta_h->end()),
                    dealii::ExcInternalError()
            );

            DTM::pout << "\tn = " << n << std::endl;

            const auto n_active_cells_on_slab{slab->tria->n_global_active_cells()};
            DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
            K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;

            if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                    "global") == 0 || parameter_set->dwr.refine_and_coarsen.space.top_fraction1 == 1.0) {
                    // global refinement in space
                    slab->tria->refine_global(1);
            }
            else {
                const unsigned int dofs_per_cell_pu = slab->pu->fe->dofs_per_cell;
                std::vector< unsigned int > local_dof_indices(dofs_per_cell_pu);
                unsigned int max_n = n_active_cells_on_slab *
                    parameter_set->dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells;

                typename dealii::DoFHandler<dim>::active_cell_iterator
                cell{slab->pu->dof->begin_active()},
                endc{slab->pu->dof->end()};

                dealii::Vector<double> indicators (slab->tria->n_active_cells());
                indicators = 0;
                for (unsigned int cell_no{0} ; cell != endc; ++ cell, ++cell_no){
                    cell->get_dof_indices(local_dof_indices);

                    for ( unsigned int i = 0 ; i < dofs_per_cell_pu ; i++){
                        indicators[cell_no] += (*eta_it->x[0])(local_dof_indices[i])/dofs_per_cell_pu;
                    }
                }
                if ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                                                                "RichterWick") == 0 ){
                        double threshold = eta_it->x[0]->mean_value() *
                                                           parameter_set->dwr.refine_and_coarsen.space.riwi_alpha;

                        dealii::GridRefinement::refine(
                                *slab->tria,
                                indicators,
                                threshold,
                                max_n
                        );
                } else {
                    const double top_fraction{ slab->refine_in_time ?
                        parameter_set->dwr.refine_and_coarsen.space.top_fraction1 :
                        parameter_set->dwr.refine_and_coarsen.space.top_fraction2
                    };

                    if ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                        "fixed_fraction") == 0 ){

                        dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
                            *slab->tria,
                            indicators,
                            top_fraction,
                            parameter_set->dwr.refine_and_coarsen.space.bottom_fraction,
                            max_n
                        );
                    } else if ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                                "fixed_number") == 0 ){

                        dealii::GridRefinement::refine_and_coarsen_fixed_number(
                            *slab->tria,
                            indicators,
                            top_fraction,
                            parameter_set->dwr.refine_and_coarsen.space.bottom_fraction,
                            max_n
                        );
                    }else{
                        AssertThrow(false,dealii::ExcMessage("unknown spatial refinement"));
                    }
                }
                // execute refinement in space under the conditions of mesh smoothing
                slab->tria->execute_coarsening_and_refinement();
            }

            // refine in time
            if (slab->refine_in_time) {
                grid->refine_slab_in_time(slab);
                slab->refine_in_time = false;
            }
        }

        DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
    }
}

template <int dim>
void
Diffusion_PU_DWR<dim>::
refine_and_coarsen_space_time_grid_sv_dof() {

    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    std::vector<double> eta_k(N);

    for ( unsigned int n{0} ; n < N ; ++n){
        eta_k[n] = 0;
    }
    double eta_k_global {0.};
    double eta_h_global {0.};

    // compute eta^n on I_n for n=1..N as well as global estimators
    {
        auto eta_it{error_estimator.storage.primal.eta_k->begin()};
        auto eta_it_minus{error_estimator.storage.primal.eta_k_minus->begin()};
        for (unsigned int n{0}; n < N; ++n, ++eta_it, ++eta_it_minus) {
            Assert(
                (eta_it != error_estimator.storage.primal.eta_k->end()),
                dealii::ExcInternalError()
            );
            Assert(
                (eta_it_minus != error_estimator.storage.primal.eta_k_minus->end()),
                dealii::ExcInternalError()
            );

            double eta_k_K = std::accumulate(
                eta_it->x[0]->begin(),
                eta_it->x[0]->end(),
                0.
            );
            eta_k_global += eta_k_K;
            if ( parameter_set->dwr.use_cG1){
                eta_k[n] += eta_k_K;
                if ( n < (N-1) ){
                    eta_k[n+1] += eta_k_K;
                }
                double eta_k_K_minus = std::accumulate(
                    eta_it_minus->x[0]->begin(),
                    eta_it_minus->x[0]->end(),
                    0.
                );
                eta_k_global += eta_k_K_minus;
                eta_k[n] += eta_k_K_minus;
                if ( n > 0 ){
                    eta_k[n-1] += eta_k_K_minus;
                }
            }
            else{
                eta_k[n] = std::abs(eta_k_K);
            }
        }
    }
    if ( parameter_set->dwr.use_cG1){
        for ( unsigned int n{0}; n < N ; ++n){
            eta_k[n] = std::abs(eta_k[n]);
        }
    }

    {
        auto eta_it{error_estimator.storage.primal.eta_h->begin()};
        auto eta_it_minus{error_estimator.storage.primal.eta_h_minus->begin()};
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};
        for (unsigned n{0}; n < N; ++n, ++eta_it, ++eta_it_minus ,++slab) {
            Assert(
                (eta_it != error_estimator.storage.primal.eta_h->end()),
                dealii::ExcInternalError()
            );
            Assert(
                (eta_it_minus != error_estimator.storage.primal.eta_h_minus->end()),
                dealii::ExcInternalError()
            );
            eta_h_global += parameter_set->T/(N*slab->tau_n())*
                std::accumulate(
                    eta_it->x[0]->begin(),
                    eta_it->x[0]->end(),
                    0.
                );

            if ( parameter_set->dwr.use_cG1){
                eta_h_global += parameter_set->T/(N*slab->tau_n())*
                std::accumulate(
                    eta_it_minus->x[0]->begin(),
                    eta_it_minus->x[0]->end(),
                    0.
                );
            }
        }
    }

    /*
     * Choose if temporal or spatial discretization should be refined
     * according to Algorithm 4.1 in Schmich & Vexler
     *
     */
    double equilibration_factor{2.5e10};

    // mark for temporal refinement
    if ( std::abs(eta_k_global)*equilibration_factor >= std::abs(eta_h_global))
    {
        Assert(
            ((parameter_set->dwr.refine_and_coarsen.time.top_fraction >= 0.) &&
            (parameter_set->dwr.refine_and_coarsen.time.top_fraction <= 1.)),
            dealii::ExcMessage(
                "parameter_set->dwr.refine_and_coarsen.time.top_fraction "
                "must be in [0,1]"
            )
        );

        if (parameter_set->dwr.refine_and_coarsen.time.top_fraction > 0.) {
            std::vector<double> eta_sorted(eta_k);
            std::sort(eta_sorted.begin(), eta_sorted.end(),std::greater<double>());

            double threshold = 0.;
            //do Doerfler marking
            if ( parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_fraction") == 0)
            {
                double D_goal = std::accumulate(
                    eta_k.begin(),
                    eta_k.end(),
                    0.
                ) * parameter_set->dwr.refine_and_coarsen.time.top_fraction;

                double D_sum = 0.;
                for ( unsigned int n{0} ; n < N ; n++){
                    D_sum += eta_sorted[n];
                    if ( D_sum >= D_goal){
                        threshold = eta_sorted[n];
                        n = N;
                    }
                }
            } else if ( parameter_set -> dwr.refine_and_coarsen.time.strategy.compare("fixed_number")== 0){
                // check if index for eta_criterium_for_mark_time_refinement is valid
                Assert(
                    ( static_cast<int>(std::ceil(static_cast<double>(N)
                         * parameter_set->dwr.refine_and_coarsen.time.top_fraction)) ) >= 0,
                    dealii::ExcInternalError()
                );

                unsigned int index_for_mark_time_refinement {
                    static_cast<unsigned int> (
                        static_cast<int>(std::ceil(
                            static_cast<double>(N)
                            * parameter_set->dwr.refine_and_coarsen.time.top_fraction
                        ))
                    )
                };

                threshold = eta_sorted[ index_for_mark_time_refinement < N ?
                    index_for_mark_time_refinement : N-1] ;
            } else {
                AssertThrow(
                    false,
                    dealii::ExcMessage( "parameter_set->dwr.refine_and_coarsen.time.strategy unknown" )
                );
            }


            auto slab{grid->slabs.begin()};
            auto ends{grid->slabs.end()};
            //do this as primal temporal estimator on first interval
            //is equal to zero. This refines first and second slab
            //if second slab should be refined
            if(parameter_set->dwr.use_cG1){
                if (eta_k[0]>= threshold){
                    slab->set_refine_in_time_flag();
                }
                ++slab;
                if(eta_k[1] >= threshold){
                    slab->set_refine_in_time_flag();
                }
                ++slab;
            } else{
                if ( eta_k[1] >= threshold){
                    slab->set_refine_in_time_flag();
                    ++slab;
                    slab->set_refine_in_time_flag();
                    ++slab;
                } else {
                    ++slab;
                    ++slab;
                }
            }
            for (unsigned int n{2} ; slab != ends; ++slab, ++n) {
                Assert((n < N), dealii::ExcInternalError());

                if (eta_k[n] >= threshold) {
                    slab->set_refine_in_time_flag();
                }
            }
        }
    }

    // spatial refinement
    if ( std::abs(eta_k_global) <= equilibration_factor*std::abs(eta_h_global))
    {
        unsigned int K_max{0};
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};
        auto eta_it{error_estimator.storage.primal.eta_h->begin()};
        auto eta_it_minus{error_estimator.storage.primal.eta_h_minus->begin()};

        //temporary vectors to interpolate previous and or next slab estimator parts into
        dealii::Vector<double> tmp_prev;
        tmp_prev.reinit(slab->pu->dof->n_dofs());
        tmp_prev = 0;
        dealii::Vector<double> tmp_next;

        for (unsigned int n{0} ; slab != ends; ++slab, ++eta_it, ++eta_it_minus, ++n) {
            Assert((n < N), dealii::ExcInternalError());

            Assert(
                (eta_it != error_estimator.storage.primal.eta_h->end()),
                dealii::ExcInternalError()
            );
            Assert(
                (eta_it_minus != error_estimator.storage.primal.eta_h_minus->end()),
                dealii::ExcInternalError()
            );

            DTM::pout << "\tn = " << n << std::endl;

            const auto n_active_cells_on_slab{slab->tria->n_global_active_cells()};
            DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
            K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;

            if ( parameter_set->dwr.refine_and_coarsen.space.top_fraction1 == 1.0){
                slab->tria->refine_global(1);
            }
            else {
                const unsigned int dofs_per_cell_pu = slab->pu->fe->dofs_per_cell;
                std::vector< unsigned int > local_dof_indices(dofs_per_cell_pu);

                if(parameter_set->dwr.use_cG1){
                    tmp_next.reinit(slab->pu->dof->n_dofs());

                    if ( n < (N-1)){
                        dealii::VectorTools::interpolate_to_different_mesh(
                            /* I_{n+1} */ *std::next(slab)->pu->dof, *std::next(eta_it_minus)->x[0],
                            /* I_n     */ *slab->pu->dof, *slab->pu->constraints, tmp_next
                        );
                    } else {
                        tmp_next = 0;
                    }
                }

                typename dealii::DoFHandler<dim>::active_cell_iterator
                cell{slab->pu->dof->begin_active()},
                endc{slab->pu->dof->end()};

                dealii::Vector<double> indicators (slab->tria->n_active_cells());
                indicators = 0;
                for (unsigned int cell_no{0} ; cell != endc; ++ cell, ++cell_no){
                    cell->get_dof_indices(local_dof_indices);

                    for ( unsigned int i = 0 ; i < dofs_per_cell_pu ; i++){
                        indicators[cell_no] += (*eta_it->x[0])(local_dof_indices[i]);
                        if( parameter_set->dwr.use_cG1){
                            indicators[cell_no] += tmp_prev(local_dof_indices[i]);
                            indicators[cell_no] += tmp_next(local_dof_indices[i]);
                            indicators[cell_no] += (*eta_it_minus->x[0])(local_dof_indices[i]);
                        }
                    }
                    indicators[cell_no] = std::abs(indicators[cell_no]);
                }
                if(parameter_set->dwr.use_cG1){

                    if ( n < (N-1)){
                        tmp_prev.reinit(std::next(slab)->pu->dof->n_dofs());

                        dealii::VectorTools::interpolate_to_different_mesh(
                            /* I_{n}   */ *slab->pu->dof, *eta_it->x[0],
                            /* I_{n+1} */ *std::next(slab)->pu->dof, *std::next(slab)->pu->constraints, tmp_prev
                        );
                    }
                }

                if ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                    "fixed_fraction") == 0 ){
                    const double top_fraction{ slab->refine_in_time ?
                        parameter_set->dwr.refine_and_coarsen.space.top_fraction1 :
                        parameter_set->dwr.refine_and_coarsen.space.top_fraction2
                    };

                    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
                        *slab->tria,
                        indicators,
                        top_fraction,
                        parameter_set->dwr.refine_and_coarsen.space.bottom_fraction
                    );
                } else if ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                    "fixed_number") == 0 ){
                    const double top_fraction{ slab->refine_in_time ?
                        parameter_set->dwr.refine_and_coarsen.space.top_fraction1 :
                        parameter_set->dwr.refine_and_coarsen.space.top_fraction2
                    };

                    dealii::GridRefinement::refine_and_coarsen_fixed_number(
                        *slab->tria,
                        indicators,
                        top_fraction,
                        parameter_set->dwr.refine_and_coarsen.space.bottom_fraction
                    );
                } else if ( parameter_set->dwr.refine_and_coarsen.space.strategy.compare(
                    "RichterWick") == 0 ){
                    cell =slab->pu->dof->begin_active();
                    double mean_error_indicator = eta_it->x[0]->mean_value()*
                        parameter_set->dwr.refine_and_coarsen.space.riwi_alpha;
                    for (unsigned int cell_no{0} ; cell != endc; ++ cell, ++cell_no){
                        if ( indicators[cell_no] > mean_error_indicator ){
                            cell -> set_refine_flag(
                                dealii::RefinementCase<dim>::isotropic_refinement
                            );
                        }
                    }
                }
                // execute refinement in space under the conditions of mesh smoothing
                slab->tria->execute_coarsening_and_refinement();
            }
        }
        std::cout << "K_max = " << K_max << std::endl;
    }
    //do actual refine in time loop
    {
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};
        for (; slab != ends; ++slab) {
            if (slab->refine_in_time) {
                grid->refine_slab_in_time(slab);
                slab->refine_in_time = false;
            }
        }
    }

}
////////////////////////////////////////////////////////////////////////////////
// other

template<int dim>
void
Diffusion_PU_DWR<dim>::
write_convergence_table_to_tex_file() {
    convergence_table.set_precision("primal_error", 5);
    convergence_table.set_precision("eta", 5);
    convergence_table.set_precision("I_eff", 3);

    convergence_table.set_scientific("primal_error", true);
    convergence_table.set_scientific("eta", true);
    convergence_table.set_scientific("eta_h",true);
    convergence_table.set_scientific("eta_k", true);

	
    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    // Set tex captions and formation of respective columns
    convergence_table.set_tex_caption("DWR-loop","DWR-loop");
    convergence_table.set_tex_caption("N_max","$N_{\\text{max}}$");
    convergence_table.set_tex_caption("K_max","$K_{\\text{max}}$");
    convergence_table.set_tex_caption(
            "primal_error","$\\|e\\|_{(0,T)\\times\\Omega}$"
    );
    convergence_table.set_tex_caption("eta_k", "$\\eta_k$");
    convergence_table.set_tex_caption("eta_h", "$\\eta_h$");
    convergence_table.set_tex_caption("eta","$\\eta$");
    convergence_table.set_tex_caption("I_eff","I$_{\\text{eff}}$");
    convergence_table.set_tex_format("DWR-loop","c");
    convergence_table.set_tex_format("N_max","r");
    convergence_table.set_tex_format("K_max","r");
    convergence_table.set_tex_format("primal_error","c");
    convergence_table.set_tex_format("eta_k", "c");
    convergence_table.set_tex_format("eta_h", "c");
    convergence_table.set_tex_format("eta","c");
    convergence_table.set_tex_format("I_eff","c");
	
    std::vector<std::string> new_order;
    new_order.push_back("DWR-loop");
    new_order.push_back("N_max");
    new_order.push_back("K_max");
    new_order.push_back("primal_error");
    new_order.push_back("eta");
    new_order.push_back("eta_k");
    new_order.push_back("eta_h");
    new_order.push_back("I_eff");
    convergence_table.set_column_order (new_order);

    convergence_table.evaluate_convergence_rates(
        "primal_error",
        dealii::ConvergenceTable::reduction_rate
    );
    convergence_table.evaluate_convergence_rates(
        "primal_error",
        dealii::ConvergenceTable::reduction_rate_log2
    );
	
    // write TeX/LaTeX file of the convergence table with deal.II
    {
        std::string filename = "convergence-table.tex";
        std::ofstream out(filename.c_str());
        convergence_table.write_tex(out);
    }

    // read/write TeX/LaTeX file to make pdflatex *.tex working for our headers
    {
        std::ifstream in("convergence-table.tex");

        std::string filename = "my-convergence-table.tex";
        std::ofstream out(filename.c_str());

        std::string line;
        std::getline(in, line);
        out << line << std::endl;
        // add the missing amsmath latex package
        out << "\\usepackage{amsmath}" << std::endl;

        for ( ; std::getline(in, line) ; )
            out << line << std::endl;
        out.close();
    }
}

} // namespace

#include "Diffusion_PU_DWR.inst.in"
