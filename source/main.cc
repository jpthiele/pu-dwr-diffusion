/**
 * @file main.cc
 *
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @brief DTM++.Project/DWR/DWR-Diffusion: Solve the diffusion-eq with DWR.
 * 
 * @mainpage
 * The pu-dwr-diffusion subproject simulates the diffusion equation
 * \f[
 * \rho\, \partial_t u - \nabla \cdot (\epsilon\, \nabla u) = f\,,
 * \f]
 * in the space-time cyclinder \f$ \Omega \times I \f$,
 * \f$ \Omega \subset \mathbb{R}^d \f$, \f$ I = (0,T) \f$ with \f$ T < \infty \f$,
 * equipped with appropriate inital and boundary values as given by
 * \f[ \begin{array}{r@{\,}c@{\,}l@{\,}l@{\,}l}
 * u(\boldsymbol x, 0) &=& u_0(\boldsymbol x) & \text{in} & \Omega \times \{0\}\,,\\[1ex]
 * u(\boldsymbol x, t) &=& g(\boldsymbol x, t) & \text{on} & \Gamma_D \times I\,,\\[1ex]
 * \end{array}
 * \f]
 * with the the Dirichlet boundary \f$\Gamma_D = \partial \Omega\f$,
 * using goal-oriented space-time partition-of-unity dual-weighted residual methods.
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

// DEFINES

////////////////////////////////////////////////////////////////////////////////
#define MPIX_THREADS 1
////////////////////////////////////////////////////////////////////////////////

// PROJECT includes
#include <DTM++/base/LogStream.hh>
#include <DTM++/base/Problem.hh>

#include <diffusion/parameters/ParameterHandler.hh>
#include <diffusion/Diffusion_PU_DWR.tpl.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <memory>

int main(int argc, char *argv[]) {
    // Init MPI (or MPI+X)
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, MPIX_THREADS);

    // EVALUATE wall time now.
    auto wall_time_start = MPI_Wtime();

    // Prepare DTM++ process logging to file
    DTM::pout.open();

    // Get MPI Variables
    const unsigned int MyPID(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    const unsigned int NumProc(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

    ////////////////////////////////////////////////////////////////////////////
    // Restrict usage of pu-dwr-diffusion to a single process only.
    // NOTE: we use the deal.II MPI framework to control non-distributed
    //       threading facilities precisely and to give a general approach
    //       which can be extended to a MPI-version.
    //
    // see the project pu-dwr-combustion for an example using MPI with
    // Trilinos linear algebra classes.
    //

    AssertThrow(NumProc == 1, dealii::ExcMessage("MPI mode not supported."));

    //
    ////////////////////////////////////////////////////////////////////////////

    try {
        ////////////////////////////////////////////////////////////////////////
        // Init application
        //

        // Attach deallog to process output
        dealii::deallog.attach(DTM::pout);
        dealii::deallog.depth_console(0);
        DTM::pout
                << "****************************************"
                << "****************************************"
                << std::endl;

        DTM::pout
                << "Hej, here is process " << MyPID+1 << " from " << NumProc << std::endl;

        // Check input arguments
        AssertThrow(
            !(argc < 2),
            dealii::ExcMessage (
                std::string ("===>\tUSAGE: ./pu-dwr-diffusion <Input_Parameter_File.prm>"))
        );

        // Check if the given input parameter file can be opened
        const std::string input_parameter_filename(argv[1]);
        {
            std::ifstream input_parameter_file(input_parameter_filename.c_str());
            AssertThrow(
                input_parameter_file,
                dealii::ExcMessage (
                    std::string ("===>\tERROR: Input parameter file <")
                    + input_parameter_filename + "> not found."
                )
            );
        }

        // Prepare input parameter handling:
        auto parameter_handler =
            std::make_shared< diffusion::dwr::ParameterHandler > ();
        parameter_handler->parse_input(argv[1]);

        // Get minimal set of input parameters to drive the correct simulator
        unsigned int dimension;
        {
            // Problem dimension
            dimension = static_cast<unsigned int> (
                parameter_handler->get_integer("dim")
            );
        }

        //
        ////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // Begin application
        //

        // create simulator
        std::shared_ptr< DTM::Problem > problem;

        // select simulator
        {
            switch (dimension) {
            case 2: {
                problem =
                    std::make_shared< diffusion::Diffusion_PU_DWR<2> > ();
                break;
            }

            case 3: {
                problem =
                    std::make_shared< diffusion::Diffusion_PU_DWR<3> > ();
                break;
            }

            default:
                dealii::ExcNotImplemented();
            }
        }

        DTM::pout
            << "pu-dwr-diffusion: dimension dim = " << dimension << std::endl
            << std::endl;

        DTM::pout
            << std::endl
            << "*********************************************************"
            << std::endl << std::endl;

        // run the simulator
        problem->set_input_parameters(parameter_handler);
        problem->run();

        DTM::pout << std::endl << "Goodbye." << std::endl;

        //
        // End application
        ////////////////////////////////////////////////////////////////////////////<
    }
    catch (std::exception &exc) {
        // EVALUATE program run time in terms of the consumed wall time.
        auto wall_time_end = MPI_Wtime();
        DTM::pout
            << std::endl
            << "Elapsed wall time: " << wall_time_end-wall_time_start
            << std::endl
        ;

        if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
            std::cerr
                << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl
                << "An EXCEPTION occured: Please READ the following output CAREFULLY!"
                << std::endl;

            std::cerr << exc.what() << std::endl;

            std::cerr
                << std::endl
                << "APPLICATION TERMINATED unexpectedly due to an exception."
                << std::endl << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl;
        }

        // LOG error message to individual process output file.
        DTM::pout
            << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl
            << "An EXCEPTION occured: Please READ the following output CAREFULLY!"
            << std::endl;

        DTM::pout << exc.what() << std::endl;

        DTM::pout
            << std::endl
            << "APPLICATION TERMINATED unexpectedly due to an exception."
            << std::endl << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl;

        // Close output file stream
        DTM::pout.close();

        return 1;
    }
    catch (...) {
        // EVALUATE program run time in terms of the consumed wall time.
        auto wall_time_end = MPI_Wtime();
        DTM::pout
            << std::endl
            << "Elapsed wall time: " << wall_time_end-wall_time_start
            << std::endl;

        if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
            std::cerr
                << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl
                << "An UNKNOWN EXCEPTION occured!"
                << std::endl;

            std::cerr
                << std::endl
                << "----------------------------------------"
                << "----------------------------------------"
                << std::endl << std::endl
                << "Further information:" << std::endl
                << "\tThe main() function catched an exception"
                << std::endl
                << "\twhich is not inherited from std::exception."
                << std::endl
                << "\tYou have probably called 'throw' somewhere,"
                << std::endl
                << "\tif you do not have done this, please contact the authors!"
                << std::endl << std::endl
                << "----------------------------------------"
                << "----------------------------------------"
                << std::endl;

            std::cerr
                << std::endl
                << "APPLICATION TERMINATED unexpectedly due to an exception."
                << std::endl << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl;
        }

        // LOG error message to individual process output file.
        DTM::pout
            << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl
            << "An UNKNOWN EXCEPTION occured!"
            << std::endl;

        DTM::pout
            << std::endl
            << "----------------------------------------"
            << "----------------------------------------"
            << std::endl << std::endl
            << "Further information:" << std::endl
            << "\tThe main() function catched an exception"
            << std::endl
            << "\twhich is not inherited from std::exception."
            << std::endl
            << "\tYou have probably called 'throw' somewhere,"
            << std::endl
            << "\tif you do not have done this, please contact the authors!"
            << std::endl << std::endl
            << "----------------------------------------"
            << "----------------------------------------"
            << std::endl;

        DTM::pout
            << std::endl
            << "APPLICATION TERMINATED unexpectedly due to an exception."
            << std::endl << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl;

        // Close output file stream
        DTM::pout.close();

        return 1;
    }

    // EVALUATE program run time in terms of the consumed wall time.
    auto wall_time_end = MPI_Wtime();
    DTM::pout
        << std::endl
        << "Elapsed wall time: " << wall_time_end-wall_time_start
        << std::endl;

    // Close output file stream
    DTM::pout.close();

    return 0;
}
