/**
 * @file ExactSolution_Selector.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
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

#include <DTM++/base/LogStream.hh>

#include <diffusion/ExactSolution/ExactSolution_Selector.tpl.hh>
#include <diffusion/ExactSolution/ExactSolutions.hh>

// C++ includes
#include <vector>

namespace diffusion {
namespace exact_solution {

template<int dim>
void
Selector<dim>::
create_function(
    const std::string &_type,
    const std::string &_options,
    std::shared_ptr< dealii::Function<dim> > &function
) const {
	
    ////////////////////////////////////////////////////////////////////////////
    // parse the input string, arguments are splitted with spaces
    //
    std::string argument;
    std::vector< std::string > options;
    for (auto &character : _options) {
        if (!std::isspace(character) && (character!='\"') ) {
            argument += character;
        }
        else {
            if (argument.size()) {
                options.push_back(argument);
                argument.clear();
            }
        }
    }

    if (argument.size()) {
        options.push_back(argument);
        argument.clear();
    }
    ////////////////////////////////////////////////////////////////////////////
    //

    DTM::pout << "* found configuration: exact_solution function = " << _type << std::endl;
    DTM::pout << "* found configuration: exact_solution options = " << std::endl;
    for (auto &option : options) {
        DTM::pout << "\t" << option << std::endl;
    }
    DTM::pout << std::endl;

    DTM::pout << "* generating function" << std::endl;
	
    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("ZeroFunction") == 0) {
        AssertThrow(
            options.size() == 0,
            dealii::ExcMessage(
                "exact_solution options invalid, "
                "please check your input file data."
            )
        );

        function =
            std::make_shared< dealii::Functions::ZeroFunction<dim> > (1);

        DTM::pout
            << "exact_solution selector: created zero function" << std::endl
            << std::endl;

        return;
    }
	
    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("ConstantFunction") == 0) {
        AssertThrow(
            options.size() == 1,
            dealii::ExcMessage(
                "exact_solution options invalid, "
                "please check your input file data."
            )
        );

        function = std::make_shared<
            dealii::Functions::ConstantFunction<dim> > (
            std::stod(options.at(0)),
            1
        );

        DTM::pout
            << "exact_solution selector: created ConstantFunction "
            << "as exact_solution function, with " << std::endl
            << "\tf(1) = " << std::stod(options.at(0)) << " . " << std::endl
            << std::endl;

        return;
    }
	
    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("ExactSolution_Simple") == 0) {
        AssertThrow(
            options.size() == 0,
            dealii::ExcMessage(
                "exact_solution options invalid, "
                "please check your input file data."
            )
        );

        function =
            std::make_shared< diffusion::exact_solution::Simple<dim> > ();

        DTM::pout
            << "exact_solution selector: created Simple solution function" << std::endl
            << std::endl;

        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("ExactSolution_Hartmann142") == 0) {
        AssertThrow(
            options.size() == 1,
            dealii::ExcMessage(
                "exact_solution options invalid, "
                "please check your input file data."
            )
        );

        function = std::make_shared< diffusion::exact_solution::Hartmann142<dim> >(
            std::stod(options.at(0)) // a
        );

        DTM::pout
            << "exact_solution selector: created Hartmann142 "
            << "as exact_solution function, with " << std::endl
            << "\ta = " << std::stod(options.at(0)) << std::endl
            << std::endl;

        return;
    }
	
    ////////////////////////////////////////////////////////////////////////////
    //
    AssertThrow(
        false,
        dealii::ExcMessage("exact_solution function unknown, please check your input file data.")
    );
}

}} //namespaces

#include "ExactSolution_Selector.inst.in"
