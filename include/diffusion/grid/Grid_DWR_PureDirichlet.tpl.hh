/**
 * @file Grid_DWR_PureDirichlet.tpl.hh
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

#ifndef __Grid_DWR_PureDirichlet_tpl_hh
#define __Grid_DWR_PureDirichlet_tpl_hh

// PROJECT includes
#include <diffusion/grid/Grid_DWR.tpl.hh>

// DEAL.II includes

// C++ includes

namespace diffusion {
namespace grid {

/**
 * Colorises the boundary \f$ \Gamma = \partial \Omega \f$ for the application
 * of a Dirichlet type boundary independently of the geometry of \f$ \Omega \f$.
 */
template<int dim, int spacedim>
class Grid_DWR_PureDirichlet : public diffusion::Grid_DWR<dim,spacedim> {
public:
    Grid_DWR_PureDirichlet(
        const std::string &Grid_Class_Options,
        const std::string &TriaGenerator,
        const std::string &TriaGenerator_Options) :
        diffusion::Grid_DWR<dim,spacedim> (TriaGenerator, TriaGenerator_Options),
        Grid_Class_Options(Grid_Class_Options) { };

    virtual ~Grid_DWR_PureDirichlet() = default;

    virtual void set_boundary_indicators();
	
private:
    const std::string Grid_Class_Options;
};

}} // namespace

#endif
