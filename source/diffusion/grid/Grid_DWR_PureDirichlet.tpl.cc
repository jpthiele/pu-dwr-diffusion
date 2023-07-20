/**
 * @file Grid_DWR_PureDirichlet.tpl.cc
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
#include <diffusion/grid/Grid_DWR_PureDirichlet.tpl.hh>

// DTM++ includes

// DEAL.II includes

// C++ includes

namespace diffusion {
namespace grid {

template<int dim, int spacedim>
void
Grid_DWR_PureDirichlet<dim,spacedim>::
set_boundary_indicators() {
    // set boundary indicators
    auto slab(this->slabs.begin());
    auto ends(this->slabs.end());

    for (; slab != ends; ++slab) {
        auto cell(slab->tria->begin_active());
        auto endc(slab->tria->end());

        for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
        for (unsigned int face(0); face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                cell->face(face)->set_boundary_id(
                    static_cast<dealii::types::boundary_id> (
                            diffusion::types::boundary_id::Dirichlet)
                );
            }
        }}}
    }
}

}} // namespaces

#include "Grid_DWR_PureDirichlet.inst.in"
