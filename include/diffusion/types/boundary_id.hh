/**
 * @file boundary_id.hh
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


#ifndef __boundary_id_hh
#define __boundary_id_hh

namespace diffusion {
namespace types {

enum class boundary_id : unsigned int {
    forbidden = 0,

    /**
     * Colour for marking \f$ \Gamma_D \f$ from the partition
     * \f$ \partial \Omega = \Gamma_D \cup \Gamma_N \f$ with
     * \f$ \Gamma_D \neq \emptyset \f$.
     */
    Dirichlet = 2 << 0,

    /**
     * Colour for marking \f$ \Gamma_N \f$ from the partition
     * \f$ \partial \Omega = \Gamma_D \cup \Gamma_N \f$ with
     * \f$ \Gamma_D \neq \emptyset \f$.
     */
    Neumann   = 2 << 1
};

}}

#endif
